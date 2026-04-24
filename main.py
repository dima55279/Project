import os
import re
import fitz
import pytesseract
import pandas as pd
from tqdm import tqdm
from PIL import Image

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import pymorphy3

from functools import lru_cache
from openai import OpenAI
import pickle

# =========================
# 🔥 API клиент
# =========================

client = OpenAI(
    base_url="https://polza.ai/api/v1",
    api_key="pza_D5xEW88zif5GhGYdvkmDYT3m_bwV_Utb",
)

EMBED_MODEL = "openai/text-embedding-3-large"  # можно заменить на large
EMBED_CACHE_PATH = "embeddings_cache.pkl"

# =========================
# 0. Нормализация
# =========================

morph = pymorphy3.MorphAnalyzer()

@lru_cache(maxsize=100000)
def normalize_word(word):
    return morph.parse(word)[0].normal_form

def tokenize(text):
    words = re.findall(r'\w+', text.lower())
    return [normalize_word(w) for w in words]


# =========================
# 🔥 Embeddings
# =========================

def get_embeddings_batch(texts):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [np.array(e.embedding) for e in response.data]


# =========================
# 🔥 Query Expansion
# =========================

def expand_query(query):
    try:
        prompt = f"""
Переформулируй вопрос для поиска по юридическим документам.
Дай 3 варианта.

Вопрос:
{query}
"""

        response = client.chat.completions.create(
            model="openai/gpt-5.4-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        text = response.choices[0].message.content

        variants = [
            v.strip("-• ").strip()
            for v in text.split("\n")
            if len(v.strip()) > 5
        ]

        return [query] + variants

    except:
        return [query]


# =========================
# 1. PDF
# =========================

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    texts = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        if not text.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="rus")

        texts.append({
            "text": text,
            "page": page_num,
            "source": os.path.basename(path)
        })

    return texts


# =========================
# 2. Чанкинг
# =========================

def split_into_chunks(pages):
    chunks = []

    for page in pages:
        parts = re.split(r'\n{2,}', page["text"])

        for part in parts:
            part = part.strip()

            if len(part) < 150:
                continue

            chunks.append({
                "text": part,
                "source": page["source"],
                "page": page["page"]
            })

    return chunks


# =========================
# 3. Search Engine
# =========================

class SearchEngine:
    def __init__(self, chunks):
        self.texts = [c["text"] for c in chunks]
        self.meta = chunks

        print("Tokenizing...")
        self.tokenized = [tokenize(t) for t in tqdm(self.texts)]

        self.bm25 = BM25Okapi(self.tokenized)

        # =========================
        # 🔥 Embeddings (с кешем)
        # =========================

        if os.path.exists(EMBED_CACHE_PATH):
            print("Loading embeddings from cache...")
            with open(EMBED_CACHE_PATH, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            print("Encoding with OpenAI embeddings...")

            all_embeddings = []
            batch_size = 64

            for i in tqdm(range(0, len(self.texts), batch_size)):
                batch = self.texts[i:i+batch_size]
                emb = get_embeddings_batch(batch)
                all_embeddings.extend(emb)

            self.embeddings = np.vstack(all_embeddings)

            with open(EMBED_CACHE_PATH, "wb") as f:
                pickle.dump(self.embeddings, f)

        print("Loading reranker...")
        self.reranker = CrossEncoder(
            "jinaai/jina-reranker-v2-base-multilingual"
        )

    def search(self, query, k=60):
        queries = expand_query(query)

        all_scores = np.zeros(len(self.texts))

        for q in queries:
            q_tokens = tokenize(q)

            bm25_scores = self.bm25.get_scores(q_tokens)

            q_emb = get_embeddings_batch([q])[0]
            emb_scores = np.dot(self.embeddings, q_emb)

            bm25_scores /= (bm25_scores.max() + 1e-6)
            emb_scores /= (np.max(emb_scores) + 1e-6)

            all_scores += 0.5 * bm25_scores + 0.5 * emb_scores

        top_idx = np.argpartition(all_scores, -k)[-k:]
        candidates = [self.meta[i] for i in top_idx]

        pairs = [(query, c["text"]) for c in candidates]

        rerank_scores = self.reranker.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False
        )

        reranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [r[0] for r in reranked[:10]]


# =========================
# 4. Источники
# =========================

def collect_sources(chunks):
    return ", ".join(
        sorted({f'{c["source"]}:page_{c["page"]}' for c in chunks})
    )


# =========================
# 5. LLM
# =========================

def generate_answer(question, chunks):
    context = "\n\n".join([c["text"][:1500] for c in chunks])

    prompt = f"""
Ты помощник, работающий с юридическими документами.

Контекст:
{context}

Вопрос:
{question}

Инструкция:
- Найди ответ в тексте
- Если есть частичный ответ — дай его
- Если нет информации — напиши "нет в документах"
- Не выдумывай

Ответ:
"""

    response = client.chat.completions.create(
        model="openai/gpt-5.4-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


# =========================
# 6. MAIN
# =========================

def main():
    docs_path = "data/docs"
    questions_path = "data/questions.csv"

    all_pages = []

    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            all_pages.extend(
                extract_text_from_pdf(os.path.join(docs_path, file))
            )

    chunks = split_into_chunks(all_pages)

    print(f"Chunks: {len(chunks)}")

    engine = SearchEngine(chunks)

    df = pd.read_csv(questions_path)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering"):
        q = row["question"]

        found = engine.search(q)
        answer = generate_answer(q, found)
        sources = collect_sources(found)

        results.append({
            "question": q,
            "answer": answer,
            "sources": sources
        })

    pd.DataFrame(results).to_csv("answers_GPT_v2.csv", index=False)

    print("DONE")


if __name__ == "__main__":
    main()