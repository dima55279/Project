import os
import re
import fitz
import pytesseract
import pandas as pd
from tqdm import tqdm
from PIL import Image

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import pymorphy3
import ollama

# =========================
# Нормализация
# =========================

morph = pymorphy3.MorphAnalyzer()

def tokenize(text):
    words = re.findall(r'\w+', text.lower())
    return [morph.parse(w)[0].normal_form for w in words]


# =========================
# PDF
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
# Чанкинг
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
# SEARCH + RERANK
# =========================

class SearchEngine:
    def __init__(self, chunks):
        self.texts = [c["text"] for c in chunks]
        self.meta = chunks

        self.tokenized = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

        self.model = SentenceTransformer("intfloat/multilingual-e5-small")
        self.embeddings = self.model.encode(self.texts)

        # 🔥 лучше для русского
        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    def search(self, query, k=20):
        bm25_scores = self.bm25.get_scores(tokenize(query))

        q_emb = self.model.encode([query])[0]
        emb_scores = np.dot(self.embeddings, q_emb)

        bm25_scores /= (bm25_scores.max() + 1e-6)
        emb_scores /= (np.max(emb_scores) + 1e-6)

        scores = 0.5 * bm25_scores + 0.5 * emb_scores

        top_idx = np.argsort(scores)[::-1][:k]
        candidates = [self.meta[i] for i in top_idx]

        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

        return [r[0] for r in reranked[:5]]


# =========================
# LLM ответ
# =========================

def generate_answer(question, chunks):
    context = "\n\n".join([c["text"] for c in chunks])

    prompt = f"""
Ты помощник, отвечающий строго по документам.

Контекст:
{context}

Вопрос:
{question}

Ответь кратко и по делу. Если ответа нет — скажи "нет информации".
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# =========================
# MAIN
# =========================

def main():
    docs_path = "data/docs"
    questions_path = "data/questions.csv"

    all_pages = []

    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            all_pages.extend(extract_text_from_pdf(os.path.join(docs_path, file)))

    chunks = split_into_chunks(all_pages)

    engine = SearchEngine(chunks)

    df = pd.read_csv(questions_path)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = row["question"]

        found = engine.search(q)
        answer = generate_answer(q, found)

        results.append({
            "question": q,
            "answer": answer
        })

    pd.DataFrame(results).to_csv("answers_llm.csv", index=False)

    print("DONE")


if __name__ == "__main__":
    main()
