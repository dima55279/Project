import os
import re
import fitz  # PyMuPDF
import pytesseract
import pandas as pd
from tqdm import tqdm
from PIL import Image

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# pytesseract.pytesseract.tesseract_cmd = r"F:\Tesseract\tesseract.exe"

# =========================
# 1. Загрузка документов
# =========================

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    texts = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        # если пусто — OCR
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
# 2. Чанкинг (по пунктам)
# =========================

def split_into_chunks(pages):
    chunks = []

    for page in pages:
        text = page["text"]

        if not text:
            continue

        parts = re.split(r"(п\.\s?\d+(?:\.\d+)*)", text)

        for part in parts:
            if not part:
                continue

            part = part.strip()

            if len(part) < 50:
                continue

            chunks.append({
                "text": part,
                "source": page["source"],
                "page": page["page"]
            })

    return chunks


# =========================
# 3. Индексация
# =========================

class SearchEngine:
    def __init__(self, chunks):
        self.texts = [c["text"] for c in chunks]
        self.meta = chunks

        print("Loading embedding model...")
        self.model = SentenceTransformer("intfloat/multilingual-e5-base")

        print("Embedding...")
        embeddings = self.model.encode(self.texts, show_progress_bar=True)

        self.embeddings = np.array(embeddings).astype("float32")

        # FAISS
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        # BM25
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=10):
        # semantic
        q_emb = self.model.encode([query]).astype("float32")
        _, idxs = self.index.search(q_emb, k)

        semantic_results = [self.meta[i] for i in idxs[0]]

        # BM25
        scores = self.bm25.get_scores(query.lower().split())
        top_bm25 = np.argsort(scores)[::-1][:k]
        bm25_results = [self.meta[i] for i in top_bm25]

        # объединение
        results = semantic_results + bm25_results

        # deduplicate
        seen = set()
        final = []
        for r in results:
            key = r["text"][:100]
            if key not in seen:
                seen.add(key)
                final.append(r)

        return final[:k]


# =========================
# 4. Извлечение ответа
# =========================

def extract_answer(question, chunks):
    # разбиваем на предложения
    sentences = []

    for c in chunks:
        sents = sent_tokenize(c["text"])
        for s in sents:
            sentences.append({
                "sentence": s,
                "meta": c
            })

    # простая релевантность
    best = None
    best_score = 0

    q_words = set(question.lower().split())

    for s in sentences:
        s_words = set(s["sentence"].lower().split())
        score = len(q_words & s_words)

        if score > best_score:
            best_score = score
            best = s

    if best:
        return {
            "answer": best["sentence"],
            "source": best["meta"]["source"],
            "page": best["meta"]["page"]
        }

    return {"answer": "нет в документе"}


# =========================
# 5. Основной pipeline
# =========================

def main():
    docs_path = "data/docs"
    questions_path = "data/questions.csv"

    print("Loading documents...")
    all_pages = []

    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            pages = extract_text_from_pdf(os.path.join(docs_path, file))
            all_pages.extend(pages)

    print("Chunking...")
    chunks = split_into_chunks(all_pages)

    print(f"Total chunks: {len(chunks)}")

    print("Building search engine...")
    engine = SearchEngine(chunks)

    print("Loading questions...")
    df = pd.read_csv(questions_path)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = row["question"]  # поправь под свой CSV

        found_chunks = engine.search(q, k=10)
        answer = extract_answer(q, found_chunks)

        results.append({
            "question": q,
            "answer": answer["answer"],
            "source": answer.get("source", ""),
            "page": answer.get("page", "")
        })

    out = pd.DataFrame(results)
    out.to_csv("answers.csv", index=False)

    print("Done! answers.csv saved.")


if __name__ == "__main__":
    main()