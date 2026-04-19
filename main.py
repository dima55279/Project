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

# =========================
# 0. Нормализация
# =========================

morph = pymorphy3.MorphAnalyzer()

def tokenize(text):
    words = re.findall(r'\w+', text.lower())
    return [morph.parse(w)[0].normal_form for w in words]


# =========================
# 1. Загрузка PDF
# =========================

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    texts = []

    for page_num, page in tqdm(enumerate(doc), total=len(doc), desc=f"OCR {os.path.basename(path)}"):
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
# 2. Чанкинг (лучше)
# =========================

def split_into_chunks(pages):
    chunks = []

    for page in tqdm(pages, desc="Chunking"):
        text = page["text"]

        parts = re.split(r'\n{2,}', text)

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
# 3. Hybrid + Reranker
# =========================

class SearchEngine:
    def __init__(self, chunks):
        self.texts = [c["text"] for c in chunks]
        self.meta = chunks

        print("Tokenizing...")
        self.tokenized = [tokenize(t) for t in tqdm(self.texts)]
        self.bm25 = BM25Okapi(self.tokenized)

        print("Embedding model...")
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")

        print("Encoding...")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)

        print("Loading reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def search(self, query, k=20):
        # BM25
        bm25_scores = self.bm25.get_scores(tokenize(query))

        # Embeddings
        q_emb = self.model.encode([query])[0]
        emb_scores = np.dot(self.embeddings, q_emb)

        # normalize
        bm25_scores /= (bm25_scores.max() + 1e-6)
        emb_scores /= (np.max(emb_scores) + 1e-6)

        scores = 0.5 * bm25_scores + 0.5 * emb_scores

        top_idx = np.argsort(scores)[::-1][:k]

        candidates = [self.meta[i] for i in top_idx]

        # 🔥 reranking
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

        return [r[0] for r in reranked[:5]]


# =========================
# 4. Ответ (лучшее предложение)
# =========================

def extract_answer(question, chunks):
    best = None
    best_score = -1

    for c in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', c["text"])

        for s in sentences:
            if len(s) < 30:
                continue

            score = len(set(tokenize(question)) & set(tokenize(s)))

            if score > best_score:
                best_score = score
                best = (s, c)

    if best:
        s, meta = best
        return {
            "answer": s,
            "source": meta["source"],
            "page": meta["page"]
        }

    return {"answer": "нет ответа"}


# =========================
# 5. MAIN
# =========================

def main():
    docs_path = "data/docs"
    questions_path = "data/questions.csv"

    print("Loading documents...")
    all_pages = []

    pdfs = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]

    for f in tqdm(pdfs):
        pages = extract_text_from_pdf(os.path.join(docs_path, f))
        all_pages.extend(pages)

    chunks = split_into_chunks(all_pages)

    print(f"Chunks: {len(chunks)}")

    engine = SearchEngine(chunks)

    df = pd.read_csv(questions_path)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Answering"):
        q = row["question"]

        found = engine.search(q)
        ans = extract_answer(q, found)

        results.append({
            "question": q,
            "answer": ans["answer"],
            "source": ans.get("source", ""),
            "page": ans.get("page", "")
        })

    pd.DataFrame(results).to_csv("answers_final.csv", index=False)

    print("DONE")


if __name__ == "__main__":
    main()
