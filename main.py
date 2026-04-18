import os
import re
import fitz  # PyMuPDF
import pytesseract
import pandas as pd
from tqdm import tqdm
from PIL import Image

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import pymorphy3

# =========================
# 0. Нормализация
# =========================

morph = pymorphy3.MorphAnalyzer()

def tokenize(text):
    words = re.findall(r'\w+', text.lower())
    return [morph.parse(w)[0].normal_form for w in words]


# =========================
# 1. Загрузка документов
# =========================

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    texts = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        # OCR если пусто
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

            if len(part) < 100:
                continue

            chunks.append({
                "text": part,
                "source": page["source"],
                "page": page["page"]
            })

    return chunks


# =========================
# 3. Hybrid Search Engine
# =========================

class SearchEngine:
    def __init__(self, chunks):
        self.texts = [c["text"] for c in chunks]
        self.meta = chunks

        print("Tokenizing...")
        self.tokenized = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

        print("Loading embedding model...")
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")

        print("Encoding...")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def search(self, query, k=10):
        # BM25
        bm25_scores = self.bm25.get_scores(tokenize(query))

        # embeddings
        q_emb = self.model.encode([query])[0]
        emb_scores = np.dot(self.embeddings, q_emb)

        # нормализация
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-6)
        emb_scores = emb_scores / (np.max(emb_scores) + 1e-6)

        # hybrid score
        scores = 0.5 * bm25_scores + 0.5 * emb_scores

        top_idx = np.argsort(scores)[::-1][:k]

        return [self.meta[i] for i in top_idx]


# =========================
# 4. Извлечение ответа
# =========================

def extract_answer(question, chunks):
    sentences = []
    metas = []

    for c in chunks:
        sents = re.split(r'(?<=[.!?])\s+', c["text"])

        for s in sents:
            if len(s) < 30:
                continue

            sentences.append(tokenize(s))
            metas.append((s, c))

    bm25 = BM25Okapi(sentences)
    scores = bm25.get_scores(tokenize(question))

    best_idx = int(np.argmax(scores))
    best_sent, meta = metas[best_idx]

    return {
        "answer": best_sent,
        "source": meta["source"],
        "page": meta["page"]
    }


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
        q = row["question"]

        found_chunks = engine.search(q, k=10)
        answer = extract_answer(q, found_chunks)

        results.append({
            "question": q,
            "answer": answer["answer"],
            "source": answer.get("source", ""),
            "page": answer.get("page", "")
        })

    out = pd.DataFrame(results)
    out.to_csv("new_answers.csv", index=False)

    print("Done! new_answers.csv saved.")


if __name__ == "__main__":
    main()
