from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class EmbeddingIndexer:

    def __init__(self):
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )

    def build(self, documents):
        self.documents = documents

        vectors = self.model.encode(documents)

        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(np.array(vectors).astype("float32"))

    def search(self, query, top_k=5):
        q = self.model.encode([query]).astype("float32")

        distances, ids = self.index.search(q, top_k)

        return [self.documents[i] for i in ids[0]]
