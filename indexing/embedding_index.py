from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class EmbeddingIndexer:

    def __init__(self,
                 model_name="intfloat/multilingual-e5-base"):

        self.model = SentenceTransformer(
            model_name
        )

    def build(self,
              documents):

        self.documents = documents

        texts = [
            d["text"]
            for d in documents
        ]

        vectors = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.index = faiss.IndexFlatL2(
            vectors.shape[1]
        )

        self.index.add(
            np.array(vectors).astype("float32")
        )

    def search(self,
               query,
               top_k=10):

        q = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, ids = self.index.search(
            np.array(q).astype("float32"),
            top_k
        )

        results = []

        for i, idx in enumerate(ids[0]):

            item = self.documents[idx]

            results.append({
                **item,
                "score": float(distances[0][i])
            })

        return results

    def save(self,
             path):

        faiss.write_index(
            self.index,
            path
        )

    def load(self,
             path,
             documents):

        self.index = faiss.read_index(path)
        self.documents = documents
