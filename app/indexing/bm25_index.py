from rank_bm25 import BM25Okapi
import pickle
import os


class BM25Indexer:

    def __init__(self):
        self.corpus = []
        self.metadata = []
        self.bm25 = None

    def add_document(self,
                     text,
                     metadata):

        tokens = text.lower().split()

        self.corpus.append(tokens)
        self.metadata.append(metadata)

    def build(self):
        self.bm25 = BM25Okapi(self.corpus)

    def search(self,
               query,
               top_k=10):

        tokens = query.lower().split()

        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            zip(scores, self.metadata),
            reverse=True,
            key=lambda x: x[0]
        )

        return ranked[:top_k]

    def save(self, path="bm25.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path="bm25.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
