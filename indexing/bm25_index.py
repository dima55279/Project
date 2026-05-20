from rank_bm25 import BM25Okapi
import pickle


class BM25Indexer:

    def __init__(self,
                 documents):

        self.documents = documents

        tokenized = [
            d["text"].split()
            for d in documents
        ]

        self.bm25 = BM25Okapi(tokenized)

    def search(self,
               query,
               top_k=10):

        scores = self.bm25.get_scores(
            query.split()
        )

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                **doc,
                "score": float(score)
            }
            for doc, score in ranked[:top_k]
        ]

    def save(self,
             path):

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):

        with open(path, "rb") as f:
            return pickle.load(f)
