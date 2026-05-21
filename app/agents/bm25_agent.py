from app.indexing.bm25_index import BM25Indexer


class BM25Agent:

    def __init__(self):
        self.index = BM25Indexer.load()

    def retrieve(self,
                 question,
                 top_k=10):

        return self.index.search(question, top_k)
