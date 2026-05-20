class RetrievalTools:

    def __init__(self, retriever):
        self.retriever = retriever

    def bm25_lookup(self, query):
        return self.retriever.bm25.search(query)

    def vector_lookup(self, query):
        return self.retriever.embedding.search(query)

    def graph_lookup(self, query):
        return self.retriever.graph_search(query)
