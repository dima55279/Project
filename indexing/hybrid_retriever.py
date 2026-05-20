class HybridRetriever:

    def __init__(self,
                 bm25,
                 embedding,
                 neo4j_driver):

        self.bm25 = bm25
        self.embedding = embedding
        self.neo4j_driver = neo4j_driver

    def retrieve(self, query):

        bm25_results = self.bm25.search(query)
        vector_results = self.embedding.search(query)
        graph_results = self.graph_search(query)

        return {
            "bm25": bm25_results,
            "vector": vector_results,
            "graph": graph_results
        }

    def graph_search(self, query):

        cypher = """
        MATCH (a:Article)
        WHERE toLower(a.text)
        CONTAINS toLower($query)
        OPTIONAL MATCH (a)-[:REFERS_TO]->(b)
        RETURN a.text,
               a.law,
               collect(b.article_id)
        LIMIT 5
        """

        with self.neo4j_driver.session() as session:
            rows = session.run(cypher, query=query)
            return [dict(r) for r in rows]
