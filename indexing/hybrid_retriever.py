class HybridRetriever:

    def __init__(self,
                 bm25,
                 embedding,
                 neo4j_driver):

        self.bm25 = bm25
        self.embedding = embedding
        self.neo4j_driver = neo4j_driver

    def graph_expand(self,
                     article_id):

        query = """
        MATCH (a:Article {article_id:$article_id})
        OPTIONAL MATCH (a)-[:REFERS_TO]->(b)

        RETURN b.law as law,
               b.article_id as article_id,
               b.text as text
        LIMIT 5
        """

        with self.neo4j_driver.session() as session:

            rows = session.run(
                query,
                article_id=article_id
            )

            return [dict(r) for r in rows]

    def retrieve(self,
                 query,
                 top_k=5):

        bm25_results = self.bm25.search(
            query,
            top_k
        )

        vector_results = self.embedding.search(
            query,
            top_k
        )

        merged = {}

        for item in bm25_results + vector_results:

            key = item["chunk_id"]

            if key not in merged:
                merged[key] = item

        final_results = list(merged.values())

        expanded = []

        for item in final_results:
            expanded.extend(
                self.graph_expand(
                    item["article_id"]
                )
            )

        return {
            "primary": final_results,
            "expanded": expanded
        }
