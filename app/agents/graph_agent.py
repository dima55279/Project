from app.db.neo4j_client import Neo4jClient


class GraphAgent:

    def __init__(self):
        self.db = Neo4jClient()

    def retrieve(self,
                 terms,
                 limit=20):

        query = """
        MATCH (a:Article)
        WHERE any(term IN $terms
                  WHERE toLower(a.text)
                  CONTAINS term)

        RETURN a.text as text,
               a.document as document,
               a.article_number as article

        LIMIT $limit
        """

        result = self.db.execute(
            query,
            {
                "terms": terms,
                "limit": limit
            }
        )

        return [dict(r) for r in result]
