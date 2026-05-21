# app/agents/graph_agent.py
from app.db.neo4j_client import Neo4jClient


class GraphAgent:
    def __init__(self):
        self.db = Neo4jClient()   # теперь singleton

    def retrieve(self, terms: list, limit: int = 20):
        if not terms:
            return []

        query = """
        MATCH (a:Article)
        WHERE any(term IN $terms 
                  WHERE toLower(a.text) CONTAINS toLower(term))
        RETURN 
            a.text AS text,
            a.document AS document,
            a.article_number AS article
        LIMIT $limit
        """

        return self.db.execute(query, {
            "terms": terms,
            "limit": limit
        })
