from neo4j import GraphDatabase


class Neo4jClient:

    def __init__(self,
                 uri="bolt://localhost:7687",
                 user="neo4j",
                 password="password"):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

    def close(self):
        self.driver.close()

    def execute(self, query, params=None):
        with self.driver.session() as session:
            return session.run(query, params or {})
