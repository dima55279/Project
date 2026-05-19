from neo4j import GraphDatabase

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD
)


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)



def run_query(query, params=None):

    with driver.session() as session:

        result = session.run(
            query,
            params or {}
        )

        return [dict(r) for r in result]
