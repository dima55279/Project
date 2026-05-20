from graphdb.neo4j_client import run_query

def retrieve_entities(question: str):
    query = """
    MATCH (e:Entity)
    WHERE toLower(e.id) CONTAINS toLower($question)
       OR toLower(e.description) CONTAINS toLower($question)
    RETURN e.id as entity
    LIMIT 20
    """
    rows = run_query(query, {"question": question})
    return [r["entity"] for r in rows]


def expand_neighbors(entities: list):
    if not entities:
        return []
    query = """
    MATCH (e:Entity)-[r:RELATED_TO]-(n:Entity)
    WHERE e.id IN $entities
    RETURN e.id as source, type(r) as relation, n.id as target
    LIMIT 60
    """
    return run_query(query, {"entities": entities})