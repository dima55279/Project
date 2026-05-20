from retrieval.graph_search import retrieve_entities, expand_neighbors
from retrieval.evidence_collector import collect_evidence


def graph_reasoning(question: str):
    entities = retrieve_entities(question)
    
    if not entities:
        entities = retrieve_entities_fallback(question)
    
    neighbors = expand_neighbors(entities)
    evidence = collect_evidence(entities)
    
    return {
        "entities": entities,
        "neighbors": neighbors,
        "evidence": evidence
    }


def retrieve_entities_fallback(question: str):
    from graphdb.neo4j_client import run_query
    query = """
    MATCH (e:Entity)
    WHERE toLower(e.id) CONTAINS toLower($term)
    RETURN e.id as entity
    LIMIT 15
    """
    result = run_query(query, {"term": question})
    return [r["entity"] for r in result]