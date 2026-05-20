from graphdb.neo4j_client import run_query

def collect_evidence(entities):
    query = """
    MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
    WHERE e.id IN $entities
    OPTIONAL MATCH (f:Fact)-[:MENTIONS]->(e)
    RETURN DISTINCT
        d.name as document,
        d.content as content,
        collect(DISTINCT e.id) as entities,
        collect(DISTINCT f.statement) as facts
    """
    return run_query(query, {"entities": entities})