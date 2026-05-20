from graphdb.neo4j_client import (
    run_query
)


def collect_evidence(entities):

    if not entities:

        return []

    query = """
    MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)

    WHERE e.id IN $entities

    RETURN DISTINCT
        d.name as document,
        d.content as content
    LIMIT 10
    """

    rows = run_query(query, {
        "entities": entities
    })

    return rows
