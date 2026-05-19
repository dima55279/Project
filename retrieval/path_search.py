from graphdb.neo4j_client import run_query



def shortest_paths(entity_a, entity_b):

    query = """
    MATCH p = shortestPath(
        (a:Entity {id:$a})-[*..5]-(b:Entity {id:$b})
    )

    RETURN p
    """

    return run_query(query, {
        "a": entity_a,
        "b": entity_b
    })
