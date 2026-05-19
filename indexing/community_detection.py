from graphdb.neo4j_client import run_query



def build_communities():

    query = """
    MATCH (e:Entity)

    RETURN e.id as entity
    """

    rows = run_query(query)

    entities = [r["entity"] for r in rows]

    communities = {}

    chunk_size = 20

    cid = 0

    for i in range(0, len(entities), chunk_size):

        communities[str(cid)] = entities[i:i + chunk_size]

        cid += 1

    return communities
