# indexing/community_detection.py
from graphdb.neo4j_client import run_query
from utils.storage import save_json
from config import GRAPH_EXPORT_DIR


def build_communities():
    """Простое, но рабочее разбиение на сообщества"""
    query = """
    MATCH (e:Entity)
    RETURN e.id as entity
    """
    rows = run_query(query)
    entities = [r["entity"] for r in rows]

    # Простое чанкирование + можно потом улучшить через Louvain
    communities = {}
    chunk_size = 25
    for i in range(0, len(entities), chunk_size):
        cid = str(i // chunk_size)
        communities[cid] = entities[i:i + chunk_size]

    save_json(GRAPH_EXPORT_DIR / "communities.json", communities)
    return communities