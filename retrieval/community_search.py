from utils.storage import load_json

from config import GRAPH_EXPORT_DIR


communities = load_json(
    GRAPH_EXPORT_DIR / "communities.json"
)

summaries = load_json(
    GRAPH_EXPORT_DIR / "summaries.json"
)



def retrieve_communities(entities):

    matched = []

    for cid, ents in communities.items():

        if any(e in ents for e in entities):

            matched.append({
                "community": cid,
                "summary": summaries[cid]
            })

    return matched
