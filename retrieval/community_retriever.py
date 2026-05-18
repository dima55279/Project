import json

from config import (
    COMMUNITIES_PATH,
    SUMMARIES_PATH
)

with open(COMMUNITIES_PATH, "r", encoding="utf-8") as f:
    communities = json.load(f)

with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
    summaries = json.load(f)

def retrieve_community_summaries(entities):
    matched = []

    for cid, nodes in communities.items():

        for entity in entities:
            
            if entity in nodes:
                matched.append({
                    "community": cid,
                    "summary": summaries.get(cid, "")
                })

    return matched
