import networkx as nx

def build_context(
    query,
    entities,
    subgraph,
    summaries,
    chunks
):
    graph_relations = []

    for u, v, data in subgraph.edges(data=True):
        graph_relations.append(
            f"{u} --[{data.get('relation')}]--> {v}"
        )

    chunk_text = "\n\n".join([
        c.page_content
        for c in chunks[:10]
    ])

    summary_text = "\n\n".join([
        s["summary"]
        for s in summaries
    ])

    context = f"""
GLOBAL COMMUNITY CONTEXT:
{summary_text}

ENTITIES:
{entities}

GRAPH RELATIONSHIPS:
{graph_relations}

RELEVANT CHUNKS:
{chunk_text}
"""

    return context
