import community.community_louvain as community_louvain

def detect_communities(graph):
    undirected = graph.to_undirected()
    partition = community_louvain.best_partition(
        undirected
    )
    communities = {}

    for node, community_id in partition.items():
        communities.setdefault(
            community_id,
            []
        ).append(node)

    return communities
