from retrieval.graph_search import (
    retrieve_entities,
    expand_neighbors
)



def graph_reasoning(question):

    entities = retrieve_entities(question)

    neighbors = expand_neighbors(entities)

    return {
        "entities": entities,
        "neighbors": neighbors
    }
