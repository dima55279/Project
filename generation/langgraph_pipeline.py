from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    END
)

from generation.graph_pipeline import (
    local_search,
    global_search
)

from generation.router import (
    route_query
)


class GraphState(TypedDict):
    question: str
    route: str
    answer: str
    entities: list


# ROUTER

def router_node(state):
    route = route_query(
        state["question"]
    )
    return {
        "route": route
    }


# LOCAL SEARCH

def local_node(state):
    result = local_search(
        state["question"]
    )
    return {
        "answer": result["answer"],
        "entities": result["entities"]
    }


# GLOBAL SEARCH

def global_node(state):
    result = global_search(
        state["question"]
    )
    return {
        "answer": result["answer"],
        "entities": result["entities"]
    }


# ROUTING

def routing(state):
    return state["route"]


# BUILD GRAPH

graph = StateGraph(GraphState)

graph.add_node("router", router_node)
graph.add_node("local", local_node)
graph.add_node("global", global_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    routing,
    {
        "local": "local",
        "global": "global"
    }
)

graph.add_edge("local", END)
graph.add_edge("global", END)

app = graph.compile()
