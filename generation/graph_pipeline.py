import json
import pickle

from retrieval.entity_retriever import (
    find_relevant_entities
)

from retrieval.graph_retriever import (
    retrieve_subgraph
)

from retrieval.community_retriever import (
    retrieve_community_summaries
)

from retrieval.hybrid_retriever import (
    retrieve_chunks
)

from retrieval.context_builder import (
    build_context
)

from generation.generator import (
    generate_answer
)

from config import GRAPH_PATH

with open(GRAPH_PATH, "rb") as f:
    graph = pickle.load(f)


def run_graphrag(question):
    entities = find_relevant_entities(
        graph,
        question
    )
    subgraph = retrieve_subgraph(
        graph,
        entities
    )
    summaries = retrieve_community_summaries(
        entities
    )
    chunks = retrieve_chunks(question)
    context = build_context(
        query=question,
        entities=entities,
        subgraph=subgraph,
        summaries=summaries,
        chunks=chunks
    )
    answer = generate_answer(
        question,
        context
    )
    return {
        "question": question,
        "entities": entities,
        "answer": answer
    }
