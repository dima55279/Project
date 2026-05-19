import json



def build_output(
    question,
    answer,
    documents,
    entities,
    graph_paths,
    evidence,
    mode
):

    return {
        "question": question,
        "answer": answer,
        "document": json.dumps(
            documents,
            ensure_ascii=False
        ),
        "supporting_entities": json.dumps(
            entities,
            ensure_ascii=False
        ),
        "graph_paths": json.dumps(
            graph_paths,
            ensure_ascii=False
        ),
        "evidence": json.dumps(
            evidence,
            ensure_ascii=False
        ),
        "reasoning_mode": mode
    }
