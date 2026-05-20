import json


def build_output(
    question,
    answer,
    documents
):

    return {
        "question": question,
        "answer": answer,
        "document": ", ".join(
            list(set(documents))
        ) if documents else "-"
    }