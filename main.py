import pandas as pd

from agents.planner import (
    choose_strategy
)

from agents.graph_agent import (
    graph_reasoning
)

from agents.synthesis_agent import (
    synthesize_answer
)

from agents.citation_agent import (
    build_citations
)

from retrieval.evidence_collector import (
    collect_evidence
)

from generation.final_answer import (
    build_output
)

from config import OUTPUT_DIR


INPUT_FILE = "questions.csv"



def process_question(question):

    strategy = choose_strategy(question)

    graph_result = graph_reasoning(question)

    entities = graph_result["entities"]

    relations = graph_result["neighbors"]

    evidence = collect_evidence(entities)

    citations = build_citations(evidence)

    answer = synthesize_answer(
        question,
        entities,
        relations,
        citations["evidence"]
    )

    graph_paths = []

    for r in relations:

        graph_paths.append(
            f'{r["source"]} -> {r["relation"]} -> {r["target"]}'
        )

    return build_output(
        question=question,
        answer=answer,
        documents=citations["documents"],
        entities=entities,
        graph_paths=graph_paths,
        evidence=citations["evidence"],
        mode=strategy
    )



def main():

    df = pd.read_csv(INPUT_FILE)

    results = []

    for _, row in df.iterrows():

        result = process_question(
            row["question"]
        )

        results.append(result)

    output = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_DIR / "results.csv",
        index=False
    )


if __name__ == "__main__":
    main()
