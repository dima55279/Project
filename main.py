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

from tqdm import tqdm

INPUT_FILE = "questions_test.csv"


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
        documents=citations["documents"]
    )



def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Загружено вопросов: {len(df)}")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка вопросов", unit="q"):
        result = process_question(row["question"])
        results.append(result)

    output = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_DIR / "results.csv", index=False)

    print(f"✅ Готово! Результаты сохранены в {OUTPUT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
