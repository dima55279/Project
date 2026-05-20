# main.py
import pandas as pd
from tqdm import tqdm
from rich.console import Console

from agents.planner import choose_strategy
from agents.graph_agent import graph_reasoning
from agents.synthesis_agent import synthesize_answer
from agents.citation_agent import build_citations
from generation.final_answer import build_output
from config import OUTPUT_DIR

console = Console()
INPUT_FILE = "questions_test.csv"


def process_question(question: str):
    console.print(f"\n[bold cyan]→ Вопрос:[/bold cyan] {question}")

    strategy = choose_strategy(question)
    graph_result = graph_reasoning(question)

    entities = graph_result.get("entities", [])
    evidence_raw = graph_result.get("evidence", [])

    citations = build_citations(evidence_raw)

    # Подготовка фактов
    facts = []
    for item in evidence_raw:
        if item.get("facts"):
            facts.extend([{"statement": f} for f in item["facts"] if f])

    # Генерация ответа
    answer = synthesize_answer(
        question=question,
        entities=[{"id": e} for e in entities],
        facts=facts,
        evidence=citations.get("evidence", [])
    )

    return build_output(
        question=question,
        answer=answer.strip(),
        documents=citations.get("documents", [])
    )


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Загружено вопросов: {len(df)}")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка вопросов"):
        try:
            result = process_question(row["question"])
            results.append(result)
        except Exception as e:
            console.print(f"[red]Ошибка: {e}[/red]")
            results.append({
                "question": row["question"],
                "answer": "Ошибка при обработке вопроса.",
                "document": "[]"
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(results)
    
    # Оставляем ТОЛЬКО 3 колонки
    output_df = output_df[["question", "answer", "document"]]
    
    output_df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    console.print(f"\n[bold green]✅ Готово! Результат сохранён: {OUTPUT_DIR / 'results.csv'}[/bold green]")
    console.print(f"   Колонки: question, answer, document")


if __name__ == "__main__":
    main()