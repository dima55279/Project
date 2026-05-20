import time
from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate
from utils.llm import llm

console = Console()

PROMPT = ChatPromptTemplate.from_template("""
Ты — высококвалифицированный юридический и технический эксперт по российскому законодательству.

Ответь на вопрос **на русском языке**, подробно, структурировано и точно, опираясь только на предоставленную информацию.

QUESTION:
{question}

КЛЮЧЕВЫЕ СУЩНОСТИ ИЗ ГРАФА:
{entities}

ВАЖНЫЕ ОТНОШЕНИЯ И СВЯЗИ:
{relations}

ДОКАЗАТЕЛЬСТВА ИЗ ДОКУМЕНТОВ:
{evidence}

---
**Обязательные требования к ответу:**
- Отвечай **только на русском языке**.
- Делай ответ структурированным (используй нумерацию, заголовки, списки).
- Приводи конкретные ссылки на статьи, документы или сущности, если они есть.
- Если информации недостаточно — прямо скажи об этом.
- Будь максимально точным и профессиональным.

Ответь:
""")


def synthesize_answer(question, entities, relations, evidence):
    start_time = time.time()
    
    console.print(f"\n[bold magenta]ГЕНЕРАЦИЯ ОТВЕТА[/bold magenta]")
    console.print(f"[bold]Вопрос:[/bold] {question}")

    # Мягкое ограничение контекста
    entities_str = str(entities)[:10000]
    relations_str = str(relations)[:7000]
    evidence_str = str(evidence)[:14000]

    try:
        console.print("[yellow]→ Генерация ответа...[/yellow]")
        llm_start = time.time()

        result = (PROMPT | llm).invoke({
            "question": question,
            "entities": entities_str,
            "relations": relations_str,
            "evidence": evidence_str
        })

        duration = time.time() - start_time
        answer = result.content.strip()

        console.print(f"[bold green]✓ Ответ готов за {duration:.1f} сек[/bold green]")
        console.print(f"Длина ответа: {len(answer)} символов\n")

        return answer

    except Exception as e:
        console.print(f"[bold red]Ошибка генерации: {e}[/bold red]")
        return "Не удалось сгенерировать ответ. Попробуйте задать вопрос ещё раз."