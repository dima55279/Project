import time
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.llm import llm
from config import DEBUG_DIR, OLLAMA_MODEL

console = Console()

DEBUG_DIR.mkdir(parents=True, exist_ok=True)

class Entity(BaseModel):
    id: str
    type: str
    description: str = ""

class Relationship(BaseModel):
    source: str
    target: str
    relation: str
    description: str = ""

class GraphExtraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

parser = PydanticOutputParser(pydantic_object=GraphExtraction)

PROMPT = ChatPromptTemplate.from_template("""
Ты — эксперт по извлечению knowledge graph из нормативно-правовых и технических документов.

{format_instructions}

Текст раздела:
{text}
""")


def extract_graph_batch(texts: list, batch_idx: int = 0, batch_names: list = None) -> dict:
    start_time = time.time()
    batch_names = batch_names or [f"doc_{i}" for i in range(len(texts))]
    
    total_chars = sum(len(t) for t in texts)
    
    console.rule(f"[bold cyan]БАТЧ {batch_idx} [/bold cyan]")
    console.print(f"Документов: [yellow]{len(texts)}[/yellow] | Символов: [yellow]{total_chars:,}[/yellow]")
    
    for name in batch_names:
        console.print(f"   • {name}")
    
    # Подготовка текста
    combined = []
    for i, text in enumerate(texts):
        truncated = text[:8500]
        combined.append(f"--- SECTION {i+1}: {batch_names[i]} ---\n{truncated}\n\n")
    
    final_text = "".join(combined)

    try:
        console.print("[yellow]→ Отправка запроса к LLM...[/yellow]")
        llm_start = time.time()

        chain = PROMPT | llm | parser
        result = chain.invoke({
            "text": final_text,
            "format_instructions": parser.get_format_instructions()
        })

        duration = time.time() - start_time
        llm_time = time.time() - llm_start

        entities_count = len(result.entities)
        rels_count = len(result.relationships)

        console.print(f"[bold green]✓ БАТЧ {batch_idx} УСПЕШНО ЗАВЕРШЁН[/bold green]")
        console.print(f"   Общее время: {duration:.1f} сек | LLM: {llm_time:.1f} сек")
        console.print(f"   Извлечено: [green]{entities_count} сущностей[/green], [green]{rels_count} отношений[/green]\n")

        # Сохранение отладочной информации
        debug_data = {
            "batch_idx": batch_idx,
            "model": OLLAMA_MODEL,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_names": batch_names,
            "input_chars": total_chars,
            "duration_sec": round(duration, 2),
            "llm_time_sec": round(llm_time, 2),
            "entities_count": entities_count,
            "relationships_count": rels_count,
            "result": result.model_dump()
        }

        with open(DEBUG_DIR / f"batch_{batch_idx:03d}.json", "w", encoding="utf-8") as f:
            json.dump(debug_data, f, ensure_ascii=False, indent=2)

        return result.model_dump()

    except Exception as e:
        duration = time.time() - start_time
        console.print(f"[bold red]✗ БАТЧ {batch_idx} ОШИБКА[/bold red] ({duration:.1f} сек)")
        console.print(f"[red]Ошибка: {type(e).__name__} - {e}[/red]")

        error_data = {
            "batch_idx": batch_idx,
            "error": str(e),
            "error_type": type(e).__name__,
            "batch_names": batch_names,
            "duration_sec": round(duration, 2),
            "input_preview": final_text[:2000]
        }
        with open(DEBUG_DIR / f"batch_{batch_idx:03d}_ERROR.json", "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)
        
        return {"entities": [], "relationships": []}