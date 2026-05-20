import time
import json
import re
from pathlib import Path
from rich.console import Console

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.llm import llm
from config import DEBUG_DIR, OLLAMA_MODEL

console = Console()
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


class LegalEntity(BaseModel):
    id: str
    type: str = "Term"  # Definition, Requirement, Prohibition, Permission, Threshold, Authority, Object, Procedure, DocumentRef
    description: str = ""
    article_reference: str = ""


class LegalRelationship(BaseModel):
    source: str
    target: str
    relation: str
    description: str = ""
    strength: float = 1.0


class ExtractedFact(BaseModel):
    statement: str
    entities: list[str] = Field(default_factory=list)
    article_reference: str = ""


class LegalGraphExtraction(BaseModel):
    entities: list[LegalEntity] = Field(default_factory=list)
    relationships: list[LegalRelationship] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)


parser = PydanticOutputParser(pydantic_object=LegalGraphExtraction)


FEW_SHOT = """
Пример 1:
Текст: "Под затонувшим имуществом понимается судно, иное плавучее средство..."
Ожидаемый результат: {"id": "затонувшее имущество", "type": "Definition", "article_reference": "ст. 3"}

Пример 2:
Текст: "Особо крупный размер ущерба составляет один миллион рублей."
Ожидаемый результат: {"id": "особо крупный размер", "type": "Threshold"}
"""


PROMPT = ChatPromptTemplate.from_template("""
Ты — эксперт по извлечению юридического knowledge graph из российского законодательства.

{format_instructions}

{few_shot}

Текст раздела:
{text}

Правила:
- id сущности — полноценное юридическое понятие (желательно 2+ слова)
- Извлекай определения, требования, пороги, органы, объекты регулирования
- Избегай шума: одиночные цифры, номера страниц, общие слова
""")


def normalize_entity_id(text: str) -> str | None:
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'^[«"\'(]\s*|\s*[»"\')\],.]$', '', text)
    if len(text) < 4 or text.isdigit() or text.lower() in {"да", "нет", "руб", "рублей", "№"}:
        return None
    return text


def clean_extraction(raw: dict) -> dict:
    entities = []
    seen = set()

    for e in raw.get("entities", []):
        clean_id = normalize_entity_id(e.get("id", ""))
        if not clean_id or clean_id.lower() in seen:
            continue
        seen.add(clean_id.lower())
        e["id"] = clean_id
        entities.append(e)

    raw["entities"] = entities
    return raw


def extract_graph_batch(texts: list, batch_idx: int = 0, batch_names: list = None) -> dict:
    start_time = time.time()
    batch_names = batch_names or [f"section_{i}" for i in range(len(texts))]

    total_chars = sum(len(t) for t in texts)
    console.rule(f"[bold cyan]БАТЧ {batch_idx}[/bold cyan]")
    console.print(f"Документов: {len(texts)} | Символов: {total_chars:,}")

    combined = [f"--- SECTION {i+1}: {name} ---\n{text[:8500]}\n\n" 
                for i, (text, name) in enumerate(zip(texts, batch_names))]
    final_text = "".join(combined)

    try:
        chain = PROMPT | llm | parser
        result = chain.invoke({
            "text": final_text,
            "format_instructions": parser.get_format_instructions(),
            "few_shot": FEW_SHOT
        })

        cleaned = clean_extraction(result.model_dump())

        duration = time.time() - start_time
        console.print(f"[bold green]✓ БАТЧ {batch_idx} УСПЕШНО[/bold green] — Сущностей: {len(cleaned['entities'])} | Фактов: {len(cleaned.get('facts', []))}")

        with open(DEBUG_DIR / f"batch_{batch_idx:03d}.json", "w", encoding="utf-8") as f:
            json.dump({
                "batch_idx": batch_idx,
                "entities_count": len(cleaned["entities"]),
                "facts_count": len(cleaned.get("facts", [])),
                "result": cleaned
            }, f, ensure_ascii=False, indent=2)

        return cleaned

    except Exception as e:
        console.print(f"[bold red]✗ БАТЧ {batch_idx} ОШИБКА: {e}[/bold red]")
        return {"entities": [], "relationships": [], "facts": []}