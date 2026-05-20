from langchain_core.prompts import (
    ChatPromptTemplate
)

from utils.llm import llm


PROMPT = ChatPromptTemplate.from_template("""
Ты — высококвалифицированный эксперт по российскому законодательству.

QUESTION: {question}

КЛЮЧЕВЫЕ СУЩНОСТИ:
{entities}

ФАКТЫ ИЗ ДОКУМЕНТОВ:
{facts}

ДОКАЗАТЕЛЬСТВА:
{evidence}

Отвечай **только на русском**, структурировано, с ссылками на статьи где возможно.
""")


def synthesize_answer(question, entities, facts, evidence):
    entities_str = "\n".join([f"- {e['id']} ({e.get('type','')})" for e in entities[:30]])
    facts_str = "\n".join([f"- {f['statement']}" for f in facts[:20]])
    evidence_str = str(evidence)[:15000]

    result = (PROMPT | llm).invoke({
        "question": question,
        "entities": entities_str,
        "facts": facts_str,
        "evidence": evidence_str
    })
    return result.content.strip()