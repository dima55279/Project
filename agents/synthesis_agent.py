from langchain_core.prompts import (
    ChatPromptTemplate
)

from utils.llm import llm


PROMPT = ChatPromptTemplate.from_template("""
Ты retrieval QA система.

Отвечай строго ТОЛЬКО по предоставленным fragments.

Запрещено:
- придумывать информацию
- делать выводы вне текста
- писать JSON
- писать markdown
- создавать списки
- создавать разделы
- добавлять пояснения
- писать рассуждения

Требования:
- короткий factual answer
- plain text
- только информация из fragments

Если информации недостаточно:
ответь:
Недостаточно информации.

QUESTION:
{question}

GRAPH ENTITIES:
{entities}

GRAPH RELATIONS:
{relations}

TEXT FRAGMENTS:
{evidence}
""")


def synthesize_answer(
    question,
    entities,
    relations,
    evidence
):

    chain = PROMPT | llm

    result = chain.invoke({
        "question": question,
        "entities": str(entities),
        "relations": str(relations),
        "evidence": "\n\n".join(evidence)
    })

    text = result.content.strip()

    text = text.replace("\n", " ")

    text = text.replace('"', "'")

    text = " ".join(text.split())

    return text