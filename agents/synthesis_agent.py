from langchain_core.prompts import (
    ChatPromptTemplate
)

from utils.llm import llm


PROMPT = ChatPromptTemplate.from_template("""
Ты система поиска ответов по документам.

ТВОЯ ЗАДАЧА:
дать краткий ответ ТОЛЬКО на основе fragments.

СТРОГО ЗАПРЕЩЕНО:
- придумывать информацию
- использовать знания вне fragments
- переводить текст
- писать JSON
- писать markdown
- делать списки
- делать разделы
- делать пояснения
- рассуждать

ФОРМАТ ОТВЕТА:
- обычный текст
- только русский язык

Если fragments не содержат ответа:
ответь:
Недостаточно информации.

QUESTION:
{question}

FRAGMENTS:
{evidence}
""")


def synthesize_answer(
    question,
    entities,
    relations,
    evidence
):

    if not evidence:

        return "Недостаточно информации."

    chain = PROMPT | llm

    result = chain.invoke({
        "question": question,
        "evidence": "\n\n".join(evidence[:5])
    })

    text = result.content.strip()

    text = text.replace("\n", " ")

    text = text.replace('"', "'")

    text = " ".join(text.split())

    if text.startswith("{"):
        return "Недостаточно информации."

    return text