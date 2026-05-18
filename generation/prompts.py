from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_template("""
Определи тип запроса.

LOCAL:
- factual lookup
- entity lookup
- specific question

GLOBAL:
- summarize
- trends
- themes
- insights
- analysis

Ответь:
LOCAL
или
GLOBAL

Question:
{question}
""")


GENERATION_PROMPT = ChatPromptTemplate.from_template("""
Ты GraphRAG AI assistant.

Используй:
- graph relationships
- community summaries
- retrieved chunks

Если информации недостаточно:
скажи об этом.

CONTEXT:
{context}

QUESTION:
{question}
""")
