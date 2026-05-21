from app.llm.ollama_client import OllamaClient


class SynthesisAgent:

    def __init__(self):
        self.llm = OllamaClient()

    def synthesize(self,
                   question,
                   evidence):

        context = ""

        documents = set()

        for item in evidence:

            data = item["data"]

            documents.add(data["document"])

            context += f"""

DOCUMENT: {data['document']}
ARTICLE: {data['article']}
TEXT:
{data['text']}

"""

        prompt = f"""
Ты юридический ассистент.

Отвечай ТОЛЬКО на основе контекста.

Если данных недостаточно —
напиши:
Недостаточно данных в нормативной базе.

ВОПРОС:
{question}

КОНТЕКСТ:
{context}
"""

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "documents": list(documents)
        }
