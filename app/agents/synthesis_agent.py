# app/agents/synthesis_agent.py
from app.llm.ollama_client import OllamaClient


class SynthesisAgent:

    def __init__(self):
        self.llm = OllamaClient(model="mistral")

    def synthesize(self, question: str, evidence: list):
        if not evidence:
            return {
                "answer": "Недостаточно данных в нормативной базе.",
                "documents": []
            }

        context = ""
        documents = set()

        for item in evidence:
            data = item.get("data", {})
            doc_name = data.get("document", "Неизвестный документ")
            documents.add(doc_name)

            context += f"""
DOCUMENT: {doc_name}
ARTICLE: {data.get('article', '')}
TEXT:
{data.get('text', '')}

"""

        system_prompt = """
Ты — опытный юридический ассистент по нормативным актам РФ.
Отвечай строго на основе предоставленного контекста.
Если информации недостаточно — пиши: "Недостаточно данных в нормативной базе."
"""

        user_prompt = f"""
ВОПРОС: {question}

КОНТЕКСТ:
{context}
"""

        try:
            answer = self.llm.generate_with_system(system_prompt, user_prompt)
        except Exception as e:
            answer = f"Ошибка генерации ответа: {str(e)}"

        return {
            "answer": answer,
            "documents": list(documents)
        }