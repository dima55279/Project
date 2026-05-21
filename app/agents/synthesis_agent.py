# app/agents/synthesis_agent.py
from app.llm.ollama_client import OllamaClient


class SynthesisAgent:

    def __init__(self):
        self.llm = OllamaClient(model="mistral")

    def synthesize(self, question: str, evidence: list):
        context = ""
        documents = set()

        for item in evidence:
            data = item.get("data", {}) if isinstance(item, dict) else {}
            doc_name = data.get("document", "Unknown")
            documents.add(doc_name)

            context += f"""
DOCUMENT: {doc_name}
ARTICLE: {data.get('article', '')}
TEXT:
{data.get('text', '')}

"""

        system_prompt = """
Ты — юридический ассистент по нормативным актам РФ.
Отвечай строго на основе контекста.
Если данных недостаточно — отвечай: "Недостаточно данных в нормативной базе."
"""

        user_prompt = f"ВОПРОС: {question}\n\nКОНТЕКСТ:\n{context}"

        try:
            answer = self.llm.generate_with_system(system_prompt, user_prompt)
        except Exception as e:
            answer = f"Ошибка генерации: {str(e)}"

        return {
            "answer": answer,
            "documents": list(documents)
        }