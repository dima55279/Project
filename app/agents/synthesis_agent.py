# app/agents/synthesis_agent.py
from app.llm.ollama_client import OllamaClient


class SynthesisAgent:

    def __init__(self):
        self.llm = OllamaClient(model="mistral")

    def synthesize(self, question: str, evidence: list):
        context = ""

        documents = set()

        for item in evidence:
            data = item["data"]
            documents.add(data.get("document", "Unknown"))
            
            context += f"""
DOCUMENT: {data.get('document', '')}
ARTICLE: {data.get('article', '')}
TEXT:
{data.get('text', '')}

"""

        system_prompt = """
Ты — опытный юридический ассистент, специализирующийся на нормативно-правовых актах РФ.
Отвечай строго на основе предоставленного контекста.
Если информации недостаточно — честно напиши: "Недостаточно данных в нормативной базе."
"""

        user_prompt = f"""
ВОПРОС: {question}

КОНТЕКСТ:
{context}
"""

        answer = self.llm.generate_with_system(system_prompt, user_prompt)

        return {
            "answer": answer,
            "documents": list(documents)
        }