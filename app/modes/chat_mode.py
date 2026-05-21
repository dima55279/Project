# app/modes/chat_mode.py
from app.retrieval.hybrid_retriever import HybridRetriever
from app.agents.synthesis_agent import SynthesisAgent


class ChatMode:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.synthesis = SynthesisAgent()

    def ask(self, question: str):
        try:
            evidence = self.retriever.retrieve(question)
            return self.synthesis.synthesize(question, evidence)
        except Exception as e:
            print(f"❌ Ошибка в ChatMode.ask(): {e}")
            return {
                "answer": f"Системная ошибка: {str(e)}",
                "documents": []
            }
