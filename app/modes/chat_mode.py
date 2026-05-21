from app.retrieval.hybrid_retriever import HybridRetriever
from app.agents.synthesis_agent import SynthesisAgent


class ChatMode:

    def __init__(self):

        self.retriever = HybridRetriever()
        self.synthesis = SynthesisAgent()

    def ask(self,
            question):

        evidence = self.retriever.retrieve(question)

        return self.synthesis.synthesize(
            question,
            evidence
        )
