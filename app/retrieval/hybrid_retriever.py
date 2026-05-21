# app/retrieval/hybrid_retriever.py
from app.agents.query_agent import QueryAgent
from app.agents.graph_agent import GraphAgent
from app.agents.bm25_agent import BM25Agent
from app.agents.evidence_agent import EvidenceAgent


class HybridRetriever:

    def __init__(self):
        self.query_agent = QueryAgent()
        self.graph_agent = GraphAgent()
        self.bm25_agent = BM25Agent()
        self.evidence_agent = EvidenceAgent()

    def retrieve(self, question):
        try:
            query_data = self.query_agent.run(question)

            graph_results = self.graph_agent.retrieve(query_data["terms"])

            bm25_results = self.bm25_agent.retrieve(question)

            return self.evidence_agent.rank(graph_results, bm25_results)
        except Exception as e:
            print(f"❌ Ошибка в HybridRetriever: {e}")
            return []