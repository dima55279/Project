import json


class GroundedAnswerBuilder:

    def build_context(self, retrieval_results):

        contexts = []
        documents = set()

        for item in retrieval_results.get("graph_lookup", []):
            contexts.append(item["a.text"])
            documents.add(item["a.law"])

        return {
            "context": "\n".join(contexts),
            "documents": list(documents)
        }

    def format_output(self,
                      question,
                      answer,
                      documents):

        return {
            "question": question,
            "answer": answer,
            "document": documents
        }
