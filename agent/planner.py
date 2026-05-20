class LegalPlanner:

    def plan(self, question):

        question = question.lower()

        plan = []

        if "что означает" in question:
            plan.append("graph_lookup")
            plan.append("bm25_lookup")
            plan.append("vector_lookup")

        elif "какая статья" in question:
            plan.append("graph_lookup")
            plan.append("bm25_lookup")

        else:
            plan.append("vector_lookup")

        return plan
