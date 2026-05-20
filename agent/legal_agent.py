class LegalAgent:

    def __init__(self, planner, tools):
        self.planner = planner
        self.tools = tools

    def run(self, question):

        plan = self.planner.plan(question)

        results = {}

        for step in plan:
            tool = getattr(self.tools, step)
            results[step] = tool(question)

        return results
