class LegalAgent:

    def __init__(self,
                 planner,
                 tools):

        self.planner = planner
        self.tools = tools

    def run(self,
            question):

        plan = self.planner.plan(question)

        result = None

        for step in plan:

            if hasattr(self.tools, step):
                tool = getattr(self.tools, step)
                result = tool(question)

        return result

