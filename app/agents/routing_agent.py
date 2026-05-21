class RoutingAgent:

    def run(self, query_data):

        question = query_data["question"].lower()

        if "уголов" in question:
            return "Уголовный"

        if "граждан" in question:
            return "Гражданский"

        if "налог" in question:
            return "Налоговый"

        return None
