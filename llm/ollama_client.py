import ollama


class OllamaClient:

    def __init__(self,
                 model="mistral"):

        self.model = model

    def generate(self,
                 system_prompt,
                 user_prompt):

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        return response[
            "message"
        ]["content"]
