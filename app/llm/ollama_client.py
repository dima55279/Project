# app/llm/ollama_client.py
import ollama

class OllamaClient:
    def __init__(self, model: str = "mistral"):
        self.model = model

    def generate_with_system(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.3, "num_ctx": 8192}
            )
            return response['message']['content']
        except Exception as e:
            print(f"❌ Ollama error: {type(e).__name__}: {e}")
            return f"Ошибка Ollama: {str(e)}"