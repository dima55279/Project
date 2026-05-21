# app/llm/ollama_client.py
import ollama
from typing import Optional


class OllamaClient:
    def __init__(self, model: str = "mistral"):
        self.model = model

    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.3) -> str:
        """
        Генерация ответа через официальную библиотеку ollama
        """
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_ctx": 8192,        # контекстное окно
                }
            )

            return response['message']['content']

        except Exception as e:
            print(f"❌ Ollama ошибка: {e}")
            return f"Ошибка при обращении к Ollama: {str(e)}"


    def generate_with_system(self, system_prompt: str, user_prompt: str) -> str:
        """Удобный метод для system + user"""
        return self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )