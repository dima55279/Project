# generation/final_answer.py
import json

def build_output(question: str, answer: str, documents: list):
    """
    Возвращает ТОЛЬКО 3 поля: question, answer, document
    """
    return {
        "question": question,
        "answer": answer,
        "document": json.dumps(documents, ensure_ascii=False)  # список файлов как JSON-строка
    }