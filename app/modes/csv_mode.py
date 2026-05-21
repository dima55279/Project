import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from app.modes.chat_mode import ChatMode


class CSVMode:
    def __init__(self):
        self.chat = ChatMode()

    def process_row(self, row: dict):
        """Обрабатываем одну строку с обработкой ошибок"""
        try:
            result = self.chat.ask(row["question"])
            return {
                "question": row["question"],
                "answer": result["answer"],
                "documents": result.get("documents", [])
            }
        except Exception as e:
            return {
                "question": row["question"],
                "answer": f"ОШИБКА: {str(e)}",
                "documents": []
            }

    def run(self, input_csv, output_csv="output/results.csv", max_workers=8):
        df = pd.read_csv(input_csv)
        rows = df.to_dict("records")

        print(f"🚀 Запуск обработки {len(rows)} вопросов (потоков: {max_workers})...")

        results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Используем as_completed вместо map — лучше контроль
            future_to_row = {executor.submit(self.process_row, row): row for row in rows}
            
            for future in tqdm(
                as_completed(future_to_row),
                total=len(rows),
                desc="PROCESSING"
            ):
                try:
                    result = future.result(timeout=180)  # таймаут 3 минуты на вопрос
                    results.append(result)
                except Exception as e:
                    row = future_to_row[future]
                    results.append({
                        "question": row["question"],
                        "answer": f"ТАЙМАУТ/ОШИБКА: {str(e)}",
                        "documents": []
                    })

        # Сохраняем результат
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_csv, index=False)

        elapsed = time.time() - start_time
        print(f"\n✅ Обработка завершена за {elapsed:.1f} секунд")
        print(f"📁 Результат сохранён в {output_csv}")
