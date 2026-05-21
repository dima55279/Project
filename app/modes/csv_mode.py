import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from app.modes.chat_mode import ChatMode


class CSVMode:
    def __init__(self):
        self.chat = ChatMode()

    def process_row(self, row: dict):
        """Обработка одной строки с защитой от сбоев"""
        question = row.get("question", "").strip()
        if not question:
            return {"question": question, "answer": "Пустой вопрос", "documents": []}

        try:
            start_time = time.time()
            
            result = self.chat.ask(question)
            
            duration = time.time() - start_time
            
            print(f"✅ [{duration:5.1f}s] {question[:70]:70}")

            return {
                "question": question,
                "answer": result["answer"],
                "documents": result.get("documents", [])
            }

        except Exception as e:
            print(f"❌ ОШИБКА: {question[:60]}")
            return {
                "question": question,
                "answer": f"ОШИБКА: {str(e)}",
                "documents": []
            }

    def run(self, 
            input_csv: str, 
            output_csv: str = "output/results.csv", 
            max_workers: int = 4):   # ← Важно: не больше 4-6!
        
        df = pd.read_csv(input_csv)
        rows = df.to_dict("records")

        print(f"🚀 Запуск обработки {len(rows)} вопросов (потоков: {max_workers})...\n")

        results = []
        total_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(self.process_row, row): row for row in rows}

            for future in tqdm(
                as_completed(future_to_row),
                total=len(rows),
                desc="PROCESSING"
            ):
                try:
                    results.append(result)
                except Exception as e:
                    row = future_to_row[future]
                    results.append({
                        "question": row.get("question", ""),
                        "answer": f"ТАЙМАУТ/КРИТИЧЕСКАЯ ОШИБКА: {str(e)}",
                        "documents": []
                    })

        # Сохранение результата
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_csv, index=False)

        total_time = time.time() - total_start
        print(f"\n🎉 Обработка завершена за {total_time/60:.1f} минут")
        print(f"📁 Результат сохранён в → {output_csv}")