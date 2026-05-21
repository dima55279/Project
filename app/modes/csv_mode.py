import pandas as pd

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from app.modes.chat_mode import ChatMode


class CSVMode:

    def __init__(self):
        self.chat = ChatMode()

    def process_row(self, row):

        result = self.chat.ask(row["question"])

        return {
            "question": row["question"],
            "answer": result["answer"],
            "document": result["documents"]
        }

    def run(self,
            input_csv,
            output_csv="output/results.csv"):

        df = pd.read_csv(input_csv)

        rows = df.to_dict("records")

        results = []

        with ThreadPoolExecutor(max_workers=16) as executor:

            for result in tqdm(
                    executor.map(self.process_row, rows),
                    total=len(rows),
                    desc="PROCESSING CSV"):

                results.append(result)

        out_df = pd.DataFrame(results)

        out_df.to_csv(
            output_csv,
            index=False
        )
