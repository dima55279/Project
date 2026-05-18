import pandas as pd
from tqdm import tqdm

from generation.graph_pipeline import (
    run_graphrag
)

from config import (
    QUESTIONS_FILE,
    RESULTS_DIR
)

questions = pd.read_csv(
    QUESTIONS_FILE
)

results = []

for _, row in tqdm(questions.iterrows(), total=len(questions)):
    result = run_graphrag(
        row["question"]
    )
    results.append(result)

output = pd.DataFrame(results)

output.to_csv(
    f"{RESULTS_DIR}/results.csv",
    index=False
)

print("DONE")
