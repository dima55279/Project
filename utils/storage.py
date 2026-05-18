import os
import json
import pickle

def save_json(path, data):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )



def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def save_pickle(path, data):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "wb") as f:
        pickle.dump(data, f)



def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
