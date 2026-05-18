import re

def normalize_entity(text):
    text = text.lower()
    text = re.sub(
        r"[^a-zA-Zа-яА-Я0-9 ]",
        "",
        text
    )
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()
