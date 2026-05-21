import re


def normalize_text(text: str):

    text = text.lower()

    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^\w\sа-яА-Я]", " ", text)

    return text.strip()


def tokenize(text: str):

    text = normalize_text(text)

    return text.split()


def clean_markdown(text: str):

    text = re.sub(r"#+", "", text)

    text = re.sub(r"\*", "", text)

    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    text = re.sub(r"`+", "", text)

    return text


def extract_article_references(text: str):

    pattern = r"(статья|ст\.)\s*([0-9\.]+)"

    matches = re.findall(
        pattern,
        text.lower()
    )

    return [m[1] for m in matches]