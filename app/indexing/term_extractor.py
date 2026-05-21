import re
import pymorphy3


class TermExtractor:

    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def extract(self, text: str):

        words = re.findall(r"[А-Яа-яA-Za-z-]+", text.lower())

        terms = []

        for word in words:

            if len(word) < 3:
                continue

            parsed = self.morph.parse(word)[0]

            if parsed.tag.POS in ["NOUN", "ADJF"]:
                terms.append(parsed.normal_form)

        return list(set(terms))
