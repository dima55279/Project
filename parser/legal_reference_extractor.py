import re

REFERENCE_RE = r"стать[ьяи]\s+(\d+)"


class LegalReferenceExtractor:

    def extract(self, article_text: str):
        refs = re.findall(REFERENCE_RE, article_text.lower())
        return list(set(refs))
