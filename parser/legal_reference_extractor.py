import re

REFERENCE_RE = r"стать[ьяи]\s+(\d+)"


class LegalReferenceExtractor:

    def extract(self, text):
        refs = re.findall(
            REFERENCE_RE,
            text.lower()
        )

        return list(set(refs))
