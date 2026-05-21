import re
from pathlib import Path


SECTION_PATTERN = r"^##\s+(.+)"
ARTICLE_PATTERN = r"^(Статья|Ст\.)\s*([0-9\.]+)"
SUBARTICLE_PATTERN = r"^([0-9]+\.[0-9]+\.)"


class MarkdownParser:

    def parse(self, file_path: str):

        text = Path(file_path).read_text(
            encoding="utf-8",
            errors="ignore"
        )

        lines = text.splitlines()

        document = {
            "title": Path(file_path).stem,
            "sections": []
        }

        current_section = None
        current_article = None

        for line in lines:

            line = line.strip()

            section_match = re.match(SECTION_PATTERN, line)

            if section_match:
                current_section = {
                    "title": section_match.group(1),
                    "articles": []
                }

                document["sections"].append(current_section)
                continue

            article_match = re.match(ARTICLE_PATTERN, line)

            if article_match:
                current_article = {
                    "number": article_match.group(2),
                    "title": line,
                    "text": ""
                }

                if current_section is None:
                    current_section = {
                        "title": "ROOT",
                        "articles": []
                    }
                    document["sections"].append(current_section)

                current_section["articles"].append(current_article)
                continue

            if current_article:
                current_article["text"] += line + "\n"

        return document
