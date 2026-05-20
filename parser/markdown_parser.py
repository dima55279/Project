import re
from pathlib import Path

ARTICLE_RE = r"^##\s+Статья\s+(\d+)"
CHAPTER_RE = r"^#\s+Глава\s+(\d+)"


class MarkdownLawParser:

    def parse(self, filepath):

        text = Path(filepath).read_text(
            encoding="utf-8"
        )

        lines = text.split("\n")

        law = {
            "articles": []
        }

        current_article = None
        current_chapter = None
        article_buffer = []

        for line in lines:

            chapter_match = re.match(CHAPTER_RE, line)

            if chapter_match:
                current_chapter = chapter_match.group(1)
                continue

            article_match = re.match(ARTICLE_RE, line)

            if article_match:

                if current_article:
                    law["articles"].append({
                        "article_id": current_article,
                        "chapter": current_chapter,
                        "text": "\n".join(article_buffer)
                    })

                current_article = article_match.group(1)
                article_buffer = []
                continue

            article_buffer.append(line)

        if current_article:
            law["articles"].append({
                "article_id": current_article,
                "chapter": current_chapter,
                "text": "\n".join(article_buffer)
            })

        return law
