from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from config import NUM_WORKERS
from rich.console import Console

console = Console()

def get_loader(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix in [".txt", ".md"]:
        return TextLoader(str(path), encoding="utf-8")
    if suffix == ".docx":
        return Docx2txtLoader(str(path))
    return None


def smart_split_markdown(text: str, source_name: str):
    """Умное разделение по главам и статьям"""
    # Разделяем по крупным заголовкам
    parts = re.split(r'(^Глава\s+\d+|^Статья\s+\d+|^##\s+|^###\s+)', text, flags=re.MULTILINE)
    
    chunks = []
    current = ""
    
    for part in parts:
        if len(current) > 22000 and len(part.strip()) > 1000:
            if current.strip():
                chunks.append(current.strip())
            current = part
        else:
            current += part
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks


def load_single_file(file_path: Path):
    loader = get_loader(file_path)
    if loader is None:
        return []
    
    try:
        docs = loader.load()
        all_chunks = []

        for doc in docs:
            source = file_path.name
            content = doc.page_content.strip()

            if len(content) > 8000 and file_path.suffix.lower() in {".md", ".txt"}:
                console.print(f"[yellow]✂️ Умное разделение:[/yellow] {source} ({len(content):,} символов)")
                split_items = smart_split_markdown(content, source)
                
                for item in split_items:
                    new_doc = doc.copy()
                    new_doc.page_content = item
                    new_doc.metadata.update({
                        "source": source,
                        "filepath": str(file_path),
                        "is_chunked": True,
                        "section_title": item[:200]
                    })
                    all_chunks.append(new_doc)
            else:
                doc.metadata["source"] = source
                doc.metadata["filepath"] = str(file_path)
                doc.metadata["is_chunked"] = False
                all_chunks.append(doc)

        return all_chunks
    except Exception as e:
        console.print(f"[red]Ошибка загрузки {file_path.name}: {e}[/red]")
        return []


def load_documents(folder):
    folder = Path(folder)
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}]
    
    console.print(f"[bold]Найдено файлов: {len(files)}[/bold]")

    all_docs = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {executor.submit(load_single_file, f): f for f in files}
        
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="Загрузка документов"):
            docs = future.result()
            all_docs.extend(docs)
    
    console.print(f"[bold green]Загружено документов/секций: {len(all_docs)}[/bold green]")
    return all_docs
