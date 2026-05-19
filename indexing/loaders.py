from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)

from config import NUM_WORKERS


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
    """Умное разделение юридических .md документов"""
    # Разделяем по основным заголовкам
    sections = re.split(r'(^##\s+|^###\s+|^####\s+)', text, flags=re.MULTILINE)
    
    chunks = []
    current_chunk = ""
    current_title = ""

    for i in range(len(sections)):
        part = sections[i].strip()
        if not part:
            continue
            
        if re.match(r'^#{2,4}\s+', part):  # это заголовок
            current_title = part
            continue

        if len(current_chunk) > 15000 or (current_chunk and len(part) > 10000):
            if current_chunk.strip():
                chunks.append({
                    "content": current_title + "\n\n" + current_chunk,
                    "metadata": {
                        "source": source_name,
                        "section_title": current_title,
                        "is_chunked": True
                    }
                })
            current_chunk = part
        else:
            current_chunk += "\n\n" + part

    # Последний кусок
    if current_chunk.strip():
        chunks.append({
            "content": current_title + "\n\n" + current_chunk,
            "metadata": {
                "source": source_name,
                "section_title": current_title,
                "is_chunked": True
            }
        })

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
            content = doc.page_content

            if len(content) > 8000 and file_path.suffix.lower() in {".md", ".txt"}:
                print(f"✂️ Умное разделение: {source} ({len(content):,} символов)")
                split_docs = smart_split_markdown(content, source)
                
                for item in split_docs:
                    new_doc = doc.copy()
                    new_doc.page_content = item["content"]
                    new_doc.metadata.update(item["metadata"])
                    all_chunks.append(new_doc)
            else:
                doc.metadata["source"] = source
                doc.metadata["filepath"] = str(file_path)
                doc.metadata["is_chunked"] = False
                all_chunks.append(doc)

        return all_chunks
    except Exception as e:
        print(f"⚠️ Ошибка загрузки {file_path.name}: {e}")
        return []


def load_documents(folder):
    folder = Path(folder)
    files = [f for f in folder.iterdir() 
             if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}]
    
    print(f"📁 Найдено файлов: {len(files)}")

    all_docs = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {executor.submit(load_single_file, f): f for f in files}
        
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="Загрузка + умное разделение"):
            docs = future.result()
            all_chunks.extend(docs)   # all_docs -> all_chunks (опечатка исправлена)
    
    print(f"✅ Загружено документов/секций: {len(all_docs)}")
    return all_docs