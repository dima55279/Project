from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)


def get_loader(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix in [".txt", ".md"]:
        return TextLoader(str(path), encoding="utf-8")
    if suffix == ".docx":
        return Docx2txtLoader(str(path))
    return None


def load_single_file(file_path: Path):
    """Загружает один файл"""
    loader = get_loader(file_path)
    if loader is None:
        return []
    
    try:
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = file_path.name
            d.metadata["filepath"] = str(file_path)
        return docs
    except Exception as e:
        print(f"Ошибка загрузки {file_path.name}: {e}")
        return []


def load_documents(folder):
    folder = Path(folder)
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}]
    
    print(f"Найдено файлов: {len(files)}")
    
    all_docs = []
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {executor.submit(load_single_file, f): f for f in files}
        
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="Загрузка файлов"):
            docs = future.result()
            all_docs.extend(docs)
    
    print(f"Успешно загружено документов: {len(all_docs)}")
    return all_docs
