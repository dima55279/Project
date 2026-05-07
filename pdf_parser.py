import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import pymupdf  # pip install pymupdf
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown

# =========================================================
# CONFIG
# =========================================================

PDF_PATH = "data/docs/Data1_10_10855_index.pdf"
OUTPUT_MD = "Data1_10_10855_index.md"

# ---------- IMAGE ----------
DPI = 110
MAX_SIDE = 1280

# ---------- MODEL ----------
MODEL_NAME = "datalab-to/chandra-ocr-2"

# A100 80GB:
BATCH_SIZE = 16

# RTX 3090:
# BATCH_SIZE = 6

PROMPT_TYPE = "ocr"

# "ocr_layout" = сильно медленнее
# "ocr" = быстрее

USE_FLASH_ATTN = True
USE_TORCH_COMPILE = True

# =========================================================


# =========================================================
# FAST PDF RENDER
# =========================================================

def render_page(page, dpi=110, max_side=1280):
    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )

    w, h = img.size

    scale = min(max_side / w, max_side / h)

    if scale < 1:
        img = img.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS,
        )

    return img


def pdf_to_images(pdf_path):
    print("\n[1/5] Rendering PDF...")

    t0 = time.time()

    doc = pymupdf.open(pdf_path)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        pages = list(
            tqdm(
                ex.map(
                    lambda p: render_page(
                        p,
                        dpi=DPI,
                        max_side=MAX_SIDE,
                    ),
                    doc,
                ),
                total=len(doc),
                desc="Rendering pages",
            )
        )

    dt = time.time() - t0

    print(f"PDF render done in {dt:.2f}s")
    print(f"Pages: {len(pages)}")

    return pages


# =========================================================
# MODEL
# =========================================================

def load_model():
    print("\n[2/5] Loading model...")

    t0 = time.time()

    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    if USE_FLASH_ATTN:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        **kwargs,
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    processor.tokenizer.padding_side = "left"

    model.processor = processor

    model.eval()

    if USE_TORCH_COMPILE:
        print("Compiling model...")
        model = torch.compile(model)

    dt = time.time() - t0

    print(f"Model loaded in {dt:.2f}s")
    print(f"Device: {next(model.parameters()).device}")

    return model


# =========================================================
# WARMUP
# =========================================================

def warmup(model, sample_page):
    print("\n[3/5] Warmup...")

    t0 = time.time()

    with torch.inference_mode():
        _ = generate_hf(
            [
                BatchInputItem(
                    image=sample_page,
                    prompt_type=PROMPT_TYPE,
                )
            ],
            model,
        )

    torch.cuda.synchronize()

    dt = time.time() - t0

    print(f"Warmup done in {dt:.2f}s")


# =========================================================
# OCR
# =========================================================

def process_pages(model, pages):
    print("\n[4/5] OCR processing...")

    all_markdown = []

    total_batches = (len(pages) + BATCH_SIZE - 1) // BATCH_SIZE

    inference_total = 0
    parsing_total = 0

    with torch.inference_mode():

        for i in tqdm(
            range(0, len(pages), BATCH_SIZE),
            total=total_batches,
            desc="Batches",
        ):

            batch_pages = pages[i:i + BATCH_SIZE]

            batch = [
                BatchInputItem(
                    image=page,
                    prompt_type=PROMPT_TYPE,
                )
                for page in batch_pages
            ]

            # ---------------- INFERENCE ----------------

            t0 = time.time()

            results = generate_hf(batch, model)

            torch.cuda.synchronize()

            inference_dt = time.time() - t0
            inference_total += inference_dt

            # ---------------- PARSE ----------------

            t0 = time.time()

            for j, result in enumerate(results):
                page_num = i + j + 1

                markdown = parse_markdown(result.raw)

                all_markdown.append(
                    f"# Page {page_num}\n\n{markdown}\n"
                )

            parsing_dt = time.time() - t0
            parsing_total += parsing_dt

    print("\n==============================")
    print(f"Inference time : {inference_total:.2f}s")
    print(f"Markdown parse : {parsing_total:.2f}s")
    print("==============================")

    pages_per_sec = len(pages) / inference_total

    print(f"\nOCR throughput: {pages_per_sec:.2f} pages/sec")

    return all_markdown


# =========================================================
# SAVE
# =========================================================

def save_markdown(all_markdown, output_path):
    print("\n[5/5] Saving markdown...")

    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_markdown))

    dt = time.time() - t0

    print(f"Saved in {dt:.2f}s")


# =========================================================
# MAIN
# =========================================================

def main():
    total_t0 = time.time()

    model = load_model()

    pages = pdf_to_images(PDF_PATH)

    if not pages:
        print("Empty PDF")
        return

    warmup(model, pages[0])

    markdown_pages = process_pages(model, pages)

    save_markdown(markdown_pages, OUTPUT_MD)

    total_dt = time.time() - total_t0

    print("\n===================================")
    print(f"TOTAL TIME: {total_dt:.2f}s")
    print(f"OUTPUT: {OUTPUT_MD}")
    print("===================================")


if __name__ == "__main__":
    main()