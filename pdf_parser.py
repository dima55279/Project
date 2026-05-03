import os
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm

from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown

# ================= CONFIG =================
PDF_PATH = "data/docs/Data1_10_10855_index.pdf"
OUTPUT_MD = "output.md"

DPI = 200
BATCH_SIZE = 4
RESIZE_FACTOR = 1.0
# ==========================================


def load_model():
    print("Loading model...")

    model = AutoModelForImageTextToText.from_pretrained(
        "datalab-to/chandra-ocr-2",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained("datalab-to/chandra-ocr-2")
    processor.tokenizer.padding_side = "left"

    model.processor = processor

    print(f"Model device: {next(model.parameters()).device}")
    return model


def pdf_to_images(pdf_path):
    print("Converting PDF to images...")

    pages = convert_from_path(pdf_path, dpi=DPI)

    processed_pages = []
    for page in tqdm(pages, desc="Preparing pages"):
        if RESIZE_FACTOR != 1.0:
            new_size = (
                int(page.width * RESIZE_FACTOR),
                int(page.height * RESIZE_FACTOR),
            )
            page = page.resize(new_size)

        processed_pages.append(page)

    print(f"Total pages: {len(processed_pages)}")
    return processed_pages


def warmup(model, sample_page):
    print("Warming up model...")
    with torch.inference_mode():
        _ = generate_hf(
            [BatchInputItem(image=sample_page, prompt_type="ocr_layout")],
            model
        )


def process_pages(model, pages):
    print("Processing pages...")

    all_markdown = []
    total_batches = (len(pages) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.inference_mode():
        for i in tqdm(range(0, len(pages), BATCH_SIZE), desc="Batches", total=total_batches):
            batch_pages = pages[i:i + BATCH_SIZE]

            batch = [
                BatchInputItem(image=page, prompt_type="ocr_layout")
                for page in batch_pages
            ]

            results = generate_hf(batch, model)

            for j, result in enumerate(results):
                page_num = i + j + 1
                markdown = parse_markdown(result.raw)

                all_markdown.append(
                    f"# Page {page_num}\n\n{markdown}\n"
                )

    return all_markdown


def save_markdown(all_markdown, output_path):
    print("Saving result...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_markdown))


def main():
    model = load_model()
    pages = pdf_to_images(PDF_PATH)

    if len(pages) == 0:
        print("Empty PDF!")
        return

    warmup(model, pages[0])

    markdown_pages = process_pages(model, pages)

    save_markdown(markdown_pages, OUTPUT_MD)

    print(f"\nDone! Saved to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()