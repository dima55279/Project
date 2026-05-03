from transformers import AutoModelForImageTextToText, AutoProcessor
from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown
from pdf2image import convert_from_path
from PIL import Image
import torch
import os

# === CONFIG ===
PDF_PATH = "data/docs/Data1_10_10855_index.pdf"
OUTPUT_MD = "output.md"
DPI = 300  # качество рендера страниц

# === LOAD MODEL ===
model = AutoModelForImageTextToText.from_pretrained(
    "datalab-to/chandra-ocr-2",
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

processor = AutoProcessor.from_pretrained("datalab-to/chandra-ocr-2")
processor.tokenizer.padding_side = "left"
model.processor = processor

# === PDF → IMAGES ===
print("Converting PDF to images...")
pages = convert_from_path(PDF_PATH, dpi=DPI)

all_markdown = []

# === PROCESS EACH PAGE ===
for i, page in enumerate(pages):
    print(f"Processing page {i+1}/{len(pages)}...")

    batch = [
        BatchInputItem(
            image=page,
            prompt_type="ocr_layout"
        )
    ]

    result = generate_hf(batch, model)[0]
    markdown = parse_markdown(result.raw)

    all_markdown.append(f"# Page {i+1}\n\n{markdown}\n")

# === SAVE RESULT ===
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(all_markdown))

print(f"Done! Saved to {OUTPUT_MD}")