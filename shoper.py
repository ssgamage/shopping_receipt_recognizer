import argparse
import json
import os
from parser import format_summary, parse_receipt_text

from rich.console import Console
from rich.panel import Panel

import config
from ocr_engine import ocr_image
from receipt_processor import ReceiptProcessor
from utils import ensure_dir

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Receipt OCR & Summarizer (CLI)")
    parser.add_argument("image", help="Path to a receipt image (png/jpg)")
    parser.add_argument("--save-steps", action="store_true", help="Save intermediate step images to outputs/")
    parser.add_argument("--show", action="store_true", help="Show intermediate windows (press a key to continue)")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive thresholding (otherwise Otsu)")
    parser.add_argument("--psm", type=int, default=config.TESSERACT_PSM, help="Tesseract PSM (page segmentation mode)")
    args = parser.parse_args()

    ensure_dir(config.OUTPUT_DIR)

    console.print(f"[bold cyan][INFO][/bold cyan] Loading image: {args.image}")
    rp = ReceiptProcessor(save_steps=args.save_steps, show=args.show, adaptive=args.adaptive)
    base, result = rp.process(args.image)

    console.print("[bold cyan][STEP][/bold cyan] OCR running...")
    # Pass the whole ProcessResult instead of just the ocr_ready image
    text = ocr_image(result, psm=args.psm)

    console.print("[bold cyan][STEP][/bold cyan] Parsing text...")
    summary = parse_receipt_text(text)

    console.print(Panel.fit(" SUMMARY ", style="green"))
    console.print(format_summary(summary))

    json_path = os.path.join(config.OUTPUT_DIR, f"{base}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    console.print(f"[bold cyan]Saved JSON:[/bold cyan] {json_path}")
    if args.save_steps:
        step_list = ", ".join(sorted([f for f in result.steps.values() if f]))
        console.print(f"[bold cyan]Saved steps:[/bold cyan] {step_list}")

if __name__ == "__main__":
    main()
