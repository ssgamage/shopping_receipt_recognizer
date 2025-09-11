# shoper_multi.py
import argparse
import json
import os
from parser import format_summary, parse_receipt_text

from rich.console import Console
from rich.panel import Panel

import config
from multi_receipt_processor import MultiReceiptProcessor
from ocr_engine import ocr_image

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Multi-Receipt OCR (CLI)")
    parser.add_argument("image", help="Path to an image with multiple receipts")
    parser.add_argument("--save-steps", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--psm", type=int, default=config.TESSERACT_PSM)
    args = parser.parse_args()

    mrp = MultiReceiptProcessor(save_steps=args.save_steps, adaptive=args.adaptive)
    results = mrp.process_all(args.image)

    combined = []
    for base, process_result in results:
        # ✅ Pass full ProcessResult
        text = ocr_image(process_result, psm=args.psm)
        summary = parse_receipt_text(text)

        console.print(Panel.fit(f" SUMMARY ({base}) ", style="green"))
        console.print(format_summary(summary))

        # Save per-receipt JSON
        json_path = os.path.join(config.OUTPUT_DIR, f"{base}_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        console.print(f"[cyan]Saved:[/cyan] {json_path}")

        combined.append({"receipt": base, "summary": summary})

    # Save combined JSON
    combined_path = os.path.join(config.OUTPUT_DIR, f"{os.path.splitext(os.path.basename(args.image))[0]}_multi.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    console.print(f"[bold magenta]Combined JSON saved:[/bold magenta] {combined_path}")


if __name__ == "__main__":
    main()
