import re
from typing import Any, Dict, List


def split_receipts_by_text(text: str) -> List[str]:
    """
    Split OCR text into chunks by markers like 'Sub Total', 'TOTAL', 'Cash', etc.
    Useful for separating multiple receipts OCR'd into one big block.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    receipts = []
    current = []

    for ln in lines:
        current.append(ln)
        # Look for markers that usually end a receipt
        if re.search(r"(Sub\s*Total|TOTAL|Cash|Change)", ln, re.IGNORECASE):
            receipts.append("\n".join(current))
            current = []

    # Any leftovers
    if current:
        receipts.append("\n".join(current))

    return receipts


def parse_receipt_text(text: str) -> Dict[str, Any]:
    """
    Parse OCR'd receipt text into structured fields.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: Dict[str, Any] = {
        "raw_text": text,
        "merchant": None,
        "cashier": None,
        "bill_no": None,
        "items": [],  # list of {name, qty, price}
        "subtotal": None,
        "total": None,
        "cash": None,
        "change": None,
        "other_fields": {}
    }

    # Merchant heuristic: pick the first uppercase-ish line
    if lines:
        for ln in lines[:6]:
            if len(ln) >= 3 and re.search(r"[A-Za-z]", ln):
                out["merchant"] = ln
                break

    # Cashier (catch OCR variants: "Cashiersn", "Cashier", etc.)
    m = re.search(r"(Cashier\w*\s*[:#]?\s*([A-Za-z0-9\-]+))", text, re.IGNORECASE)
    if m:
        out["cashier"] = m.group(2)

    # Bill number (catch OCR variants like "Billt#")
    m = re.search(r"(Bill\w*\s*[:#]?\s*([A-Za-z0-9\-]+))", text, re.IGNORECASE)
    if m:
        out["bill_no"] = m.group(2)

    # Totals-like fields (allow int or float)
    def find_amount(label_variants):
        for lab in label_variants:
            pat = rf"{lab}\s*[:#=]?\s*([0-9]+(?:[\,\.][0-9]{{1,2}})?)"
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).replace(',', '.')
                try:
                    return float(val)
                except:
                    return None
        return None

    out["subtotal"] = find_amount([r"Sub\s*Total", r"Subtotal", r"Sub-?Total"])
    out["total"]    = find_amount([r"Total", r"TOTAL"])
    out["cash"]     = find_amount([r"Cash", r"CASH"])
    out["change"]   = find_amount([r"Change", r"Balance", r"Return"])

    # Items
    # Match lines like: "Fried Chicken 2 1000" OR "Water 2 2.00"
    item_re = re.compile(r"(.+?)\s+([0-9]{1,3})\s+([0-9]+(?:[\,\.][0-9]{1,2})?)$")

    for ln in lines:
        m = item_re.match(ln)
        if m:
            name = m.group(1).strip()
            try:
                qty = int(m.group(2))
            except:
                qty = 1
            try:
                price = float(m.group(3).replace(',', '.'))
            except:
                price = None
            if name and any(c.isalpha() for c in name) and price is not None:
                out["items"].append({"name": name, "qty": qty, "price": price})

    return out


def format_summary(summary: Dict[str, Any]) -> str:
    """
    Create a pretty string summary for console printing.
    """
    lines = []
    lines.append(f"Merchant: {summary.get('merchant') or '-'}")
    if summary.get('cashier'):
        lines.append(f"Cashier: {summary['cashier']}")
    if summary.get('bill_no'):
        lines.append(f"Bill: {summary['bill_no']}")

    lines.append("Items:")
    items = summary.get('items') or []
    if not items:
        lines.append("  (No structured items parsed; see raw text)")
    else:
        for it in items:
            nm = it.get('name', '').strip()
            qty = it.get('qty', 1)
            price = it.get('price', 0.0)
            lines.append(f"  - {nm:<15} x{qty:<3} {price:.2f}")

    footer = []
    if summary.get('subtotal') is not None: footer.append(f"Sub Total: {summary['subtotal']:.2f}")
    if summary.get('total')    is not None: footer.append(f"Total: {summary['total']:.2f}")
    if summary.get('cash')     is not None: footer.append(f"Cash: {summary['cash']:.2f}")
    if summary.get('change')   is not None: footer.append(f"Change: {summary['change']:.2f}")
    if footer:
        lines.append(" | ".join(footer))

    return "\n".join(lines)
