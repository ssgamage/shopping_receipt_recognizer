# shopping_receipt_recognizer
CLI/GUI application that can recognize various shopping receipts


# 📄 Receipt OCR CLI + GUI

This project is a **Receipt OCR Application** developed in Python.  
It can extract text, items, and totals from receipt images using **OpenCV + Tesseract OCR**, with both a **CLI tool** and a **GUI (PySide6)**.

---

## ⚙️ 1. Prerequisites

- **Windows 11**
- **Python 3.13.1** (or Python ≥3.10)
- **Tesseract OCR** installed → [Download](https://github.com/UB-Mannheim/tesseract/wiki)

  - Add its path to system environment or update `config.py`, e.g.:

  ```python
  TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

- **VS Code** (recommended)

---

## 📂 2. Project Setup

Open PowerShell inside your project folder:

```powershell
cd "Your path"

# Create project folder
mkdir receipt_ocr_cli
cd receipt_ocr_cli
```

---

## 🐍 3. Virtual Environment

```powershell
# Create venv
python -m venv venv

# Activate venv
& "Your path/receipt_ocr_cli/venv/Scripts/Activate.ps1"
```

---

## 📦 4. Install Dependencies

```powershell
pip install opencv-python pillow pytesseract numpy
pip install rich
pip install PySide6
pip install pyinstaller
```

---

## ▶️ 5. Running the Application

### (a) CLI – Single Receipt

```powershell
python shoper.py samples\receipt1.png --save-steps --adaptive
```

### (b) CLI – Multi Receipt (experimental)

```powershell
python shoper_multi.py samples\receiptall.png --save-steps --adaptive
```

### (c) GUI

```powershell
python gui_app.py
```

---

## 🖼️ 6. Application Icon

Project uses an icon file in `assets/icon.png` (or `.ico`).  
To change → replace with your own file and update in `gui_app.py`:

```python
from PySide6.QtGui import QIcon
app.setWindowIcon(QIcon("assets/icon.png"))
```

---

## 📦 7. Build Executable

To package GUI app as a Windows `.exe`:

```powershell
pyinstaller --onefile --name=CGV_Group_9 --windowed --icon=assets/icon.ico gui_app.py
```

Executable will be in the `dist/` folder.

---

## 📝 8. Project Structure

```
receipt_ocr_cli/
│
├── assets/                # icons, images
│   └── icon.ico
│
├── config.py              # configuration (tesseract path, params)
├── utils.py               # helpers
├── receipt_processor.py   # single receipt pipeline
├── multi_receipt_processor.py # multi receipt pipeline
├── ocr_engine.py          # OCR wrapper
├── parser.py              # text parsing
├── shoper.py              # CLI single receipt
├── shoper_multi.py        # CLI multi receipt
├── gui_app.py             # GUI app (PySide6)
└── outputs/               # generated OCR steps + JSON
```

---

## 🎓 9. Notes

- CLI + GUI both implemented (pipeline reused).
- Preprocessing uses **grayscale → CLAHE → blur → edge detection → perspective transform → threshold → morphology**.
- Supports both **adaptive** and **Otsu** thresholding (toggle in CLI/GUI).
- **Multi-receipt OCR** is partially working, button in GUI is disabled until stabilized.
- Packaged into standalone EXE with custom icon.

