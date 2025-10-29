# ocr_engine.py
from typing import Optional, Union

import numpy as np
import pytesseract
from PIL import Image

import config
from receipt_processor import ProcessResult


def configure_tesseract(psm: Optional[int] = None) -> str:
    """
    Configure Tesseract command with OEM and PSM settings.
    """
    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    cfg = f'--oem {config.TESSERACT_OEM} --psm {psm if psm is not None else config.TESSERACT_PSM}'
    return cfg


def ocr_image(
    image_or_result: Union[ProcessResult, np.ndarray, Image.Image],
    lang: Optional[str] = None,
    psm: Optional[int] = None
) -> str:
    """
    Run Tesseract OCR.

    Parameters:
    - image_or_result: can be
        * ProcessResult → use its .ocr_ready, fallback to warped/gray if needed
        * numpy.ndarray (OpenCV image) → OCR directly
        * PIL.Image.Image → OCR directly
    - lang: language(s) to pass to Tesseract
    - psm: page segmentation mode override

    Returns:
    - Extracted text as string
    """
    cfg = configure_tesseract(psm=psm)
    language = lang or config.TESSERACT_LANG

    # --- Case 1: ProcessResult ---
    if isinstance(image_or_result, ProcessResult):
        # Preferred: thresholded result
        img = image_or_result.ocr_ready
        text = pytesseract.image_to_string(img, lang=language, config=cfg).strip()

        # Fallback: warped if no text
        if not text and image_or_result.warped is not None:
            text = pytesseract.image_to_string(image_or_result.warped, lang=language, config=cfg).strip()

        # Final fallback: gray if still empty
        if not text and hasattr(image_or_result, "gray"):
            text = pytesseract.image_to_string(image_or_result.gray, lang=language, config=cfg).strip()

        return text

    # --- Case 2: raw numpy image ---
    if isinstance(image_or_result, np.ndarray):
        return pytesseract.image_to_string(image_or_result, lang=language, config=cfg).strip()

    # --- Case 3: PIL Image ---
    if isinstance(image_or_result, Image.Image):
        return pytesseract.image_to_string(image_or_result, lang=language, config=cfg).strip()

    raise TypeError(f"Unsupported input type for OCR: {type(image_or_result)}")
