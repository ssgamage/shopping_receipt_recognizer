"""
receipt_processor.py
---------------------
Handles preprocessing of shopping receipt images for OCR.

Pipeline:
1. Load image & convert to grayscale
2. Apply CLAHE for local contrast enhancement
3. Apply Gaussian blur to reduce noise
4. Detect edges (Canny) for contour finding
5. Attempt perspective correction (deskew basics)
6. Apply thresholding (Adaptive vs Otsu)
7. Apply morphological operations (opening + closing)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config
from utils import four_point_transform, save_step, show_window, stem


@dataclass
class ProcessResult:
    """Holds intermediate results of receipt processing."""
    ocr_ready: np.ndarray              # Final processed image for OCR
    gray: np.ndarray                   # Grayscale fallback
    steps: Dict[str, str] = field(default_factory=dict)  # Saved step images
    warped: Optional[np.ndarray] = None


class ReceiptProcessor:
    def __init__(self, save_steps: bool = True, show: bool = False, adaptive: bool = True):
        """
        Initialize the processor.
        - save_steps: save intermediate images
        - show: display steps in window
        - adaptive: use Adaptive Threshold (True) or Otsu (False)
        """
        self.save_steps = save_steps
        self.show = show
        self.adaptive = adaptive

    def process(self, image_path: str) -> Tuple[str, ProcessResult]:
        """Main preprocessing pipeline for receipt images."""
        base = stem(image_path)
        steps_saved = {}
        out_dir = config.OUTPUT_DIR

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        if self.save_steps:
            steps_saved["orig"] = save_step(image, out_dir, base, "orig", png_params=config.PNG_PARAMS)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.save_steps:
            steps_saved["gray"] = save_step(gray, out_dir, base, "gray", png_params=config.PNG_PARAMS)
        if self.show:
            show_window("Grayscale", gray)

        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        if self.save_steps:
            steps_saved["clahe"] = save_step(eq, out_dir, base, "clahe", png_params=config.PNG_PARAMS)
        if self.show:
            show_window("CLAHE", eq)

        # Apply Gaussian Blur to reduce noise
        blur = cv2.GaussianBlur(eq, config.GAUSSIAN_BLUR_KSIZE, 0)
        if self.save_steps:
            steps_saved["blur"] = save_step(blur, out_dir, base, "blur", png_params=config.PNG_PARAMS)
        if self.show:
            show_window("Blur", blur)

        # Detect edges using Canny
        edges = cv2.Canny(blur, config.CANNY_THRESHOLDS[0], config.CANNY_THRESHOLDS[1])
        if self.save_steps:
            steps_saved["edges"] = save_step(edges, out_dir, base, "edges", png_params=config.PNG_PARAMS)
        if self.show:
            show_window("Edges", edges)

        # Find contours and attempt perspective correction (deskew basics)
        warped = None
        cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        receipt_quad = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  # Found quadrilateral (possible receipt boundary)
                receipt_quad = approx.reshape(4, 2).astype("float32")
                break

        if receipt_quad is not None:
            try:
                warped = four_point_transform(gray, receipt_quad)
                if self.save_steps:
                    steps_saved["warped"] = save_step(warped, out_dir, base, "warped", png_params=config.PNG_PARAMS)
                if self.show:
                    show_window("Warped", warped)
            except Exception:
                warped = None

        # Use warped image if available, otherwise fallback to grayscale
        working = warped if warped is not None else gray

        # Apply thresholding (Adaptive or Otsu)
        if self.adaptive:
            th = cv2.adaptiveThreshold(
                working, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                config.ADAPTIVE_BLOCK_SIZE,
                config.ADAPTIVE_C
            )
            th_tag = "th_adaptive"
        else:
            _, th = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th_tag = "th_otsu"

        if self.save_steps:
            steps_saved[th_tag] = save_step(th, out_dir, base, th_tag, png_params=config.PNG_PARAMS)
        if self.show:
            show_window("Threshold", th)

        # Apply morphological operations (opening then closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL_SIZE)
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=config.MORPH_OPEN_ITER)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_CLOSE_ITER)
        if self.save_steps:
            steps_saved["morph_open"] = save_step(opened, out_dir, base, "morph_open", png_params=config.PNG_PARAMS)
            steps_saved["morph_close"] = save_step(closed, out_dir, base, "morph_close", png_params=config.PNG_PARAMS)
        if self.show:
            show_window("Morph Open", opened)
            show_window("Morph Close", closed)

        # Return final processed result
        result = ProcessResult(
            ocr_ready=closed,
            gray=gray,
            steps=steps_saved,
            warped=warped
        )
        return base, result



