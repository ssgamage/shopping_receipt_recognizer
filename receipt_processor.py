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

