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
