import os
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def stem(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def save_step(image, out_dir, base_stem, tag, png_params=None):
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{base_stem}_{tag}.png")
    if image is None:
        return None
    # Normalize to 8-bit if needed
    if image.dtype != 'uint8':
        img = image
        if hasattr(img, "max") and img.max() <= 1.0:
            img = (img * 255).astype('uint8')
        else:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    else:
        img = image
    if png_params is not None:
        cv2.imwrite(out_path, img, png_params)
    else:
        cv2.imwrite(out_path, img)
    return out_path

def show_window(title, img):
    try:
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
    except cv2.error:
        # Headless environment; ignore
        pass

def order_points(pts: np.ndarray) -> np.ndarray:
    # Order corners: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
