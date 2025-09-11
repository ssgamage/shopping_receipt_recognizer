# Configuration for the receipt OCR pipeline

# If Tesseract isn't on PATH, set full path here (Windows example)
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = None  # leave None to auto-detect from PATH

# Default OCR options
TESSERACT_LANG = "eng"
TESSERACT_OEM = 3         # 3 = Default, based on what is available
TESSERACT_PSM = 6         # Assume a uniform block of text

# Preprocessing options
GAUSSIAN_BLUR_KSIZE = (3, 3)
CANNY_THRESHOLDS = (50, 150)
ADAPTIVE_BLOCK_SIZE = 31    # must be odd
ADAPTIVE_C = 10

# Morphology
MORPH_KERNEL_SIZE = (3, 3)
MORPH_OPEN_ITER = 1
MORPH_CLOSE_ITER = 1

# Output directory
OUTPUT_DIR = "outputs"

# Save 8-bit PNG for steps
# cv2.IMWRITE_PNG_COMPRESSION = 16; 0-9 (higher = smaller file)
PNG_PARAMS = [int(16), 9]
