"""
test_ocr.py
-----------
Script CLI untuk menguji OCR (EasyOCR / PaddleOCR) secara langsung
tanpa GUI, menggunakan pipeline penuh (deteksi YOLO → crop → OCR).

Jalankan dari root project:
    python test_ocr.py
    python test_ocr.py --engine paddleocr
    python test_ocr.py --engine easyocr --images dataset/test/test001.jpg dataset/test/test002.jpg
    python test_ocr.py --engine paddleocr --raw   # test hanya OCR tanpa deteksi YOLO
"""

import os
# Harus ada SEBELUM import apapun yang me-load Paddle.
# PIR executor PaddlePaddle memiliki bug pada ArrayAttribute<DoubleAttribute>.
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import argparse
import sys
import time
import logging
from pathlib import Path

import cv2

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_ocr")


# ── Argument parsing ─────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Test Multi-OCR Engine")
    parser.add_argument(
        "--engine",
        choices=["easyocr", "paddleocr"],
        default="easyocr",
        help="Pilih OCR engine (default: easyocr)",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=[
            "dataset/test/test001.jpg",
            "dataset/test/test002.jpg",
            "dataset/test/test003.jpg",
        ],
        help="Path gambar yang akan diuji",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Test hanya OCR mentah (tanpa deteksi YOLO, langsung feed gambar ke OCR)",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────
RESET  = "\033[0m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"


def color(text, c):
    return f"{c}{text}{RESET}"


def separator(char="─", n=60):
    print(color(char * n, CYAN))


# ── Test: Full Pipeline (YOLO + OCR) ────────────────────────
def test_full_pipeline(engine: str, image_paths: list[str]):
    from app.pipeline.config import PipelineConfig, OcrConfig
    from app.pipeline.pipeline import ImageProcessingPipeline

    print(color(f"\n{'='*60}", CYAN))
    print(color(f"  MODE  : Full Pipeline (YOLO → Crop → OCR)", BOLD))
    print(color(f"  ENGINE: {engine.upper()}", BOLD))
    print(color(f"{'='*60}", CYAN))

    config = PipelineConfig()
    config.ocr.engine = engine
    # Nonaktifkan fitur non-esensial agar output bersih
    config.enhancement.steps = []
    config.segmentation.steps = []
    config.watermark.enabled = False
    config.feature.enabled = False

    pipeline = ImageProcessingPipeline(config=config)

    total_ok = 0
    total_fail = 0

    for img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            print(color(f"  ✗  File tidak ditemukan: {img_path}", RED))
            total_fail += 1
            continue

        image = cv2.imread(str(p))
        if image is None:
            print(color(f"  ✗  Gagal membaca gambar: {img_path}", RED))
            total_fail += 1
            continue

        separator()
        print(color(f"  File : {p.name}", BOLD))

        t0 = time.perf_counter()
        try:
            result = pipeline.run(image)
        except Exception as e:
            print(color(f"  ✗  Error saat pipeline.run(): {e}", RED))
            total_fail += 1
            continue
        elapsed = time.perf_counter() - t0

        if not result.plates:
            print(color("  ⚠  Tidak ada plat terdeteksi.", YELLOW))
            total_fail += 1
        else:
            for plate in result.plates:
                if plate.text:
                    print(color(f"  ✓  Plat {plate.index}: {plate.text}  "
                                f"(conf OCR: {plate.text_confidence:.2f})", GREEN))
                    total_ok += 1
                else:
                    print(color(f"  ⚠  Plat {plate.index}: OCR kosong", YELLOW))
                    total_fail += 1

        print(f"     Waktu: {elapsed:.3f}s")

    separator("═")
    print(color(f"  Hasil  : {total_ok} berhasil | {total_fail} gagal", BOLD))
    separator("═")


# ── Test: Raw OCR (tanpa YOLO) ───────────────────────────────
def test_raw_ocr(engine: str, image_paths: list[str]):
    from app.pipeline.config import OcrConfig
    from app.detection.ocr import PlateReader

    print(color(f"\n{'='*60}", CYAN))
    print(color(f"  MODE  : Raw OCR (tanpa YOLO, langsung ke OCR)", BOLD))
    print(color(f"  ENGINE: {engine.upper()}", BOLD))
    print(color(f"{'='*60}", CYAN))

    config = OcrConfig()
    config.engine = engine
    reader = PlateReader(config=config)

    for img_path in image_paths:
        p = Path(img_path)
        if not p.exists():
            print(color(f"  ✗  File tidak ditemukan: {img_path}", RED))
            continue

        image = cv2.imread(str(p))
        if image is None:
            print(color(f"  ✗  Gagal membaca gambar: {img_path}", RED))
            continue

        separator()
        print(color(f"  File : {p.name}", BOLD))

        t0 = time.perf_counter()
        try:
            text, prob = reader.read_text(image)
        except Exception as e:
            print(color(f"  ✗  Error: {e}", RED))
            continue
        elapsed = time.perf_counter() - t0

        if text:
            print(color(f"  ✓  Teks: {text}  (conf: {prob:.2f})", GREEN))
        else:
            print(color("  ⚠  Hasil OCR kosong.", YELLOW))
        print(f"     Waktu: {elapsed:.3f}s")

    separator("═")


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if args.raw:
        test_raw_ocr(args.engine, args.images)
    else:
        test_full_pipeline(args.engine, args.images)
