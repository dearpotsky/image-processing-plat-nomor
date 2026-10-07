from pathlib import Path

import cv2


IMAGE_PATH = Path("dataset/input/test001.jpg")


def test_image_exists():
    assert IMAGE_PATH.exists(), f"File tidak ditemukan: {IMAGE_PATH}"


def test_image_can_be_read():
    image = cv2.imread(str(IMAGE_PATH))

    assert image is not None, "OpenCV gagal membaca gambar"
    assert image.size > 0, "Gambar kosong"


if __name__ == "__main__":
    test_image_exists()
    test_image_can_be_read()

    print("✓ File gambar ditemukan")
    print("✓ Gambar berhasil dibaca OpenCV")
    print("✓ STEP 2 berhasil")