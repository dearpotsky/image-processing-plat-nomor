from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path(
    "models/plate_detector/plate_detector_yolo11n.pt"
)

IMAGE_PATH = Path(
    "dataset/input/test001.jpg"
)


def main():
    # Cek file
    assert MODEL_PATH.exists(), (
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

    assert IMAGE_PATH.exists(), (
        f"Gambar tidak ditemukan: {IMAGE_PATH}"
    )

    print("✓ Model ditemukan")
    print("✓ Gambar ditemukan")

    # Load model
    model = YOLO(str(MODEL_PATH))

    print("✓ Model berhasil dimuat")
    print()

    print("Class model:")

    for class_id, class_name in model.names.items():
        print(f"  {class_id}: {class_name}")

    # Inference
    print()
    print("Menjalankan deteksi...")

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        save=True,
        verbose=False,
    )

    for result in results:
        print()
        print(f"Jumlah plate terdeteksi: {len(result.boxes)}")

        for index, box in enumerate(result.boxes, start=1):
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print()
            print(f"Plate #{index}")
            print(f"Confidence : {confidence:.3f}")
            print(
                f"Bounding box: "
                f"({x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f})"
            )


if __name__ == "__main__":
    main()