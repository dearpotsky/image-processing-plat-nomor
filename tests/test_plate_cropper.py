from pathlib import Path

import cv2
from ultralytics import YOLO

from app.detection.plate_cropper import PlateCropper


MODEL_PATH = Path(
    "models/plate_detector/plate_detector_yolo11n.pt"
)

IMAGE_PATH = Path(
    "dataset/input/test001.jpg"
)

OUTPUT_DIR = Path(
    "output/00_detection"
)


def main():
    # Load image
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Gambar tidak dapat dibaca: {IMAGE_PATH}"
        )

    # Load detector
    model = YOLO(str(MODEL_PATH))

    # Detection
    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=640,
        conf=0.25,
        verbose=False,
    )

    cropper = PlateCropper()

    total = 0

    for result in results:
        for index, box in enumerate(
            result.boxes,
            start=1
        ):
            coordinates = box.xyxy[0].tolist()

            crop = cropper.crop(
                image,
                coordinates
            )

            output_path = (
                OUTPUT_DIR
                / f"plate_{index:02d}.jpg"
            )

            cropper.save(
                crop,
                output_path
            )

            total += 1

            print(
                f"✓ Plate #{index} disimpan: "
                f"{output_path}"
            )

            print(
                f"  Ukuran crop: "
                f"{crop.shape[1]} x {crop.shape[0]}"
            )

    print()
    print(
        f"✓ Total crop berhasil: {total}"
    )


if __name__ == "__main__":
    main()