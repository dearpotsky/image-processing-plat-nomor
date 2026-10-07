import cv2

from app.detection.plate_detector import PlateDetector


MODEL_PATH = "models/plate_detector/plate_detector_yolo11n.pt"
IMAGE_PATH = "dataset/input/test001.jpg"


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(
            f"Gagal membaca image: {IMAGE_PATH}"
        )

    detector = PlateDetector(MODEL_PATH)

    detections = detector.detect(image)

    print(f"Jumlah plate terdeteksi: {len(detections)}")

    for index, detection in enumerate(detections, start=1):
        print(f"\nPlate #{index}")
        print(f"Confidence : {detection['confidence']:.3f}")
        print(f"Bounding box: {detection['box']}")


if __name__ == "__main__":
    main()