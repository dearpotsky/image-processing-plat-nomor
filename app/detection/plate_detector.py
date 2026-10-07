from pathlib import Path

from ultralytics import YOLO


class PlateDetector:

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        image_size: int = 640,
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.image_size = image_size

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: "
                f"{self.model_path}"
            )

        self.model = YOLO(
            str(self.model_path)
        )

    def detect(self, image):

        if image is None:
            raise ValueError(
                "Image tidak boleh None."
            )

        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            for box, confidence in zip(
                boxes,
                confidences,
            ):

                x1, y1, x2, y2 = box

                detections.append({
                    "box": (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ),
                    "confidence": float(
                        confidence
                    ),
                })

        return detections