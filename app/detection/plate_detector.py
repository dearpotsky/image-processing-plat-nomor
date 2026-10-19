from pathlib import Path

from ultralytics import YOLO


class PlateDetector:

    def __init__(
        self,
        config,
        image_size: int = 640,
    ):
        self.config = config
        self.model_path = Path(config.model_path)
        self.confidence = config.confidence
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
            
        height, width = image.shape[:2]

        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=0.45,  # Non-Maximum Suppression agar tidak mendeteksi box ganda bertumpuk
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
                
                plate_w = x2 - x1
                plate_h = y2 - y1

                # Abaikan bounding box anomali yang terlalu kecil (noise/false positive)
                if plate_w < 20 or plate_h < 8:
                    continue

                pad_v = plate_h * self.config.padding_v
                # Gunakan padding asimetris
                pad_left = plate_w * self.config.padding_left
                pad_right = plate_w * self.config.padding_right

                x1_pad = max(0, x1 - pad_left)
                y1_pad = max(0, y1 - pad_v)
                x2_pad = min(width, x2 + pad_right)
                y2_pad = min(height, y2 + pad_v)

                detections.append({
                    "box": (
                        float(x1_pad),
                        float(y1_pad),
                        float(x2_pad),
                        float(y2_pad),
                    ),
                    "confidence": float(
                        confidence
                    ),
                })

        # Urutkan plat dari kiri ke kanan (atau posisi x1) agar teratur
        detections.sort(key=lambda d: d["box"][0])

        return detections