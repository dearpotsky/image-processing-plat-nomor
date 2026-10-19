import os
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import cv2
import numpy as np
from app.pipeline.config import PipelineConfig, EnhancementStep, SegmentationStep
from app.pipeline.pipeline import ImageProcessingPipeline

def test_direct_plate_mode():
    # Buat dummy image plat
    dummy_image = np.ones((100, 300, 3), dtype=np.uint8) * 200
    cv2.putText(dummy_image, "B 1234 CD", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    config = PipelineConfig()
    config.detector.enabled = False  # BYPASS YOLO DETECTION
    config.preprocessing.resize = False # Jaga resolusi asli
    config.enhancement.steps = []
    config.segmentation.steps = [SegmentationStep(method="otsu")]
    config.ocr.engine = "easyocr"

    pipeline = ImageProcessingPipeline(config=config)
    result = pipeline.run(dummy_image)

    assert result.plate_count == 1, f"Expected 1 plate, got {result.plate_count}"
    plate = result.plates[0]
    assert plate.crop.shape == dummy_image.shape, f"Expected crop to match full image shape, got {plate.crop.shape}"
    print(f"Direct plate test success! Crop shape: {plate.crop.shape}, Text detected: '{plate.text}'")

if __name__ == "__main__":
    test_direct_plate_mode()
