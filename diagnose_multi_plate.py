import os
os.environ.setdefault("FLAGS_enable_pir_api", "0")
import cv2
from pathlib import Path
from app.pipeline.config import PipelineConfig
from app.pipeline.pipeline import ImageProcessingPipeline

def diagnose():
    config = PipelineConfig()
    config.detector.confidence = 0.2
    config.detector.enabled = True
    config.ocr.engine = "easyocr"
    
    pipeline = ImageProcessingPipeline(config=config)

    test_dir = Path("dataset/test")
    images = list(test_dir.glob("*.jpg"))[:20]
    
    multi_plate_found = False
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        detections = pipeline.detector.detect(img)
        if len(detections) > 1:
            multi_plate_found = True
            print(f"\n[MULTI-PLATE DETECTED] File: {img_path.name} -> {len(detections)} plates detected!")
            result = pipeline.run(img)
            for i, p in enumerate(result.plates, start=1):
                print(f"  Plate #{i}: Box={p.box}, YOLO Conf={p.confidence:.2f}, Text='{p.text}', TextConf={p.text_confidence:.2f}")

    if not multi_plate_found:
        print("No multi-plate found in first 20 images, scanning all 100 images...")
        all_images = list(test_dir.glob("*.jpg"))
        for img_path in all_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            detections = pipeline.detector.detect(img)
            if len(detections) > 1:
                print(f"\n[MULTI-PLATE DETECTED] File: {img_path.name} -> {len(detections)} plates detected!")
                result = pipeline.run(img)
                for i, p in enumerate(result.plates, start=1):
                    print(f"  Plate #{i}: Box={p.box}, YOLO Conf={p.confidence:.2f}, Text='{p.text}', TextConf={p.text_confidence:.2f}")
                break

if __name__ == "__main__":
    diagnose()
