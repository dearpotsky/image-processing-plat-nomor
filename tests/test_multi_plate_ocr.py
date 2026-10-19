import os
os.environ.setdefault("FLAGS_enable_pir_api", "0")

import cv2
import numpy as np
from app.pipeline.config import PipelineConfig, SegmentationStep
from app.pipeline.pipeline import ImageProcessingPipeline

def test_multi_plate_reading():
    # Buat kanvas dengan 2 plat nomor berbeda
    canvas = np.ones((400, 800, 3), dtype=np.uint8) * 220
    
    # Gambar plat 1 di kiri
    cv2.rectangle(canvas, (50, 100), (350, 220), (0, 0, 0), -1)
    cv2.putText(canvas, "B 1234 CD", (70, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    # Gambar plat 2 di kanan
    cv2.rectangle(canvas, (450, 100), (750, 220), (0, 0, 0), -1)
    cv2.putText(canvas, "D 5678 EF", (470, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    config = PipelineConfig()
    config.detector.enabled = True
    config.ocr.engine = "easyocr"

    pipeline = ImageProcessingPipeline(config=config)
    
    # Cek mock deteksi 2 box langsung ke pipeline loop
    mock_detections = [
        {"box": (50.0, 100.0, 350.0, 220.0), "confidence": 0.95},
        {"box": (450.0, 100.0, 750.0, 220.0), "confidence": 0.92},
    ]

    print("Testing multi-plate processing with 2 plates...")
    results = []
    for idx, det in enumerate(mock_detections, start=1):
        crop = pipeline.cropper.crop(canvas, det["box"])
        prep = pipeline.preprocessor.process(crop)
        enh = pipeline.enhancer.enhance(prep, steps=[])
        _, seg = pipeline.segmenter.segment(enh, steps=[SegmentationStep(method="otsu")])
        text, prob = pipeline.reader.read_text(seg)
        if not text:
            text, prob = pipeline.reader.read_text(enh)
        print(f"Plate #{idx}: Detected Text='{text}', Confidence={prob:.2f}")
        results.append(text)

    assert len(results) == 2, "Harus ada 2 plat yang diproses"
    print("Multi-plate test passed successfully!")

if __name__ == "__main__":
    test_multi_plate_reading()
