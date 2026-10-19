import os
os.environ.setdefault("FLAGS_enable_pir_api", "0")

from pathlib import Path
from app.pipeline.config import PipelineConfig, EnhancementStep, SegmentationStep
from app.pipeline.config_utils import (
    config_to_dict,
    dict_to_config,
    save_config_json,
    load_config_json,
)

def test_config_save_load():
    config = PipelineConfig()
    config.detector.enabled = False  # Direct plate mode
    config.preprocessing.resize = False # Keep original resolution
    config.ocr.engine = "paddleocr"
    config.ocr.text_threshold = 0.55
    config.enhancement.steps = [
        EnhancementStep(method="clahe", limit=2.5, tile_grid_width=8, tile_grid_height=8)
    ]
    config.segmentation.steps = [
        SegmentationStep(method="otsu", limit=0.0)
    ]
    config.experiment.name = "test_experiment"

    test_path = Path("configs/test_saved_config.json")
    save_config_json(config, test_path)
    assert test_path.exists(), "File config tidak terbentuk"

    loaded_config = load_config_json(test_path)
    assert loaded_config.detector.enabled == False, "detector.enabled mismatch"
    assert loaded_config.preprocessing.resize == False, "preprocessing.resize mismatch"
    assert loaded_config.ocr.engine == "paddleocr", "ocr.engine mismatch"
    assert loaded_config.ocr.text_threshold == 0.55, "ocr.text_threshold mismatch"
    assert len(loaded_config.enhancement.steps) == 1, "enhancement steps mismatch"
    assert loaded_config.enhancement.steps[0].method == "clahe", "enhancement method mismatch"
    assert len(loaded_config.segmentation.steps) == 1, "segmentation steps mismatch"

    # Cleanup
    if test_path.exists():
        test_path.unlink()

    print("ALL TESTS PASSED: test_config_save_load OK!")

if __name__ == "__main__":
    test_config_save_load()
