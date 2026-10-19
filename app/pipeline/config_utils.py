from dataclasses import asdict
import json
import logging
from pathlib import Path

from app.pipeline.config import (
    PipelineConfig,
    DetectorConfig,
    OcrConfig,
    PreprocessingConfig,
    EnhancementConfig,
    EnhancementStep,
    SegmentationConfig,
    SegmentationStep,
    FeatureConfig,
    WatermarkConfig,
    OutputConfig,
    ExperimentConfig,
)

logger = logging.getLogger("config_utils")

DEFAULT_CONFIG_PATH = Path("configs/default_preset.json")
LAST_CONFIG_PATH = Path("configs/last_config.json")


def config_to_dict(config: PipelineConfig) -> dict:
    """Mengubah objek PipelineConfig menjadi dictionary serializable."""
    return asdict(config)


def dict_to_config(data: dict) -> PipelineConfig:
    """Mengubah dictionary menjadi objek PipelineConfig yang valid."""
    config = PipelineConfig()

    # 1. Detector
    if "detector" in data and isinstance(data["detector"], dict):
        det_d = data["detector"]
        config.detector = DetectorConfig(
            enabled=det_d.get("enabled", True),
            model_path=det_d.get("model_path", config.detector.model_path),
            confidence=float(det_d.get("confidence", 0.25)),
            padding_v=float(det_d.get("padding_v", 0.12)),
            padding_left=float(det_d.get("padding_left", 0.25)),
            padding_right=float(det_d.get("padding_right", 0.20)),
        )

    # 2. OCR
    if "ocr" in data and isinstance(data["ocr"], dict):
        ocr_d = data["ocr"]
        config.ocr = OcrConfig(
            engine=str(ocr_d.get("engine", "easyocr")),
            text_threshold=float(ocr_d.get("text_threshold", 0.6)),
            low_text=float(ocr_d.get("low_text", 0.3)),
            mag_ratio=float(ocr_d.get("mag_ratio", 1.2)),
        )

    # 3. Preprocessing
    if "preprocessing" in data and isinstance(data["preprocessing"], dict):
        prep_d = data["preprocessing"]
        config.preprocessing = PreprocessingConfig(
            enabled=prep_d.get("enabled", True),
            resize=prep_d.get("resize", True),
            target_width=int(prep_d.get("target_width", 400)),
            target_height=int(prep_d.get("target_height", 120)),
            interpolation=str(prep_d.get("interpolation", "INTER_CUBIC")),
            grayscale=prep_d.get("grayscale", True),
            normalize=prep_d.get("normalize", True),
        )

    # 4. Enhancement
    if "enhancement" in data and isinstance(data["enhancement"], dict):
        enh_d = data["enhancement"]
        steps = []
        for s in enh_d.get("steps", []):
            if isinstance(s, dict):
                steps.append(
                    EnhancementStep(
                        method=s.get("method", "none"),
                        limit=float(s.get("limit", 0.0)),
                        tile_grid_width=int(s.get("tile_grid_width", 8)),
                        tile_grid_height=int(s.get("tile_grid_height", 8)),
                    )
                )
        config.enhancement = EnhancementConfig(
            enabled=enh_d.get("enabled", True),
            steps=steps,
        )

    # 5. Segmentation
    if "segmentation" in data and isinstance(data["segmentation"], dict):
        seg_d = data["segmentation"]
        steps = []
        for s in seg_d.get("steps", []):
            if isinstance(s, dict):
                steps.append(
                    SegmentationStep(
                        method=s.get("method", "none"),
                        limit=float(s.get("limit", 0.0)),
                    )
                )
        config.segmentation = SegmentationConfig(
            enabled=seg_d.get("enabled", True),
            steps=steps,
            block_size=int(seg_d.get("block_size", 11)),
            c=int(seg_d.get("c", 2)),
            invert=seg_d.get("invert", False),
        )

    # 6. Feature
    if "feature" in data and isinstance(data["feature"], dict):
        feat_d = data["feature"]
        config.feature = FeatureConfig(
            enabled=feat_d.get("enabled", True),
            connected_components=feat_d.get("connected_components", True),
            connectivity=int(feat_d.get("connectivity", 8)),
            min_area=int(feat_d.get("min_area", 20)),
        )

    # 7. Watermark
    if "watermark" in data and isinstance(data["watermark"], dict):
        wm_d = data["watermark"]
        config.watermark = WatermarkConfig(
            enabled=wm_d.get("enabled", True),
        )

    # 8. Output
    if "output" in data and isinstance(data["output"], dict):
        out_d = data["output"]
        config.output = OutputConfig(
            base_dir=str(out_d.get("base_dir", "output")),
            save_detection=out_d.get("save_detection", True),
            save_preprocessing=out_d.get("save_preprocessing", True),
            save_enhancement=out_d.get("save_enhancement", True),
            save_segmentation=out_d.get("save_segmentation", True),
            save_features=out_d.get("save_features", True),
            save_watermark=out_d.get("save_watermark", True),
            save_csv=out_d.get("save_csv", True),
        )

    # 9. Experiment
    if "experiment" in data and isinstance(data["experiment"], dict):
        exp_d = data["experiment"]
        config.experiment = ExperimentConfig(
            name=str(exp_d.get("name", "default")),
            save_results=exp_d.get("save_results", True),
        )

    return config


def save_config_json(config: PipelineConfig, output_path: str | Path):
    """Menyimpan PipelineConfig ke file JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config_to_dict(config)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"Konfigurasi berhasil disimpan ke {path}")


def load_config_json(input_path: str | Path) -> PipelineConfig:
    """Membaca file JSON dan mengembalikan PipelineConfig."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = dict_to_config(data)
    logger.info(f"Konfigurasi berhasil dimuat dari {path}")
    return config


def save_last_config(config: PipelineConfig, file_path: str | Path = LAST_CONFIG_PATH):
    """Menyimpan konfigurasi terakhir aplikasi."""
    try:
        save_config_json(config, file_path)
    except Exception as e:
        logger.warning(f"Gagal menyimpan last_config: {e}")


def load_last_config(file_path: str | Path = LAST_CONFIG_PATH) -> PipelineConfig | None:
    """Memuat konfigurasi terakhir aplikasi jika ada."""
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        return load_config_json(path)
    except Exception as e:
        logger.warning(f"Gagal memuat last_config dari {path}: {e}")
        return None