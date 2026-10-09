from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlateResult:
    """
    Menyimpan seluruh hasil processing untuk satu plat.
    """

    index: int
    box: tuple
    confidence: float

    crop: Any = None
    preprocessing: Any = None
    enhancement: Any = None
    segmentation: Any = None
    
    text: str = ""
    text_confidence: float = 0.0

    features: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """
    Menyimpan seluruh hasil dari satu image.
    """

    image: Any = None
    watermarked: Any = None

    plates: list[PlateResult] = field(
        default_factory=list
    )

    def add_plate(self, plate: PlateResult):
        self.plates.append(plate)

    @property
    def plate_count(self):
        return len(self.plates)


@dataclass
class BatchResult:
    """
    Menyimpan hasil processing dari banyak image.
    """

    images: list[PipelineResult] = field(
        default_factory=list
    )

    def add_image(self, result: PipelineResult):
        self.images.append(result)

    @property
    def image_count(self):
        return len(self.images)

    @property
    def total_plate_count(self):
        return sum(
            result.plate_count
            for result in self.images
        )