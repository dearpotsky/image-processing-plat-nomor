import cv2
import numpy as np


class FeatureAnalyzer:

    def __init__(
        self,
        min_area=20,
        connectivity=8,
    ):
        self.min_area = min_area
        self.connectivity = connectivity

    def analyze(self, image):

        if image is None:
            raise ValueError(
                "Image tidak boleh None."
            )

        if len(image.shape) != 2:
            raise ValueError(
                "Image harus grayscale/binary."
            )

        height, width = image.shape

        foreground_pixels = np.count_nonzero(
            image == 255
        )

        total_pixels = image.size

        foreground_ratio = (
            foreground_pixels /
            total_pixels
        )

        mean_intensity = float(
            np.mean(image)
        )

        std_intensity = float(
            np.std(image)
        )

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                image,
                connectivity=self.connectivity,
            )
        )

        components = []

        for label in range(
            1,
            num_labels,
        ):

            x = int(
                stats[
                    label,
                    cv2.CC_STAT_LEFT
                ]
            )

            y = int(
                stats[
                    label,
                    cv2.CC_STAT_TOP
                ]
            )

            component_width = int(
                stats[
                    label,
                    cv2.CC_STAT_WIDTH
                ]
            )

            component_height = int(
                stats[
                    label,
                    cv2.CC_STAT_HEIGHT
                ]
            )

            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA
                ]
            )

            if component_height == 0:
                continue

            aspect_ratio = (
                component_width /
                component_height
            )

            components.append({
                "area": area,
                "width": component_width,
                "height": component_height,
                "aspect_ratio": aspect_ratio,
                "x": x,
                "y": y,
            })

        valid_components = [
            component
            for component in components
            if component["area"]
            >= self.min_area
        ]

        return {
            "width": width,
            "height": height,
            "foreground_ratio":
                foreground_ratio,
            "mean_intensity":
                mean_intensity,
            "std_intensity":
                std_intensity,
            "total_components":
                len(components),
            "valid_components":
                len(valid_components),
            "components":
                valid_components,
        }