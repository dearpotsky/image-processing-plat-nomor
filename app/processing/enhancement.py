import cv2


class ImageEnhancer:

    def enhance(
        self,
        image,
        method="clahe",
        clip_limit=2.0,
        tile_grid_size=(8, 8),
    ):

        if method == "none":
            return image

        if method == "clahe":

            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=tile_grid_size,
            )

            return clahe.apply(image)

        raise ValueError(
            f"Metode enhancement tidak dikenal: "
            f"{method}"
        )