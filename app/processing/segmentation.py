import cv2


class ImageSegmenter:

    def segment(
        self,
        image,
        method="otsu",
        block_size=11,
        c=2,
        invert=False,
    ):

        if method == "none":
            return None, image

        if method == "otsu":

            threshold, binary = cv2.threshold(
                image,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

        elif method == "adaptive":

            if block_size % 2 == 0:
                raise ValueError(
                    "block_size adaptive "
                    "harus bernilai ganjil."
                )

            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c,
            )

            threshold = None

        else:

            raise ValueError(
                f"Metode segmentation tidak dikenal: "
                f"{method}"
            )

        if invert:

            binary = cv2.bitwise_not(
                binary
            )

        return threshold, binary