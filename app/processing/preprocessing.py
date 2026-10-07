import cv2


class Preprocessor:

    def __init__(
        self,
        target_size=(400, 120),
        grayscale=True,
        normalize=True,
        interpolation="INTER_AREA",
    ):
        self.target_size = target_size
        self.use_grayscale = grayscale
        self.use_normalize = normalize
        self.interpolation = interpolation

    def resize(self, image):

        interpolation_method = getattr(
            cv2,
            self.interpolation,
            cv2.INTER_AREA,
        )

        return cv2.resize(
            image,
            self.target_size,
            interpolation=interpolation_method,
        )

    def grayscale(self, image):

        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    def normalize(self, image):

        return cv2.normalize(
            image,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        )

    def process(self, image):

        result = self.resize(image)

        if self.use_grayscale:
            result = self.grayscale(result)

        if self.use_normalize:
            result = self.normalize(result)

        return result