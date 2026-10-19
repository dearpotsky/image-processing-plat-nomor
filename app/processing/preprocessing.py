import cv2

class Preprocessor:

    def __init__(
        self,
        target_size=(400, 120),
        grayscale=True,
        normalize=True,
        interpolation="INTER_CUBIC",  # Diubah ke INTER_CUBIC agar teks scale-up tidak pecah
        resize=True,
    ):
        self.target_size = target_size
        self.use_grayscale = grayscale
        self.use_normalize = normalize
        self.interpolation = interpolation
        self.use_resize = resize

    def resize(self, image):
        interpolation_method = getattr(
            cv2,
            self.interpolation,
            cv2.INTER_CUBIC,
        )
        
        # Pertahankan aspect ratio berdasarkan height agar karakter tidak melar/distorsi
        target_h = self.target_size[1]
        h, w = image.shape[:2]
        
        if h == 0:
            return image
            
        ratio = target_h / float(h)
        target_w = int(w * ratio)
        
        # Batasi lebar maksimal jika terlalu ekstrem (misalnya lebih dari 800px)
        # agar tidak merusak EasyOCR
        if target_w > 800:
            target_w = 800
            
        return cv2.resize(
            image,
            (target_w, target_h),
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
        if self.use_resize:
            result = self.resize(image)
        else:
            result = image.copy()

        if self.use_grayscale:
            result = self.grayscale(result)

        if self.use_normalize:
            result = self.normalize(result)

        return result
