import cv2
import numpy as np

class ImageEnhancer:

    def enhance(
        self,
        image,
        steps=None,
    ):
        if not steps:
            return image

        result = image.copy()

        for step in steps:
            method = step.method
            limit = step.limit
            tile_grid_size = step.tile_grid_size
            
            if method == "none":
                continue
                
            elif method == "clahe":
                clahe = cv2.createCLAHE(
                    clipLimit=limit,
                    tileGridSize=tile_grid_size,
                )
                result = clahe.apply(result)
                
            elif method == "contrast_stretching":
                result = cv2.normalize(
                    result, None, 0, 255, cv2.NORM_MINMAX
                )
                
            elif method == "histogram_equalization":
                result = cv2.equalizeHist(result)
                
            elif method == "lpf":
                k = int(max(1, limit))
                if k % 2 == 0:
                    k += 1
                result = cv2.GaussianBlur(result, (k, k), 0)
                
            elif method == "hpf":
                center_weight = 8 + limit
                kernel = np.array([
                    [-1, -1, -1],
                    [-1, center_weight, -1],
                    [-1, -1, -1]
                ]) / (center_weight - 8)
                result = cv2.filter2D(result, -1, kernel)
                
            elif method == "retinex":
                result_float = np.float64(result) + 1.0
                sigma = max(1, limit * 15)
                blur = cv2.GaussianBlur(result_float, (0, 0), sigma)
                retinex = np.log10(result_float) - np.log10(blur)
                result = cv2.normalize(
                    retinex, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                
            elif method == "gamma_correction":
                gamma = limit if limit > 0 else 1.0
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                result = cv2.LUT(result, table)
                
            elif method == "unsharp_masking":
                blur = cv2.GaussianBlur(result, (5, 5), limit if limit > 0 else 1.0)
                result = cv2.addWeighted(result, 1.5, blur, -0.5, 0)
                
            elif method == "morphology_open":
                # BARU: Menghapus goresan tipis/noise vertikal di plat nomor
                # Nilai k mengikuti parameter limit (direkomendasikan limit=2 atau 3)
                k = int(max(1, limit))
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
                

            else:
                raise ValueError(
                    f"Metode enhancement tidak dikenal: {method}"
                )

        return result
