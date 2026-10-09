import cv2
import numpy as np


class ImageSegmenter:

    def segment(
        self,
        image,
        steps=None,
        block_size=11,
        c=2,
        invert=False,
    ):
        if not steps:
            return None, image

        result = image.copy()
        threshold_val = None

        for step in steps:
            method = step.method
            limit = step.limit
            
            if method == "none":
                continue
                
            # Pastikan format grayscale jika metode membutuhkannya dan citra belum grayscale
            is_gray = len(result.shape) == 2
            
            if method == "global":
                t, result = cv2.threshold(
                    result, limit, 255, cv2.THRESH_BINARY
                )
                threshold_val = t
                
            elif method == "otsu":
                if not is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                t, result = cv2.threshold(
                    result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                threshold_val = t
                
            elif method == "adaptive":
                if not is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                if block_size % 2 == 0:
                    block_size += 1
                result = cv2.adaptiveThreshold(
                    result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, c
                )
                threshold_val = None
                
            elif method == "sobel":
                if not is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(result, cv2.CV_16S, 1, 0, ksize=3)
                grad_y = cv2.Sobel(result, cv2.CV_16S, 0, 1, ksize=3)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                result = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                
            elif method == "canny":
                if not is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                t1 = max(0, limit - 50)
                t2 = min(255, limit + 50)
                result = cv2.Canny(result, t1, t2)
                
            elif method == "prewitt":
                if not is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
                kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                grad_x = cv2.filter2D(result, cv2.CV_16S, kernel_x)
                grad_y = cv2.filter2D(result, cv2.CV_16S, kernel_y)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                result = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                
            elif method == "kmeans":
                k = max(2, int(limit))
                pixel_values = result.reshape((-1, 1 if is_gray else 3))
                pixel_values = np.float32(pixel_values)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, centers = cv2.kmeans(
                    pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
                )
                centers = np.uint8(centers)
                result = centers[labels.flatten()].reshape(result.shape)
                
            elif method == "watershed":
                if is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((3, 3), np.uint8)
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                sure_bg = cv2.dilate(opening, kernel, iterations=3)
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
                ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                unknown = cv2.subtract(sure_bg, sure_fg)
                ret, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                markers = cv2.watershed(result, markers)
                result[markers == -1] = [255, 0, 0]
                if is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    
            elif method == "fuzzy_cmeans":
                # FCM is slow natively, using kmeans as approximation for simplicity
                k = max(2, int(limit))
                pixel_values = result.reshape((-1, 1 if is_gray else 3)).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, centers = cv2.kmeans(
                    pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
                )
                centers = np.uint8(centers)
                result = centers[labels.flatten()].reshape(result.shape)
                
            elif method == "region_growing":
                # Simple flood fill from center
                if not is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                h, w = result.shape
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(result, mask, (w//2, h//2), 255, loDiff=limit, upDiff=limit)
                
            elif method == "region_split_merge":
                # Approximation via PyrMeanShiftFiltering
                if is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                sp = max(10, limit)
                sr = max(20, limit * 2)
                result = cv2.pyrMeanShiftFiltering(result, sp, sr)
                if is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    
            elif method == "active_contour":
                # Approximation via GrabCut for demonstration (snakes requires external lib)
                if is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                mask = np.zeros(result.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                h, w = result.shape[:2]
                rect = (10, 10, w-20, h-20)
                cv2.grabCut(result, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
                result = result * mask2[:, :, np.newaxis]
                if is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    
            elif method == "clear_border":
                # Harus dalam format grayscale biner
                if not is_gray:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                # Pastikan ini gambar hitam-putih
                _, bin_result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
                
                # Cek polaritas: gambar plat nomor bisa berupa background putih atau background hitam
                # Kita cek mayoritas piksel di pinggiran gambar untuk menentukan warna background
                h, w = bin_result.shape
                import numpy as np
                border_pixels = np.concatenate((bin_result[0, :], bin_result[-1, :], bin_result[:, 0], bin_result[:, -1]))
                white_count = np.count_nonzero(border_pixels)
                is_bg_white = white_count > (len(border_pixels) / 2)
                
                # Agar kontur teks terdeteksi (findContours mencari objek putih),
                # pastikan teks/foreground berwarna putih
                if is_bg_white:
                    # Teks adalah hitam, background putih. Kita invert.
                    foreground = cv2.bitwise_not(bin_result)
                else:
                    # Teks adalah putih, background hitam. Biarkan.
                    foreground = bin_result.copy()
                
                contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                    # Jika menyentuh pinggir gambar
                    if x <= 2 or y <= 2 or (x + w_cnt) >= (w - 2) or (y + h_cnt) >= (h - 2):
                        cv2.drawContours(foreground, [cnt], -1, 0, -1)
                        
                # Hapus noise kecil (bercak mengambang bebas)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < 40:
                        foreground[labels == i] = 0
                        
                # Kembalikan ke polaritas aslinya
                if is_bg_white:
                    result = cv2.bitwise_not(foreground)
                else:
                    result = foreground
                
            else:
                raise ValueError(
                    f"Metode segmentation tidak dikenal: {method}"
                )

        if invert:
            result = cv2.bitwise_not(result)

        return threshold_val, result