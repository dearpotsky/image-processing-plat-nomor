from pathlib import Path
import cv2

class PlateCropper:
    """
    Memotong area license plate berdasarkan bounding box hasil detector.
    """

    def crop(self, image, box):
        """
        Crop image berdasarkan bounding box [x1, y1, x2, y2].

        Parameters
        ----------
        image : numpy.ndarray
            Gambar input.
        box : list atau tuple
            Koordinat [x1, y1, x2, y2] yang sudah melalui padding di detector.

        Returns
        -------
        numpy.ndarray
            Hasil crop.
        """
        x1, y1, x2, y2 = map(float, box)
        height, width = image.shape[:2]

        # Pastikan koordinat tidak keluar dari frame
        x1 = max(0, min(int(x1), width))
        x2 = max(0, min(int(x2), width))
        y1 = max(0, min(int(y1), height))
        y2 = max(0, min(int(y2), height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box tidak valid.")

        # Potong gambar
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            raise ValueError("Hasil crop kosong.")

        return crop

    def save(self, crop, output_path):
        """
        Menyimpan hasil crop ke file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        success = cv2.imwrite(
            str(output_path),
            crop
        )

        if not success:
            raise IOError(
                f"Gagal menyimpan crop: {output_path}"
            )
