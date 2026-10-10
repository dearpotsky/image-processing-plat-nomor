import easyocr
import numpy as np

class PlateReader:
    def __init__(self, config=None):
        # Initialize with OCR configuration (fallback to defaults)
        if config is None:
            from app.pipeline.config import OcrConfig
            config = OcrConfig()
        self.config = config
        # Initialize EasyOCR reader (GPU enabled)
        self.reader = easyocr.Reader(['en'], gpu=True)

    def read_text(self, image):
        if image is None:
            return "", 0.0
            
        # Biarkan tanpa spasi agar EasyOCR memotong per blok kata (Prefix, Middle, Suffix)
        allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        # Gunakan parameter wajar untuk EasyOCR. Karena masalah L hilang ternyata dari cropping YOLO!
        # mag_ratio sedikit di atas default untuk membaca U yang kecil.
        results = self.reader.readtext(
            image,
            allowlist=allowlist,
            text_threshold=self.config.text_threshold,
            low_text=self.config.low_text,
            mag_ratio=self.config.mag_ratio
        )
        
        if not results:
            return "", 0.0
            
        valid_results = []
        
        # 1. Cari teks utama (paling tinggi) untuk menentukan garis Y-center plat nomor
        def get_h(bbox):
            return ((bbox[3][1] - bbox[0][1]) + (bbox[2][1] - bbox[1][1])) / 2
            
        def get_y_center(bbox):
            return (bbox[0][1] + bbox[3][1]) / 2

        main_res = max(results, key=lambda x: get_h(x[0]))
        main_h = get_h(main_res[0])
        main_y = get_y_center(main_res[0])
        
        # 2. Filter teks: harus berada di baris yang sama dengan teks utama (Y-center berdekatan)
        # dan memiliki tinggi minimal 30% dari teks utama agar huruf L/U yang kecil selamat.
        for res in results:
            bbox, text, prob = res
            h = get_h(bbox)
            y = get_y_center(bbox)
            
            # Toleransi Y-center: max bergeser setengah tinggi huruf utama
            if abs(y - main_y) < (main_h * 0.6) and h >= (0.30 * main_h):
                valid_results.append(res)
                    
        if not valid_results:
            return "", 0.0

        # Gabungkan semua teks utama yang tersisa tanpa spasi bawaan
        # Pastikan diurutkan dari kiri ke kanan (berdasarkan sumbu x paling kiri)
        valid_results.sort(key=lambda x: min([pt[0] for pt in x[0]]))
        raw_text = "".join([res[1].replace(" ", "") for res in valid_results])
        
        formatted_text = self._format_plate(raw_text)
        
        # Ambil rata-rata probabilitas
        prob = sum([res[2] for res in valid_results]) / len(valid_results)
        
        return formatted_text, prob

    def _format_plate(self, text):
        text = text.upper()
        if not text:
            return ""

        # Mapping error OCR yang disesuaikan untuk TNKB Indonesia
        to_number = {'O': '0', 'D': '0', 'I': '1', 'L': '1', 'S': '5', 'Z': '2', 'B': '8', 'G': '6'}
        # Karena wilayah Jatim (Surabaya) L sering terbaca 1, kita petakan 1 -> L.
        # 0 sering dari huruf D (Bandung) atau O. Kita petakan 0 -> D.
        to_letter = {'0': 'D', '5': 'S', '2': 'Z', '8': 'B', '6': 'G', '4': 'A', '1': 'L'}

        prefix = ""
        middle = ""
        suffix = ""
        
        state = 0
        for char in text:
            if state == 0:
                # 1. KODE WILAYAH (Depan: Maksimal 2 Huruf)
                if char.isalpha() or char in to_letter:
                    if len(prefix) < 2:
                        prefix += char if char.isalpha() else to_letter[char]
                elif char.isdigit() or char in to_number:
                    # Masuk ke area angka
                    state = 1
                    middle += char if char.isdigit() else to_number[char]
                    
            elif state == 1:
                # 2. NOMOR POLISI (Tengah: Maksimal 4 Angka)
                if char.isdigit() or char in to_number:
                    if len(middle) < 4:
                        middle += char if char.isdigit() else to_number[char]
                    else:
                        # Angka sudah 4, paksa pindah ke suffix (bisa jadi angka kelima ini harusnya huruf)
                        state = 2
                        if char in to_letter and len(suffix) < 3:
                            suffix += to_letter[char]
                elif char.isalpha() or char in to_letter:
                    # Ketemu huruf murni, berarti nomor polisi sudah selesai
                    state = 2
                    if len(suffix) < 3:
                        suffix += char if char.isalpha() else to_letter[char]
                        
            elif state == 2:
                # 3. SERI HURUF (Belakang: Maksimal 3 Huruf)
                if char.isalpha() or char in to_letter:
                    if len(suffix) < 3:
                        suffix += char if char.isalpha() else to_letter[char]

        # Pembersihan border plat nomor yang ikut terbaca sebagai 'I' atau 'L' di akhir
        if (suffix.endswith('I') or suffix.endswith('L')) and len(suffix) > 1:
            suffix = suffix[:-1]

        # Validasi Kode Wilayah (Prefix) berdasarkan daftar modular
        from app.detection.plate_regions import is_valid_region
        if not is_valid_region(prefix):
            return ""

        return f"{prefix} {middle} {suffix}".strip()
