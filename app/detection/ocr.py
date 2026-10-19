import os
import re
import logging
import numpy as np

# ── Paddle flags — harus di-set sebelum paddle library dimuat ──
# Bug PIR: ConvertPirAttribute2RuntimeAttribute tidak support
# ArrayAttribute<DoubleAttribute> yang digunakan angle classifier.
os.environ["FLAGS_enable_pir_api"]                   = "0"
os.environ["FLAGS_pir_apply_shape_optimization_pass"] = "0"
os.environ["PADDLE_DISABLE_PIR"]                      = "1"
os.environ.setdefault("FLAGS_use_stride_kernel",       "0")

logger = logging.getLogger("ocr")

# ============================================================
# BASE OCR (Strategy Interface)
# ============================================================

class BaseOCR:
    """Abstract base untuk semua engine OCR."""

    def readtext_raw(self, image, allowlist: str, config):
        """
        Membaca teks mentah dari gambar.

        Returns:
            list[tuple[bbox, text, prob]]:
                - bbox : list 4 titik [[x,y], ...]
                - text : str
                - prob : float
        """
        raise NotImplementedError


# ============================================================
# EASYOCR ENGINE
# ============================================================

class EasyOCREngine(BaseOCR):
    def __init__(self, gpu: bool = True):
        import easyocr
        logger.info("Memuat EasyOCR reader...")
        self._reader = easyocr.Reader(['en'], gpu=gpu)

    def readtext_raw(self, image, allowlist: str, config):
        results = self._reader.readtext(
            image,
            allowlist=allowlist,
            text_threshold=config.text_threshold,
            low_text=config.low_text,
            mag_ratio=config.mag_ratio,
            decoder='beamsearch',
            beamWidth=5,
        )
        # Format sudah sesuai [(bbox, text, prob), ...]
        return results


# ============================================================
# PADDLEOCR ENGINE
# ============================================================

class PaddleOCREngine(BaseOCR):
    """
    Engine PaddleOCR v3.x.
    - Menggunakan predict() API (bukan ocr()) yang diperkenalkan di v3.
    - use_angle_cls dihapus karena model classifier memicu bug PIR
      (ArrayAttribute<DoubleAttribute> belum diimplementasikan).
    """

    def __init__(self):
        from paddleocr import PaddleOCR
        logger.info("Memuat PaddleOCR reader...")
        # use_angle_cls sengaja TIDAK diset (default False di v3)
        # untuk menghindari bug PIR pada model classifier.
        self._reader = PaddleOCR(lang='en')

    def readtext_raw(self, image, allowlist: str, config):
        # Pastikan gambar uint8 RGB (PaddleOCR mengharapkan RGB)
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            if image.ndim == 2:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 3:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Debug: log image info
        logger.debug(f"PaddleOCR image dtype={image.dtype}, shape={image.shape}")
        # Gunakan API ocr() standar PaddleOCR (v3) – menghasilkan list[Result].
        raw = self._reader.ocr(image)
        logger.debug(f"PaddleOCR raw output count: {len(raw) if raw else 0}")
        results = self._parse_v3(raw)
        if not results:
            # fallback ke parser v2 bila format tidak sesuai atau kosong
            results = self._parse_v2(raw)
        logger.debug(f"PaddleOCR parsed results count: {len(results) if results else 0}")
        return results

    # ----------------------------------------------------------
    # Parsers
    # ----------------------------------------------------------

    def _parse_v3(self, raw) -> list:
        """
        PaddleOCR v3.x predict() mengembalikan list[Result] dengan atribut:
          .dt_polys   : list of ndarray(4,2)
          .rec_texts  : list of str
          .rec_scores : list of float
        """
        results = []
        if not raw:
            return results

        for page in raw:
            if page is None:
                continue
            # Jika atribut v3 tersedia
            if hasattr(page, 'rec_texts'):
                polys  = getattr(page, 'dt_polys',   None) or []
                texts  = getattr(page, 'rec_texts',  None) or []
                scores = getattr(page, 'rec_scores', None) or []
                for bbox, text, prob in zip(polys, texts, scores):
                    bbox_list = [list(map(float, pt)) for pt in bbox]
                    results.append((bbox_list, str(text), float(prob)))
            else:
                # predict() mengembalikan format v2 dalam wrapper
                results.extend(self._parse_v2([page]))

        return results

    def _parse_v2(self, raw) -> list:
        """
        PaddleOCR v2 ocr() mengembalikan:
          list[ list[ [bbox_4pts, (text, score)] ] ]
        """
        results = []
        if not raw:
            return results

        first = raw[0]
        page = first if isinstance(first, list) else []
        if not page:
            return results

        for item in page:
            if item is None:
                continue
            try:
                bbox, (text, prob) = item
                results.append((bbox, str(text), float(prob)))
            except (TypeError, ValueError):
                continue

        return results


# ============================================================
# FACTORY
# ============================================================

_ENGINE_REGISTRY = {
    "easyocr": EasyOCREngine,
    "paddleocr": PaddleOCREngine,
}

VALID_OCR_ENGINES = set(_ENGINE_REGISTRY.keys())


def _build_engine(engine_name: str) -> BaseOCR:
    """Membuat instance engine OCR berdasarkan nama string."""
    key = engine_name.lower().strip()
    cls = _ENGINE_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"OCR engine tidak dikenal: '{engine_name}'. "
            f"Pilihan valid: {sorted(VALID_OCR_ENGINES)}"
        )
    return cls()


# ============================================================
# PLATE READER (main entry point)
# ============================================================

class PlateReader:
    """
    Membaca teks plat nomor dari citra crop.

    Parameter:
        config (OcrConfig): konfigurasi OCR termasuk ``engine``,
            ``text_threshold``, ``low_text``, ``mag_ratio``.
    """

    def __init__(self, config=None):
        if config is None:
            from app.pipeline.config import OcrConfig
            config = OcrConfig()
        self.config = config

        engine_name = getattr(config, "engine", "easyocr")
        logger.info(f"PlateReader menggunakan engine: {engine_name}")
        self._engine: BaseOCR = _build_engine(engine_name)

    # ----------------------------------------------------------
    # PUBLIC
    # ----------------------------------------------------------

    def read_text(self, image):
        """
        Membaca dan memformat teks plat nomor dari citra.

        Returns:
            tuple[str, float]: (teks_terformat, rata_rata_probabilitas)
        """
        if image is None:
            return "", 0.0

        # Allowlist karakter legal plat Indonesia
        allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '

        results = self._engine.readtext_raw(image, allowlist, self.config)

        if not results:
            return "", 0.0

        # ── Filtering baris utama ───────────────────────────────

        def get_h(bbox):
            pts = np.array(bbox)
            return float(
                ((pts[3][1] - pts[0][1]) + (pts[2][1] - pts[1][1])) / 2
            )

        def get_y_center(bbox):
            pts = np.array(bbox)
            return float((pts[0][1] + pts[3][1]) / 2)

        main_res = max(results, key=lambda x: get_h(x[0]))
        main_h = get_h(main_res[0])
        main_y = get_y_center(main_res[0])

        valid_results = []
        for res in results:
            bbox, text, prob = res
            h = get_h(bbox)
            y = get_y_center(bbox)
            if abs(y - main_y) < (main_h * 0.6) and h >= (0.30 * main_h):
                valid_results.append(res)

        if not valid_results:
            return "", 0.0

        # Urutkan kiri ke kanan
        valid_results.sort(key=lambda x: min([pt[0] for pt in x[0]]))
        raw_text = "".join([res[1].replace(" ", "") for res in valid_results])

        formatted_text = self._format_plate(raw_text)

        prob = sum(res[2] for res in valid_results) / len(valid_results)

        return formatted_text, prob

    # ----------------------------------------------------------
    # PRIVATE
    # ----------------------------------------------------------

    def _format_plate(self, text: str) -> str:
        """
        Memformat teks hasil OCR menjadi format TNKB Indonesia yang valid:
        [KODE_WILAYAH] [NOMOR_POLISI] [SERI_HURUF]
        Contoh: "B 1234 CD", "G 4689 SN", "E 500 MA"
        """
        if not text:
            return ""

        text = text.upper()
        # Bersihkan karakter aneh selain huruf dan angka
        text = re.sub(r'[^A-Z0-9]', '', text)
        if len(text) < 3:
            return ""

        from app.detection.plate_regions import VALID_REGION_CODES, is_valid_region

        to_number = {
            'O': '0', 'D': '0', 'I': '1', 'L': '1',
            'S': '5', 'Z': '2', 'B': '8', 'G': '6',
        }
        to_letter = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '4': 'A',
            '5': 'S',
            '6': 'G',
            '8': 'B',
        }

        # 1. Cari KODE WILAYAH resmi (Prefix)
        # Seringkali di awal teks ada noise berupa baut atau lis tepi plat (misal angka 7, 1, atau simbol).
        # Kita scan dari depan (maksimal 3 karakter pertama) untuk menemukan kode wilayah yang sah.
        prefix = None
        start_idx = 0

        for i in range(min(4, len(text))):
            # Prioritas 1: Cek 2 huruf wilayah (misal AA, AB, AD, BK, KT, dll.)
            if i + 2 <= len(text):
                two_char = text[i:i+2]
                if two_char.isalpha() and two_char in VALID_REGION_CODES:
                    prefix = two_char
                    start_idx = i + 2
                    break
            # Prioritas 2: Cek 1 huruf wilayah (misal A, B, D, E, F, G, H, L, N, dll.)
            if i + 1 <= len(text):
                one_char = text[i]
                if one_char.isalpha() and one_char in VALID_REGION_CODES:
                    prefix = one_char
                    start_idx = i + 1
                    break

        # Jika tidak ditemukan kode wilayah Indonesia resmi di awal, tolak (bukan plat nomor yang valid)
        if not prefix:
            return ""

        remainder = text[start_idx:]
        if not remainder:
            return ""

        # 2. Pisahkan bagian NOMOR POLISI (1-4 digit angka) dan SUFFIX (huruf akhir)
        # Tangkap kelompok angka (atau huruf mirip angka) setelah kode wilayah
        match = re.match(r'^([0-9ODILSZBG]{1,4})(.*)$', remainder)
        if not match:
            return ""

        raw_middle = match.group(1)
        raw_suffix = match.group(2)

        # Ubah middle menjadi angka murni
        middle = "".join([c if c.isdigit() else to_number.get(c, c) for c in raw_middle])
        # Di Indonesia nomor plat adalah 1 s.d. 9999 (tidak boleh kosong atau hanya "0")
        if not middle or middle == "0" or not middle.isdigit():
            return ""

        # 3. Format bagian SUFFIX (1-3 huruf)
        # Buang masa berlaku di akhir (contoh: 0528, 1226, atau 4 digit angka di ujung)
        raw_suffix = re.sub(r'[0-9]{4}$', '', raw_suffix)
        raw_suffix = re.sub(r'[^A-Z0-9]', '', raw_suffix)

        suffix = ""
        for c in raw_suffix[:3]:
            if c.isalpha():
                suffix += c
            elif c in to_letter:
                suffix += to_letter[c]

        return f"{prefix} {middle} {suffix}".strip()