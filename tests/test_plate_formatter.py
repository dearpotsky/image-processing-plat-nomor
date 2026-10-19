from app.detection.ocr import PlateReader
from app.pipeline.config import OcrConfig

def test_formatter():
    reader = PlateReader(config=OcrConfig())

    test_cases = [
        # (input_raw, expected_output)
        ("7G4689SN", "G 4689 SN"),
        ("7G895N", "G 895 N"),
        ("E500MA", "E 500 MA"),
        ("B1234CD", "B 1234 CD"),
        ("D1889XYZ", "D 1889 XYZ"),
        ("AD9999AB", "AD 9999 AB"),
        ("KT1234AB0528", "KT 1234 AB"), # Masa berlaku 0528 harus dibuang
        ("1B1234CD", "B 1234 CD"),       # Noise 1 di depan huruf B
        ("SD0MF", ""),                  # SD bukan kode wilayah, 0 bukan nomor plat valid
        ("12345", ""),                  # Tidak ada kode wilayah
        ("XYZ", ""),                    # Bukan plat
    ]

    for raw, expected in test_cases:
        actual = reader._format_plate(raw)
        print(f"Input: '{raw}' -> Output: '{actual}' | Expected: '{expected}'")
        assert actual == expected, f"Failed for '{raw}': got '{actual}', expected '{expected}'"

    print("\nALL FORMATTER TEST CASES PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_formatter()
