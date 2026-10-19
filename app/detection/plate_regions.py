"""
Daftar Kode Wilayah Tanda Nomor Kendaraan Bermotor (TNKB) Indonesia.
Dibuat di file terpisah agar modular dan mudah ditambahkan di kemudian hari.
"""

# Gunakan set (himpunan) agar pencarian sangat cepat (O(1))
VALID_REGION_CODES = {
    # Pulau Jawa
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',
    'AA', 'AB', 'AD', 'AE', 'AG',
    
    # Pulau Sumatera
    'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'BP',
    
    # Pulau Kalimantan
    'DA', 'KB', 'KH', 'KT', 'KU',
    
    # Pulau Sulawesi
    'DB', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DP', 'DT',
    
    # Bali & Nusa Tenggara
    'DK', 'DR', 'EA', 'EB', 'ED', 'DH',
    
    # Pulau Maluku & Papua
    'DE', 'DG', 'PA', 'PB'
}

def is_valid_region(prefix: str) -> bool:
    """
    Mengecek apakah kode wilayah (prefix) valid sesuai aturan TNKB Indonesia.
    """
    return prefix.upper() in VALID_REGION_CODES
