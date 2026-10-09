"""
Daftar Kode Wilayah Tanda Nomor Kendaraan Bermotor (TNKB) Indonesia.
Dibuat di file terpisah agar modular dan mudah ditambahkan di kemudian hari.
"""

# Gunakan set (himpunan) agar pencarian sangat cepat (O(1))
VALID_REGION_CODES = {
    # Pulau Jawa
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Z',
    'AA', 'AB', 'AD',
    
    # Pulau Sumatera
    'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'BP',
    
    # Pulau Kalimantan, Sulawesi, Maluku, Nusa Tenggara, Bali
    'DA', 'DB', 'DC', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DP', 'DR', 'DT',
    'EA', 'ED',
    'KB', 'KH', 'KT', 'KU',
    
    # Pulau Papua
    'PA', 'PB'
    
    # Catatan: Jika ada plat daerah lain yang kurang (misal AG, AE, EB, dll), 
    # cukup tambahkan kode hurufnya di atas.
}

def is_valid_region(prefix: str) -> bool:
    """
    Mengecek apakah kode wilayah (prefix) valid sesuai aturan TNKB Indonesia.
    """
    return prefix.upper() in VALID_REGION_CODES
