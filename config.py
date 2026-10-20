"""
Konfigurasi dan Konstanta untuk Pengenal Isyarat BISINDO & Isyarat Informal
"""

# Indeks MediaPipe Pose Landmark yang relevan
POSE_NOSE = 0
POSE_LEFT_EYE = 2
POSE_RIGHT_EYE = 5
POSE_LEFT_EAR = 7
POSE_RIGHT_EAR = 8
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16

# Indeks MediaPipe Hand Landmark
HAND_WRIST = 0
HAND_THUMB_CMC = 1
HAND_THUMB_MCP = 2
HAND_THUMB_IP = 3
HAND_THUMB_TIP = 4
HAND_INDEX_MCP = 5
HAND_INDEX_PIP = 6
HAND_INDEX_DIP = 7
HAND_INDEX_TIP = 8
HAND_MIDDLE_MCP = 9
HAND_MIDDLE_PIP = 10
HAND_MIDDLE_DIP = 11
HAND_MIDDLE_TIP = 12
HAND_RING_MCP = 13
HAND_RING_PIP = 14
HAND_RING_DIP = 15
HAND_RING_TIP = 16
HAND_PINKY_MCP = 17
HAND_PINKY_PIP = 18
HAND_PINKY_DIP = 19
HAND_PINKY_TIP = 20

# Daftar Gestur Kustom yang didukung
CUSTOM_SIGNS = [
    "HALO (MELAMBAI & TERSENYUM)",
    "KAMU (MENUNJUK KE KAMERA)",
    "AKU (MENUNJUK KE DADA)",
    "MENYENANGKAN",
    "DEWASA",
    "FINGER HEART (SARANGHAE)",
    "LOVE (DUA TANGAN)"
]

# Daftar Alfabet BISINDO Dasar
BISINDO_ALPHABET = [
    "A", "B", "C", "D", "E", "I", "L", "O", "U", "V"
]

# Parameter Buffer Gerak Dinamis (dalam jumlah frame)
MOTION_BUFFER_SIZE = 30  # Sekitar 1 detik pada 30 FPS
DEBOUNCE_FRAMES = 5      # Jumlah frame berturut-turut untuk memvalidasi isyarat statis
