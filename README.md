# 🖐️ Hand Sign Alphabet & Informal Gesture Tracker

Sistem pengenalan real-time untuk **Alfabet Jari BISINDO** dan **Bahasa Isyarat Informal / Kustom** berbasis Python, OpenCV, dan MediaPipe Holistic (Pose, Hands, & Face Mesh).

> [!WARNING]
> **CATATAN PENTING (STATUS PROYEK):**  
> Proyek ini **masih belum sempurna** dan sedang berada dalam **tahap pengembangan aktif (Work in Progress / Eksperimental)**. Beberapa gestur mungkin masih membutuhkan penyesuaian sudut kamera, pencahayaan yang cukup, atau kalibrasi ambang batas posisi tangan lebih lanjut.

---

## 📌 Daftar Isyarat yang Didukung

### 1. Isyarat Informal / Kustom (Sehari-hari)
* 👋 **HALO / DADAH**: Satu telapak tangan terbuka lurus melambai di samping kepala/bahu. Dilengkapi deteksi senyuman (*Melambai & Tersenyum 😊👋*).
* 🫵 **KAMU**: Satu tangan dengan jari telunjuk mengarah/menunjuk lurus ke layar/kamera.
* ☝️ **AKU**: Satu tangan dengan jari telunjuk menempel/menunjuk ke area dada sendiri.
* 🙌 **MENYENANGKAN**: Kedua tangan terbuka di depan dada/leher mengayun perlahan ke arah wajah (*seperti chef mencium aroma masakan*).
* 👨🫴 **DEWASA**: Satu tangan dengan jari melengkung (*claw*) menempel di samping pelipis, diketuk 2 kali, lalu diangkat/dilepas ke atas kepala.
* 🫰 **FINGER HEART / SARANGHAE**: Ujung jempol dan telunjuk bersilangan membentuk hati kecil, 3 jari lainnya terlipat rapat.
* 🫶 **LOVE DUA TANGAN**: Kedua tangan bertemu di depan dada membentuk busur hati klasik.

### 2. Alfabet Jari BISINDO
Mendukung pengenalan otomatis huruf alfabet jari:
* **A**: 1 tangan (kepalan tinju dengan jempol tegak di samping) atau 2 tangan (segitiga atap A).
* **B**: 4 jari (telunjuk s/d kelingking) tegak lurus rapat ke atas.
* **C**: Tangan melengkung setengah lingkaran membentuk busur C.
* **I**: Hanya kelingking tegak lurus ke atas, jari lain mengepal.
* **L**: Jempol & telunjuk membentuk sudut siku 90 derajat.
* **O**: Semua ujung jari melengkung bertemu dengan ujung jempol membentuk lingkaran O.
* **U**: Telunjuk & jari tengah tegak lurus dan rapat berdampingan.
* **V**: Telunjuk & jari tengah tegak lurus dan terbuka (peace).
* **W**: Tiga jari (telunjuk, tengah, manis) tegak lurus terbuka.
* **Y**: Jempol & kelingking terbuka lebar (*shaka*), 3 jari tengah terlipat.

---

## 🛠️ Instalasi & Prasyarat

1. **Clone Repositori**:
   ```bash
   git clone https://github.com/raatuuuu26/hand-sign.git
   cd hand-sign
   ```

2. **Pasang Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Model MediaPipe Holistic akan diunduh secara otomatis dari server resmi Google pada saat program dijalankan pertama kali).*

---

## 🚀 Cara Menjalankan

### Melalui Batch Script (Windows)
Cukup klik dua kali file:
```
run.bat
```

### Melalui Terminal / Command Prompt
```bash
python app.py
```

### Kontrol Aplikasi:
* Tekan **`C`** : Mereset riwayat gerak pelacak.
* Tekan **`Q`** atau **`ESC`** : Keluar dari aplikasi.

---

## 📁 Struktur File
* `app.py`: Program visual webcam real-time dengan tampilan Heads-Up Display (HUD) dan area sasaran panduan.
* `holistic_detector.py`: Pembungkus MediaPipe Holistic untuk ekstraksi pose tubuh, wajah, dan kedua tangan.
* `gesture_rules.py`: Logika pengenalan geometris dan spasial untuk isyarat statis dan alfabet.
* `motion_tracker.py`: Pelacak gerak dinamis multi-fase (*FSM*) dengan proteksi anti-benturan (*anti-collision*).
* `config.py`: Definisi indeks landmark dan daftar isyarat.
