"""
Aplikasi Utama Pengenal Isyarat BISINDO & Isyarat Informal (Webcam Real-Time)
Versi 3.0: Sistem Pengenalan Terpadu (Unified Auto-Recognition)
Mengenali secara otomatis dan simultan:
1. Isyarat Dinamis:
   - HALO (Melambai & Tersenyum 😊👋)
   - DEWASA (Ketuk Pelipis 2x + Angkat)
   - MENYENANGKAN (Ayunan Aroma Kedua Tangan)
2. Isyarat Kustom Statis:
   - FINGER HEART / SARANGHAE (Jempol & Telunjuk Silang 🫰)
   - LOVE DUA TANGAN (Simbol Hati 🫶)
   - AKU (Telunjuk ke Dada)
3. Alfabet Jari BISINDO Lengkap:
   - A, B, C, I, L, O, U, V, W, Y
"""
import sys
import time
import cv2
import numpy as np

from holistic_detector import HolisticDetector
from gesture_rules import GestureRules
from motion_tracker import MotionTracker

def draw_guide_zones(frame: np.ndarray, results: any):
    """Menggambar visual guide area pelipis dan dada."""
    if not results.pose_landmarks:
        return

    h, w, _ = frame.shape
    pose = results.pose_landmarks
    shoulder_l = pose[11]
    shoulder_r = pose[12]
    ear_l = pose[7]
    ear_r = pose[8]

    # Target Pelipis (DEWASA)
    cv2.circle(frame, (int(ear_l.x * w), int((ear_l.y - 0.04) * h)), 38, (255, 180, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, (int(ear_r.x * w), int((ear_r.y - 0.04) * h)), 38, (255, 180, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Pelipis: DEWASA", (int(ear_r.x * w) - 60, int((ear_r.y - 0.08) * h)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 0), 1, cv2.LINE_AA)

    # Target Dada (AKU, LOVE, SARANGHAE, MENYENANGKAN)
    chest_cx = int((shoulder_l.x + shoulder_r.x) / 2.0 * w)
    chest_cy = int(((shoulder_l.y + shoulder_r.y) / 2.0 + 0.06) * h)
    chest_rx = int(abs(shoulder_l.x - shoulder_r.x) * w * 0.40)
    chest_ry = int(0.14 * h)

    cv2.rectangle(frame,
                  (chest_cx - chest_rx, chest_cy - chest_ry),
                  (chest_cx + chest_rx, chest_cy + chest_ry),
                  (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Dada: AKU / LOVE / SARANGHAE / MENYENANGKAN",
                (chest_cx - chest_rx, chest_cy - chest_ry - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

def main():
    print("=" * 75)
    print("   SISTEM PENGENAL ISYARAT BISINDO & ISYARAT INFORMAL v3.1")
    print("   Dukungan: HALO, KAMU, AKU, DEWASA, MENYENANGKAN, SARANGHAE, LOVE")
    print("   + Alfabet BISINDO: A, B, C, I, L, O, U, V, W, Y")
    print("=" * 75)
    print("Membuka kamera webcam... Tekan 'Q' atau 'ESC' untuk keluar.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera webcam!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HolisticDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    tracker = MotionTracker(max_frames=45)

    active_gesture = "Menunggu isyarat..."
    gesture_color = (180, 180, 180)
    last_detect_time = 0.0

    # Debounce filter responsif (3 frame ~ 0.1 detik)
    static_candidate = None
    static_consecutive_frames = 0
    REQUIRED_STATIC_FRAMES = 3

    fps_time = time.time()
    fps_counter = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_time = time.time()

        # Ekstraksi MediaPipe Holistic
        results = detector.process(frame)
        tracker.update(results)

        display_frame = detector.draw_landmarks(frame, results)
        draw_guide_zones(display_frame, results)

        detected_this_frame = None
        debug_message = ""

        # =====================================================================
        # 1. EVALUASI ISYARAT DINAMIS (HALO, DEWASA, MENYENANGKAN)
        # =====================================================================
        # Cek HALO (Melambai / Greeting)
        is_halo, detail_halo = tracker.check_halo(results)
        if is_halo:
            detected_this_frame = detail_halo
            debug_message = detail_halo

        # Cek DEWASA (Ketuk Pelipis 2x + Angkat)
        if not detected_this_frame:
            is_dewasa, detail_dewasa = tracker.check_dewasa(results)
            if is_dewasa:
                detected_this_frame = "DEWASA (Ketuk Pelipis 2x + Angkat)"
                debug_message = detail_dewasa
            elif "DEWASA:" in detail_dewasa:
                debug_message = detail_dewasa

        # Cek MENYENANGKAN (Ayunan Kedua Tangan)
        if not detected_this_frame:
            is_senang, detail_senang = tracker.check_menyenangkan(results)
            if is_senang:
                detected_this_frame = "MENYENANGKAN (Ayunan Aroma Masakan)"
                debug_message = detail_senang
            elif "MENYENANGKAN:" in detail_senang and not debug_message:
                debug_message = detail_senang

        # =====================================================================
        # 2. EVALUASI ISYARAT STATIS & ALFABET
        # =====================================================================
        current_static = None

        # Cek LOVE DUA TANGAN (Gambar 5 - 🫶)
        if not detected_this_frame:
            is_love, detail_love = GestureRules.check_love_two_hands(results)
            if is_love:
                current_static = detail_love
                debug_message = detail_love

        # Cek FINGER HEART / SARANGHAE (Gambar 4 - 🫰)
        if (not detected_this_frame) and (not current_static):
            is_fh, detail_fh = GestureRules.check_finger_heart(results)
            if is_fh:
                current_static = detail_fh
                debug_message = detail_fh

        # Cek AKU (Telunjuk ke Dada)
        if (not detected_this_frame) and (not current_static):
            is_aku, detail_aku = GestureRules.check_aku(results)
            if is_aku:
                current_static = detail_aku
                debug_message = detail_aku

        # Cek KAMU (Telunjuk menunjuk ke Kamera 🫵)
        if (not detected_this_frame) and (not current_static):
            is_kamu, detail_kamu = GestureRules.check_kamu(results)
            if is_kamu:
                current_static = detail_kamu
                debug_message = detail_kamu

        # Cek ALFABET BISINDO LENGKAP (A, B, C, I, L, O, U, V, W, Y)
        if (not detected_this_frame) and (not current_static):
            is_alpha, detail_alpha = GestureRules.check_alphabet(results)
            if is_alpha:
                current_static = detail_alpha
                debug_message = detail_alpha

        # Terapkan Debounce Cepat
        if current_static:
            if current_static == static_candidate:
                static_consecutive_frames += 1
            else:
                static_candidate = current_static
                static_consecutive_frames = 1

            if static_consecutive_frames >= REQUIRED_STATIC_FRAMES:
                detected_this_frame = current_static
        else:
            static_candidate = None
            static_consecutive_frames = 0

        # Kunci tampilan jika terdeteksi (tahan 2 detik)
        if detected_this_frame:
            active_gesture = detected_this_frame
            # Warna Hijau untuk Kustom, Cyan untuk Alfabet
            gesture_color = (0, 255, 120) if "Alfabet" not in detected_this_frame else (0, 220, 255)
            last_detect_time = time.time()
        else:
            if time.time() - last_detect_time > 2.0:
                active_gesture = "Menunggu isyarat..."
                gesture_color = (180, 180, 180)

        # =====================================================================
        # HUD DISPLAY DI LAYAR
        # =====================================================================
        cv2.rectangle(display_frame, (0, 0), (w, 88), (20, 22, 28), -1)
        cv2.line(display_frame, (0, 88), (w, 88), (55, 60, 75), 2)

        cv2.putText(display_frame, "ISYARAT TERDETEKSI:", (25, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 170, 185), 1, cv2.LINE_AA)
        cv2.putText(display_frame, active_gesture, (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, gesture_color, 2, cv2.LINE_AA)

        # Sensor & Status Info
        hands_count = sum([1 for h_obj in [results.left_hand_landmarks, results.right_hand_landmarks] if h_obj])
        sensor_info = f"Pose: {'OK' if results.pose_landmarks else '-'} | Tangan: {hands_count}/2 | FPS: {fps}"
        cv2.putText(display_frame, sensor_info, (w - 480, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 190, 205), 1, cv2.LINE_AA)

        if debug_message:
            cv2.putText(display_frame, f">> {debug_message}", (w - 480, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 240, 255), 1, cv2.LINE_AA)

        # Bottom Bar
        cv2.rectangle(display_frame, (0, h - 35), (w, h), (18, 20, 24), -1)
        cv2.putText(display_frame, "Tekan 'C' untuk Reset Tracker | Tekan 'Q' atau 'ESC' untuk Keluar",
                    (25, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 160, 170), 1, cv2.LINE_AA)

        cv2.imshow("BISINDO & Gesture Recognition v3.0 - Antigravity", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            print("\nMenutup aplikasi...")
            break
        elif key in [ord('c'), ord('C')]:
            tracker.reset_state_dewasa()
            tracker.history.clear()
            print("Tracker di-reset.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Aplikasi selesai.")

if __name__ == "__main__":
    main()
