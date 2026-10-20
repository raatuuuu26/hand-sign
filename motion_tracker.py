"""
Pelacak Gerak Dinamis (Temporal Motion Tracker)
Anti-Collision Engine:
- Memastikan lambaian tangan (HALO) TIDAK PERNAH tertukar dengan DEWASA atau isyarat lain
- Mendeteksi DEWASA secara akurat hanya jika jari melengkung dan menempel di pelipis
- Mendeteksi MENYENANGKAN dengan kedua tangan mengayun di dada
"""
import time
from collections import deque
import numpy as np
from typing import Dict, Any, Optional, Tuple
from holistic_detector import HolisticDetector

class MotionTracker:
    def __init__(self, max_frames: int = 45):
        self.max_frames = max_frames
        self.history = deque(maxlen=max_frames)
        
        # State machine untuk DEWASA
        self.state_dewasa = {
            "phase": "IDLE",      # IDLE -> TAPPING -> READY_RELEASE
            "tap_count": 0,
            "last_action_time": 0.0,
            "prev_hand_pos": None,
            "moving_towards_head": False
        }

    def update(self, results: Any):
        """Menambahkan frame baru ke riwayat gerakan."""
        current_data = {
            "time": time.time(),
            "pose": results.pose_landmarks,
            "face": results.face_landmarks,
            "left": results.left_hand_landmarks,
            "right": results.right_hand_landmarks
        }
        self.history.append(current_data)

    def reset_state_dewasa(self):
        self.state_dewasa["phase"] = "IDLE"
        self.state_dewasa["tap_count"] = 0
        self.state_dewasa["last_action_time"] = 0.0
        self.state_dewasa["prev_hand_pos"] = None
        self.state_dewasa["moving_towards_head"] = False

    @staticmethod
    def check_smile(face_landmarks) -> bool:
        """Mendeteksi apakah pengguna tersenyum."""
        if not face_landmarks or len(face_landmarks) < 300:
            return False

        lip_l = face_landmarks[61]
        lip_r = face_landmarks[291]
        eye_l = face_landmarks[33]
        eye_r = face_landmarks[263]
        upper_lip = face_landmarks[13]

        mouth_w = np.sqrt((lip_l.x - lip_r.x)**2 + (lip_l.y - lip_r.y)**2)
        eye_w = np.sqrt((eye_l.x - eye_r.x)**2 + (eye_l.y - eye_r.y)**2)

        if eye_w < 0.01:
            return False

        smile_ratio = mouth_w / eye_w
        avg_corner_y = (lip_l.y + lip_r.y) / 2.0
        corners_raised = avg_corner_y < (upper_lip.y + 0.035)

        return (smile_ratio > 0.44) and corners_raised

    def check_halo(self, results: Any) -> Tuple[bool, str]:
        """
        Mendeteksi isyarat HALO / MELAMBAI:
        Satu telapak tangan terbuka (semua jari lurus ke atas) terangkat di samping kepala/bahu.
        Baik melambai kiri-kanan maupun pose menyapa sambil tersenyum.
        """
        candidates = []
        if results.right_hand_landmarks:
            candidates.append(("Kanan", results.right_hand_landmarks))
        if results.left_hand_landmarks:
            candidates.append(("Kiri", results.left_hand_landmarks))

        # HALO hanya untuk 1 tangan terangkat
        if len(candidates) != 1:
            return False, "Bukan Halo"

        hand_label, hand = candidates[0]

        # 1. Pastikan kelima jari TERBUKA LURUS (Ciri khas telapak melambai)
        states = HolisticDetector.get_finger_extended_states(hand)
        open_fingers = sum([1 for f in ["index", "middle", "ring", "pinky"] if states[f]])

        if open_fingers < 3:
            return False, "Bukan Halo (Jari tidak terbuka)"

        # 2. Tangan harus terangkat di area bahu atau kepala
        wrist = hand[0]
        if results.pose_landmarks:
            shoulder_y = max(results.pose_landmarks[11].y, results.pose_landmarks[12].y)
            if wrist.y > (shoulder_y + 0.12):
                return False, "Bukan Halo (Tangan terlalu rendah)"

        # 3. Analisis gerak horizontal (lambaian tangan)
        recent = list(self.history)[-16:]
        xs = [f[hand_label.lower()][8].x for f in recent if f.get(hand_label.lower())]

        waving = False
        if len(xs) >= 6:
            x_std = float(np.std(xs))
            diffs = np.diff(xs)
            flips = int(np.count_nonzero(np.diff(np.sign(diffs))))
            waving = (x_std > 0.006 and flips >= 1) or (x_std > 0.012)

        smiling = self.check_smile(results.face_landmarks)

        # Jika tangan terbuka terangkat dan (melambai atau tersenyum atau dipertahankan terbuka)
        if open_fingers >= 4:
            if smiling:
                return True, "HALO (Melambai & Tersenyum 😊👋)"
            elif waving:
                return True, "HALO (Melambai 👋)"
            elif len(xs) >= 8:
                return True, "HALO (Lambaian Tangan 👋)"

        return False, "Bukan Halo"

    def check_dewasa(self, results: Any) -> Tuple[bool, str]:
        """
        Mendeteksi isyarat DEWASA:
        Satu tangan menempel di pelipis/rambut samping dengan JARI MELENGKUNG (claw/curved),
        bukan telapak terbuka lurus. Diketuk 2 kali lalu diangkat ke atas kepala.
        """
        now = time.time()

        if self.state_dewasa["phase"] != "IDLE" and (now - self.state_dewasa["last_action_time"] > 3.0):
            self.reset_state_dewasa()

        if not results.pose_landmarks:
            return False, "Menunggu pose tubuh..."

        nose = results.pose_landmarks[0]
        ear_l = results.pose_landmarks[7]
        ear_r = results.pose_landmarks[8]

        candidates = []
        if results.left_hand_landmarks:
            candidates.append((results.left_hand_landmarks, ear_l))
        if results.right_hand_landmarks:
            candidates.append((results.right_hand_landmarks, ear_r))

        active_hand = None
        ref_ear = None

        for h, ear in candidates:
            # Jarak pergelangan tangan ke telinga/pelipis harus DEKAT (< 0.16)
            dist_to_ear = np.sqrt((h[0].x - ear.x)**2 + (h[0].y - ear.y)**2)
            if dist_to_ear < 0.16 and h[0].y <= (nose.y + 0.12):
                active_hand = h
                ref_ear = ear
                break

        # ANTI-COLLISION: Jika 4 jari terbuka lurus ke atas, ini ADALAH LAMBAIAN/HALO, BUKAN DEWASA!
        if active_hand:
            states = HolisticDetector.get_finger_extended_states(active_hand)
            all_open = states["index"] and states["middle"] and states["ring"] and states["pinky"]
            if all_open:
                self.reset_state_dewasa()
                return False, "Tangan terbuka lebar (bukan Dewasa)"

        if not active_hand:
            if self.state_dewasa["phase"] == "READY_RELEASE":
                any_hand = results.left_hand_landmarks or results.right_hand_landmarks
                if any_hand and any_hand[0].y < (nose.y - 0.04):
                    self.reset_state_dewasa()
                    return True, "DEWASA (Ketuk Pelipis 2x + Angkat Tangan)"

            if self.state_dewasa["phase"] != "READY_RELEASE":
                self.reset_state_dewasa()
            return False, "DEWASA: Letakkan tangan di pelipis"

        wrist = active_hand[0]
        cur_y = wrist.y

        prev_pos = self.state_dewasa["prev_hand_pos"]
        self.state_dewasa["prev_hand_pos"] = (wrist.x, wrist.y)

        if prev_pos is not None:
            dy = cur_y - prev_pos[1]
            dx = wrist.x - prev_pos[0]
            motion_delta = np.sqrt(dx**2 + dy**2)

            if self.state_dewasa["phase"] in ["IDLE", "TAPPING"]:
                if motion_delta > 0.007:
                    if dy > 0.005 and not self.state_dewasa["moving_towards_head"]:
                        self.state_dewasa["moving_towards_head"] = True
                        self.state_dewasa["last_action_time"] = now
                    elif dy < -0.005 and self.state_dewasa["moving_towards_head"]:
                        self.state_dewasa["moving_towards_head"] = False
                        self.state_dewasa["tap_count"] += 1
                        self.state_dewasa["phase"] = "TAPPING"
                        self.state_dewasa["last_action_time"] = now

                if self.state_dewasa["tap_count"] >= 2:
                    self.state_dewasa["phase"] = "READY_RELEASE"
                    self.state_dewasa["last_action_time"] = now

            elif self.state_dewasa["phase"] == "READY_RELEASE":
                if dy < -0.012 or cur_y < (nose.y - 0.08):
                    self.reset_state_dewasa()
                    return True, "DEWASA (Ketuk Pelipis 2x + Angkat Tangan)"

        if self.state_dewasa["phase"] == "READY_RELEASE":
            status_text = "DEWASA: Bagus! Sekarang angkat tangan ke atas kepala..."
        elif self.state_dewasa["tap_count"] == 1:
            status_text = "DEWASA: Ketukan 1/2 terdeteksi, ketuk 1x lagi..."
        elif self.state_dewasa["phase"] == "TAPPING":
            status_text = "DEWASA: Sedang mengetuk pelipis..."
        else:
            status_text = "DEWASA: Siap di pelipis, ketuk 2x..."

        return False, status_text

    def check_menyenangkan(self, results: Any) -> Tuple[bool, str]:
        """
        Mendeteksi isyarat MENYENANGKAN (Gambar 1):
        WAJIB DUA TANGAN aktif di depan dada/leher, mengayun perlahan ke arah wajah.
        """
        if not (results.left_hand_landmarks and results.right_hand_landmarks and results.pose_landmarks):
            return False, "MENYENANGKAN: Perlu KEDUA TANGAN di depan dada"

        nose = results.pose_landmarks[0]
        shoulder_l = results.pose_landmarks[11]
        shoulder_r = results.pose_landmarks[12]
        chest_y = (shoulder_l.y + shoulder_r.y) / 2.0

        lw = results.left_hand_landmarks[0]
        rw = results.right_hand_landmarks[0]

        in_area_l = (nose.y - 0.05) <= lw.y <= (chest_y + 0.22)
        in_area_r = (nose.y - 0.05) <= rw.y <= (chest_y + 0.22)

        if not (in_area_l and in_area_r):
            return False, "MENYENANGKAN: Posisikan kedua tangan di depan leher/dada"

        if len(self.history) < 10:
            return False, "Mengumpulkan data ayunan..."

        recent = list(self.history)[-16:]
        pos_l = []
        pos_r = []

        for frame in recent:
            if frame["left"] and frame["right"]:
                pos_l.append((frame["left"][0].x, frame["left"][0].y))
                pos_r.append((frame["right"][0].x, frame["right"][0].y))

        if len(pos_l) < 8:
            return False, "Pertahankan kedua tangan di depan leher"

        ys_l = [p[1] for p in pos_l]
        ys_r = [p[1] for p in pos_r]
        xs_l = [p[0] for p in pos_l]
        xs_r = [p[0] for p in pos_r]

        std_motion = np.std(ys_l) + np.std(ys_r) + np.std(xs_l) + np.std(xs_r)

        if std_motion > 0.015:
            diffs = np.diff(ys_l)
            flips = np.count_nonzero(np.diff(np.sign(diffs)))
            if flips >= 1:
                return True, "MENYENANGKAN (Ayunan Aroma Kedua Tangan)"

        return False, "MENYENANGKAN: Ayunkan kedua tangan perlahan ke arah wajah"
