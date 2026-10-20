"""
Aturan Geometris untuk Pengenalan Isyarat Statis & Alfabet BISINDO
Perbaikan Khusus:
1. Anti-Salah Deteksi Huruf I:
   - Finger Heart WAJIB mensyaratkan kelingking terlipat. Jika kelingking tegak, Finger Heart DITOLAK MUTLAK!
2. Pemisahan Tegas Huruf Y vs I:
   - Huruf Y: Jempol TERBUKA lebar + Kelingking tegak (Shaka).
   - Huruf I: Jempol MELIPAT di kepalan + Kelingking tegak sendirian.
3. Dukungan Huruf B, A, C, L, O, U, V, W
4. Isyarat Kustom: KAMU, AKU, LOVE, FINGER HEART
"""
import numpy as np
from typing import Optional, Dict, Tuple, Any
from holistic_detector import HolisticDetector

class GestureRules:
    # =========================================================================
    # 1. ISYARAT KUSTOM STATIS
    # =========================================================================

    @staticmethod
    def check_aku(results: Any) -> Tuple[bool, str]:
        """
        Mendeteksi isyarat AKU:
        Satu tangan dengan jari telunjuk menempel/menunjuk ke area dada sendiri.
        """
        if not results.pose_landmarks:
            return False, "Menunggu pose tubuh..."

        shoulder_l = results.pose_landmarks[11]
        shoulder_r = results.pose_landmarks[12]
        chest_x = (shoulder_l.x + shoulder_r.x) / 2.0
        chest_y = (shoulder_l.y + shoulder_r.y) / 2.0 + 0.08
        shoulder_width = abs(shoulder_l.x - shoulder_r.x)

        candidates = []
        if results.right_hand_landmarks:
            candidates.append(("Kanan", results.right_hand_landmarks))
        if results.left_hand_landmarks:
            candidates.append(("Kiri", results.left_hand_landmarks))

        if len(candidates) != 1:
            return False, "Bukan AKU"

        hand_label, hand = candidates[0]
        index_tip = hand[8]
        wrist = hand[0]

        dist_to_chest = np.sqrt((index_tip.x - chest_x)**2 + (index_tip.y - chest_y)**2)
        chest_radius = max(0.18, shoulder_width * 0.65)

        if dist_to_chest > chest_radius:
            return False, "Bukan AKU"

        states = HolisticDetector.get_finger_extended_states(hand)
        other_curled = (not states["middle"]) and (not states["ring"]) and (not states["pinky"])

        if other_curled and (dist_to_chest < chest_radius):
            return True, f"AKU (Telunjuk {hand_label} di Dada)"

        return False, "Bukan AKU"

    @staticmethod
    def check_kamu(results: Any) -> Tuple[bool, str]:
        """
        Mendeteksi isyarat KAMU:
        Satu tangan dengan jari telunjuk menunjuk lurus ke arah KAMERA / LAYAR.
        Jari tengah, manis, kelingking terlipat rapat ke telapak.
        """
        candidates = []
        if results.right_hand_landmarks:
            candidates.append(("Kanan", results.right_hand_landmarks))
        if results.left_hand_landmarks:
            candidates.append(("Kiri", results.left_hand_landmarks))

        if len(candidates) != 1:
            return False, "Bukan KAMU"

        hand_label, hand = candidates[0]
        index_tip = hand[8]
        index_mcp = hand[5]
        wrist = hand[0]

        states = HolisticDetector.get_finger_extended_states(hand)
        other_curled = (not states["middle"]) and (not states["ring"]) and (not states["pinky"])

        if not other_curled:
            return False, "Bukan KAMU"

        # Telunjuk mengarah ke kamera
        pointing_forward_z = (index_tip.z < index_mcp.z - 0.015) or (index_tip.z < wrist.z - 0.02)

        not_at_chest = True
        if results.pose_landmarks:
            shoulder_l = results.pose_landmarks[11]
            shoulder_r = results.pose_landmarks[12]
            chest_x = (shoulder_l.x + shoulder_r.x) / 2.0
            chest_y = (shoulder_l.y + shoulder_r.y) / 2.0 + 0.08
            dist_to_chest = np.sqrt((index_tip.x - chest_x)**2 + (index_tip.y - chest_y)**2)
            not_at_chest = dist_to_chest > 0.14 or (index_tip.z < -0.05)

        index_projected = HolisticDetector.calc_distance_2d(index_tip, wrist)
        middle_curled = HolisticDetector.calc_distance_2d(hand[12], wrist)

        if other_curled and pointing_forward_z and (index_projected > middle_curled * 1.15) and not_at_chest:
            return True, f"KAMU (Telunjuk {hand_label} ke Kamera 🫵)"

        return False, "Bukan KAMU"

    @staticmethod
    def check_finger_heart(results: Any) -> Tuple[bool, str]:
        """
        Mendeteksi FINGER HEART / SARANGHAE (Gambar 4 - 🫰):
        Ujung jempol dan telunjuk bersilangan membentuk hati kecil.
        MUTLAK: Jari kelingking, manis, dan tengah HARUS terlipat.
        Jika kelingking tegak (seperti huruf I), FINGER HEART DITOLAK!
        """
        candidates = []
        if results.right_hand_landmarks:
            candidates.append(results.right_hand_landmarks)
        if results.left_hand_landmarks:
            candidates.append(results.left_hand_landmarks)

        if len(candidates) != 1:
            return False, "Bukan Finger Heart"

        hand = candidates[0]
        states = HolisticDetector.get_finger_extended_states(hand)

        # ANTI-COLLISION: Jika kelingking tegak (huruf I atau Y), BUKAN FINGER HEART!
        if states["pinky"] or states["middle"] or states["ring"]:
            return False, "Bukan Finger Heart (Jari lain tidak mengepal)"

        thumb_tip = hand[4]
        index_tip = hand[8]
        middle_tip = hand[12]
        wrist = hand[0]

        gap_thumb_index = HolisticDetector.calc_distance_2d(thumb_tip, index_tip)

        dist_mid = HolisticDetector.calc_distance_2d(middle_tip, wrist)
        dist_mcp = HolisticDetector.calc_distance_2d(hand[9], wrist)

        other_curled = dist_mid < dist_mcp * 1.35
        tips_elevated = (thumb_tip.y < middle_tip.y) or (index_tip.y < middle_tip.y)

        if (gap_thumb_index < 0.075) and other_curled and tips_elevated:
            return True, "FINGER HEART (Saranghae 🫰)"

        return False, "Bukan Finger Heart"

    @staticmethod
    def check_love_two_hands(results: Any) -> Tuple[bool, str]:
        """Mendeteksi LOVE DUA TANGAN (🫶)"""
        if not (results.left_hand_landmarks and results.right_hand_landmarks):
            return False, "Bukan Love"

        left = results.left_hand_landmarks
        right = results.right_hand_landmarks

        gap_thumbs = HolisticDetector.calc_distance_2d(left[4], right[4])
        gap_index = HolisticDetector.calc_distance_2d(left[8], right[8])
        gap_middle = HolisticDetector.calc_distance_2d(left[12], right[12])
        top_gap = min(gap_index, gap_middle)

        avg_index_y = (left[8].y + right[8].y) / 2.0
        avg_thumb_y = (left[4].y + right[4].y) / 2.0
        wrist_distance = HolisticDetector.calc_distance_2d(left[0], right[0])

        is_top = top_gap < 0.13
        is_bottom = gap_thumbs < 0.13
        is_orientation = avg_index_y < (avg_thumb_y + 0.05)
        is_spread = wrist_distance > 0.08

        if is_top and is_bottom and is_orientation and is_spread:
            return True, "LOVE (Simbol Hati Dua Tangan 🫶)"

        return False, "Bukan Love"

    # =========================================================================
    # 2. ALFABET JARI BISINDO LENGKAP
    # =========================================================================

    @staticmethod
    def check_alphabet(results: Any) -> Tuple[bool, str]:
        """
        Mengenali huruf alfabet jari BISINDO secara akurat:
        I, Y, B, A, C, L, V, U, W, O
        """
        # -------------------------------------------------------------
        # Huruf 'A' (Dua Tangan - BISINDO Tradisional)
        # -------------------------------------------------------------
        if results.left_hand_landmarks and results.right_hand_landmarks:
            left = results.left_hand_landmarks
            right = results.right_hand_landmarks
            st_l = HolisticDetector.get_finger_extended_states(left)
            st_r = HolisticDetector.get_finger_extended_states(right)

            pillar_l = st_l["index"] and (not st_l["middle"])
            pillar_r = st_r["index"] and (not st_r["middle"])

            if pillar_l or pillar_r:
                pillar = left if pillar_l else right
                lean = right if pillar_l else left
                dist_apex = min(
                    HolisticDetector.calc_distance_2d(pillar[8], lean[8]),
                    HolisticDetector.calc_distance_2d(pillar[8], lean[4])
                )
                if dist_apex < 0.18:
                    return True, "Huruf BISINDO: A"

        # Evaluasi Huruf 1 Tangan
        hands = []
        if results.right_hand_landmarks:
            hands.append(results.right_hand_landmarks)
        if results.left_hand_landmarks:
            hands.append(results.left_hand_landmarks)

        if len(hands) != 1:
            return False, "Bukan Alfabet"

        h = hands[0]
        states = HolisticDetector.get_finger_extended_states(h)
        wrist = h[0]
        thumb_tip = h[4]
        index_tip = h[8]
        middle_tip = h[12]
        ring_tip = h[16]
        pinky_tip = h[20]

        th = states["thumb"]
        idx = states["index"]
        mid = states["middle"]
        rng = states["ring"]
        pnk = states["pinky"]

        # Jarak ujung jempol ke pangkal jari telunjuk
        dist_thumb_index_base = HolisticDetector.calc_distance_2d(thumb_tip, h[5])
        thumb_extended_wide = dist_thumb_index_base > 0.085

        # -------------------------------------------------------------
        # Huruf 'Y': Jempol dan Kelingking TERBUKA LEBAR (Shaka), 3 jari tengah mengepal
        # DI-EVALUASI SEBELUM HURUF 'I'!
        # -------------------------------------------------------------
        if pnk and thumb_extended_wide and (not idx) and (not mid) and (not rng):
            dist_thumb_pinky = HolisticDetector.calc_distance_2d(thumb_tip, pinky_tip)
            if dist_thumb_pinky > 0.10:
                return True, "Huruf BISINDO: Y"

        # -------------------------------------------------------------
        # Huruf 'I': HANYA jari kelingking tegak ke atas, Jempol TIDAK terbuka lebar
        # -------------------------------------------------------------
        if pnk and (not thumb_extended_wide) and (not idx) and (not mid) and (not rng):
            # Kelingking berdiri tegak melampaui jari-jari lainnya
            if pinky_tip.y < index_tip.y and pinky_tip.y < middle_tip.y:
                return True, "Huruf BISINDO: I"

        # -------------------------------------------------------------
        # Huruf 'B': Keempat jari (Telunjuk s/d Kelingking) tegak lurus rapat
        # -------------------------------------------------------------
        if idx and mid and rng and pnk:
            # Pastikan keempat jari berdiri tegak ke atas
            if (index_tip.y < wrist.y) and (pinky_tip.y < wrist.y):
                return True, "Huruf BISINDO: B"

        # -------------------------------------------------------------
        # Huruf 'A' (1 Tangan - Kepalan tangan dengan jempol tegak di samping)
        # -------------------------------------------------------------
        if (not idx) and (not mid) and (not rng) and (not pnk):
            # Semua 4 jari terlipat membentuk kepalan
            dist_mid_wrist = HolisticDetector.calc_distance_2d(middle_tip, wrist)
            dist_mcp_wrist = HolisticDetector.calc_distance_2d(h[9], wrist)
            is_fist = dist_mid_wrist < dist_mcp_wrist * 1.35
            
            # Jempol tegak di samping telunjuk
            thumb_up = thumb_tip.y < h[5].y + 0.05
            if is_fist and thumb_up:
                return True, "Huruf BISINDO: A"

        # -------------------------------------------------------------
        # Huruf 'L': Jempol dan telunjuk terbuka membentuk sudut siku 90 derajat
        # -------------------------------------------------------------
        if idx and thumb_extended_wide and (not mid) and (not rng) and (not pnk):
            angle_gap = HolisticDetector.calc_distance_2d(thumb_tip, index_tip)
            if angle_gap > 0.08:
                return True, "Huruf BISINDO: L"

        # -------------------------------------------------------------
        # Huruf 'V' dan 'U': Telunjuk dan Jari Tengah tegak ke atas
        # -------------------------------------------------------------
        if idx and mid and (not rng) and (not pnk):
            dist_index_middle = HolisticDetector.calc_distance_2d(index_tip, middle_tip)
            if dist_index_middle > 0.050:
                return True, "Huruf BISINDO: V"
            else:
                return True, "Huruf BISINDO: U"

        # -------------------------------------------------------------
        # Huruf 'W': Tiga jari (Telunjuk, Tengah, Manis) tegak terbuka
        # -------------------------------------------------------------
        if idx and mid and rng and (not pnk):
            return True, "Huruf BISINDO: W"

        # -------------------------------------------------------------
        # Huruf 'O': Ujung jari melengkung bertemu dengan ujung jempol
        # -------------------------------------------------------------
        gap_thumb_index = HolisticDetector.calc_distance_2d(thumb_tip, index_tip)
        gap_thumb_middle = HolisticDetector.calc_distance_2d(thumb_tip, middle_tip)
        if (gap_thumb_index < 0.06) and (gap_thumb_middle < 0.07):
            dist_mid_wrist = HolisticDetector.calc_distance_2d(middle_tip, wrist)
            dist_mcp_wrist = HolisticDetector.calc_distance_2d(h[9], wrist)
            if dist_mid_wrist > dist_mcp_wrist * 0.90:
                return True, "Huruf BISINDO: O"

        # -------------------------------------------------------------
        # Huruf 'C': Jari melengkung membentuk busur C (ada celah jelas)
        # -------------------------------------------------------------
        gap_c = HolisticDetector.calc_distance_2d(thumb_tip, index_tip)
        dist_idx_wrist = HolisticDetector.calc_distance_2d(index_tip, wrist)
        dist_th_wrist = HolisticDetector.calc_distance_2d(thumb_tip, wrist)
        if (0.08 < gap_c < 0.22) and (dist_idx_wrist > 0.12) and (dist_th_wrist > 0.09):
            if not (idx and mid and rng and pnk):
                return True, "Huruf BISINDO: C"

        return False, "Bukan Alfabet"
