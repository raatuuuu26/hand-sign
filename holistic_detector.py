"""
Modul Hand & Body Tracking menggunakan MediaPipe Modern Tasks API (HolisticLandmarker)
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from typing import Optional, Tuple, Dict, Any, List

# Lokasi model Holistic Landmarker
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "holistic_landmarker.task")

class LandmarkList(list):
    """
    Daftar landmark yang mendukung akses indeks langsung [i]
    maupun properti .landmark[i] untuk kompatibilitas penuh.
    """
    @property
    def landmark(self):
        return self

class HolisticResults:
    """Kontainer terstandarisasi untuk hasil deteksi pose & kedua tangan."""
    def __init__(self, raw_result):
        # MediaPipe Tasks API mengembalikan list kosong [] jika tidak terdeteksi
        self.pose_landmarks = (
            LandmarkList(raw_result.pose_landmarks) if (raw_result and len(raw_result.pose_landmarks) > 0) else None
        )
        self.left_hand_landmarks = (
            LandmarkList(raw_result.left_hand_landmarks) if (raw_result and len(raw_result.left_hand_landmarks) > 0) else None
        )
        self.right_hand_landmarks = (
            LandmarkList(raw_result.right_hand_landmarks) if (raw_result and len(raw_result.right_hand_landmarks) > 0) else None
        )
        self.face_landmarks = (
            LandmarkList(raw_result.face_landmarks) if (raw_result and len(raw_result.face_landmarks) > 0) else None
        )


class HolisticDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5
    ):
        if not os.path.exists(MODEL_PATH):
            print("Mengunduh model holistic_landmarker.task dari Google...")
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("Model berhasil diunduh!")

        base_options = BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_hand_landmarks_confidence=min_tracking_confidence,
            min_face_landmarks_confidence=min_detection_confidence
        )

        self.landmarker = vision.HolisticLandmarker.create_from_options(options)
        self.drawing_utils = vision.drawing_utils
        self.drawing_styles = vision.drawing_styles
        self.hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    def process(self, frame: np.ndarray) -> HolisticResults:
        """
        Memproses frame BGR melalui MediaPipe Holistic Tasks
        dan mengembalikan objek HolisticResults.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.landmarker.detect(mp_image)
        return HolisticResults(detection_result)

    def draw_landmarks(self, frame: np.ndarray, results: HolisticResults) -> np.ndarray:
        """
        Menggambar skeleton pose dan kedua tangan ke frame BGR.
        """
        annotated_frame = frame.copy()

        # Gambar Pose (bahu, kepala, dada)
        if results.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.pose_connections,
                landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style()
            )

        # Gambar Tangan Kiri
        if results.left_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                self.hand_connections,
                landmark_drawing_spec=self.drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.drawing_styles.get_default_hand_connections_style()
            )

        # Gambar Tangan Kanan
        if results.right_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                self.hand_connections,
                landmark_drawing_spec=self.drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.drawing_styles.get_default_hand_connections_style()
            )

        return annotated_frame

    @staticmethod
    def calc_distance_3d(p1, p2) -> float:
        """Menghitung jarak euclidean 3D antara dua landmark."""
        return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2))

    @staticmethod
    def calc_distance_2d(p1, p2) -> float:
        """Menghitung jarak euclidean 2D (sumbu X dan Y)."""
        return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2))

    @staticmethod
    def get_finger_extended_states(hand_landmarks) -> Dict[str, bool]:
        """
        Menentukan secara akurat apakah tiap jari terbuka (lurus) atau menekuk (fist/curled).
        Menggunakan kombinasi perbandingan posisi TIP vs PIP dan jarak ke pangkal jari (MCP).
        """
        wrist = hand_landmarks[0]
        
        # Jari-jari: [TIP, PIP, MCP]
        fingers = {
            "index": (8, 6, 5),
            "middle": (12, 10, 9),
            "ring": (16, 14, 13),
            "pinky": (20, 18, 17)
        }
        
        states = {}
        for name, (tip_idx, pip_idx, mcp_idx) in fingers.items():
            tip = hand_landmarks[tip_idx]
            pip = hand_landmarks[pip_idx]
            mcp = hand_landmarks[mcp_idx]
            
            # Saat jari lurus: jarak TIP ke MCP jauh lebih besar daripada PIP ke MCP
            dist_tip_mcp = HolisticDetector.calc_distance_2d(tip, mcp)
            dist_pip_mcp = HolisticDetector.calc_distance_2d(pip, mcp)
            far_from_base = dist_tip_mcp > (dist_pip_mcp * 1.25)
            
            # Posisi vertikal: ujung jari lebih tinggi dari sendi PIP (untuk pose tangan menghadap atas)
            higher_than_pip = tip.y < pip.y
            
            states[name] = far_from_base and higher_than_pip

        # Khusus Ibu Jari (Thumb: TIP 4, IP 3, MCP 2, CMC 1)
        thumb_tip = hand_landmarks[4]
        index_mcp = hand_landmarks[5]
        
        # Jempol terbuka jika ujung jempol menjauh dari pangkal jari telunjuk
        dist_thumb_index = HolisticDetector.calc_distance_2d(thumb_tip, index_mcp)
        dist_thumb_wrist = HolisticDetector.calc_distance_2d(thumb_tip, wrist)
        dist_cmc_wrist = HolisticDetector.calc_distance_2d(hand_landmarks[1], wrist)
        
        states["thumb"] = (dist_thumb_index > 0.08) and (dist_thumb_wrist > dist_cmc_wrist * 1.15)

        return states

    def close(self):
        self.landmarker.close()
