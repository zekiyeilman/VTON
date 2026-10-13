import re
import json
import os
import cv2
import numpy as np
import mediapipe as mp
import requests
import time
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import sys

# Windows PowerShell üzerinde Emoji (Unicode) hatalarını kalıcı olarak çözen kod
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

import logging

logging.getLogger('selenium.webdriver.common.service').setLevel(logging.FATAL)


# ============================================================================
# 1. BÖLÜM: OTONOM CHROME SCRAPER (SNIPER MODU)
# ============================================================================

from ..paths import CALIBRATION_JSON


class MediaPipeAdaptiveCalibrator:
    def __init__(self, memory_file=str(CALIBRATION_JSON)):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=True)
        self.memory_file = memory_file

        # Biyometrik Veritabanı (Yaş ve Cinsiyet bazlı %100 Kemik Sabitleri)
        # Sadece hata payı %10'un altında olan, esnemeyen (non-deformable) kafatası verileri kullanılmıştır.
        self.BIOMETRICS = {
            "kadin": {"ipd": 6.2, "face_len": 6.5},
            "erkek": {"ipd": 6.4, "face_len": 7.2},
            "cocuk": {"ipd": 5.4, "face_len": 5.6},
            "bebek": {"ipd": 4.5, "face_len": 4.2}
        }

        self.adaptive_ipd_weight = self._load_calibration_weight()

    def _load_calibration_weight(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f).get("ipd_correction_factor", 1.0)
            except Exception:
                pass
        return 1.0

    def _save_calibration_weight(self, new_factor):
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump({"ipd_correction_factor": new_factor}, f, indent=4)
        except Exception:
            pass

    def _calculate_pose_angles(self, landmarks, w, h):
        l_sh = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        dx_sh = (l_sh.x - r_sh.x) * w
        dz_sh = (l_sh.z - r_sh.z) * w
        yaw_deg = np.degrees(np.arctan2(dz_sh, dx_sh))

        mid_sh_y = (l_sh.y + r_sh.y) / 2.0 * h
        mid_sh_z = (l_sh.z + r_sh.z) / 2.0 * w
        mid_hip_y = (l_hip.y + r_hip.y) / 2.0 * h
        mid_hip_z = (l_hip.z + r_hip.z) / 2.0 * w

        dy_spine = mid_hip_y - mid_sh_y
        dz_spine = mid_hip_z - mid_sh_z
        pitch_deg = np.degrees(np.arctan2(dz_spine, dy_spine))

        return abs(yaw_deg), abs(pitch_deg)

    def process_image(self, image_bgr, ground_truth=None, demografi="kadin"):
        if image_bgr is None:
            return {"durum": "HATA", "mesaj": "Görüntü verisi boş."}

        bio = self.BIOMETRICS.get(demografi, self.BIOMETRICS["kadin"])
        current_ipd_cm = bio["ipd"]
        current_face_cm = bio["face_len"]

        h, w, _ = image_bgr.shape
        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)

        if not results.pose_landmarks:
            return {"durum": "UYARI", "mesaj": "İskelet tespit edilemedi."}

        lm = results.pose_landmarks.landmark
        yaw_deg, pitch_deg = self._calculate_pose_angles(lm, w, h)
        cos_yaw = max(np.cos(np.radians(yaw_deg)), 0.35)
        cos_pitch = max(np.cos(np.radians(pitch_deg)), 0.50)

        cm_per_pixel = None
        kalibrasyon_kaynagi = ""

        if ground_truth and ('manken_boy' in ground_truth or 'urun_on_boy' in ground_truth):
            if 'manken_boy' in ground_truth:
                gt_boy = ground_truth['manken_boy']
                burun = lm[self.mp_pose.PoseLandmark.NOSE]
                topuk_y = max(lm[self.mp_pose.PoseLandmark.LEFT_HEEL].y, lm[self.mp_pose.PoseLandmark.RIGHT_HEEL].y) * h
                iskelet_px = (topuk_y - (burun.y * h)) * 1.14
                cm_per_pixel = gt_boy / iskelet_px
                kalibrasyon_kaynagi = f"[OK] Ground Truth (Manken Boyu: {gt_boy} cm) -> Piksel Kilitli"

            elif 'urun_on_boy' in ground_truth:
                gt_on_boy = ground_truth['urun_on_boy']
                mid_sh_y = (lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm[
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2 * h
                mid_hip_y = (lm[self.mp_pose.PoseLandmark.LEFT_HIP].y + lm[
                    self.mp_pose.PoseLandmark.RIGHT_HIP].y) / 2 * h
                cm_per_pixel = gt_on_boy / (abs(mid_hip_y - mid_sh_y) * 1.15)
                kalibrasyon_kaynagi = f"[OK] Ground Truth (Ürün Ön Boyu: {gt_on_boy} cm) -> Piksel Kilitli"

            r_eye = lm[self.mp_pose.PoseLandmark.RIGHT_EYE]
            l_eye = lm[self.mp_pose.PoseLandmark.LEFT_EYE]
            if r_eye.visibility > 0.65 and l_eye.visibility > 0.65:
                raw_ipd_px = np.linalg.norm(np.array([r_eye.x * w, r_eye.y * h]) - np.array([l_eye.x * w, l_eye.y * h]))
                estimated_ipd = (raw_ipd_px / cos_yaw) * cm_per_pixel
                yeni_carpan = estimated_ipd / current_ipd_cm
                self.adaptive_ipd_weight = (0.8 * self.adaptive_ipd_weight) + (0.2 * yeni_carpan)
                self._save_calibration_weight(self.adaptive_ipd_weight)

        else:
            # --- [AI] SENSOR FUSION / WEIGHTED ENSEMBLE ---
            # HATA PAYI >%10 KURALI: Omuz genişliği duruşa ve açıya göre %15-20 oranında hata verdiği için 
            # sensör füzyonundan tamamen çıkarılmıştır. Sadece hata payı <%5 olan rijit kafatası kemikleri kalmıştır.
            
            # 1. IPD (Gözler Arası Mesafe) - Baz Ağırlık: %60
            r_eye = lm[self.mp_pose.PoseLandmark.RIGHT_EYE]
            l_eye = lm[self.mp_pose.PoseLandmark.LEFT_EYE]
            ipd_cm_per_pixel = 0
            ipd_weight = 0
            if r_eye.visibility > 0.65 and l_eye.visibility > 0.65:
                raw_ipd_px = np.linalg.norm(np.array([r_eye.x * w, r_eye.y * h]) - np.array([l_eye.x * w, l_eye.y * h]))
                if raw_ipd_px > 0:
                    ipd_cm_per_pixel = (current_ipd_cm * self.adaptive_ipd_weight) / (raw_ipd_px / cos_yaw)
                    ipd_weight = 0.60

            # 2. Yüz Uzunluğu (Göz hizasından Dudak hizasına dikey mesafe) - Baz Ağırlık: %40
            l_mouth = lm[self.mp_pose.PoseLandmark.MOUTH_LEFT]
            r_mouth = lm[self.mp_pose.PoseLandmark.MOUTH_RIGHT]
            face_cm_per_pixel = 0
            face_weight = 0
            
            if l_eye.visibility > 0.6 and r_eye.visibility > 0.6 and l_mouth.visibility > 0.6 and r_mouth.visibility > 0.6:
                mid_eye_y = (l_eye.y + r_eye.y) / 2 * h
                mid_eye_x = (l_eye.x + r_eye.x) / 2 * w
                mid_mouth_y = (l_mouth.y + r_mouth.y) / 2 * h
                mid_mouth_x = (l_mouth.x + r_mouth.x) / 2 * w
                
                face_px_dist = np.linalg.norm(np.array([mid_eye_x, mid_eye_y]) - np.array([mid_mouth_x, mid_mouth_y]))
                
                if face_px_dist > 0:
                    face_cm_per_pixel = current_face_cm / (face_px_dist / cos_pitch)
                    face_weight = 0.40
                    
            # Dinamik Yeniden Dağıtım (Ağırlıklı Füzyon)
            total_weight = ipd_weight + face_weight
            
            if total_weight > 0:
                # Kayıp veri varsa kalanların ağırlığını %100'e tamamlayacak şekilde normalize et
                ipd_norm = ipd_weight / total_weight
                face_norm = face_weight / total_weight
                
                cm_per_pixel = (ipd_cm_per_pixel * ipd_norm) + (face_cm_per_pixel * face_norm)
                
                katilanlar = []
                if ipd_weight > 0: katilanlar.append(f"Göz(%{int(ipd_norm*100)})")
                if face_weight > 0: katilanlar.append(f"Yüz Uzunluğu(%{int(face_norm*100)})")
                
                kalibrasyon_kaynagi = "[AI] AI Füzyon: " + " + ".join(katilanlar)
            else:
                return {"durum": "HATA", "mesaj": "Mankenin kafatası referans noktaları görünmüyor."}

        # Sadece raporlama için omuz bilgisini hesapla (Füzyona dahil edilmez, hata payı >%10)
        l_sh_pt = np.array(
            [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
        r_sh_pt = np.array(
            [lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])
        omuz_cm = (np.linalg.norm(l_sh_pt - r_sh_pt) / cos_yaw) * cm_per_pixel if cm_per_pixel else 0

        rapor = {
            "kalibrasyon_stratejisi": kalibrasyon_kaynagi,
            "cm_per_pixel": round(cm_per_pixel, 5),
            "hesaplanan_omuz_cm": round(omuz_cm, 2)
        }
        if ground_truth: rapor["kazilan_gercek_olculer"] = ground_truth

        return rapor


# ============================================================================
# 3. BÖLÜM: ÇALIŞTIRMA HATTI
# ============================================================================

