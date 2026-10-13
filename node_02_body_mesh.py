import os
import cv2
import numpy as np
import mediapipe as mp
import math

class BodyMeshNode:
    def __init__(self):
        print("[Node-02] 3D Vücut Haritalama (Body Mesh) Modülü Yükleniyor...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, 
            model_complexity=2, # En yüksek çözünürlüklü Z derinliği için Model=2
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        print("[Node-02] 3D Mesh Modeli Yüklendi.")

    def calculate_3d_slice(self, p1, p2, w, h, yaw_rad=0.0):
        """
        Gövdenin belirli bir kesitindeki (örn. göğüs, bel) 3D genişliği hesaplar.
        4D-Humans (SMPL) mantığında olduğu gibi, X ve Z eksenlerindeki derinliği hesaba katar.
        """
        # x ve y piksellere çevrilir. z ise genellikle omuz genişliğine göre orantılıdır
        dx = (p1.x - p2.x) * w
        dy = (p1.y - p2.y) * h
        dz = (p1.z - p2.z) * w # z ekseni w üzerinden orantılıdır
        
        # 3 Boyutlu Öklid Mesafesi
        width_3d_px = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Yaw (Dönüş) açısı düzeltmesi: Eğer kişi yan duruyorsa kesiti düzeltir
        corrected_width = width_3d_px * (1.0 + abs(math.sin(yaw_rad)) * 0.3)
        return corrected_width

    def generate_body_tensor(self, pose_landmarks, w, h):
        """
        Kullanıcı pose landmark'larını alır ve 3D gövde tensörünü oluşturur.
        Çıktı olarak W_body (3D kesit genişlikleri matrisi) ve Yaw açısı döndürür.
        """
        if not pose_landmarks:
            print("[Node-02] UYARI: İnsan bedeni bulunamadı.")
            return None
            
        lm = pose_landmarks.landmark
        
        # 1. Yaw (Y Ekseni Etrafında Dönüş) Açısı Hesaplama
        # Omuzların Z derinlik farkına göre kişinin ne kadar döndüğünü (Yan Duruş) anlar
        l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        dz = l_sh.z - r_sh.z
        dx = l_sh.x - r_sh.x
        yaw_rad = math.atan2(dz, dx)
        yaw_deg = math.degrees(yaw_rad)
        
        is_side_profile = abs(yaw_deg) > 40.0
        
        # 2. SMPL Slice (3D Gövde Kesit Genişlikleri) - ADIM 04
        chest_3d_width = self.calculate_3d_slice(l_sh, r_sh, w, h, yaw_rad)
        
        l_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
        hips_3d_width = self.calculate_3d_slice(l_hip, r_hip, w, h, yaw_rad)
        
        # Bel, omuz ve kalça arasındadır
        waist_3d_width = (chest_3d_width * 0.4) + (hips_3d_width * 0.6)
        
        w_body_tensor = {
            "chest_px_3d": chest_3d_width,
            "waist_px_3d": waist_3d_width,
            "hips_px_3d": hips_3d_width,
            "yaw_angle_rad": yaw_rad,
            "yaw_angle_deg": yaw_deg,
            "is_side_profile": is_side_profile
        }
        
        print(f"[Node-02] Başarılı! 3D Tensor Çıkarıldı. (Yan Duruş: {is_side_profile}, Açı: {yaw_deg:.1f}°)")
        return w_body_tensor

if __name__ == "__main__":
    node = BodyMeshNode()
    if os.path.exists("goruntu/temp_user.jpg"):
        tensor = node.generate_body_tensor("goruntu/temp_user.jpg")
        print("Çıkarılan Gövde Matrisi:", tensor)
