import cv2
import numpy as np
import math

class FitHeatmapNode:
    def __init__(self):
        print("[Node-03] Fit Isı Haritası (Heatmap) Modülü Yükleniyor...")
        
    def get_tension_color(self, garment_cm, user_cm):
        """
        Matematiksel Katman Farkı Simülasyonu (ΔW = Wgarment − Wbody)
        Kırmızı: Dar (<0.95), Sarı: Tam (0.95-1.05), Yeşil: Bol (>1.05)
        """
        garment_circumference = garment_cm * 2.0 
        ratio = garment_circumference / user_cm
        
        if ratio < 0.95: 
            return (0, 0, 255), f"Dar ({ratio:.2f}x)" # Kırmızı
        elif 0.95 <= ratio <= 1.05: 
            return (0, 255, 255), f"Tam ({ratio:.2f}x)" # Sarı
        else: 
            return (0, 255, 0), f"Bol ({ratio:.2f}x)" # Yeşil

    def apply_heatmap_overlay(self, image, body_tensor, garment_width_cm, user_measurements, garment_type="Üst Giyim"):
        """
        Adım 06: OpenCV ColorMap mantığıyla (Kıyafet Katmanı - Gövde Katmanı) tensör farkını alır, 
        darlık/genişlik simülasyonunu görüntünün üzerine bindirir.
        """
        print("[Node-03] Isı Haritası Üretiliyor...")
        overlay = image.copy()
        h, w = image.shape[:2]
        
        # Gövde merkezlerini (y ekseni) body_tensor verilerine göre yaklaşık belirliyoruz
        shoulder_mid_y = int(h * 0.25)  
        hip_mid_y = int(h * 0.60)
        
        chest_y = shoulder_mid_y + int((hip_mid_y - shoulder_mid_y) * 0.25)
        waist_y = shoulder_mid_y + int((hip_mid_y - shoulder_mid_y) * 0.60)
        basen_y = hip_mid_y + int((hip_mid_y - shoulder_mid_y) * 0.20)
        
        # 3D Tensor'dan genişlik piksellerini çek
        chest_px = body_tensor.get("chest_px_3d", w * 0.2)
        waist_px = body_tensor.get("waist_px_3d", w * 0.18)
        hips_px  = body_tensor.get("hips_px_3d", w * 0.22)
        
        # Kullanıcının beyan ettiği gerçek çevre ölçüleri
        user_chest = user_measurements.get('chest', 90)
        user_waist = user_measurements.get('waist', 75)
        user_hips = user_measurements.get('hips', 95)
        
        center_x = int(w / 2)
        alpha = 0.5
        y_offset = 30
        
        # 1. GÖĞÜS (Üst Giyim ve Elbise)
        if garment_type in ["Üst Giyim", "Elbise"]:
            chest_color, chest_txt = self.get_tension_color(garment_width_cm, user_chest)
            cv2.ellipse(overlay, (center_x, chest_y), (int(chest_px*0.7), int(chest_px*0.3)), 0, 0, 360, chest_color, -1)
            cv2.putText(image, f"Gogus: {chest_txt}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, chest_color, 2)
            y_offset += 30
            
        # 2. BEL (Tümü)
        garment_waist_cm = garment_width_cm * 0.95 if garment_type != "Alt Giyim" else garment_width_cm
        waist_color, waist_txt = self.get_tension_color(garment_waist_cm, user_waist)
        cv2.ellipse(overlay, (center_x, waist_y), (int(waist_px*0.6), int(waist_px*0.25)), 0, 0, 360, waist_color, -1)
        cv2.putText(image, f"Bel: {waist_txt}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, waist_color, 2)
        y_offset += 30
        
        # 3. BASEN (Alt Giyim ve Elbise)
        if garment_type in ["Alt Giyim", "Elbise"]:
            garment_hips_cm = garment_width_cm * 1.10 if garment_type == "Elbise" else garment_width_cm * 1.05
            hips_color, hips_txt = self.get_tension_color(garment_hips_cm, user_hips)
            cv2.ellipse(overlay, (center_x, basen_y), (int(hips_px*0.8), int(hips_px*0.35)), 0, 0, 360, hips_color, -1)
            cv2.putText(image, f"Basen: {hips_txt}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hips_color, 2)
            
        # Pürüzsüzleştir ve ana görüntüyle birleştir
        overlay = cv2.GaussianBlur(overlay, (99, 99), 0)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        print("[Node-03] Isı Haritası Başarıyla Uygulandı.")
        return image
