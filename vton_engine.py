import cv2
import numpy as np
import mediapipe as mp
import json
import math
import os

class VTONEngine:
    def __init__(self, calibration_file="vton_calibration_memory.json"):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, 
            model_complexity=1, 
            enable_segmentation=True, 
            min_detection_confidence=0.5
        )
        self.calibration_data = {}
        self.load_calibration(calibration_file)

    def load_calibration(self, memory_file):
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                self.calibration_data = json.load(f)
        except:
            pass

    def calculate_scale_from_person(self, h, w, landmarks, person_height_cm=None):
        lm = landmarks.landmark
        # Eğer boy verilmemişse ortalama 170cm kabul edelim
        if not person_height_cm:
            person_height_cm = 170.0
            
        nose_y = lm[self.mp_pose.PoseLandmark.NOSE].y * h
        l_heel_y = lm[self.mp_pose.PoseLandmark.LEFT_HEEL].y * h
        r_heel_y = lm[self.mp_pose.PoseLandmark.RIGHT_HEEL].y * h
        heel_y = max(l_heel_y, r_heel_y)
        
        if lm[self.mp_pose.PoseLandmark.LEFT_EYE].visibility > 0.5:
            l_eye = lm[self.mp_pose.PoseLandmark.LEFT_EYE]
            l_mouth = lm[self.mp_pose.PoseLandmark.MOUTH_LEFT]
        else:
            l_eye = lm[self.mp_pose.PoseLandmark.LEFT_EAR]
            l_mouth = lm[self.mp_pose.PoseLandmark.MOUTH_LEFT]

        face_len_px = math.hypot((l_eye.x - l_mouth.x)*w, (l_eye.y - l_mouth.y)*h)
        top_of_head_y = nose_y - (face_len_px * 2.0)
        height_px = heel_y - top_of_head_y
        
        # Eğer ayaklar görünmüyorsa gövde üzerinden tahmin yap (7.5 baş oranı)
        if lm[self.mp_pose.PoseLandmark.LEFT_HEEL].visibility < 0.5 and lm[self.mp_pose.PoseLandmark.RIGHT_HEEL].visibility < 0.5:
            l_hip_y = lm[self.mp_pose.PoseLandmark.LEFT_HIP].y * h
            r_hip_y = lm[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h
            hip_y = (l_hip_y + r_hip_y) / 2
            torso_px = hip_y - top_of_head_y
            height_px = torso_px / 0.48 
            
        return float(person_height_cm) / height_px

    def measure_baseline_from_model(self, garment_bgra, original_on_jpg_path):
        """
        KULLANICI İSTEĞİ: "Manken neyse önce ona göre analiz yap."
        Bu fonksiyon, ayiklanmis_on.png'nin orjinal kaynağı olan on.jpg'ye bakar.
        Eğer on.jpg'de manken varsa, onun piksel/cm oranını çıkarıp kıyafeti kusursuz ölçer.
        """
        if not os.path.exists(original_on_jpg_path):
            return None
            
        on_bgr = cv2.imread(original_on_jpg_path)
        if on_bgr is None: return None
        
        h, w = on_bgr.shape[:2]
        on_rgb = cv2.cvtColor(on_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(on_rgb)
        
        if not results.pose_landmarks:
            print("[INFO] Referans fotoğrafta insan bulunamadı (Flat-Lay). Manken analizi atlanıyor.")
            return None
            
        # Mankenin fotoğrafındaki cm/px oranını buluyoruz
        model_cm_per_px = self.calculate_scale_from_person(h, w, results.pose_landmarks, person_height_cm=175.0)
        
        # ayiklanmis_on.png'nin alpha kanalından şeffaf olmayan sınırları buluyoruz
        alpha = garment_bgra[:, :, 3]
        y_indices, x_indices = np.where(alpha > 0)
        
        if len(x_indices) > 0:
            g_w_px = np.max(x_indices) - np.min(x_indices)
            base_cm = g_w_px * model_cm_per_px
            print(f"[MODEL ANALİZİ] Manken üzerindeki Orijinal Tişört Genişliği Ölçüldü: {base_cm:.1f} cm")
            return base_cm
            
        return None

    def calculate_garment_cm(self, user_shoulder_px, user_cm_per_pixel, garment_bgra, original_on_jpg_path, desired_size="M", model_worn_size="M"):
        # 1. ÖNCE MANKEN ÜZERİNDEN GERÇEK ÖLÇÜYÜ ÇIKAR (Kullanıcı İsteği)
        base_cm = self.measure_baseline_from_model(garment_bgra, original_on_jpg_path)
        
        # 2. Eğer mankensiz fotoğrafsa (Flat-Lay) veya ölçüm başarısızsa Fallback (Omuz bazlı)
        if base_cm is None:
            user_shoulder_cm = user_shoulder_px * user_cm_per_pixel
            base_cm = user_shoulder_cm * 1.35 # Regular fallback
            print(f"[FALLBACK ANALİZ] Manken ölçülemedi, omuza göre {base_cm:.1f} cm varsayıldı.")
        
        # 3. Dinamik Beden Tablosu Farkları
        size_offsets = {
            "XS": -6.0,
            "S": -3.0,
            "M": 0.0,
            "L": +3.0,
            "XL": +6.0
        }
        
        model_offset = size_offsets.get(model_worn_size.upper(), 0.0)
        desired_offset = size_offsets.get(desired_size.upper(), 0.0)
        
        modifier = desired_offset - model_offset
        target_cm = base_cm + modifier
        
        hesap_raporu = f"Manken ({model_worn_size.upper()}): {base_cm:.1f}cm | Sen: {desired_size.upper()} ({modifier:+.1f}cm) -> Uygulanan: {target_cm:.1f}cm"
        print(f"[Beden Simülasyonu] {hesap_raporu}")
        return target_cm, base_cm, modifier, hesap_raporu

    def resize_garment_for_user(self, garment_bgra, garment_cm_width, user_cm_per_pixel, is_side_profile=False):
        alpha_channel = garment_bgra[:, :, 3]
        y_indices, x_indices = np.where(alpha_channel > 0)
        if len(y_indices) > 0:
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            cropped_garment = garment_bgra[y_min:y_max+1, x_min:x_max+1]
        else:
            cropped_garment = garment_bgra
            
        g_h, g_w = cropped_garment.shape[:2]
        aspect_ratio = g_h / float(g_w)
        
        if is_side_profile:
            garment_cm_width = garment_cm_width * 0.60
            
        target_w_px = int(garment_cm_width / user_cm_per_pixel)
        target_h_px = int(target_w_px * aspect_ratio)
            
        return cv2.resize(cropped_garment, (target_w_px, target_h_px), interpolation=cv2.INTER_AREA)

    def overlay_and_blend(self, user_bgr, garment_bgra, landmarks, is_side_profile=False):
        h, w = user_bgr.shape[:2]
        lm = landmarks.landmark
        
        l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        l_vis = l_sh.visibility
        r_vis = r_sh.visibility
        
        if is_side_profile:
            if l_vis > r_vis:
                mid_x, mid_y = int(l_sh.x * w), int(l_sh.y * h)
                hip_x, hip_y = int(l_hip.x * w), int(l_hip.y * h)
            else:
                mid_x, mid_y = int(r_sh.x * w), int(r_sh.y * h)
                hip_x, hip_y = int(r_hip.x * w), int(r_hip.y * h)
            
            mid_y = mid_y + int((hip_y - mid_y) * 0.05)
            dx, dy = hip_x - mid_x, hip_y - mid_y
            angle_deg = -math.degrees(math.atan2(dx, dy))
        else:
            l_sh_x, l_sh_y = int(l_sh.x * w), int(l_sh.y * h)
            r_sh_x, r_sh_y = int(r_sh.x * w), int(r_sh.y * h)
            
            dx = (l_sh_x - r_sh_x) if l_sh_x > r_sh_x else (r_sh_x - l_sh_x)
            dy = (l_sh_y - r_sh_y) if l_sh_x > r_sh_x else (r_sh_y - l_sh_y)
            mid_x, mid_y = (l_sh_x + r_sh_x) // 2, (l_sh_y + r_sh_y) // 2
            shoulder_width = math.hypot(dx, dy)
            mid_y = int(mid_y - (shoulder_width * 0.15))
            angle_deg = -math.degrees(math.atan2(dy, dx))
        
        g_h, g_w = garment_bgra.shape[:2]
        M = cv2.getRotationMatrix2D((g_w / 2.0, g_h * 0.05), angle_deg, 1.0)
        M[0, 2] += (mid_x - (g_w / 2.0))
        M[1, 2] += (mid_y - (g_h * 0.05))
        
        warped_garment = cv2.warpAffine(garment_bgra, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        result = user_bgr.copy()
        mask_img = np.zeros((h, w, 4), dtype=np.uint8)
        alpha = warped_garment[:, :, 3] / 255.0
        
        y_indices, x_indices = np.where(alpha > 0)
        if len(y_indices) > 0:
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            
            mask_img[y_indices, x_indices, 0:3] = 255
            mask_img[y_indices, x_indices, 3] = 255
            mask_img[cv2.dilate(mask_img[:, :, 3], np.ones((35, 35), np.uint8), iterations=2) > 0] = 255
            
            alpha_crop = alpha[y_min:y_max+1, x_min:x_max+1]
            for c in range(3):
                result[y_min:y_max+1, x_min:x_max+1, c] = (
                    alpha_crop * warped_garment[y_min:y_max+1, x_min:x_max+1, c] +
                    (1 - alpha_crop) * result[y_min:y_max+1, x_min:x_max+1, c]
                )
        return result, mask_img

    def get_tension_color(self, garment_cm, user_cm):
        garment_circumference = garment_cm * 2.0 
        ratio = garment_circumference / user_cm
        # Matematiksel Katman Farkı Simülasyonu (ΔW = Wgarment − Wbody)
        if ratio < 0.95: return (0, 0, 255), f"Dar ({ratio:.2f})" # Kırmızı
        elif 0.95 <= ratio <= 1.05: return (0, 255, 255), f"Tam ({ratio:.2f})" # Sarı
        else: return (0, 255, 0), f"Bol ({ratio:.2f})" # Yeşil

    def draw_heatmap(self, image, landmarks, garment_width_cm, user_measurements, garment_type="Üst Giyim"):
        # Bu fonksiyon iptal edildi, artık Node-03 (FitHeatmapNode) kullanılıyor!
        pass

    def run(self, user_image_path, garment_image_path, original_on_jpg_path, output_name, user_measurements=None, desired_size="M", model_worn_size="M", garment_type="Üst Giyim", use_ai_render=True, colab_url=""):
        print(f"\n--- [VTON] GİYDİRME BAŞLIYOR: {user_image_path} (Beden: {desired_size}) ---")
        user_bgr = cv2.imread(user_image_path)
        garment_bgra = cv2.imread(garment_image_path, cv2.IMREAD_UNCHANGED)
        
        if user_bgr is None or garment_bgra is None: return "Hata: Fotoğraflar okunamadı."
            
        user_rgb = cv2.cvtColor(user_bgr, cv2.COLOR_BGR2RGB)
        user_results = self.pose.process(user_rgb)
        if not user_results.pose_landmarks: return "Hata: Kullanıcı pozisyonu bulunamadı."
            
        user_h, user_w = user_bgr.shape[:2]
        
        # -------------------------------------------------------------------
        # TENSOR PIPELINE ENTEGRASYONU: Node-02 ve Node-03
        # -------------------------------------------------------------------
        if not hasattr(self, 'node_02'):
            from node_02_body_mesh import BodyMeshNode
            self.node_02 = BodyMeshNode()
        if not hasattr(self, 'node_03'):
            from node_03_heatmap import FitHeatmapNode
            self.node_03 = FitHeatmapNode()
        
        # ADIM 04: 3D Body Tensor (Yan duruş açısı ve 3D gövde genişlikleri)
        body_tensor = self.node_02.generate_body_tensor(user_results.pose_landmarks, user_w, user_h)
        is_side_profile = body_tensor.get("is_side_profile", False)
        # -------------------------------------------------------------------
        
        l_sh = user_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = user_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        u_height = user_measurements.get('height') if user_measurements else None
        user_cm_per_pixel = self.calculate_scale_from_person(user_h, user_w, user_results.pose_landmarks, u_height)
        user_shoulder_px = math.hypot((l_sh.x - r_sh.x)*user_w, (l_sh.y - r_sh.y)*user_h)
        
        garment_cm_width, base_cm, modifier, hesap_raporu = self.calculate_garment_cm(user_shoulder_px, user_cm_per_pixel, garment_bgra, original_on_jpg_path, desired_size, model_worn_size)
            
        # -------------------------------------------------------------------
        # ANSUR 3D -> 2D CONVERTER (Kıyafet Maskesi Ölçeklendirme)
        # -------------------------------------------------------------------
        from ansur_model import Ansur3DTo2DConverter
        ansur = Ansur3DTo2DConverter()
        
        if garment_type == "Üst Giyim":
            target_3d = user_measurements.get('chest', 95.0)
            ansur_2d_width = ansur.circumference_to_width(target_3d, "chest")
        elif garment_type == "Alt Giyim":
            target_3d = user_measurements.get('hips', 98.0)
            ansur_2d_width = ansur.circumference_to_width(target_3d, "hips")
        else: # Elbise
            target_3d = max(user_measurements.get('chest', 95.0), user_measurements.get('hips', 98.0))
            ansur_2d_width = ansur.circumference_to_width(target_3d, "chest")
            
        # ANSUR 2D genişliğini beden farkı (modifier) ile birleştirerek maskeyi ölçeklendir
        ansur_garment_cm_width = ansur_2d_width + modifier
        hesap_raporu += f"\n[ANSUR 3D->2D] Kullanıcının {target_3d}cm ({garment_type}) 3D çevresi -> {ansur_2d_width:.1f}cm 2D Genişliğe çevrildi.\n(Kıyafet Maskesi Hedef Genişliği: {ansur_garment_cm_width:.1f}cm)"
        
        # Maske artık Ansur modelinin verdiği değere göre büyütülüp küçültülüyor
        scaled_garment = self.resize_garment_for_user(garment_bgra, ansur_garment_cm_width, user_cm_per_pixel, is_side_profile)
        final_image, mask_image = self.overlay_and_blend(user_bgr, scaled_garment, user_results.pose_landmarks, is_side_profile)
        
        mask_output_name = output_name.replace(".jpg", "_mask.png")
        cv2.imwrite(output_name, final_image)
        cv2.imwrite(mask_output_name, mask_image)
        
        if user_measurements and not is_side_profile:
            heatmap_image = final_image.copy()
            # ADIM 06: Isı Haritası Node-03'e Devredildi!
            heatmap_image = self.node_03.apply_heatmap_overlay(heatmap_image, body_tensor, garment_cm_width, user_measurements, garment_type)
            heatmap_output_name = output_name.replace(".jpg", "_heatmap.jpg")
            cv2.imwrite(heatmap_output_name, heatmap_image)

        mesaj = f"Kıyafet Analizi:\n{hesap_raporu}\n\n"
        
        if use_ai_render:
            try:
                from vton_ai_renderer import render_with_idm_vton
                hd_output_name = output_name.replace(".jpg", "_HD_RENDER.jpg")
                success = render_with_idm_vton(user_image_path, garment_image_path, hd_output_name, mask_output_name, colab_url=colab_url)
                if success:
                    mesaj += "Google Colab / HD Render API işlemi tamamlandı."
            except Exception as e:
                mesaj += f"HD Render Hatası: {e}"
        else:
            mesaj += "Hızlı mod (Sadece OpenCV) kullanıldı."
            
        return mesaj

if __name__ == "__main__":
    engine = VTONEngine()
    # Test
    pass
