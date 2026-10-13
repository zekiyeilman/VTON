import cv2
import numpy as np
import rembg
import mediapipe as mp
import os

def ayikla_ust_kiyafet(input_image_path, output_image_path):
    print(f"[INFO] '{input_image_path}' yükleniyor...")
    
    # Resmi oku
    img = cv2.imread(input_image_path)
    if img is None:
        print("[ERROR] Dosya okunamadı!")
        return

    # ---------------------------------------------------------
    # 1. AŞAMA: MEDIAPIPE İLE İSKELETİ BUL (Person vs Flat-Lay)
    # ---------------------------------------------------------
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("[INFO] İskelet analizi yapılıyor (Person/Flat-Lay ayrımı için)...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results = pose.process(img_rgb)
    
    # ---------------------------------------------------------
    # 2. AŞAMA: REMBG İLE KIYAFETİ AYIKLA
    # ---------------------------------------------------------
    if results.pose_landmarks:
        print("[INFO] Manken algılandı! AI Modeli: u2net_cloth_seg (Sadece kıyafet)...")
        session = rembg.new_session("u2net_cloth_seg")
    else:
        print("[INFO] Manken bulunamadı (Flat-Lay). AI Modeli: u2net (Tüm obje)...")
        session = rembg.new_session("u2net")
        
    print("[INFO] Kıyafet maskesi çıkarılıyor...")
    rgba_cloth = rembg.remove(img_rgb, session=session)
    rgba_cloth = cv2.cvtColor(rgba_cloth, cv2.COLOR_RGBA2BGRA)
    
    if results.pose_landmarks:
        h, w, _ = img.shape
        lm = results.pose_landmarks.landmark
        
        # Kalça noktalarını bul
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Omuz noktalarını bul (Yaka kısmını korumak için)
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        if l_hip.visibility > 0.5 or r_hip.visibility > 0.5:
            # Kalçaların ortalama Y pozisyonunu piksel cinsinden hesapla
            hip_y = int(((l_hip.y + r_hip.y) / 2) * h)
            
            # Tişörtler kalçanın biraz altına kadar inebilir, bu yüzden bir tolerans ekleyelim
            # Omuz ve kalça arasındaki mesafenin %20'si kadar aşağıdan keselim
            shoulder_y = int(((l_sh.y + r_sh.y) / 2) * h)
            torso_length = hip_y - shoulder_y
            kesme_hizasi_y = hip_y + int(torso_length * 0.25)
            
            print(f"[INFO] Kalça hizası tespit edildi. Y={kesme_hizasi_y} pikselinden aşağısı (Pantolon) siliniyor...")
            
            # Maskenin Y=kesme_hizasi_y altındaki her şeyin Alpha kanalını 0 (şeffaf) yap
            # Sadece alpha kanalına müdahale ediyoruz
            if kesme_hizasi_y < h:
                rgba_cloth[kesme_hizasi_y:, :, 3] = 0
        else:
            print("[WARN] Kalça noktaları net görülmüyor, pantolon kesimi yapılamayabilir.")
    else:
        print("[WARN] İskelet bulunamadı, sadece rembg maskesi kaydedilecek.")

    # Sonucu kaydet
    cv2.imwrite(output_image_path, rgba_cloth)
    print(f"[SUCCESS] Üst kıyafet başarıyla ayrıştırıldı ve kaydedildi: {output_image_path}")
if __name__ == "__main__":
    # python -m vton.extractor
    from vton.paths import goruntu
    hedef_dosyalar = {
        goruntu("on.jpg"): goruntu("ayiklanmis_on.png"),
        goruntu("arka.jpg"): goruntu("ayiklanmis_arka.png")
    }
    
    islem_yapildi = False
    for input_path, output_path in hedef_dosyalar.items():
        if os.path.exists(input_path):
            print(f"\n--- [İŞLEM BAŞLIYOR] {input_path} ---")
            ayikla_ust_kiyafet(input_path, output_path)
            islem_yapildi = True
        else:
            print(f"[SKIP] {input_path} bulunamadı, atlanıyor.")
            
    if not islem_yapildi:
        print("\n[WARN] İşlenecek hiçbir fotoğraf bulunamadı! Önce scraper'ı çalıştırdığına emin ol.")
