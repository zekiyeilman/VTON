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

from scraper_trendyol import TrendyolOtonomScraper
from scraper_calibration import MediaPipeAdaptiveCalibrator

def vton_otonom_boru_hatti(urun_url):
    print("=" * 65)
    print("[START] VTON TAM OTONOM VERİ & GÖRSEL İŞLEME BORU HATTI")
    print("=" * 65)

    scraper = TrendyolOtonomScraper()

    try:
        ground_truth, gorseller, beden_tablosu, demografi = scraper.scrape(urun_url)

        print("\n--- [INFO] KAZINAN GROUND TRUTH BİLGİLERİ ---")
        print(f"Demografi Tespiti: {demografi.upper()}")
        if ground_truth:
            for k, v in ground_truth.items():
                print(f"[OK] {k.upper()}: {v} cm")
        else:
            print("[WARN] Ürün açıklamasında anatomik ölçü bulunamadı.")
            
        if beden_tablosu:
            print(f"\n--- 📊 STATİK BEDEN TABLOSU (Hedef: {demografi}) ---")
            print(f"{' | '.join(beden_tablosu['sutunlar'])}")
            for satir in beden_tablosu['veriler']:
                print(f"{satir['beden']:<7} | {satir['gogus'][0]}-{satir['gogus'][1]}       | {satir['bel'][0]}-{satir['bel'][1]}     | {satir['basen'][0]}-{satir['basen'][1]}")

        if not gorseller:
            print("[ERROR] Bu ürüne ait görsel bulunamadı.")
        else:
            print(f"\n[IMG] Sadece bu ürüne ait olan orjinal {len(gorseller)} adet görsel bulundu.")
            
            calibrator = MediaPipeAdaptiveCalibrator()
            
            # KULLANICI İSTEĞİ: Akıllı Görsel Seçimi (Ölçü Ref, Ön, Arka ve Düz Kıyafet için ayrı ayrı)
            print("[INFO] Gözlerin, kıyafetin en net olduğu ve dik açıyla çekilmiş fotoğraf aranıyor...")
            en_uygun_gorsel_ref = None
            en_yuksek_skor_ref = -9999
            
            en_uygun_gorsel_on = None
            en_yuksek_skor_on = -9999
            
            en_uygun_gorsel_arka = None
            en_yuksek_skor_arka = -9999
            
            en_uygun_gorsel_yan = None
            en_yuksek_skor_yan = -9999
            
            duz_gorseller = []
            
            hedef_gorsel_url = gorseller[0]
            olcu_var_mi = ground_truth and ('manken_boy' in ground_truth or 'manken_gogus' in ground_truth or 'manken_bel' in ground_truth)
            
            if olcu_var_mi:
                print("[INFO] Model ölçüleri tespit edildi. Yine de en doğru hizalama için görseller AI ile taranıyor...")
            
            for i, url in enumerate(gorseller[:8]): # İlk 8 resmi tara
                print(f"  -> Aday görsel {i+1} inceleniyor...")
                temp_img = scraper.gorsel_indir(url)
                if temp_img is None: continue
                    
                rgb_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                results = calibrator.pose.process(rgb_img)
                
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    h, w, _ = temp_img.shape
                    
                    r_eye = lm[calibrator.mp_pose.PoseLandmark.RIGHT_EYE]
                    l_eye = lm[calibrator.mp_pose.PoseLandmark.LEFT_EYE]
                    l_sh = lm[calibrator.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    r_sh = lm[calibrator.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    l_hip = lm[calibrator.mp_pose.PoseLandmark.LEFT_HIP]
                    r_hip = lm[calibrator.mp_pose.PoseLandmark.RIGHT_HIP]
                    
                    # Temel Duruş Skoru (Sadece Görsellik)
                    durus_skoru = 0
                    
                    if r_eye.visibility > 0.6 and l_eye.visibility > 0.6:
                        durus_skoru += 20
                        
                    if l_sh.visibility > 0.6 and r_sh.visibility > 0.6 and l_hip.visibility > 0.6 and r_hip.visibility > 0.6:
                        durus_skoru += 40
                        
                    yaw, pitch = calibrator._calculate_pose_angles(lm, w, h)
                    durus_skoru -= min(abs(pitch) * 2, 30) 
                    durus_skoru -= min(abs(yaw), 20)
                    
                    # Ölçü Referans Skoru (İlk resim avantajı dahil)
                    ref_skoru = durus_skoru
                    if olcu_var_mi and i == 0:
                        ref_skoru += 15
                        
                    is_front = r_sh.x < l_sh.x
                    yon_str = "ÖN" if is_front else "ARKA"
                    print(f"     [Pose: {durus_skoru:.1f} | Ref: {ref_skoru:.1f}] (Yön: {yon_str} | Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
                    
                    # Ölçü Referansı İçin En İyiyi Seç
                    if ref_skoru > en_yuksek_skor_ref:
                        en_yuksek_skor_ref = ref_skoru
                        en_uygun_gorsel_ref = temp_img
                        
                    is_side_profile = abs(yaw) > 40.0 or abs(pitch) > 40.0

                    # Giydirme (VTON) için En İyi Ön, Arka ve Yanı Seç
                    if is_side_profile:
                        if durus_skoru > en_yuksek_skor_yan:
                            en_yuksek_skor_yan = durus_skoru
                            en_uygun_gorsel_yan = temp_img
                    elif is_front:
                        if durus_skoru > en_yuksek_skor_on:
                            en_yuksek_skor_on = durus_skoru
                            en_uygun_gorsel_on = temp_img
                    else:
                        if durus_skoru > en_yuksek_skor_arka:
                            en_yuksek_skor_arka = durus_skoru
                            en_uygun_gorsel_arka = temp_img
                else:
                    print("     [Skor: RED] İskelet bulunamadı -> (Düz Zemin / İnsansız Kıyafet Adayı)")
                    duz_gorseller.append(temp_img)

            # Fallbackler (Manken fotoğrafları)
            manken_resmi_ref = en_uygun_gorsel_ref if en_uygun_gorsel_ref is not None else scraper.gorsel_indir(gorseller[0])
            manken_resmi_on = en_uygun_gorsel_on if en_uygun_gorsel_on is not None else manken_resmi_ref
            
            if len(duz_gorseller) > 0:
                print(f"     [INFO] {len(duz_gorseller)} adet mankensiz temiz kıyafet (Flat-lay) fotoğrafı bulundu! Sınır Varyans Algoritması ile taranıyor...")
                
                gecerli_duz_gorseller = []
                for img in duz_gorseller:
                    try:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Fotoğrafın en dışındaki 10 piksellik çerçeveyi al
                        top = gray[0:10, :].flatten()
                        bottom = gray[-10:, :].flatten()
                        left = gray[:, 0:10].flatten()
                        right = gray[:, -10:].flatten()
                        
                        # Çerçevedeki tüm pikselleri birleştir
                        borders = np.concatenate((top, bottom, left, right))
                        
                        # Standart Sapma (Varyans) hesapla
                        std_dev = np.std(borders)
                        
                        # Eğer kenarlar dümdüz gri/beyaz arka plansa std_dev 1-5 arası çıkar.
                        # Eğer kıyafet kenarlara değiyorsa (yakın çekimse) dokudan dolayı std_dev çok yüksek çıkar (>15)
                        is_close_up = False
                        if std_dev > 15:
                            is_close_up = True
                                
                        if is_close_up:
                            print(f"     [INFO] Hızlı Tespit (Varyans: {std_dev:.1f}): Bu bir YAKIN ÇEKİM (Detay) fotoğrafı. VTON için eleniyor.")
                        else:
                            print(f"     [INFO] Mükemmel Flat-lay (Varyans: {std_dev:.1f}). Onaylandı!")
                            gecerli_duz_gorseller.append(img)
                    except Exception as e:
                        print("     [WARN] Hızlı analiz başarısız oldu, resim varsayılan olarak kabul ediliyor.", e)
                        gecerli_duz_gorseller.append(img)
                
                # Geçerli flat-lay resimler varsa mankenlileri ez!
                if len(gecerli_duz_gorseller) > 0:
                    print(f"     [INFO] {len(gecerli_duz_gorseller)} adet mükemmel kalitede mankensiz kıyafet bulundu. Ön/Arka mankenler eziliyor!")
                    manken_resmi_on = gecerli_duz_gorseller[0]
                    if len(gecerli_duz_gorseller) > 1:
                        en_uygun_gorsel_arka = gecerli_duz_gorseller[1]
                    else:
                        en_uygun_gorsel_arka = None

            print(f"[SUCCESS] Otonom Sınıflandırma Tamamlandı!")
            
            if manken_resmi_ref is not None:
                cv2.imwrite("goruntu/olcu_ref.jpg", manken_resmi_ref)
                print("[SAVE] Ölçüm Referans görseli 'goruntu/olcu_ref.jpg' olarak kaydedildi.")
                
            if manken_resmi_on is not None:
                cv2.imwrite("goruntu/on.jpg", manken_resmi_on)
                print("[SAVE] En iyi ÖN duruş 'goruntu/on.jpg' olarak kaydedildi.")
                
            if en_uygun_gorsel_arka is not None:
                cv2.imwrite("goruntu/arka.jpg", en_uygun_gorsel_arka)
                print("[SAVE] En iyi ARKA duruş 'goruntu/arka.jpg' olarak kaydedildi.")

            if en_uygun_gorsel_yan is not None:
                cv2.imwrite("goruntu/yan.jpg", en_uygun_gorsel_yan)
                print("[SAVE] En iyi YAN duruş 'goruntu/yan.jpg' olarak kaydedildi.")
                
            if manken_resmi_ref is not None:
                print("\n--- [CALC] MEDİAPİPE PERSPEKTİF & KALİBRASYON ANALİZİ ---")
                try:
                    analiz_sonucu = calibrator.process_image(manken_resmi_ref, ground_truth=ground_truth, demografi=demografi)
                    print(json.dumps(analiz_sonucu, indent=4, ensure_ascii=True))
                    
                    # Kalibrasyon verisini motor için kaydet
                    with open("vton_calibration_memory.json", "w", encoding="utf-8") as f:
                        json.dump(analiz_sonucu, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    import traceback
                    print("MEDİAPİPE VEYA JSON CRASH:", e)
                    traceback.print_exc()

            # Yeni Eklenen: Arayüze veri döndürmek için
            return {
                "ground_truth": ground_truth,
                "beden_tablosu": beden_tablosu,
                "demografi": demografi,
                "ref_image_path": "goruntu/olcu_ref.jpg" if manken_resmi_ref is not None else None,
                "on_image_path": "goruntu/on.jpg" if manken_resmi_on is not None else None,
                "arka_image_path": "goruntu/arka.jpg" if en_uygun_gorsel_arka is not None else None,
                "yan_image_path": "goruntu/yan.jpg" if en_uygun_gorsel_yan is not None else None
            }
    except Exception as ex:
        print("[ERROR] Boru hattı çalışırken hata:", ex)
        return None
    finally:
        scraper.kapat()
        print("=" * 65)


if __name__ == "__main__":
    HEDEF_URL = "https://www.trendyol-milla.com/trendyolmilla/antrasit-100-pamuk-oversize-genis-kalip-baskili-orme-t-shirt-twoss25ts00148-p-896485014"
    vton_otonom_boru_hatti(HEDEF_URL)