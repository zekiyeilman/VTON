"""
kalibrasyon.py
Virtual Try-On Projesi — ANSUR & MediaPipe Kalibrasyon Motoru
"""

import math
import pickle
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ─────────────────────────────────────────────
# 1. MediaPipe: Omuz Piksel Mesafesi
# ─────────────────────────────────────────────

def omuz_piksel_mesafesi(goruntu_yolu: str) -> Optional[float]:
    """
    Verilen görüntüdeki MediaPipe Pose Landmark 11 (sol omuz)
    ve 12 (sağ omuz) arasındaki Öklid mesafesini piksel cinsinden döndürür.
    Landmark tespit edilemezse None döner.
    """
    goruntu = cv2.imread(goruntu_yolu)
    if goruntu is None:
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {goruntu_yolu}")

    yukseklik, genislik = goruntu.shape[:2]
    rgb = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        sonuc = pose.process(rgb)

    if not sonuc.pose_landmarks:
        return None

    landmarks = sonuc.pose_landmarks.landmark

    sol_omuz = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]   # Landmark 11
    sag_omuz = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]  # Landmark 12

    # Normalize koordinatları gerçek piksele çevir
    sol_x = sol_omuz.x * genislik
    sol_y = sol_omuz.y * yukseklik
    sag_x = sag_omuz.x * genislik
    sag_y = sag_omuz.y * yukseklik

    mesafe = math.hypot(sag_x - sol_x, sag_y - sol_y)
    return mesafe


# ─────────────────────────────────────────────
# 2. RandomForest: Hedef Piksel Genişliği
# ─────────────────────────────────────────────

def model_yukle(model_yolu: str = "model.pkl"):
    """
    scikit-learn RandomForest modelini (model.pkl) diskten yükler.
    """
    yol = Path(model_yolu)
    if not yol.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_yolu}")

    with open(yol, "rb") as f:
        model = pickle.load(f)

    return model


def hedef_piksel_genisligi(
    model,
    mezura_olculeri: dict,
    ozellik_sirasi: Optional[list] = None,
) -> float:
    """
    RandomForest modeline mezura ölçülerini (cm) vererek
    tahmini hedef piksel genişliğini döndürür.

    Parameters
    ----------
    model          : Yüklü sklearn modeli
    mezura_olculeri: Örnek → {"gogus": 90, "bel": 75, "kalca": 95}
    ozellik_sirasi : Modelin beklediği sütun sırası.
                     None ise mezura_olculeri'nin ekleme sırası kullanılır.
    """
    if ozellik_sirasi is not None:
        ozellik_vektoru = [mezura_olculeri[k] for k in ozellik_sirasi]
    else:
        ozellik_vektoru = list(mezura_olculeri.values())

    X = np.array(ozellik_vektoru, dtype=np.float32).reshape(1, -1)
    tahmin = model.predict(X)
    return float(tahmin[0])


# ─────────────────────────────────────────────
# 3. Delta Piksel Hesaplama
# ─────────────────────────────────────────────

def delta_piksel_hesapla(
    mevcut_piksel: float,
    hedef_piksel: float,
) -> int:
    """
    Genişleme/Daralma Delta Pikseli.

    Pozitif değer → giysi genişletilmeli  (hedef > mevcut)
    Negatif değer → giysi daraltılmalı    (hedef < mevcut)
    Sıfır         → değişiklik gerekmez
    """
    delta = hedef_piksel - mevcut_piksel
    return int(round(delta))


# ─────────────────────────────────────────────
# 4. Ana Kalibrasyon Akışı
# ─────────────────────────────────────────────

def kalibre_et(
    goruntu_yolu: str,
    mezura_olculeri: dict,
    model_yolu: str = "model.pkl",
    ozellik_sirasi: Optional[list] = None,
    verbose: bool = True,
) -> dict:
    """
    Tam kalibrasyon akışını çalıştırır ve sonuç sözlüğü döndürür.

    Returns
    -------
    {
        "mevcut_piksel_genisligi" : float,
        "hedef_piksel_genisligi"  : float,
        "delta_piksel"            : int,   # Genişleme/Daralma Delta Pikseli
    }
    """
    # 1. Omuz tespiti
    mevcut = omuz_piksel_mesafesi(goruntu_yolu)
    if mevcut is None:
        raise RuntimeError(
            "MediaPipe görüntüdeki omuz noktalarını tespit edemedi. "
            "Daha net bir tam/yarım boy fotoğraf deneyin."
        )

    # 2. RF modeli tahmini
    model = model_yukle(model_yolu)
    hedef = hedef_piksel_genisligi(model, mezura_olculeri, ozellik_sirasi)

    # 3. Delta
    delta = delta_piksel_hesapla(mevcut, hedef)

    if verbose:
        print(f"[Kalibrasyon]")
        print(f"  Mevcut Omuz Piksel Genişliği : {mevcut:.2f} px")
        print(f"  Hedef  Omuz Piksel Genişliği : {hedef:.2f} px")
        print(f"  Genişleme/Daralma Delta Pikseli : {delta:+d} px")
        if delta > 0:
            print(f"  → Giysi {abs(delta)} piksel GENİŞLETİLMELİ")
        elif delta < 0:
            print(f"  → Giysi {abs(delta)} piksel DARALTILMALI")
        else:
            print(f"  → Değişiklik gerekmez")

    return {
        "mevcut_piksel_genisligi": mevcut,
        "hedef_piksel_genisligi": hedef,
        "delta_piksel": delta,
    }


# ─────────────────────────────────────────────
# 5. Örnek Kullanım (Colab / doğrudan çalıştırma)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    GORUNTU_YOLU = "ornek_kullanici.jpg"   # Kullanıcı fotoğrafı
    MODEL_YOLU   = "model.pkl"             # Eğitilmiş RF modeli

    # Kullanıcının girdiği mezura ölçüleri (cm)
    MEZURA = {
        "gogus" : 90,
        "bel"   : 75,
        "kalca" : 95,
    }

    # Modelin beklediği sütun sırası (eğitim sırasında kullanılan sıra)
    OZELLIK_SIRASI = ["gogus", "bel", "kalca"]

    sonuc = kalibre_et(
        goruntu_yolu   = GORUNTU_YOLU,
        mezura_olculeri = MEZURA,
        model_yolu     = MODEL_YOLU,
        ozellik_sirasi = OZELLIK_SIRASI,
        verbose        = True,
    )

    print("\n[Sonuç Sözlüğü]", sonuc)
