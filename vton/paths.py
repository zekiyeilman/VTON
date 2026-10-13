"""
Proje genelindeki dosya yollarının tek merkezi.

Tüm modüller göreli yol ("goruntu/on.jpg") yerine buradaki sabitleri kullanır.
Böylece kod; PyCharm'dan, Colab'den veya farklı bir çalışma dizininden
çağrıldığında da aynı klasörleri bulur.

Varsayılan kök dizin: bu paketin bir üst klasörü (repo kökü).
Farklı bir yer kullanmak için VTON_HOME ortam değişkenini ayarlayın.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("VTON_HOME", Path(__file__).resolve().parent.parent))

GORUNTU_DIR = PROJECT_ROOT / "goruntu"
DATASET_DIR = PROJECT_ROOT / "dataset"
ANSUR_CSV = DATASET_DIR / "ansur" / "ANSUR II MALE Public.csv"

# Beden tablosu önbelleği: marka/demografi/kategori bazında bir kez çekilir, sonra buradan okunur
BEDEN_TABLOSU_JSON = PROJECT_ROOT / "beden_tablolari.json"
CALIBRATION_JSON = PROJECT_ROOT / "vton_calibration_memory.json"
CHROME_PROFILE_DIR = PROJECT_ROOT / "VTON_Chrome_Profil"

GORUNTU_DIR.mkdir(parents=True, exist_ok=True)


def goruntu(dosya_adi: str) -> str:
    """goruntu/ klasöründeki bir dosyanın mutlak yolunu döndürür."""
    return str(GORUNTU_DIR / dosya_adi)
