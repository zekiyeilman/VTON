# 👗 VTON - Beden Duyarlı Sanal Giydirme

3B vücut haritalama, segmentasyon, ANSUR tabanlı ölçeklendirme ve ısı haritası (darlık/bolluk) analizi yapan modüler bir Virtual Try-On pipeline'ı.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zekiyeilman/VTON/blob/main/colab_run.ipynb)

---

## 🚀 Google Colab

En kolayı: yukarıdaki **Open in Colab** rozetine tıklayıp hücreleri sırayla çalıştırın.

Kendi notebook'unuzda kullanmak için:

```python
!git clone https://github.com/zekiyeilman/VTON.git /content/VTON
%cd /content/VTON
!pip install -q -e .

import sys; sys.path.insert(0, "/content/VTON")

from vton import VTONEngine, SegFormerNode, Ansur3DTo2DConverter
from vton.scraper import vton_otonom_boru_hatti
from vton.paths import goruntu
```

Arayüzü public link ile açmak için:

```python
import app
app.demo.launch(share=True, debug=True)
```

> [!NOTE]
> - HD render için Hugging Face token'ını Colab **Secrets** bölümüne `HF_TOKEN` olarak ekleyin. Token'ı koda yazmayın.
> - "Linkten veri çek" özelliği Colab'de headless Chrome kullanır; notebook'taki 4. hücre Chrome'u kurar.

---

## 📂 Paket Yapısı

```text
VTON/
├── vton/                      # pip install -e . ile kurulan paket
│   ├── __init__.py            # VTONEngine, SegFormerNode ... (lazy import)
│   ├── paths.py               # Tüm dosya yollarının tek merkezi
│   ├── engine.py              # VTONEngine - orkestrasyon
│   ├── renderer.py            # IDM-VTON HD render (HF_TOKEN ortam değişkeni)
│   ├── extractor.py           # rembg ile kıyafet ayıklama
│   ├── calibration.py         # RandomForest kalibrasyon yardımcıları
│   ├── nodes/
│   │   ├── segmentation.py    # Node-01: SegFormer segmentasyon
│   │   ├── body_mesh.py       # Node-02: 3B gövde tensörü
│   │   └── heatmap.py         # Node-03: darlık/bolluk ısı haritası
│   ├── scraper/
│   │   ├── pipeline.py        # vton_otonom_boru_hatti
│   │   ├── trendyol.py        # Selenium scraper + beden tablosu önbelleği
│   │   └── calibration.py     # Manken cm/piksel kalibrasyonu
│   └── models/
│       └── ansur.py           # ANSUR 3B çevre -> 2B genişlik
├── app.py                     # Gradio arayüzü
├── colab_run.ipynb            # Colab notebook'u
├── pyproject.toml
├── requirements.txt
├── beden_tablolari.json       # Marka bazlı beden tablosu önbelleği
├── vton_calibration_memory.json
├── dataset/ansur/             # ANSUR II veri seti
└── goruntu/                   # Kullanıcı ön/arka/yan + manken görselleri ve maskeler
```

Kök dizindeki `node_01_segmentation.py`, `vton_engine.py` gibi dosyalar sadece **geriye dönük uyumluluk** içindir; gerçek kod `vton/` altındadır. Eski `from vton_engine import VTONEngine` import'ları çalışmaya devam eder.

Dosya yolları `vton/paths.py` üzerinden repo köküne göre çözülür. Farklı bir veri klasörü için `VTON_HOME` ortam değişkenini ayarlayabilirsiniz.

---

## 💻 Yerel Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python app.py
```

Tek bir modülü test etmek için: `python -m vton.nodes.segmentation`, `python -m vton.scraper.pipeline` vb.
