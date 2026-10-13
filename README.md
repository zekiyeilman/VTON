# 👗 Virtual Try-On (VTON) Pipeline

Bu proje; 3B vücut haritalama, segmentasyon, giysi çıkarma ve ısı haritası (fit/darlık/bolluk) analizi sunan modüler bir Sanal Giydirme (Virtual Try-On) motorudur.

---

## 🚀 Google Colab Üzerinde Çalıştırma

Projeyi Google Colab GPU ortamında tek tıkla çalıştırmak için:

1. [Google Colab](https://colab.research.google.com)'i açın.
2. `colab_run.ipynb` dosyasını Colab'e yükleyin veya bir kod hücresinde şu adımları izleyin:

```bash
# 1. Depoyu klonlayın
!git clone https://github.com/zekiyeilman/VTON.git
%cd VTON

# 2. Gereksinimleri yükleyin
!pip install -r requirements.txt

# 3. Gradio Arayüzünü genel link (share link) ile başlatın
!GRADIO_SHARE=true python app.py
```

---

## 📂 Modüler Proje Mimarisi

- `app.py`: Gradio tabanlı kullanıcı etkileşim arayüzü.
- `vton_engine.py`: VTON orkestrasyon motoru (Ön, arka ve yan profil işleme).
- `node_01_segmentation.py`: SegFormer tabanlı giysi ve vücut segmentasyonu.
- `node_02_body_mesh.py`: MediaPipe ile 3B vücut mesh noktaları ve ölçüm çıkarımı.
- `node_03_heatmap.py`: Beden ve esneme tabanlı sıkılık/bolluk ısı haritası üretimi.
- `garment_extractor.py`: Ürün fotoğraflarından otomatik giysi ayıklama (rembg).
- `ansur_model.py`: Antropometrik vücut ölçü regresyon modeli (ANSUR II).
- `scraper_pipeline.py` & `scraper_trendyol.py`: E-ticaret ürün detay ve beden tablosu çekici.
- `vton_ai_renderer.py`: Difüzyon tabanlı HD yapay zeka render motoru.
- `dataset/`: ANSUR II veri seti ve önbelleklenen beden tabloları.
- `goruntu/`: Model referans ve kullanıcı fotoğrafları.

---

## 💻 Yerel Kurulum (Local Setup)

```bash
# Sanal ortam oluşturup aktif edin
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Arayüzü başlatın
python app.py
```
