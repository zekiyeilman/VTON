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

class TrendyolOtonomScraper:
    def __init__(self):
        self.chrome_process = None
        self._chrome_otomatik_baslat()

        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

        for i in range(5):
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
                print("[OK] Otonom Chrome tarayıcısına başarıyla bağlanıldı!")
                break
            except Exception:
                time.sleep(1)
        else:
            raise Exception("Chrome bağlantısı kurulamadı.")

    def _chrome_otomatik_baslat(self):
        print("[SYSTEM] Sistem kendi özel tarayıcısını başlatıyor...")
        olasi_yollar = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe")
        ]
        chrome_yolu = next((yol for yol in olasi_yollar if os.path.exists(yol)), None)
        if not chrome_yolu:
            raise Exception("Chrome.exe bulunamadı!")

        profil_dizini = os.path.join(os.getcwd(), "VTON_Chrome_Profil")
        komut = [
            chrome_yolu,
            "--remote-debugging-port=9222",
            f"--user-data-dir={profil_dizini}",
            "--no-first-run",
            "--no-default-browser-check"
        ]
        self.chrome_process = subprocess.Popen(komut)
        time.sleep(2)

    def kapat(self):
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
            if self.chrome_process:
                self.chrome_process.terminate()
        except:
            pass

    def extract_measurements(self, text):
        olcumler = {}

        # Çok daha esnek ve güçlü Keskin Nişancı Regex
        patterns = {
            'manken_boy': r'Boy(?:uzunluğu)?[\s:]*(\d[.,]\d{2}|\d{3})\s*(?:cm)?',
            'manken_kilo': r'Kilo[\s:]*(\d{2,3})\s*(?:kg)?',
            'manken_gogus': r'Göğüs[\s:]*(\d{2,3})\s*(?:cm)?',
            'manken_bel': r'Bel[\s:]*(\d{2,3})\s*(?:cm)?',
            'manken_kalca': r'Kalça[\s:]*(\d{2,3})\s*(?:cm)?',
            'urun_on_boy': r'[ÖoO]n\s*Boyu?[\s:]*(\d{2,3}(?:[.,]\d+)?)\s*(?:cm)?',
            'manken_ic_bacak': r'İç\s*Bacak[\s:]*(\d{2,3}(?:[.,]\d+)?)\s*(?:cm)?'
        }

        validasyon = {
            'manken_boy': (140.0, 205.0),
            'manken_kilo': (30.0, 150.0),
            'manken_gogus': (60.0, 135.0),
            'manken_bel': (50.0, 115.0),
            'manken_kalca': (70.0, 145.0),
            'urun_on_boy': (25.0, 160.0),
            'manken_ic_bacak': (50.0, 110.0)
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val_str = match.group(1).replace(',', '.')
                try:
                    val = float(val_str)
                    # "1.89" gibi metre cinsinden girildiyse cm'ye çevir
                    if key == 'manken_boy' and val < 3.0:
                        val *= 100
                    
                    if validasyon[key][0] <= val <= validasyon[key][1]:
                        olcumler[key] = val
                except ValueError:
                    continue
        return olcumler

    def _kategori_ve_marka_belirle(self, url, tam_metin):
        marka = "bilinmeyen_marka"
        if "trendyolmilla" in url.lower():
            marka = "trendyolmilla"
        
        url_lower = url.lower()
        
        # Demografi (Yaş ve Cinsiyet) Tespiti
        demografi = "kadin" # Varsayılan
        if "erkek-cocuk" in url_lower or "kiz-cocuk" in url_lower or "-cocuk-" in url_lower or "çocuk" in url_lower:
            demografi = "cocuk"
        elif "-bebek-" in url_lower or "bebek" in url_lower:
            demografi = "bebek"
        elif "erkek" in url_lower and "kadin" not in url_lower:
            demografi = "erkek"
            
        kategori = "diger"
        if any(k in url_lower for k in ["t-shirt", "tshirt", "elbise", "bluz", "sweatshirt", "gomlek", "gömlek", "kazak"]):
            kategori = "ust_giyim"
        elif any(k in url_lower for k in ["pantolon", "jean", "etek", "sort", "şort", "esofman"]):
            kategori = "alt_giyim"
            
        return marka, kategori, demografi

    def _sayfadan_beden_tablosu_ogren(self):
        try:
            # 1. Beden tablosu butonunu bul ve tıkla
            butonlar = self.driver.find_elements(By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZÖÇŞİĞÜ', 'abcdefghijklmnopqrstuvwxyzöçşiğü'), 'beden tablosu') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZÖÇŞİĞÜ', 'abcdefghijklmnopqrstuvwxyzöçşiğü'), 'ölçü tablosu')]")
            tiklandi = False
            for btn in butonlar:
                if btn.is_displayed():
                    self.driver.execute_script("arguments[0].click();", btn)
                    tiklandi = True
                    time.sleep(2)
                    break
            
            if not tiklandi:
                return None
                
            # 2. Modal içerisindeki tabloyu veya grid'i bul
            modal_text = ""
            modallar = self.driver.find_elements(By.CSS_SELECTOR, "[class*='modal'], [class*='popup'], [class*='size'], [role='dialog']")
            for modal in modallar:
                if modal.is_displayed() and "Göğüs" in modal.text:
                    modal_text = modal.text
                    break
                    
            if not modal_text:
                return None
                
            # 3. Metni Satır Satır Analiz Et (Dinamik Regex)
            lines = modal_text.split('\n')
            yeni_tablo = {
                "sutunlar": ["Beden", "Göğüs (cm)", "Bel (cm)", "Basen (cm)"],
                "veriler": []
            }
            
            for line in lines:
                araliklar = re.findall(r'(\d+)\s*-\s*(\d+)', line)
                if len(araliklar) >= 3: 
                    kelimeler = line.split()
                    beden_isim = kelimeler[0]
                    if len(kelimeler) > 1 and kelimeler[1].isdigit():
                        beden_isim += f"/{kelimeler[1]}"
                        
                    yeni_tablo["veriler"].append({
                        "beden": beden_isim,
                        "gogus": [int(araliklar[0][0]), int(araliklar[0][1])],
                        "bel": [int(araliklar[1][0]), int(araliklar[1][1])],
                        "basen": [int(araliklar[2][0]), int(araliklar[2][1])]
                    })
                    
            if len(yeni_tablo["veriler"]) > 0:
                try:
                    close_btns = self.driver.find_elements(By.CSS_SELECTOR, "[class*='close']")
                    for cb in close_btns:
                        if cb.is_displayed():
                            self.driver.execute_script("arguments[0].click();", cb)
                            break
                except: pass
                return yeni_tablo
                
            return None
        except Exception as e:
            print(f"[ERROR] Sayfadan tablo öğrenme hatası: {e}")
            return None

    def extract_beden_tablosu(self, tam_metin, url):
        """JSON'da varsa getirir, yoksa sayfadan öğrenip JSON'a kaydeder."""
        marka, kategori, demografi = self._kategori_ve_marka_belirle(url, tam_metin)
        
        try:
            with open("beden_tablolari.json", "r", encoding="utf-8") as f:
                tablolar = json.load(f)
        except Exception:
            tablolar = {}
        
        mevcut_tablo = tablolar.get(marka, {}).get(demografi, {}).get(kategori, {}).get("tablo")
        if mevcut_tablo:
            return mevcut_tablo
            
        print(f"[WARN] {marka} - {demografi} - {kategori} için beden tablosu JSON'da yok. Sayfadan öğreniliyor...")
        yeni_tablo = self._sayfadan_beden_tablosu_ogren()
        
        if yeni_tablo:
            if marka not in tablolar:
                tablolar[marka] = {}
            if demografi not in tablolar[marka]:
                tablolar[marka][demografi] = {}
            if kategori not in tablolar[marka][demografi]:
                tablolar[marka][demografi][kategori] = {}
            
            tablolar[marka][demografi][kategori]["tablo"] = yeni_tablo
            
            try:
                with open("beden_tablolari.json", "w", encoding="utf-8") as f:
                    json.dump(tablolar, f, ensure_ascii=False, indent=4)
                print(f"[OK] Yeni beden tablosu başarıyla öğrenildi ve JSON'a kaydedildi! ({marka}/{demografi}/{kategori})")
            except Exception as e:
                print(f"[ERROR] JSON'a yazma hatası: {e}")
                
            return yeni_tablo
            
        return None

    def scrape(self, url):
        print(f"\n[WEB] [Scraper] Ürün sayfasına gidiliyor: {url}")
        try:
            self.driver.get(url)
            time.sleep(3)  # Sayfanın oturmasını bekle

            olcumler = {}
            urun_gorselleri = []
            tam_metin = ""

            # ---------------------------------------------------------
            # 🎯 0. AŞAMA: DOM Üzerindeki Gerçek Carousel (Varyant Doğruluğu İçin)
            # ---------------------------------------------------------
            try:
                thumbs = self.driver.find_elements(By.CSS_SELECTOR, "img[class*='carouselThumbsImage'], .gallery-item img, img[itemprop='image']")
                for img in thumbs:
                    src = img.get_attribute("src") or img.get_attribute("data-src")
                    if src and str(src).startswith("http"):
                        if "mnresize" in src:
                            high_res = re.sub(r'mnresize/\d+/\d+', 'mnresize/1200/1800', src)
                        elif "hepsiburada.net" in src and "/s/" in src: # Hepsiburada yüksek çözünürlük
                            # Örn: .../s/123/224/... -> .../s/123/1500/...
                            high_res = re.sub(r'/s/\d+/\d+/', lambda m: m.group(0).replace(m.group(0).split('/')[3], '1500'), src)
                        else:
                            high_res = src
                        if high_res not in urun_gorselleri:
                            urun_gorselleri.append(high_res)
                if urun_gorselleri:
                    print(f"[OK] Görseller doğrudan DOM Carousel'den çekildi: {len(urun_gorselleri)} adet")
            except Exception:
                pass

            # ---------------------------------------------------------
            # 🎯 1. AŞAMA: JavaScript Belleğine Sızıyoruz (Açıklamalar & Yedek Resimler)
            # ---------------------------------------------------------
            try:
                # Sayfa yüklendikten sonra doğrudan Trendyol'un iç veri tabanına bağlan
                state = self.driver.execute_script("return window.__PRODUCT_DETAIL_APP_INITIAL_STATE__ || window.__INITIAL_STATE__;")
                product = None
                if state:
                    if 'product' in state:
                        product = state['product']
                    elif 'productDetail' in state and 'product' in state['productDetail']:
                        product = state['productDetail']['product']

                if product:
                    if not urun_gorselleri:
                        for img in product.get('images', []):
                            if not str(img).startswith('http'):
                                urun_gorselleri.append(f"https://cdn.dsmcdn.com/mnresize/1200/1800{img}")
                            else:
                                urun_gorselleri.append(img)

                    # Sadece satıcının yazdığı açıklamayı al (Müşteri yorumları karışmaz)
                    text_parts = [product.get('description', '')]
                    for cd in product.get('contentDescriptions', []):
                        if isinstance(cd, dict):
                            text_parts.append(cd.get('description', ''))
                        elif isinstance(cd, str):
                            text_parts.append(cd)

                    tam_metin = " ".join(text_parts)
                    print("[OK] Veriler JavaScript belleğinden tertemiz çekildi!")
            except Exception as e:
                pass

            # 3. YENİ MOD: SEO (ld+json) verisinden %100 Doğru Görselleri Çekme
            if not urun_gorselleri:
                try:
                    ld_scripts = self.driver.find_elements(By.CSS_SELECTOR, "script[type='application/ld+json']")
                    for s in ld_scripts:
                        text = s.get_attribute("innerHTML")
                        if text and "image" in text and "Product" in text:
                            data = json.loads(text)
                            if isinstance(data, dict) and data.get("@type") == "Product":
                                images = data.get("image", [])
                                raw_urls = []
                                if isinstance(images, str):
                                    raw_urls.append(images)
                                elif isinstance(images, dict):
                                    raw_urls.append(images.get("url", ""))
                                elif isinstance(images, list):
                                    for img in images:
                                        if isinstance(img, str): raw_urls.append(img)
                                        elif isinstance(img, dict): raw_urls.append(img.get("url", ""))
                                        
                                for img_url in raw_urls:
                                    if not img_url: continue
                                    if "mnresize" in img_url:
                                        high_res = re.sub(r'mnresize/\d+/\d+', 'mnresize/1200/1800', img_url)
                                    else:
                                        high_res = img_url
                                    if high_res not in urun_gorselleri:
                                        urun_gorselleri.append(high_res)
                                if urun_gorselleri:
                                    break
                except Exception:
                    pass

            # ---------------------------------------------------------
            # 🎯 4. EVRENSEL FALLBACK: Hiçbir şey bulunamazsa Ekrandaki Büyük Resimleri Tara
            # ---------------------------------------------------------
            if not urun_gorselleri:
                try:
                    script = """
                    var imgs = document.getElementsByTagName('img');
                    var valid = [];
                    for (var i=0; i<imgs.length; i++) {
                        var img = imgs[i];
                        if (img.src && img.src.startsWith('http') && !img.src.includes('.svg') && !img.src.includes('.gif')) {
                            // Eni VEYA boyu 400 pikselden büyük olanlar ana ürün resmidir (ikonları eler)
                            // Ayrıce benzer ürünleri elemeyi garantilemek için Y-ekseninde en üstte (ilk 1500px) olanlara bak
                            var absoluteY = window.scrollY + img.getBoundingClientRect().top;
                            if ((img.naturalWidth > 400 || img.naturalHeight > 400 || img.width > 400 || img.height > 400) && absoluteY < 1500) {
                                valid.push(img.src);
                            }
                        }
                    }
                    return Array.from(new Set(valid));
                    """
                    b_imgs = self.driver.execute_script(script)
                    if b_imgs:
                        # Hepsiburada gibi sitelerde resimleri HQ yap
                        for src in b_imgs:
                            if "hepsiburada.net" in src and "/s/" in src:
                                src = re.sub(r'/s/\d+/\d+/', lambda m: m.group(0).replace(m.group(0).split('/')[3], '1500'), src)
                            if src not in urun_gorselleri:
                                urun_gorselleri.append(src)
                        print(f"[OK] Fallback (Evrensel): Sayfadan {len(urun_gorselleri)} adet yüksek çözünürlüklü resim çekildi.")
                except Exception:
                    pass            # Sayfayı aşağı kaydırarak (Lazy Load) gizli içeriklerin yüklenmesini tetikle
            try:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                time.sleep(1.5)
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)
                
                # Tüm butonlara tıklamayı dene ("Daha Fazla Göster" vb.)
                buttons = self.driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    if btn.is_displayed() and ("daha" in btn.text.lower() or "tüm" in btn.text.lower() or "özellik" in btn.text.lower()):
                        try:
                            self.driver.execute_script("arguments[0].click();", btn)
                            time.sleep(1)
                        except:
                            pass
            except Exception:
                pass
                
            # JS State ve SEO'dan çekilen metinlere ek olarak HER ZAMAN DOM Fallback (Agresif Mod) kullan
            try:
                body_text = self.driver.find_element(By.TAG_NAME, "body").get_attribute("textContent")
                page_src = self.driver.page_source
                clean_html = re.sub(r'<[^>]+>', ' ', page_src)
                tam_metin = str(tam_metin) + " \n " + str(body_text) + " \n " + clean_html
            except Exception:
                pass

            if not urun_gorselleri:
                # Sayfadaki tüm resimleri tara ve sadece ürün/manken resimlerini süz
                # Benzer ürün resimlerini engellemek için sadece ürün galerisine öncelik ver
                tum_resimler = self.driver.find_elements(By.CSS_SELECTOR, ".gallery-container img, .product-image-container img, .base-product-image img, .detail-main-image-container img")
                if not tum_resimler:
                    tum_resimler = self.driver.find_elements(By.TAG_NAME, "img")
                    
                for img in tum_resimler:
                    src = img.get_attribute("src") or img.get_attribute("data-src")
                    if src and "cdn.dsmcdn.com" in src and "/ty" in src:
                        # Kargo, kampanya, logo, review resimlerini dışla
                        if any(x in src.lower() for x in ["stamp", "icon", "logo", "avatar", "badge", "rating"]):
                            continue
                        
                        # Resmi yüksek çözünürlüğe (1200x1800) zorla
                        if "mnresize" in src:
                            high_res = re.sub(r'mnresize/\d+/\d+', 'mnresize/1200/1800', src)
                        else:
                            high_res = src
                            
                        if high_res not in urun_gorselleri:
                            urun_gorselleri.append(high_res)
                            
                # Çok fazla alakasız resim geldiyse sadece ürün galerisini (ilk 6 resim) tut
                urun_gorselleri = urun_gorselleri[:6]

            # Süzgeci çalıştır
            olcumler = self.extract_measurements(tam_metin)
            beden_tablosu = self.extract_beden_tablosu(tam_metin, url)
            _, _, demografi = self._kategori_ve_marka_belirle(url, tam_metin)

            return olcumler, urun_gorselleri, beden_tablosu, demografi

        except Exception as e:
            print(f"[ERROR] [Scraper Hatası]: {e}")
            return {}, [], None, "kadin"

    def gorsel_indir(self, img_url):
        print(f"[DL] [Image Engine] Görsel indiriliyor: {img_url}")
        
        # 1. Aşama: Selenium üzerinden indirmeyi dene (Eğer Python DNS hatası veriyorsa Chrome üzerinden dolaylı indirme yapar)
        try:
            if hasattr(self, 'driver'):
                self.driver.set_script_timeout(15)
                script = """
                var done = arguments[0];
                fetch('%s')
                    .then(response => response.blob())
                    .then(blob => {
                        var reader = new FileReader();
                        reader.onload = function() { done(reader.result); };
                        reader.readAsDataURL(blob);
                    })
                    .catch(err => done('ERROR: ' + err.message));
                """ % img_url
                b64_str = self.driver.execute_async_script(script)
                if b64_str and not str(b64_str).startswith('ERROR'):
                    b64_data = b64_str.split(',')[1]
                    import base64
                    img_arr = np.frombuffer(base64.b64decode(b64_data), np.uint8)
                    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                else:
                    print(f"[WARN] Selenium indirmesi başarısız oldu: {b64_str}")
        except Exception as e:
            print(f"[WARN] Selenium ile resim çekilemedi: {e}")

        # 2. Aşama: Fallback olarak requests kullan
        print("[WARN] Chrome ile indirme başarısız. Klasik yöntem deneniyor...")
        for attempt in range(3):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                res = requests.get(img_url, headers=headers, timeout=10)
                if res.status_code == 200:
                    img_arr = np.asarray(bytearray(res.content), dtype=np.uint8)
                    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"[WARN] (Deneme {attempt+1}) requests hatası: {e}")
                time.sleep(1)
        
        # 3. Aşama: urllib
        print("[WARN] requests ile indirme başarısız oldu. urllib üzerinden deneniyor...")
        try:
            import urllib.request
            req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                img_arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
                return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[ERROR] urllib ile de indirilemedi: {e}")
        return None


# ============================================================================
# 2. BÖLÜM: MEDIAPIPE KALİBRATÖRÜ (KADEME 1: GERÇEK VERİ, KADEME 2: OMUZ/GÖZ)
# ============================================================================

