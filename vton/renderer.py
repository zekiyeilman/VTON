import os
import shutil

def render_with_idm_vton(person_img_path, garment_img_path, output_path, mask_img_path=None, garment_desc="A t-shirt", colab_url=""):
    print("\n" + "="*50)
    print("[API] Yüksek Çözünürlüklü (HD) IDM-VTON Render Katmanı Başlatılıyor...")
    print("[API] İşlem sunucu bağlantınıza göre 15-45 saniye sürebilir, lütfen bekleyin...")
    print("="*50)
    
    # SSL Sertifika Hatalarını Atlamak İçin (Güvenlik duvarı/Antivirüs engeline takılmamak adına)
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        import httpx
        original_init = httpx.Client.__init__
        def patched_init(self, *args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = httpx.Timeout(300.0) # 5 dakika bekleme süresi
            original_init(self, *args, **kwargs)
        httpx.Client.__init__ = patched_init
    except:
        pass
    
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("[ERROR] gradio_client kütüphanesi bulunamadı! Lütfen 'pip install gradio_client' çalıştırın.")
        return False
        
    try:
        # Google Colab veya yisol/IDM-VTON public yapay zeka sunucusuna bağlan
        if colab_url and colab_url.strip() != "":
            print(f"[API] Google Colab URL'sine bağlanılıyor: {colab_url}")
            client = Client(colab_url.strip())
        else:
            print("[API] Colab URL girilmediği için varsayılan yisol/IDM-VTON bağlanılıyor...")
            hf_token = os.environ.get("HF_TOKEN")
            client = Client("yisol/IDM-VTON", token=hf_token)
        
        # Eğer özel maske yolladıysak, ImageEditor'ün "layers" kısmına koyuyoruz.
        # Bu sayede IDM-VTON kendi maskesini üretip tişörtü crop top yapmayacak! Bizim OpenCV oranlarımızı kullanacak.
        layers = [handle_file(mask_img_path)] if mask_img_path else []
        is_checked = False if mask_img_path else True # Özel maske varsa, otomatik maskelemeyi kapat
        
        result = client.predict(
            dict({"background": handle_file(person_img_path), "layers": layers, "composite": None}),
            handle_file(garment_img_path),
            garment_desc,
            is_checked, # is_checked (Otomatik maskeleme - maskemiz varsa False)
            False,      # is_checked_crop (Kıyafeti otomatik kırpma - kesinlikle False)
            30,         # Denoise steps
            42,         # Seed
            api_name="/tryon"
        )
        
        # Sonuç [image_url, mask_url] olarak döner
        if isinstance(result, list) or isinstance(result, tuple):
            result_img_path = result[0]
        else:
            result_img_path = result
            
        # Dönen HD resmi projemizdeki hedef yola kopyala
        shutil.copy(result_img_path, output_path)
        print(f"\n[SUCCESS] Fotogerçekçi HD Render Başarıyla Tamamlandı!")
        print(f"-> Mükemmel sonuç buraya kaydedildi: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] IDM-VTON API Hatası: {e}")
        print("[WARN] Sunucu yoğun olabilir veya kısıtlamaya takılmış olabilirsiniz. Lütfen 1-2 dakika sonra tekrar deneyin.")
        return False

if __name__ == "__main__":
    # Test Senaryosu  ->  python -m vton.renderer
    from vton.paths import goruntu
    print("Test için goruntu/user_on.jpg ve goruntu/ayiklanmis_on.png aranıyor...")
    if os.path.exists(goruntu("user_on.jpg")) and os.path.exists(goruntu("ayiklanmis_on.png")):
        render_with_idm_vton(goruntu("user_on.jpg"), goruntu("ayiklanmis_on.png"), goruntu("vton_hd_sonuc_on.jpg"))
    else:
        print("Test görselleri bulunamadı.")
