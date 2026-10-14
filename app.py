import gradio as gr
import os
import cv2
import json
from vton import VTONEngine, SegFormerNode, vton_otonom_boru_hatti
from vton.paths import goruntu


def _varsa(yol):
    """Dosya varsa yolunu, yoksa None döndürür (Gradio varsayılan görselleri için)."""
    return yol if os.path.exists(yol) else None


engine = VTONEngine()
segformer_node = SegFormerNode()

def linkten_veri_cek(url):
    if not url or "trendyol" not in url:
        return None, "❌ Geçerli bir Trendyol linki girmelisiniz.", "M", gr.update(), []
        
    try:
        sonuc = vton_otonom_boru_hatti(url)
        if sonuc is None:
            return None, "❌ Scraper bir hata ile karşılaştı. Terminal loglarını inceleyin.", "M", gr.update(), []
            
        on_img_path = sonuc.get("on_image_path")
        garment_img = None
        if on_img_path and os.path.exists(on_img_path):
            garment_img = cv2.cvtColor(cv2.imread(on_img_path), cv2.COLOR_BGR2RGB)
            
        gt = sonuc.get("ground_truth") or {}
        bt = sonuc.get("beden_tablosu") or {}
        
        info_md = "### 📊 Linkten Çekilen Manken ve Beden Tablosu Analizi\n"
        if gt:
            info_md += "**Manken Gerçek Verileri (Ground Truth):**\n"
            for k, v in gt.items():
                info_md += f"- **{k.replace('_', ' ').title()}:** {v}\n"
        else:
            info_md += "Manken ölçüleri açıklamalarda bulunamadı.\n"
            
        info_md += "\n**Beden Tablosu:**\n"
        if bt and 'veriler' in bt:
            info_md += f"| Beden | Göğüs | Bel | Basen |\n|---|---|---|---|\n"
            for satir in bt['veriler']:
                info_md += f"| {satir.get('beden')} | {satir.get('gogus',[0,0])[0]}-{satir.get('gogus',[0,0])[1]} | {satir.get('bel',[0,0])[0]}-{satir.get('bel',[0,0])[1]} | {satir.get('basen',[0,0])[0]}-{satir.get('basen',[0,0])[1]} |\n"
                
        # Mankeni otomatik seçmeye çalış
        manken_beden = "M"
        if gt and "manken_bedeni" in gt:
            manken_beden = str(gt["manken_bedeni"]).upper()
            if manken_beden not in ["XS", "S", "M", "L", "XL"]:
                manken_beden = "M"
                
        # Galeride göstermek üzere tüm resimleri topla
        galeri_resimleri = []
        for path_key, label in [("ref_image_path", "Referans (İlk)"), ("on_image_path", "Ön Duruş"), ("arka_image_path", "Arka Duruş"), ("yan_image_path", "Yan Duruş")]:
            img_p = sonuc.get(path_key)
            if img_p and os.path.exists(img_p):
                galeri_resimleri.append((img_p, label))
                
        return garment_img, info_md, manken_beden, gr.update(value=manken_beden), galeri_resimleri
    except Exception as e:
        return None, f"❌ Hata: {str(e)}", "M", gr.update(), []

def islem_baslat(user_front, user_back, user_side, garment_img, height, chest, waist, hips, garment_type, model_worn_size, desired_size, use_ai, colab_url):
    user_measurements = {"height": float(height), "chest": float(chest), "waist": float(waist), "hips": float(hips)}
    
    vton_results = []
    heatmap_results = []
    mesajlar = []
    
    # Eşleşecek açılar: (User Img, Garment Path, Yön İsmi)
    angles = [
        (user_front, goruntu("on.jpg"), "Ön"),
        (user_back, goruntu("arka.jpg"), "Arka"),
        (user_side, goruntu("yan.jpg"), "Yan")
    ]
    
    for u_img, g_path, label in angles:
        try:
            if u_img is None:
                continue
            
            if not os.path.exists(g_path):
                mesajlar.append(f"{label} yönü için kıyafet fotoğrafı bulunamadı. Atlanıyor.")
                continue
                
            # Kullanıcı fotoğrafını kaydet
            u_path = goruntu(f"temp_user_{label}.jpg")
            cv2.imwrite(u_path, cv2.cvtColor(u_img, cv2.COLOR_RGB2BGR))
            
            # Kıyafeti ayıkla
            g_extracted_path = goruntu(f"temp_garment_extracted_{label}.png")
            cropped_garment, mask = segformer_node.process_image(g_path, garment_type=garment_type)
            
            if cropped_garment is None:
                mesajlar.append(f"Hata: {label} yönü için segmentasyon başarısız oldu.")
                continue
                
            cv2.imwrite(g_extracted_path, cropped_garment)
            
            out_name = goruntu(f"temp_result_{label}.jpg")
            
            mesaj = engine.run(
                user_image_path=u_path,
                garment_image_path=g_extracted_path,
                original_on_jpg_path=g_path, 
                output_name=out_name,
                user_measurements=user_measurements,
                desired_size=desired_size,
                model_worn_size=model_worn_size,
                garment_type=garment_type,
                use_ai_render=use_ai,
                colab_url=colab_url
            )
            mesajlar.append(f"[{label} Duruş]:\n" + mesaj)
            
            heatmap_name = out_name.replace(".jpg", "_heatmap.jpg")
            hd_name = out_name.replace(".jpg", "_HD_RENDER.jpg")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"CRITICAL ERROR in loop: {error_trace}")
            mesajlar.append(f"Kritik Hata Oluştu ({label}): {e}")
            continue
        
        if use_ai and os.path.exists(hd_name):
            final_img = cv2.cvtColor(cv2.imread(hd_name), cv2.COLOR_BGR2RGB)
        else:
            final_img = cv2.cvtColor(cv2.imread(out_name), cv2.COLOR_BGR2RGB)
            
        vton_results.append((final_img, label))
        
        if os.path.exists(heatmap_name):
            h_img = cv2.cvtColor(cv2.imread(heatmap_name), cv2.COLOR_BGR2RGB)
            heatmap_results.append((h_img, label))
            
    if not vton_results:
        return None, None, "İşlenecek geçerli bir fotoğraf eşleşmesi bulunamadı. Hem kullanıcı hem kıyafetin aynı yönleri olmalı."
        
    return vton_results, heatmap_results, "\n\n".join(mesajlar)

with gr.Blocks(title="VTON Beden Duyarlı Sanal Kabin") as demo:
    gr.Markdown("# 👕 Otonom Size-Aware VTON Sistemi")
    gr.Markdown("Linkten manken özelliklerini çeker, antropometrik analiz yapar ve kullanıcının vücuduna ısı haritasıyla oturtur.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Ürün Seçimi")
            url_in = gr.Textbox(value="https://www.trendyol-milla.com/trendyolmilla/antrasit-100-pamuk-oversize-genis-kalip-baskili-orme-t-shirt-twoss25ts00148-p-896485014", placeholder="Trendyol Ürün Linki Yapıştırın...", label="Ürün Linki")
            fetch_btn = gr.Button("🔗 Linkten Veri Çek ve Analiz Et", variant="secondary")
            
            scraper_info = gr.Markdown("Manken ve Beden Tablosu Verileri Burada Gösterilecek...")
            garment_gallery = gr.Gallery(label="Kazınan Ürün Görselleri (Referans, Ön, Arka, Yan)", height=250, columns=4)
            garment_in = gr.Image(value=_varsa(goruntu("on.jpg")), label="Giydirilecek Kıyafet (Galeriden seçip veya manuel yükleyebilirsiniz)")
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Kullanıcı Verileri")
            with gr.Row():
                user_front = gr.Image(value=_varsa(goruntu("user_on.jpg")), label="Senin Ön Fotoğrafın")
                user_back = gr.Image(value=_varsa(goruntu("user_arka.jpg")), label="Senin Arka Fotoğrafın")
                user_side = gr.Image(value=_varsa(goruntu("user_yan.jpg")), label="Senin Yan Fotoğrafın")
            
            with gr.Row():
                height_in = gr.Number(value=168.0, label="Boy (cm)")
                chest_in = gr.Number(value=95.0, label="Göğüs Çevresi")
                waist_in = gr.Number(value=70.0, label="Bel Çevresi")
                hips_in = gr.Number(value=98.0, label="Basen Çevresi")
                
            with gr.Row():
                garment_type_in = gr.Dropdown(choices=["Üst Giyim", "Alt Giyim", "Elbise"], value="Üst Giyim", label="Kıyafet Türü")
                model_worn_in = gr.Dropdown(choices=["S", "M", "L", "XL"], value="M", label="Mankenin Giydiği Beden")
                size_in = gr.Dropdown(choices=["S", "M", "L", "XL"], value="M", label="Denemek İstediğin Beden")
                
            use_ai_in = gr.Checkbox(value=True, label="🚀 Google Colab (HD Render) API Kullan")
            colab_url_in = gr.Textbox(placeholder="Google Colab Gradio URL (örn: https://xxxx.gradio.live)", label="Colab API URL", visible=False)
            
            def toggle_colab_url(is_checked):
                return gr.update(visible=is_checked)
                
            use_ai_in.change(fn=toggle_colab_url, inputs=[use_ai_in], outputs=[colab_url_in])
            
            submit_btn = gr.Button("Analiz Et ve Giydir", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 3. Sonuçlar")
            out_img = gr.Gallery(label="Giydirilmiş Sonuçlar (Ön, Arka, Yan)", columns=3, height=300)
            out_heatmap = gr.Gallery(label="Isı Haritaları (Darlık/Bolluk)", columns=3, height=300)
            out_text = gr.Textbox(label="Sistem Mesajı")
            
    # Bağlantılar
    fetch_btn.click(
        fn=linkten_veri_cek,
        inputs=[url_in],
        outputs=[garment_in, scraper_info, model_worn_in, size_in, garment_gallery]
    )
    
    submit_btn.click(
        fn=islem_baslat,
        inputs=[user_front, user_back, user_side, garment_in, height_in, chest_in, waist_in, hips_in, garment_type_in, model_worn_in, size_in, use_ai_in, colab_url_in],
        outputs=[out_img, out_heatmap, out_text]
    )

if __name__ == "__main__":
    share_mode = os.environ.get("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(inbrowser=not share_mode, share=share_mode)
