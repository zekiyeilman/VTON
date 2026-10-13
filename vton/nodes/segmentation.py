import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import rembg
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image

class SegFormerNode:
    def __init__(self, model_id="mattmdjaga/segformer_b2_clothes"):
        print(f"[Node-01] SegFormer Modeli Yükleniyor... ({model_id})")
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[Node-01] Model Yüklendi! Cihaz: {self.device}")

        # SegFormer Class ID'leri:
        # 0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes, 5: Skirt, 
        # 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe, 10: Right-shoe, 11: Face, 
        # 12: Left-leg, 13: Right-leg, 14: Left-arm, 15: Right-arm, 16: Bag, 17: Scarf
        self.garment_class_map = {
            "Üst Giyim": [4],
            "Alt Giyim": [5, 6],  # Etek veya Pantolon
            "Elbise": [7]
        }
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.rembg_session = None

    def process_image(self, image_path, garment_type="Üst Giyim"):
        """
        Görüntüyü alır, SegFormer'dan geçirir ve SADECE seçili kıyafet türünü (Maskesini) döndürür.
        Böylece tişört istiyorsak pantolonu/kolları asla kesmez.
        """
        print(f"[Node-01] SegFormer Görüntü Analizi Başladı: {image_path} (Hedef: {garment_type})")
        
        image = Image.open(image_path).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Mankensiz (Flat-Lay) mi yoksa insan mı kontrolü
        pose_results = self.pose.process(np.array(image))
        is_person = pose_results.pose_landmarks is not None
        
        if not is_person:
            print(f"[Node-01] Manken bulunamadı (Flat-Lay). Rembg u2net ile tüm obje ayıklanıyor...")
            if self.rembg_session is None:
                self.rembg_session = rembg.new_session("u2net")
            rgba_image = rembg.remove(image_cv, session=self.rembg_session)
            
            # Sadece opak piksellerin bounding box'ını al
            y_indices, x_indices = np.where(rgba_image[:, :, 3] > 0)
            if len(y_indices) > 0:
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                cropped_garment = rgba_image[y_min:y_max+1, x_min:x_max+1]
                print("[Node-01] Flat-Lay ayıklama başarılı.")
                return cropped_garment, (rgba_image[:, :, 3] > 0).astype(np.uint8) * 255
            return None, None

        print("[Node-01] Manken algılandı. SegFormer ile anatomik segmentasyon yapılıyor...")
        target_classes = self.garment_class_map.get(garment_type, [4])

        # 1. Modeli Çalıştır
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 2. Orijinal boyuta geri ölçekle (Upsampling)
        logits = outputs.logits.cpu()
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        # 3. Piksel bazlı sınıflandırma (0-17 arası değerler)
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
        
        # 4. Sadece hedef sınıfların maskesini oluştur
        mask = np.zeros_like(pred_seg, dtype=np.uint8)
        for class_id in target_classes:
            mask[pred_seg == class_id] = 255
            
        # Eğer hedef maske boşsa (Manken yoksa veya algılanamadıysa) hata dön
        if not np.any(mask):
            print(f"[Node-01] UYARI: {garment_type} bulunamadı! SegFormer başarısız.")
            return None, None
            
        # 5. Orijinal Görüntü (CV2 BGR) üzerinden maskeyi uygula
        rgba_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2BGRA)
        rgba_image[:, :, 3] = mask # Sadece kıyafet opak kalır, gerisi şeffaf
        
        # Sadece kıyafetin olduğu kutuyu (Bounding Box) kırp
        y_indices, x_indices = np.where(mask > 0)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        cropped_garment = rgba_image[y_min:y_max+1, x_min:x_max+1]
        
        print(f"[Node-01] Başarılı! {garment_type} segmentasyonu tamamlandı.")
        return cropped_garment, mask

if __name__ == "__main__":
    # Test bloğu  ->  python -m vton.nodes.segmentation
    from vton.paths import goruntu
    node = SegFormerNode()
    # Örnek test (Eğer dosya varsa)
    if os.path.exists(goruntu("temp_garment.jpg")):
        result, _ = node.process_image(goruntu("temp_garment.jpg"), "Üst Giyim")
        if result is not None:
            cv2.imwrite(goruntu("test_segformer_output.png"), result)
            print("[TEST] Çıktı goruntu/test_segformer_output.png olarak kaydedildi.")
