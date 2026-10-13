import math

class Ansur3DTo2DConverter:
    """
    ANSUR (Anthropometric Survey of US Army Personnel) tabanlı model.
    3 Boyutlu Çevre (Circumference) ölçülerini, 2 Boyutlu Ön Genişliğe (Front Width) dönüştürür.
    Özellikle VTON (Sanal Giydirme) işlemlerinde, çıkarılan kıyafet maskesinin 
    kullanıcının gerçek vücut genişliğine (2D) tam oturmasını sağlar.
    """
    def __init__(self):
        # Derinlik / Genişlik oranları (Depth-to-Width ratios)
        # ANSUR verilerine göre tipik insan vücudu oranları
        self.ratios = {
            "chest": 0.70,
            "waist": 0.65,
            "hips": 0.75,
            "neck": 0.90, # Boyun neredeyse daireseldir
        }

    def circumference_to_width(self, circumference_cm, body_part="chest"):
        """
        Elips çevre formülü baz alınarak 3D çevreyi 2D genişliğe çevirir.
        Formül: C ≈ pi * sqrt(2 * ((W/2)^2 + (D/2)^2))
        """
        if body_part not in self.ratios:
            k = 0.70 # Varsayılan oran
        else:
            k = self.ratios[body_part]

        # W = C / (pi * sqrt((1 + k^2)/2))
        denominator = math.pi * math.sqrt((1 + k**2) / 2.0)
        width_2d = circumference_cm / denominator
        return width_2d
