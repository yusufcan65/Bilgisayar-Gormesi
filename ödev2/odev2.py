import cv2
import numpy as np
import requests
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')

# URL'den fotografları indirip inceleyen fonksiyon
def analyze_profile_image(profile_url):
    try:
        # URL'den profil resmini çek
        response = requests.get(profile_url, stream=True)

        if response.status_code == 200:
            img_data = response.content
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                return "Resim yüklenemedi"

            # YOLO ile insan olup olmadığını kontrol et
            results = model(img)
            confidence_threshold = 0.5  # Minimum güven eşiği

            # İnsan tespit etme
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])  # Sınıf kimliği
                    confidence = box.conf[0]  # Güven oranı
                    if model.names[class_id] == 'person' and confidence > confidence_threshold:
                        return f"İNSAN (Güven: {confidence:.2f})"
            return "BASKA BİR VARLIK"
        else:
            return "Resim indirilemedi"
    except Exception as e:
        return f"Hata: {str(e)}"

# Fotograf URL'leri
photo_urls = [
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/27/671dd30c95bf3856.jpg",
    "https://icdn.ensonhaber.com/crop/1200x0/resimler/diger/kok/2024/10/27/671e015e8291d232__w1200xh800.jpg",
    "https://icdn.ensonhaber.com/crop/1200x0/resimler/diger/kok/2024/10/27/671e015cbb9b5228__w1200xh800.jpg",
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/27/671dfe490f6a5916.jpg",
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/25/671b86fb53934615.jpg",
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/27/671d702c3a8d1985.jpg",
    "https://icdn.ensonhaber.com/crop/1200x0/resimler/diger/kok/2024/10/27/671d703402fd2737__w1200xh881.jpg",
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/27/671e0467998ad435.jpg",
    "https://icdn.ensonhaber.com/crop/1200x0/resimler/diger/kok/2024/10/27/671e0492227c1415__w1200xh900.jpg",
    "https://icdn.ensonhaber.com/crop/1200x0/resimler/diger/kok/2024/10/27/671e0495b7042387__w1200xh800.jpg",
]

# Her bir fotograf için analiz yap
for url in photo_urls:
    result = analyze_profile_image(url)
    print(f"{url}: {result}")
