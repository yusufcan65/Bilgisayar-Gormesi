from ultralytics import YOLO
import cv2

# Eğitilmiş modeli yükle
model = YOLO('runs/detect/train3/weights/best.pt')  # Eğitilen model yolunu belirtin

# Webcam'i başlat
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Hata: Web kamerası açılamadı.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Modeli çalıştır
    results = model(frame)

    # Tahminleri çizin
    annotated_frame = results[0].plot()

    # Çıkışı göster
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Çıkmak için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
