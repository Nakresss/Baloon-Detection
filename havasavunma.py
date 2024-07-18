import cv2
import numpy as np
import math
import torch

# YOLOv5 modelini yükleme
model = torch.hub.load('ultralytics/yolov5', 'custom', path='YoloV5Nakres/bestnakres.pt', force_reload=True)

# Kamerayı başlatma
cap = cv2.VideoCapture(0)  # 0, ilk kamerayı belirtir, birden fazla kamera varsa numarasını değiştirebilirsiniz
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Ekran çözünürlüğü
resolution_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
resolution_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Aim (hedef işareti) özellikleri
aim_size = 10
aim_color = (0, 255, 0)  # Yeşil renk

# "Ateşe Hazır" yazısı özellikleri
ready_text = "Atese Hazir"
ready_text_color = (0, 255, 255)  # Sarı renk
ready_text_font = cv2.FONT_HERSHEY_SIMPLEX
ready_text_scale = 0.7
ready_text_thickness = 2
ready_text_position = (resolution_width - 300, 50)  # Sağ üst köşe

# Toplam hedef sayısı
total_targets = 0

def draw_aim(frame, center_x, center_y):
    # Tespit edilen nesnenin merkezine aim (hedef işareti) çizme
    cv2.line(frame, (center_x - aim_size, center_y), (center_x + aim_size, center_y), aim_color, 2)
    cv2.line(frame, (center_x, center_y - aim_size), (center_x, center_y + aim_size), aim_color, 2)

def pixels_to_cm_distance(pixels):
    # Piksel başına gerçek dünya mesafesi (varsayılan olarak 0.0264 cm/piksel)
    pixels_to_cm = 0.0264
    return pixels * pixels_to_cm

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare okunamadı!")
        break

    # YOLOv5 modeli ile nesne tespiti
    results = model(frame)  # YOLOv5 modelinden tespit sonuçlarını al

    # Tespit edilen nesneleri işleme
    detections = results.xyxy[0].numpy()
    total_targets = len(detections)  # Toplam tespit edilen hedef sayısını güncelle

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Tespit edilen nesnenin merkezini hesapla
        object_center_x = (x1 + x2) // 2
        object_center_y = (y1 + y2) // 2

        # Merkez ile ekranın ortası arasındaki mesafeyi hesapla (piksel cinsinden)
        distance_pixels = math.sqrt((object_center_x - resolution_width // 2)**2 + (object_center_y - resolution_height // 2)**2)

        # Mesafeyi cm cinsine çevirme
        distance_cm = pixels_to_cm_distance(distance_pixels)

        # Tespit edilen nesnenin merkezi ile aim (hedef işareti) çizme
        draw_aim(frame, object_center_x, object_center_y)

        # Nesne adını (etiketini) dikdörtgenin üstüne yazdırma
        label = f"{model.names[int(cls)]}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tespit edilen nesnenin merkezi ile aim arasına mesafe çizme
        cv2.line(frame, (object_center_x, object_center_y), (resolution_width // 2, resolution_height // 2), (0, 255, 0), 2)

        # Mesafeyi çizgi üzerine yazdırma
        cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (object_center_x, object_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Nesne dikdörtgenini çizme
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # "Ateşe Hazır" yazısını ekrana sadece hedef kutusuna girildiğinde yazdırma
    if any(x1 < resolution_width // 2 < x2 and y1 < resolution_height // 2 < y2 for (x1, y1, x2, y2, _, _) in detections):
        cv2.putText(frame, ready_text, ready_text_position, ready_text_font, ready_text_scale, ready_text_color, ready_text_thickness)
    # Toplam hedef sayısını ekrana her döngüde yazdırma
    cv2.putText(frame, f"Toplam Hedef: {total_targets}", (10, 50), ready_text_font, ready_text_scale, ready_text_color, ready_text_thickness)

    # Ekranda gösterme
    cv2.imshow('YOLOv5 Detection', frame)

    # Çıkış için 'q' tuşuna basma kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
