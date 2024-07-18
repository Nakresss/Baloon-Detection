import cv2
import numpy as np
import math
import torch
from ultralytics import YOLO

# YOLOv5 modelini yükleme
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/shade/OneDrive/Masaüstü/HSS_main/serkan_yolov5.pt', force_reload=True)

model_path = "C:/Users/shade/OneDrive/Masaüstü/HSS_main/serkan_yolov8.pt"
model = YOLO(model_path)

# Start the camera
cap = cv2.VideoCapture(0)  # 0 indicates the first camera
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# Screen resolution
resolution_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
resolution_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Aim properties
aim_size = 10
aim_color = (0, 255, 0)  # Green color

# "Ateşe Hazır" text properties
ready_text = "Atese Hazir"
ready_text_color = (0, 255, 255)  # Yellow color
ready_text_font = cv2.FONT_HERSHEY_SIMPLEX
ready_text_scale = 0.7
ready_text_thickness = 2
ready_text_position = (resolution_width - 300, 50)  # Top-right corner

# Total target count
total_targets = 0

def draw_aim(frame, center_x, center_y):
    # Draw aim (target sign) at the center of the detected object
    cv2.line(frame, (center_x - aim_size, center_y), (center_x + aim_size, center_y), aim_color, 2)
    cv2.line(frame, (center_x, center_y - aim_size), (center_x, center_y + aim_size), aim_color, 2)

def pixels_to_cm_distance(pixels):
    # Real-world distance per pixel (default is 0.0264 cm/pixel)
    pixels_to_cm = 0.0264
    return pixels * pixels_to_cm

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare okunamadı!")
        break

    # Object detection using YOLOv8 model
    results = model(frame)

    # Process detection results
    detections = results[0].boxes.data.cpu().numpy()  # Get detection results as numpy array
    total_targets = len(detections)  # Update total detected target count

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Calculate the center of the detected object
        object_center_x = (x1 + x2) // 2
        object_center_y = (y1 + y2) // 2

        # Calculate the distance between the center and the screen center (in pixels)
        distance_pixels = math.sqrt((object_center_x - resolution_width // 2) ** 2 + (object_center_y - resolution_height // 2) ** 2)

        # Convert the distance to cm
        distance_cm = pixels_to_cm_distance(distance_pixels)

        # Draw aim (target sign) at the center of the detected object
        draw_aim(frame, object_center_x, object_center_y)

        # Write the object name (label) on top of the rectangle
        label = f"{model.names[int(cls)]}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw a line between the center of the detected object and the aim
        cv2.line(frame, (object_center_x, object_center_y), (resolution_width // 2, resolution_height // 2), (0, 255, 0), 2)

        # Write the distance on the line
        cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (object_center_x, object_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the rectangle around the detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the "Ateşe Hazır" text only when the target is in the center
    if any(x1 < resolution_width // 2 < x2 and y1 < resolution_height // 2 < y2 for (x1, y1, x2, y2, _, _) in detections):
        cv2.putText(frame, ready_text, ready_text_position, ready_text_font, ready_text_scale, ready_text_color, ready_text_thickness)

    # Display the total target count on the screen in each loop
    cv2.putText(frame, f"Toplam Hedef: {total_targets}", (10, 50), ready_text_font, ready_text_scale, ready_text_color, ready_text_thickness)

    # Display the frame
    cv2.imshow('YOLOv8 Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()