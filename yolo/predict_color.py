from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
MODEL_PATH = "runs/detect/train/weights/best.pt" 
IMAGE_PATH = "12.jpg" 
model = YOLO(MODEL_PATH)
results = model.predict(IMAGE_PATH, conf=0.1)
img_draw = results[0].orig_img.copy() 
boxes = results[0].boxes

print(f"Tổng số mối hàn phát hiện: {len(boxes)}")
for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    conf = float(box.conf[0].cpu().numpy())
    
    if conf > 0.7:
        
        color = (0, 255, 0)
        label_text = f"GOOD: {conf:.2f}"
    elif 0.5 <= conf <= 0.7:
        
        color = (0, 255, 255)
        label_text = f"CHECK: {conf:.2f}"
    else:
       
        color = (0, 0, 255)
        label_text = f"BAD: {conf:.2f}"

    cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + w, y1), color, -1)
    
    cv2.putText(img_draw, label_text, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
plt.title("Phan loai moi han theo do tin cay")
plt.axis('off')
plt.show()
cv2.imwrite("ket_qua_mau.jpg", img_draw)