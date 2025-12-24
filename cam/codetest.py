import cv2
import numpy as np
import matplotlib.pyplot as plt
PIXELS_PER_MM = 24.0    
SPEC_MIN_MM = 0.25
SPEC_MAX_MM = 0.6
MAX_SOLDER_AREA = 500
MAX_ASPECT_RATIO = 2.5    
img = cv2.imread("1.tiff")
assert img is not None, "loi anh"
h, w = img.shape[:2]
scale = 900 / max(h, w)
img = cv2.resize(img, (int(w * scale), int(h * scale)))
orig = img.copy()
output = img.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_hsv = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 80, 255]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_th)
_, mask_th = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
mask = cv2.bitwise_and(mask_hsv, mask_th)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
error_crops = [] 
count_ok = 0
count_error = 0
for c in contours:
    area = cv2.contourArea(c)
    if area < 20 or area > MAX_SOLDER_AREA: continue
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), angle = rect
    if rw == 0 or rh == 0: continue
    ratio = max(rw, rh) / min(rw, rh)
    if ratio > MAX_ASPECT_RATIO: continue
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    if circularity < 0.3: continue
    x, y, w_b, h_b = cv2.boundingRect(c)
    roi_dist = dist_map[y:y+h_b, x:x+w_b]
    if roi_dist.size == 0: continue
    width_px = roi_dist.max() * 2.0
    width_mm = width_px / PIXELS_PER_MM
    is_error = False
    if width_mm < SPEC_MIN_MM:
        color, label, status = (0, 0, 255), f"{width_mm:.2f}", "THIEU"
        is_error = True
    elif width_mm > SPEC_MAX_MM:
        color, label, status = (255, 0, 0), f"{width_mm:.2f}", "DU"
        is_error = True
    else:
        color, label, status = (0, 255, 0), "", "OK"
    if is_error:
        count_error += 1
        pad = 10
        y1, y2 = max(0, y-pad), min(img.shape[0], y+h_b+pad)
        x1, x2 = max(0, x-pad), min(img.shape[1], x+w_b+pad)
        crop = orig[y1:y2, x1:x2].copy()
        error_crops.append((crop, status, width_mm))
    else:
        count_ok += 1
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(output, [box], 0, color, 1) 
    if label != "":
        cv2.putText(output, label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
print(f"Thong ke: OK: {count_ok} | LOI: {count_error}")
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"AOI Result - Found {count_ok + count_error} joints")
plt.axis("off")
plt.show()
if error_crops:
    num_errors = len(error_crops)
    cols = 5
    rows = (num_errors // cols) + (1 if num_errors % cols != 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    fig.suptitle("DANH SACH MOI HAN LOI (THIEU/DU)", fontsize=16, color='red')
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < num_errors:
            crop, status, val = error_crops[i]
            axes[i].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            color_text = 'red' if status == "THIEU" else 'blue'
            axes[i].set_title(f"{status}: {val:.2f}mm", color=color_text, fontsize=10)
        axes[i].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()