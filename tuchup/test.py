import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
PIXELS_PER_MM = 32.75   
SPEC_MIN_MM = 0.20
SPEC_MAX_MM = 0.40
img = cv2.imread("1.jpg")
assert img is not None, "loi anh"
h, w = img.shape[:2]
scale = 900 / max(h, w)
img = cv2.resize(img, (int(w * scale), int(h * scale)))
orig = img.copy()
output = img.copy()
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 160])
upper = np.array([180, 80, 255])
mask_hsv = cv2.inRange(hsv, lower, upper)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel_th = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_th)

_, mask_th = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
mask = cv2.bitwise_and(mask_hsv, mask_th)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
report_list = []
error_crops = []
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area < 20:
        continue
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), angle = rect
    if rw == 0 or rh == 0:
        continue
    ratio = max(rw, rh) / min(rw, rh)
    perimeter = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    if circularity > 0.92 and area > 80:
        continue
    if ratio > 10:
        continue
    x, y, w_b, h_b = cv2.boundingRect(c)
    roi_dist = dist_map[y:y+h_b, x:x+w_b]
    if roi_dist.size == 0:
        continue

    width_px = roi_dist.max() * 2.0
    width_mm = width_px / PIXELS_PER_MM
    status = "OK"
    if width_mm < SPEC_MIN_MM:
        color = (0, 0, 255)    
        label = f"{width_mm:.2f} "
        status = "THIEU"
    elif width_mm > SPEC_MAX_MM:
        color = (255, 0, 0)     
        label = f"{width_mm:.2f} "
        status = "DU"
    else:
        color = (0, 255, 0)     
        label = f"{width_mm:.2f}"
        status = "OK"
    report_list.append({
        "ID": i + 1,
        "Tam X (px)": int(cx),
        "Tam Y (px)": int(cy),
        "Rong (mm)": round(width_mm, 3),
        "Trang thai": status
    })
    if status != "OK":
        pad = 15
        y1, y2 = max(0, y-pad), min(orig.shape[0], y+h_b+pad)
        x1, x2 = max(0, x-pad), min(orig.shape[1], x+w_b+pad)
        error_crops.append((orig[y1:y2, x1:x2].copy(), status, width_mm))

    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(output, [box], 0, color, 2)
    cv2.putText(output, label, (int(cx), int(cy)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
df = pd.DataFrame(report_list)
df.to_excel("baocaoloi.xlsx", index=False)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.subplot(1, 3, 2)
plt.title("Binary Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.title(" Result")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.tight_layout()
plt.show()
if error_crops:
    num_err = len(error_crops)
    cols = 5
    rows = (num_err // cols) + (1 if num_err % cols != 0 else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < num_err:
            crop, stt, val = error_crops[i]
            axes[i].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"{stt}: {val:.2f}")
        axes[i].axis("off")
    plt.suptitle("DANH SACH MOI HAN LOI", fontsize=16)
    plt.tight_layout()
    plt.show()