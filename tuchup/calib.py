import cv2
import numpy as np
filename = "1.jpg"
img = cv2.imread(filename)
assert img is not None, "khong doc duoc anh"
h, w = img.shape[:2]
display_scale = 800 / h 
img_resized = cv2.resize(img, None, fx=display_scale, fy=display_scale)
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_resized, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow('Calibration Tool', img_resized)       
        points.append((x, y))
        
        if len(points) == 2:
            dist_display = np.sqrt((points[0][0] - points[1][0])**2 + 
                                   (points[0][1] - points[1][1])**2)
            
            dist_original = dist_display / display_scale
            
            REAL_DIST_MM = 2.54 
            px_per_mm_original = dist_original / REAL_DIST_MM

            print(f"pixel_per_mm = {px_per_mm_original:.4f}")     
            cv2.line(img_resized, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow('Calibration Tool', img_resized)           
            points.clear()
cv2.imshow('Calibration Tool', img_resized)
cv2.setMouseCallback('Calibration Tool', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()