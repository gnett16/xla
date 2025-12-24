from ultralytics import YOLO
import os

if __name__ == '__main__':
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    yaml_path = os.path.join(current_folder, 'dataset', 'data.yaml')

    yaml_path = yaml_path.replace(os.sep, '/')

    print(f"--- Đang sử dụng file cấu hình tại: {yaml_path} ---")
    if os.path.exists(yaml_path):
        model = YOLO('yolov8n.pt')
        model.train(data=yaml_path, epochs=100, imgsz=1024, workers=0)
    else:
        print("Không thấy file data.yaml")
