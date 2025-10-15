# This is a sample Python script.

# pip install ultralytics
# !git clone https://github.com/alsombra/Mask_RCNN-TF2
from ultralytics import YOLO

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # smallest model (fast)
    model.train(data='trash_taco.yaml', epochs=50, imgsz=640)

