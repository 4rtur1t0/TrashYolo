# This is a sample Python script.
# pip install ultralytics
# !git clone https://github.com/alsombra/Mask_RCNN-TF2
import cv2
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # tiny model
    # model.train(data='trash.yaml', epochs=50, imgsz=640)
    # model = YOLO('runs/detect/train/weights/best.pt')
    cv2.waitKey(delay=5)
    # image_path= '../TACO/data/batch_1/000000.jpg'
    # image_path = '../TACO/data/people/dog.jpg'
    image_path = '../TACO/data/people/tennis.jpg'
    results = model.predict(source=image_path, conf=0.5, show=True, save=True)
    print(results)
    for result in results:
        result.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
