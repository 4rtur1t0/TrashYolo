# This is a sample Python script.
# pip install ultralytics
# !git clone https://github.com/alsombra/Mask_RCNN-TF2
# import cv2
from ultralytics import YOLO


if __name__ == '__main__':
    # Select
    # tiny default model and some test images
    # image_path = '../TACO/data/people/dog.jpg'
    # image_path = '../TACO/data/people/tennis.jpg'
    # model = YOLO('yolov8n.pt')

    # tiny model trained with TACO
    image_path = 'datasets/taco/images/val/13_000039.jpg'
    model = YOLO('runs/detect/train3/weights/last.pt')

    results = model.predict(source=image_path, conf=0.5, show=True, save=False)
    print(results)
    for result in results:
        result.show()




