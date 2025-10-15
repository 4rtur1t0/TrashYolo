import os
from ultralytics import YOLO
import cv2


if __name__ == '__main__':
    # Select the model
    # model = YOLO('yolov8n.pt')
    model = YOLO('runs/detect/train3/weights/best.pt')

    # tiny model trained with TACO
    image_path = 'datasets/video/video2/'
    # inference on every image.
    images = [f for f in os.listdir(image_path) if f.endswith(".jpg")]
    images = sorted(images)
    for image in images:
        results = model.predict(source=image_path+image, conf=0.1, show=False, save=False)
        # print(results)
        annotated_frame = results[0].plot()
        cv2.imshow('Yolo trash detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        # for result in results:
        #     result.show()
    cv2.destroyAllWindows()






