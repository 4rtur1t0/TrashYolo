import os
from ultralytics import YOLO
import cv2


if __name__ == '__main__':
    # Select the model
    # model = YOLO('yolov8n.pt')
    model = YOLO('runs/detect/train3/weights/best.pt')

    display_size = (1000, 800)
    #########################
    image_path = 'datasets/people/'
    delay=1500

    # tiny model trained with TACO
    # image_path = 'datasets/taco/images/train/'
    # image_path = 'datasets/taco/images/val/'
    # delay=1500
    #########################################
    # VIDEOS EVAL
    # image_path = 'datasets/video/video1/'
    # image_path = 'datasets/video/video2/'
    # delay = 150

    # inference on every of the dataset.
    images = [f for f in os.listdir(image_path) if f.endswith(".jpg")]
    images = sorted(images)

    for image in images:
        results = model.predict(source=image_path+image, conf=0.1, show=False, save=False)
        # print(results)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, display_size)
        cv2.imshow('Yolo trash detection', annotated_frame)
        if cv2.waitKey(delay=delay) & 0xFF==ord('q'):
            break
        # for result in results:
        #     result.show()
    cv2.destroyAllWindows()






