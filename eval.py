import os
from ultralytics import YOLO
import cv2

weights_path = 'runs/detect/train5/weights/best.pt'

def yolo_orig_on_people():
    """
    EVALUAMOS EL MODELO ORIGINAL DE YOLO SOBRE PERSONAS
    :return:
    """
    image_path = 'datasets/people/'
    model = YOLO('yolov8n.pt')
    delay = 1500
    return image_path, model, delay


def yolo_trash_on_people():
    """
    EVALUAMOS EL MODELO REENTRENADO DE YOLO (con TRASH)  SOBRE PERSONAS
    :return:
    """
    image_path = 'datasets/people/'
    model = YOLO(weights_path)
    delay = 1500
    return image_path, model, delay


def yolo_trash_on_trash_train():
    """
    EVALUAMOS EL MODELO REENTRENADO DE YOLO (con TRASH) SOBRE BASURA (dataset de entrenamiento)
    :return:
    """
    image_path = 'datasets/taco/images/train/'
    model = YOLO(weights_path)
    delay = 1500
    return image_path, model, delay

def yolo_trash_on_trash_val():
    """
    EVALUAMOS EL MODELO REENTRENADO DE YOLO (con TRASH) SOBRE BASURA (dataset de test)
    :return:
    """
    image_path = 'datasets/taco/images/val/'
    model = YOLO(weights_path)
    delay = 1500
    return image_path, model, delay


def yolo_trash_on_trash_video1():
    """
    EVALUAMOS EL MODELO REENTRENADO DE YOLO (con TRASH) EN EL ESPACIO DE TRABAJO
    :return:
    """
    image_path = 'datasets/video/video1/'
    model = YOLO(weights_path)
    delay = 100
    return image_path, model, delay


def yolo_trash_on_trash_video2():
    """
    EVALUAMOS EL MODELO REENTRENADO DE YOLO (con TRASH) EN EL ESPACIO DE TRABAJO
    :return:
    """
    image_path = 'datasets/video/video2/'
    model = YOLO(weights_path)
    delay = 100
    return image_path, model, delay


def inference(image_path, model, delay, conf=0.5, title='Yolo trash detection'):
    # inference on every of the dataset.
    images = [f for f in os.listdir(image_path) if f.endswith(".jpg")]
    images = sorted(images)

    for image in images:
        results = model.predict(source=image_path + image, conf=conf, show=False, save=False)
        # print(results)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, display_size)
        cv2.imshow(title, annotated_frame)
        if cv2.waitKey(delay=delay) & 0xFF == ord('q'):
            break
        # for result in results:
        #     result.show()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    display_size = (1200, 1000)

    image_path, model, delay = yolo_orig_on_people()
    inference(image_path=image_path, model=model, delay=delay, title='ORIG. YOLO ON PEOPLE')

    image_path, model, delay = yolo_trash_on_people()
    inference(image_path=image_path, model=model, delay=delay, title='TRASH YOLO ON PEOPLE')

    image_path, model, delay = yolo_trash_on_trash_train()
    inference(image_path=image_path, model=model, conf=0.4, delay=delay, title='TRASH YOLO ON TRASH TRAIN')

    image_path, model, delay = yolo_trash_on_trash_val()
    inference(image_path=image_path, model=model, conf=0.4, delay=delay, title='TRASH YOLO ON TRASH EVAL')

    image_path, model, delay = yolo_trash_on_trash_video1()
    inference(image_path=image_path, model=model, conf=0.4, delay=delay, title='TRASH YOLO ON VIDEO 1')

    image_path, model, delay = yolo_trash_on_trash_video2()
    inference(image_path=image_path, model=model, conf=0.2, delay=delay, title='TRASH YOLO ON VIDEO 2')








