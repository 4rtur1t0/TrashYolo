import os
from ultralytics import YOLO
import cv2


if __name__ == '__main__':

    #--------------------------------
    # Select the model
    # Se pueden utilizar redes con más o menos parámetros --> mejores o peores resultados
    # a costa de mayor coste de computación
    #--------------------------------
    # model = YOLO('yolov8n.pt') # tiny
    # model = YOLO('yolov8s.pt') # small
    # model = YOLO('yolov8m.pt') # medium
    # model = YOLO('yolov8l.pt') # large
    model = YOLO('yolov8x.pt') # XL

    #--------------------------------
    # Modelos para otras tareas
    # --------------------------------
    # Detección de objetos (boxes)
    # model = YOLO('yolov8n.pt')
    # Segmentación de instancias (etiquetado de cada píxel)
    # model = YOLO('yolov8n-seg.pt')
    # Pose estimation: Estimación de la pose de una persona
    # model = YOLO('yolov8n-pose.pt')
    # image classification: clasificación de
    # model = YOLO('yolov8n-cls.pt')

    display_size = (1200, 1200)
    # Inicializar webcam (0 = cámara por defecto)
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/dev/video4")
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Ejecutar YOLO sobre el frame (BGR de OpenCV es aceptado)
        results = model(frame, conf=0.3, verbose=False)
        # Dibujar las detecciones en la imagen
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, display_size)
        # Mostrar resultado
        cv2.imshow('YOLO + OpenCV', annotated_frame)

        # Salir     con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



