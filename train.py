from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # smallest model (fast)
    # train 5 best
    # model.train(data='datasets/trash_taco.yaml', epochs=50, imgsz=640, freeze=0)
    model.train(data='datasets/trash_taco.yaml', epochs=50, imgsz=640, freeze=10)


