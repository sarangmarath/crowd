import cv2
import os
from ultralytics import YOLO

path = "Crowd.mp4"
print("path", path)
print("exists", os.path.exists(path))
cap = cv2.VideoCapture(path)
print("opened", cap.isOpened())
print("fps", cap.get(cv2.CAP_PROP_FPS), "count", cap.get(cv2.CAP_PROP_FRAME_COUNT))
ok, frame = cap.read()
print("read", ok)
if ok:
    print("frame shape", frame.shape)
    # Save the frame to check
    cv2.imwrite("debug_frame.jpg", frame)
    print("saved debug_frame.jpg")
    model = YOLO("yolov8n.pt")  # Use default model
    res = model(frame, conf=0.1, classes=[0])  # Lower conf, only person
    print("results length", len(res))
    if len(res) > 0:
        boxes = res[0].boxes
        try:
            xyxy = boxes.xyxy.cpu().tolist()
            cls = boxes.cls.cpu().tolist()
            conf = boxes.conf.cpu().tolist()
            print("xyxy", xyxy)
            print("cls", cls)
            print("conf", conf)
        except Exception as e:
            print("error reading boxes", e)
cap.release()
