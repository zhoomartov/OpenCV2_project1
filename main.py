import cv2
import datetime
import os
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

video_path = 'video'
os.makedirs(video_path, exist_ok=True)
media_path = 'media'
os.makedirs(media_path, exist_ok=True)

cap = cv2.VideoCapture('airplane_video1.mp4')

video = None

classes = ['airplane', 'car', 'dog', 'cat']

if not cap.isOpened():
    print('Камера не найдено')
    exit()

video_date = datetime.datetime.now().strftime('%D_%M_%Y %H:%M:%S')
type_video = cv2.VideoWriter_fourcc(*'mp4v')
image_type = datetime.datetime.now().strftime('%d_%m_%y %H:%M:%S')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cap.get(cv2.CAP_PROP_FPS))

if frame_fps == 0:
    frame_fps = 30.0


#   0: person
#   1: bicycle
#   2: car
#   3: motorcycle
#   4: airplane
#   5: bus
#   6: train
#   7: truck
#   8: boat
#   9: traffic light
#   10: fire hydrant
#   11: stop sign
#   12: parking meter
#   13: bench
#   14: bird
#   15: cat
#   16: dog
#   17: horse
#   18: sheep
#   19: cow
#   20: elephant
#   21: bear
#   22: zebra
#   23: giraffe
#   24: backpack
#   25: umbrella
#   26: handbag
#   27: tie
#   28: suitcase
#   29: frisbee
#   30: skis
#   31: snowboard
#   32: sports ball
#   33: kite
#   34: baseball bat
#   35: baseball glove
#   36: skateboard
#   37: surfboard
#   38: tennis racket
#   39: bottle
#   40: wine glass
#   41: cup
#   42: fork
#   43: knife
#   44: spoon
#   45: bowl
#   46: banana
#   47: apple
#   48: sandwich
#   49: orange
#   50: broccoli
#   51: carrot
#   52: hot dog
#   53: pizza
#   54: donut
#   55: cake
#   56: chair
#   57: couch
#   58: potted plant
#   59: bed
#   60: dining table
#   61: toilet
#   62: tv
#   63: laptop
#   64: mouse
#   65: remote
#   66: keyboard
#   67: cell phone
#   68: microwave
#   69: oven
#   70: toaster
#   71: sink
#   72: refrigerator
#   73: book
#   74: clock
#   75: vase
#   76: scissors
#   77: teddy bear
#   78: hair drier
#   79: toothbrush

while True:
    fps_start = time.time()

    ret, frame = cap.read()

    ii_8 = 'II8'
    vremya = datetime.datetime.now().strftime('%d-%m-%y %H-%M-%S')

    if not ret:
        print('Frame not found')
        break

    result = model(frame, conf=0.3)
    boxes = result[0].boxes
   

    for i in boxes:
        cls = int(i.cls[0])
        label = model.names[cls]
        if label not in classes:
            continue

        conf = round(float(i.conf[0]) * 100)
        x, y, w, h = map(int, i.xyxy[0])

        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
        cv2.putText(frame, f'{label} {conf}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, ii_8, (300,30),cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)
    cv2.putText(frame, vremya, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)
    fps_end = time.time()
    fps = 1 / (fps_end - fps_start)
    cv2.putText(frame, f'FPS : {round(fps, 1)}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    key = cv2.waitKey(1) & 0xff

    if video is not None:
        video.write(frame)

    if key == ord('q'):
        break

    elif key == ord('v'):
        video_date = datetime.datetime.now().strftime('%d_%m_%Y_%H-%M-%S')
        video_name = f'{video_path}/video_{video_date}.mp4'
        video = cv2.VideoWriter(video_name, type_video, frame_fps,(frame_width, frame_height))

    elif key == ord('s'):
        image_name = f"{media_path}/photo_{image_type}.jpg"
        cv2.imwrite(image_name, frame)

    else:
        cv2.imshow('CAMERA', frame)

cap.release()
cv2.destroyAllWindows()
