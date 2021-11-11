import mss
import mss.tools
import cv2 as cv
import time
import pyautogui as pg
import numpy as np

x1, x2, x3, x4 = 0, 0, 0, 0

Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

class_name = []
with open('CSv3.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net = cv.dnn.readNet('CSv3_6000.weights', 'CSv3.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


with mss.mss() as sct: 
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)
    
    starting_time = time.time()
    frame_counter = 0

    while True:

        sct_img = sct.grab(monitor)

        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        print(output)

        cap = cv.imread(output)

        x1, x2, x3, x4 = 0, 0, 0, 0

        frame = cap
        frame_counter += 1
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_name[classid[0]], score)
            cv.rectangle(frame, box, color, 1)

            x1, x2, x3, x4 = box[0], box[2], box[1], box[3]

            print (box[0], box[2], box[1], box[3])


            cv.putText(frame, label, (box[0], box[1]-10),
            cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)


        if x1 > 0:
            pg.moveTo(x1+x2/2, x3+x4/3)
        else:
            pass


        endingTime = time.time() - starting_time
        fps = frame_counter/endingTime

        cv.putText(frame, f'FPS: {fps}', (20, 50),
                   cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow('frame', frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()