import cv2
import numpy as np

def capture_training_data():
    cap = cv2.VideoCapture(0)
    filenumber = 0
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame, (384,0), (610,250), (255, 0, 0), 2)
        roi_hand = gray[0:250, 384:610]
        out = cv2.resize(roi_hand, (28, 28))
        cv2.imwrite(r'images\%s.jpg' %filenumber, out)
        cv2.imshow('frame', frame)
        filenumber+=1
        k = cv2.waitKey(30) & 0xFF
        if k == 27 or filenumber == 1100:
            break
    cap.release()
    cv2.destroyAllWindows()