import cv2
import numpy as np

def capture_training_data():
    cap = cv2.VideoCapture(0)
    filenumber = 0
    while True:
        _, frame = cap.read()
        kernel = np.ones((3, 3), np.uint8)

        lower_skin = np.array([0, 20, 70], dtype = np.uint8)
        upper_skin = np.array([20, 255, 255], dtype = np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask, kernel, iterations = 4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        cv2.rectangle(frame, (384,0), (610,250), (255, 0, 0), 2)
        roi_hand = mask[0:250, 384:610]
        out = cv2.resize(roi_hand, (28, 28))

        cv2.imwrite(r'images\%s.jpg' %filenumber, out)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        filenumber+=1
        k = cv2.waitKey(30) & 0xFF
        if k == 27 or filenumber == 1100:
            break
    cap.release()
    cv2.destroyAllWindows()