from keras.models import load_model
import cv2
import numpy as np

def predicting_realtime_data():
    model = load_model(r'synthetic_pmate.h5')
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame, (384, 0), (610, 250), (255, 0, 0), 2)
        roi_hand = gray[0:250, 384:610]
        out = cv2.resize(roi_hand, (28, 28))
        out = out.reshape(1, 28, 28, 1)
        y_pred = np.argmax(model.predict(out))
        if y_pred.round() == 0:
            print("Paper")
        elif y_pred.round() == 1:
            print("Rock")
        elif y_pred.round() == 2:
            print("Scissor")
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

predicting_realtime_data()