from keras.models import load_model
import cv2
import numpy as np
import random

def predicting_realtime_data():
    model = load_model(r'synthetic_pmate.h5')
    cap = cv2.VideoCapture(0)

    #Player choice text
    font = cv2.FONT_HERSHEY_SIMPLEX
    upperLeftCornerOfText = (10,30)
    fontScale = 1
    fontColor = (255,0,0)
    lineType = 2

    #Pal choice text
    upperLeftCornerOfText_pal = (10,70)
    fontColor_pal = (255,255,0)

    #Result text victory
    fontColor_victory = (0,255,0)

    #Result text loss
    fontColor_loss = (0,0,255)

    #Result draw
    fontColor_draw = (0,255,255)

    #Result position
    upperLeftCornerOfText_result = (10,110)

    count = 0

    pal_choice_prev = 0
    y_pred_prev = 0

    while True:
        _, frame = cap.read()
        kernel = np.ones((3, 3), np.uint8)

        count += 1

        lower_skin = np.array([0, 20, 70], dtype = np.uint8)
        upper_skin = np.array([20, 255, 255], dtype = np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.dilate(mask, kernel, iterations = 4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        cv2.rectangle(frame, (384, 0), (610, 250), (255, 0, 0), 2)
        roi_hand = mask[0:250, 384:610]
        out = cv2.resize(roi_hand, (28, 28))
        out = out.reshape(1, 28, 28, 1)

        pal_choice = random.randint(0, 3)
        y_pred = np.argmax(model.predict(out))

        #Player choice is Paper
        if y_pred.round() == 0 and count >= 100:
            count = 0
            y_pred_prev = y_pred.round()
            pal_choice_prev = pal_choice
            cv2.putText(frame,'You chose Paper', upperLeftCornerOfText, font, fontScale, fontColor, lineType)
            if pal_choice == 1: #Pal choice Rock
                cv2.putText(frame,'Pal choice is Rock', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                cv2.putText(frame,'You Win!', upperLeftCornerOfText_result, font, fontScale, fontColor_victory, lineType)
            elif pal_choice == 2: #Pal choice Scissor
                cv2.putText(frame,'Pal choice is Scissor', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                cv2.putText(frame,'You Loose!', upperLeftCornerOfText_result, font, fontScale, fontColor_loss, lineType)
            else:                 #Pal choice Paper
                cv2.putText(frame,'Pal choice is Paper', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                cv2.putText(frame,'No one wins!', upperLeftCornerOfText_result, font, fontScale, fontColor_draw, lineType)
        #Player choice is Rock
        elif y_pred.round() == 1 and count >= 100:
            count = 0
            y_pred_prev = y_pred.round()
            pal_choice_prev = pal_choice
            cv2.putText(frame,'You chose Rock', upperLeftCornerOfText, font, fontScale, fontColor, lineType)
            if pal_choice == 1: #Pal choice Rock
                cv2.putText(frame,'Pal choice is Rock', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                cv2.putText(frame,'No one wins!', upperLeftCornerOfText_result, font, fontScale, fontColor_draw, lineType)
            elif pal_choice == 2: #Pal choice Scissor
                cv2.putText(frame,'Pal choice is Scissor', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                cv2.putText(frame,'You Win!', upperLeftCornerOfText_result, font, fontScale, fontColor_victory, lineType)
            else:                 #Pal choice Paper
                cv2.putText(frame,'Pal choice is Paper', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                cv2.putText(frame,'You Loose!', upperLeftCornerOfText_result, font, fontScale, fontColor_loss, lineType)
        #Player choice is Scissor
        elif y_pred.round() == 2 and count >= 100:
            count = 0
            y_pred_prev = y_pred.round()
            pal_choice_prev = pal_choice
            cv2.putText(frame,'You chose Scissor', upperLeftCornerOfText, font, fontScale, fontColor, lineType)
            if pal_choice == 1: #Pal choice Rock
                cv2.putText(frame,'Pal choice is Rock', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                cv2.putText(frame,'You Loose!', upperLeftCornerOfText_result, font, fontScale, fontColor_loss, lineType)
            elif pal_choice == 2: #Pal choice Scissor
                cv2.putText(frame,'Pal choice is Scissor', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                cv2.putText(frame,'No one wins!', upperLeftCornerOfText_result, font, fontScale, fontColor_draw, lineType)
            else:                 #Pal choice Paper
                cv2.putText(frame,'Pal choice is Paper', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                cv2.putText(frame,'You Win!', upperLeftCornerOfText_result, font, fontScale, fontColor_victory, lineType)
        #Persistent text
        else:
            if y_pred_prev == 0: #Your choice was Paper
                cv2.putText(frame,'You chose Paper', upperLeftCornerOfText, font, fontScale, fontColor, lineType)
                if pal_choice_prev == 1:
                    cv2.putText(frame,'Pal choice is Rock', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                    cv2.putText(frame,'You Win!', upperLeftCornerOfText_result, font, fontScale, fontColor_victory, lineType)
                elif pal_choice_prev == 2:
                    cv2.putText(frame,'Pal choice is Scissor', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                    cv2.putText(frame,'You Loose!', upperLeftCornerOfText_result, font, fontScale, fontColor_loss, lineType)
                else:
                    cv2.putText(frame,'Pal choice is Paper', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                    cv2.putText(frame,'No one wins!', upperLeftCornerOfText_result, font, fontScale, fontColor_draw, lineType)
            elif y_pred_prev == 1: #Your choice was Rock
                cv2.putText(frame,'You chose Rock', upperLeftCornerOfText, font, fontScale, fontColor, lineType)
                if pal_choice_prev == 1:
                    cv2.putText(frame,'Pal choice is Rock', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                    cv2.putText(frame,'No one wins!', upperLeftCornerOfText_result, font, fontScale, fontColor_draw, lineType)
                elif pal_choice_prev == 2:
                    cv2.putText(frame,'Pal choice is Scissor', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                    cv2.putText(frame,'You Win!', upperLeftCornerOfText_result, font, fontScale, fontColor_victory, lineType)
                else:
                    cv2.putText(frame,'Pal choice is Paper', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                    cv2.putText(frame,'You Loose!', upperLeftCornerOfText_result, font, fontScale, fontColor_loss, lineType)
            elif y_pred_prev == 2: #Your choice was Scissor
                cv2.putText(frame,'You chose Scissor', upperLeftCornerOfText, font, fontScale, fontColor, lineType)
                if pal_choice_prev == 1:
                    cv2.putText(frame,'Pal choice is Rock', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                    cv2.putText(frame,'You Loose!', upperLeftCornerOfText_result, font, fontScale, fontColor_loss, lineType)
                elif pal_choice_prev == 2:
                    cv2.putText(frame,'Pal choice is Scissor', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                    cv2.putText(frame,'No one wins!', upperLeftCornerOfText_result, font, fontScale, fontColor_draw, lineType)
                else:
                    cv2.putText(frame,'Pal choice is Paper', upperLeftCornerOfText_pal, font, fontScale, fontColor_pal, lineType)
                    cv2.putText(frame,'You Win!', upperLeftCornerOfText_result, font, fontScale, fontColor_victory, lineType)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

predicting_realtime_data()
