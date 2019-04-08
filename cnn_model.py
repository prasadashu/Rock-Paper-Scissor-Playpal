import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import cv2

dataset_df = pd.read_csv('created_dataset\dataset_df.csv')

labels = dataset_df['label'].values
unique_val = np.array(labels)

dataset_df.drop('label', axis = 1, inplace = True)

images = dataset_df.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)

# Hyperparameters
batch_size = 15
num_classes = 3
epochs = 50

# Normalizing the pixel values
X_train = X_train / 255
X_test = X_test / 255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# THE MODEL
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=epochs, batch_size=batch_size)
print("Model trained\n")

#model.save('synthetic_pmate.h5')
#print("Model saved\n")

def predicting_realtime_data():
    #model = load_model('synthetic_pmate.h5')
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame, (384, 0), (610, 250), (255, 0, 0), 2)
        roi_hand = gray[0:250, 384:610]
        out = cv2.resize(roi_hand, (28, 28))
        out = out.reshape(1, 28, 28, 1)
        y_pred = model.predict(out)
        print(np.argmax(y_pred.round()))
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


predicting_realtime_data()

