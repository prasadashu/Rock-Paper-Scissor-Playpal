# Rock Paper Scissor Playpal
This is my first CNN based project wherein I train a neural network that attempts to classify the gesture made by the hand as either rock, paper or scissor. The computer (or pal!) makes a choice of its own by choosing among the three i.e. rock, paper or scissor and consequently competes with the player.

## capture_training_data.py
The above file upon execution accesses the video camera on the client system to capture images of the users hand gestures which serves as the training set for the deep learning algorithm. One can edit the path in the file to save the images in the desired location. By default all images captured get stored in the *image* directory. The images captured are stored in a 28x28 pixel format.

## creating_dataset.py
This file runs on the premise that the images of the hand gestures for each of the classes is stored in the respective(class name) directory within the *image* directory. It converts the images into a 784x1 vector and labels each image with a class identifier. Scissor is marked as 2, Rock as 1 and Paper as 0. The vectors along with the labels are stored in the form of a dataframe and stored in the *created_dataset* directory.

## cnn_model.py
This contains the CNN model. The image data is retrieved from the dataset and passed through the model. Finally the model is exported and saved in the home directory by the name *synthetic_pmate*(just liked the name!) in an h5 format.

## predict_realtime_data.py
Finally, upon execution of the above file, the video camera feed is captured and the gestures made by the hand is attempted to be classified as one of the three classes. The predicted class can be viewed in the console. The program runs in an infinite loop and can be terminated by pressing the *esc* key. The file imports the *synthetic_pmate.h5* model and passes each image through it after reshaping it again into a 28x28 pixel format.

## synthetic_pmate.h5
As stated above, this contains the model exported by the *cnn_model.py* file. This is a practice of model deployment wherein one mustn't train the entire network, instead a pre-trained model is used to predict the classes of the images.

## Note:
In order to execute the code, one must run only the *predict_realtime_data.py* file. Do rename the path in the file for the saved model if any error occurs.

## Dependencies
- Numpy
- Pandas
- Keras
- Sklearn
- OpenCV
- OS
- Matplotlib(Only for visualization purpose)
