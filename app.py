import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import pickle

# Load the age and gender classification models
age_model = load_model('models/age-model-final.pkl')
gender_model = load_model('models/sex-model-final.pkl')

# Load the emotion detection model
emotion_model = load_model('models/emotion-model-final.pkl')

# Define the labels for age and gender
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_labels = ['Male', 'Female']

# Define the labels for emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define the color for age and gender text
age_color = (255, 255, 255)
gender_color = (147, 20, 255)

# Define the font for age and gender text
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the video capture device
cap = cv2.VideoCapture(0)

# Loop through frames from the video capture device
while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Resize the frame to 224x224
    resized_frame = cv2.resize(frame, (224, 224))

    # Preprocess the frame for age and gender classification
    age_gender_input = np.expand_dims(resized_frame, axis=0)
    age_gender_input = age_gender_input / 255.0

    # Predict the age and gender from the preprocessed frame
    age_prediction = age_labels[np.argmax(age_model.predict(age_gender_input))]
    gender_prediction = gender_labels[np.argmax(gender_model.predict(age_gender_input))]

    # Draw the age and gender text on the frame
    cv2.putText(frame, "Age: " + age_prediction, (10, 50), font, 1, age_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Gender: " + gender_prediction, (10, 100), font, 1, gender_color, 2, cv2.LINE_AA)

    # Preprocess the frame for emotion detection
    emotion_input = image.img_to_array(resized_frame)
    emotion_input = np.expand_dims(emotion_input, axis=0)
    emotion_input = emotion_input / 255.0

    # Predict the emotion from the preprocessed frame
    emotion_prediction = emotion_labels[np.argmax(emotion_model.predict(emotion_input))]

    # Draw the emotion text on the frame
    cv2.putText(frame, "Emotion: " + emotion_prediction, (10, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with the predicted age, gender, and emotion
    cv2.imshow("Frame", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()