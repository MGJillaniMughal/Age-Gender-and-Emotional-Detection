# Age, Gender, and Emotion Detection

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green)

This project is an implementation of an age, gender, and emotion detection application using machine learning algorithms in Python. The application uses pre-trained models for age, gender, and emotion classification to analyze faces captured from a video stream or image.

## Demo

![Demo](demo.gif)

## Features

- Real-time detection of age, gender, and emotion from webcam feed
- Age classification into predefined categories (e.g., (0-3), (4-7), etc.)
- Gender prediction (Male or Female)
- Emotion detection (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- High precision and recall scores with 90%+ accuracy

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV

## Getting Started

To run the application, follow these steps:

1. Clone the repository to your local machine:

# git clone https://github.com/MGJillaniMughal/Age-Gender-and-Emotional-Detection.git


2. Install the required libraries:
!pip install tensorflow keras opencv-python

3. Run the application:


## Usage

The application will open a video feed from your webcam and display real-time predictions for age, gender, and emotion for the faces it detects in the frame.

- Age: Displays the predicted age group based on predefined categories.
- Gender: Predicts whether the detected face is male or female.
- Emotion: Predicts the dominant emotion among Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.

Press 'q' to exit the application.

## Model Information

The application uses pre-trained models for age, gender, and emotion classification. These models were trained on large datasets and achieve high accuracy in real-time prediction. The model files are available in the `models` directory.

## Contributing

Contributions to the project are welcome. If you find any issues or want to add new features, feel free to create a pull request. Please follow the coding conventions and guidelines while contributing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
