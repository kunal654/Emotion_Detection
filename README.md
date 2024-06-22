Sure, here is a more detailed README for your GitHub repository:

---

# Facial Emotion Recognition

This repository contains code for a facial emotion recognition system using a convolutional neural network (CNN). The system detects faces in an image and predicts the emotion of each detected face.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Loading the Model](#loading-the-model)
  - [Face Detection](#face-detection)
  - [Emotion Prediction](#emotion-prediction)
  - [Displaying Results](#displaying-results)
- [Model Files](#model-files)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Facial emotion recognition is a technology that identifies and processes human emotions from facial expressions. This project utilizes a pre-trained convolutional neural network model to classify emotions into one of seven categories: angry, disgust, fear, happy, neutral, sad, and surprise. The system uses OpenCV for face detection and Keras for deep learning.

## Requirements
- Python 3.x
- OpenCV
- Keras
- TensorFlow
- NumPy
- Matplotlib
- Google Colab (optional, for running on Google Colab)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kunal654/facial-emotion-recognition.git
    cd facial-emotion-recognition
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Mount Google Drive if using Google Colab:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Usage

### Loading the Model

First, load the pre-trained model from the JSON and HDF5 files:

```python
import cv2
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

# Path to the model files in Google Drive
model_json_path = "/content/drive/MyDrive/facialemotionmodel.json"
model_weights_path = "/content/drive/MyDrive/facialemotionmodel.h5"
image_path = "/content/drive/MyDrive/Face/am.jpg"

# Load the model
with open(model_json_path, "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(model_weights_path)
```

### Face Detection

Use OpenCV's Haar Cascade Classifier to detect faces in the input image:

```python
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Load an image for testing
im = cv2.imread(image_path)
if im is None:
    raise FileNotFoundError(f"Image at path {image_path} not found.")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Check if faces are detected
if len(faces) == 0:
    raise ValueError("No faces detected in the image.")
```

### Emotion Prediction

Define a function to preprocess the detected face image and predict the emotion using the loaded model:

```python
def extract_features(image):
    image = cv2.resize(image, (48, 48))
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

for (p, q, r, s) in faces:
    face_image = gray[q:q+s, p:p+r]
    cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

    # Preprocess the face image
    img = extract_features(face_image)

    # Predict emotion
    pred = model.predict(img)
    prediction_label = labels[np.argmax(pred)]

    # Print predictions and the predicted label
    print("Predictions:", pred)
    print("Predicted label:", prediction_label)

    # Annotate the image with the predicted emotion
    cv2.putText(im, '%s' % (prediction_label), (p, q-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
```

### Displaying Results

Finally, display the output image with detected faces and annotated emotions:

```python
# Display the output image
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

## Model Files

- `facialemotionmodel.json`: The JSON file containing the model architecture.
- `facialemotionmodel.h5`: The HDF5 file containing the model weights.

Ensure these files are placed in the correct paths as specified in the code.

## Results

The system will display the input image with detected faces annotated with the predicted emotion labels. Each face will have a rectangle around it, and the predicted emotion will be displayed above the rectangle.

## Acknowledgements

- The facial emotion recognition model is based on a convolutional neural network.
- Face detection is performed using OpenCV's Haar Cascade Classifier.
- The project utilizes pre-trained models and standard datasets for training.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize the README further according to your project specifications or additional details you may want to include.
