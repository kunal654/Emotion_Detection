import cv2
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Path to the model files in Google Drive
model_json_path = "/content/drive/MyDrive/facialemotionmodel.json"
model_weights_path = "/content/drive/MyDrive/facialemotionmodel.h5"
image_path = "/content/drive/MyDrive/Face/am.jpg"

# Load the model
with open(model_json_path, "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(model_weights_path)

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    image = cv2.resize(image, (48, 48))  # Ensure the image is resized correctly
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Load an image for testing
im = cv2.imread(image_path)
if im is None:
    raise FileNotFoundError(f"Image at path {image_path} not found.")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Check if faces are detected
if len(faces) == 0:
    raise ValueError("No faces detected in the image.")

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

# Display the output image
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
