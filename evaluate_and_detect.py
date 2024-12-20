# This script is for evaluating the model and performing live detection

# Import necessary libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# Load the trained model
model = load_model('drowsiness_detection_model.h5')

# Define paths and image parameters
val_dir = 'dataset/validation'
img_width, img_height = 224, 224

# Setup ImageDataGenerator for validation
val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_data)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Real-time drowsiness detection
def detect_drowsiness():
    # Define labels
    labels = {0: 'Alert', 1: 'Drowsy'}

    # Start video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (img_width, img_height))
        normalized_frame = resized_frame / 255.0
        reshaped_frame = np.reshape(normalized_frame, (1, img_width, img_height, 3))

        # Make prediction
        prediction = model.predict(reshaped_frame)
        predicted_class = int(prediction[0][0] > 0.5)  # Threshold at 0.5 for binary classification
        label = labels[predicted_class]

        # Display the result on the screen
        color = (0, 255, 0) if label == 'Alert' else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.imshow('Drowsiness Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Uncomment to run the detection after evaluation
detect_drowsiness()
