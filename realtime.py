import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Load the trained model
try:
    model = load_model("drowsiness_detection_model.h5")
    print("Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Based on the model summary, set the input dimensions
# The first Conv2D layer expects 64x64 input
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Define constants
CLOSED_EYES_THRESHOLD = 2.0  # Reduced threshold for testing
CONFIDENCE_THRESHOLD = 0.5  # Threshold for prediction confidence

# Initialize variables
start_time = None
alert_triggered = False
frame_count = 0

# Initialize the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    try:
        # Preprocess the frame
        resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to RGB if your model was trained on RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        normalized_frame = rgb_frame / 255.0
        
        # Reshape for model input
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Make prediction
        prediction = model.predict(input_frame, verbose=0)
        
        # Print prediction values occasionally for debugging
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Raw prediction value: {prediction[0]}")

        # Determine eye state
        if prediction.shape[-1] == 1:  # Binary classification
            confidence = prediction[0][0]
            is_drowsy = confidence > CONFIDENCE_THRESHOLD
        else:  # Multi-class
            confidence = np.max(prediction[0])
            predicted_class = np.argmax(prediction[0])
            is_drowsy = (predicted_class == 1)  # Assuming 1 is drowsy class

        # Display confidence and state
        state = "Drowsy" if is_drowsy else "Alert"
        confidence_text = f"Confidence: {confidence:.2f}"
        
        # Display prediction info
        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.putText(frame, f"State: {state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, confidence_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Drowsiness detection logic
        if is_drowsy:
            if start_time is None:
                start_time = time.time()
            
            time_closed = time.time() - start_time
            cv2.putText(frame, f"Eyes closed for: {time_closed:.1f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if time_closed >= CLOSED_EYES_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not alert_triggered:
                    print("Drowsiness Alert Triggered!")
                    alert_triggered = True
        else:
            start_time = None
            alert_triggered = False

        # Display preprocessed frame (helps in debugging)
        small_frame = cv2.resize(normalized_frame[0], (IMG_WIDTH*2, IMG_HEIGHT*2))
        cv2.imshow("Preprocessed Frame", small_frame)

    except Exception as e:
        print(f"Error during prediction: {e}")
        cv2.putText(frame, "Error during prediction", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the main frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()