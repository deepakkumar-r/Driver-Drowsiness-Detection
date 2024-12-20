import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set paths
train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# Image parameters
img_width, img_height = 224, 224
batch_size = 32
epochs = 10  # Reduced epochs for quick testing
model_checkpoint_path = 'drowsiness_model_checkpoint.keras'  # Save in .keras format for checkpoint
final_model_path = 'drowsiness_detection_model.h5'  # Final model saved in .h5 format

# Data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Build a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Check if a model checkpoint exists
if os.path.exists(final_model_path):
    model = load_model(final_model_path)
    print("Loaded model from final model.")

# Set up checkpoint and early stopping callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=model_checkpoint_path,  # Save in .keras format for checkpoint
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1,
    save_weights_only=False
)

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,  # Stop training if validation loss does not improve for 3 epochs
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Save the final trained model as .h5
model.save(final_model_path)
print(f"Final model saved as {final_model_path}")

# For live video detection
def detect_drowsiness():
    # Load the best saved model
    model = load_model(final_model_path)

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

# Uncomment this to run detection after training
detect_drowsiness()
