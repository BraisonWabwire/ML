# %%
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# %%

# Path to dataset
dataset_path = r"C:\Users\brais\Downloads\archive\mammals"

# %%
# Image size
IMG_SIZE = (64, 64)

# %%
# Load images and labels
X = []
y = []


# %%

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            img = cv2.imread(image_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype('float32') / 255.0  # Normalize
            X.append(img)
            y.append(class_name)

# %%
# Convert to numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)

# %%
# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(le.classes_)

# %%
y = keras.utils.to_categorical(y, num_classes)  # One-hot encode labels

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Define CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# %%
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
# Train the model
epochs = 25
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32)

# %%
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# %%
def classify_image(image_path):
    img = cv2.imread(image_path)  # Read the image dynamically
    if img is None:
        print(f"Error: Cannot read image at {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_label = le.classes_[np.argmax(prediction)]
    print(f"Predicted Class: {predicted_label}")
    return predicted_label

# Example usage
image_path = input("Enter the path of the image to classify: ").strip()
classify_image(image_path)


# %%



