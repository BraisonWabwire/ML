# %%
import os
import cv2
import numpy as np

# Define dataset path
dataset_path = r"C:\Users\brais\Downloads\archive\mammals"

# Initialize lists for storing image data and labels
X = []  # Feature vectors
y = []  # Labels (category names)

# Traverse through each category folder
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)

    if os.path.isdir(label_path):  # Check if it's a directory
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)

            # Read image using OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            if img is None:
                continue  # Skip unreadable images
            
            img = cv2.resize(img, (64, 64))  # Resize to 64x64
            X.append(img.flatten())  # Convert to 1D vector
            y.append(label)  # Assign class label

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Total images loaded:", len(X))
print("Feature vector shape:", X.shape)  # Should be (num_samples, 64*64)
print("Labels:", np.unique(y))  # Unique categories



