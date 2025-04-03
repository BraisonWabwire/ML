# %%
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# %%
# Path to your dataset
dataset_path = r"C:\Users\brais\Downloads\archive\mammals"

# %%
# Load images and labels
X = []  # Feature vectors
y = []  # Labels

# %%
# Image processing parameters
IMG_SIZE = (64, 64)  # Resize all images to 64x64

# %%
# Loop through each folder (each class)
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):  # Ensure it's a directory
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                continue  # Skip unreadable images

            # Convert to RGB instead of grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize the image
            img = cv2.resize(img, IMG_SIZE)

            # Normalize pixel values to [0,1]
            img = img.astype('float32') / 255.0

            # Flatten the image and add to dataset
            X.append(img.flatten())  # Convert 3D image to 1D vector
            y.append(class_name)  # Store the label


# %%
# Convert lists to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)

# %%
# Encode labels into numbers
le = LabelEncoder()
y = le.fit_transform(y)

# %%
# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Train the SVM classifier
svm_model = SVC(kernel='linear', C=1.0)  # You can also try 'rbf'
svm_model.fit(X_train, y_train)

# %%
# Make predictions
y_pred = svm_model.predict(X_test)

# %%
# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# %%
# Convert predictions back to class names
y_pred_labels = le.inverse_transform(y_pred)
print("Predicted labels:", y_pred_labels)

# %%
def classify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Cannot read image!")
        return
    
    # Convert to RGB (Ensure consistency with training data)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
    # Resize the image
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize pixel values to [0,1]
    img = img.astype('float32') / 255.0  
    
    # Flatten & reshape for SVM
    img = img.flatten().reshape(1, -1)  

    # Predict class
    prediction = svm_model.predict(img)
    predicted_label = le.inverse_transform(prediction)[0]

    print(f"Predicted Class: {predicted_label}")


# %%
# Remove leading/trailing spaces and potential quotation marks
image_path = input("Enter the path of the image to classify: ").strip().strip('"').strip("'")

classify_image(image_path)


# %%
# Define test cases (image paths and expected labels)
test_cases = [
    (r"C:\Users\brais\Downloads\archive\mammals\elephant\00000092.jpg", "elephant"),
    (r"C:\Users\brais\Downloads\archive\mammals\tiger\00000065.jpg", "tiger"),
    (r"C:\Users\brais\Downloads\archive\mammals\goat\00000099.jpg", "goat"),
    (r"C:\Users\brais\Downloads\archive\mammals\cat\00000094.jpg", "cat"),
    (r"C:\Users\brais\Downloads\archive\mammals\bear\00000068.jpg", "bear"),
]

# Run classification on test cases
for image_path, expected_label in test_cases:
    print(f"Testing: {image_path}")
    predicted_label = classify_image(image_path)
    
    # Compare prediction with expected label
    if predicted_label and predicted_label.strip().lower() == expected_label.strip().lower():
        print(f"✅ Correctly classified as: {predicted_label}")
    else:
        print(f"❌ Misclassified! Expected: {expected_label}, Got: {predicted_label}")
    print("-" * 50)  # Separator for readability


# %%



