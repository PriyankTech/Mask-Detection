# ===========================
# 1. IMPORT LIBRARIES
# ===========================
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tabulate import tabulate

# ===========================
# 2. DEFINE DIRECTORIES & CREATE FOLDERS IF NOT EXISTS
# ===========================
BASE_DIR = "./dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")  
TEST_DIR = os.path.join(BASE_DIR, "test")

# Subfolders for training and testing data
categories = ["Masked", "No Mask", "Incorrect"]

# ===========================
# 3. DATA PREPARATION & AUGMENTATION
# ===========================
# Image augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

# Rescale only for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Dynamically determine the number of classes
num_classes = len(train_generator.class_indices)
print(f"✅ Number of classes detected: {num_classes}")

# ===========================
# 4. DEFINE IMPROVED CNN MODEL
# ===========================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile model with AdamW optimizer
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks to improve accuracy
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

# ===========================
# 5. TRAIN THE MODEL
# ===========================
print("✅ Training started...")
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,  # Increased Epochs for better learning
    callbacks=[early_stop, reduce_lr]
)
print("✅ Training completed!")

# ===========================
# 6. SAVE MODEL
# ===========================
model_save_path = "mask_detection_model_optimized.h5"
model.save(model_save_path)
print(f"✅ Model saved at {model_save_path}")

# ===========================
# 7. EVALUATE MODEL
# ===========================
test_loss, test_acc = model.evaluate(test_generator)
print("========================================\n")
print(f"🎯 Test Accuracy: {test_acc*100:.2f}%")
print("\n========================================")

# ===========================
# 8. CLASSIFY RANDOM IMAGES IN VAL FOLDER
# ===========================
# Load the trained model
model = load_model(model_save_path)

# Function to classify images in validation folder
def classify_images(val_dir, model, class_labels):
    results = []

    # Iterate through all images in the validation folder
    for image_name in os.listdir(val_dir):
        image_path = os.path.join(val_dir, image_name)

        if os.path.isfile(image_path):  # Ensure it's a valid image file
            # Load and preprocess the image
            img = load_img(image_path, target_size=(150, 150))  # Resize to model input size
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict the category
            prediction = model.predict(img_array)
            label_index = np.argmax(prediction[0])  # Get predicted class index
            detected_category = class_labels[label_index]  # Map to class name

            # Append results (Filename, Predicted Category)
            results.append([image_name, detected_category])

    return results

# Run classification on random images in validation folder
results = classify_images(VAL_DIR, model, categories)

# Display results in tabular format
print(tabulate(results, headers=["Image Name", "Detected Category"], tablefmt="grid"))

print("✅ Classification completed successfully!")
