import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# ✅ 1. Define Directories (Make Sure You Have Images Here)
train_dir = "dataset/train"  # Images should be inside subfolders like "dataset/train/cat", "dataset/train/dog"
test_dir = "test"  # Same structure as train_dir

# ✅ 2. Data Augmentation for Better Training
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values
    rotation_range=30,    # Rotate images up to 30 degrees
    width_shift_range=0.2,  # Shift width
    height_shift_range=0.2,  # Shift height
    shear_range=0.2,      # Shear transformation
    zoom_range=0.2,       # Zoom in/out
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest'   # Fill missing pixels
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Just rescale test images

# ✅ 3. Load Images from Folders
train_dataset = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images
    batch_size=32,
    class_mode='sparse'  # Use 'sparse' if labels are integer-based
)

test_dataset = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)

# ✅ 4. Define the CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_dataset.class_indices), activation='softmax')  # Output neurons = Number of classes
])

# ✅ 5. Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ 6. Train the Model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# ✅ 7. Save the Model
model.save("models//Cats/Dogs/x2.h5")

print("Training complete! Model saved as image_classifier.h5")