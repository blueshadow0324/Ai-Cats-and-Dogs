import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("image_classifier.h5")

# Load the test image
img_path = "test_image.jpg"  # Change to your test image path
img = image.load_img(img_path, target_size=(150, 150))

# Convert image to array
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Load class names from dataset
train_dir = "dataset/train"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir)
class_names = train_dataset.class_names

# Print the result
predicted_label = class_names[predicted_class]
print(f"✅ Predicted Label: {predicted_label}")