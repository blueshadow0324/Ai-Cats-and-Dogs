import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tkinter import filedialog, Tk, Button, Label

# Load the trained model
model = tf.keras.models.load_model("models/x1.h5")

# Load class names from dataset
train_dir = "dataset/train"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir)
class_names = train_dataset.class_names

def classify_image():
    """Classifies an image selected by the user."""
    img_path = filedialog.askopenfilename(
        initialdir="/",
        title="Select a Cat or a Dog",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"))
    )
    if not img_path:
        return  # No file selected, do nothing

    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]

        result_label.config(text=f"✅ The Image is a: {predicted_label}")

    except Exception as e:
        result_label.config(text=f"Error: {e}")

# Create a simple window
root = Tk()
root.title("Cat or Dog Classifier")
root.geometry("260x180")

# Create a button to trigger image selection and classification
classify_button = Button(root, text="Select Image", command=classify_image, width=10, height=5)
classify_button.pack()

# Create a label to display the result
result_label = Label(root, text="")
result_label.pack()

# Run the main loop
root.mainloop()
