import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

IMG_SIZE = 64  # Set image size
BATCH_SIZE = 32
LATENT_DIM = 100  # Noise vector size

# Load dataset from directory
dataset_path = "dataset/train/dog"
train_dataset = keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    label_mode=None,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Normalize images to [-1, 1]
train_dataset = train_dataset.map(lambda x: (x / 127.5) - 1.0)

# Optional: Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
train_dataset = train_dataset.map(lambda x: data_augmentation(x, training=True))

model = tf.keras.models.load_model('generator_epoch_1000.h5')
model.fit(train_dataset, layers, epoch)