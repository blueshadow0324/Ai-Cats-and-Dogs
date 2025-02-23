import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

# ============== 1️⃣ Load & Preprocess Dataset ==============

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
# ============== 2️⃣ Build Generator Model ==============
def build_generator():
    model = keras.Sequential([
        layers.Dense(8 * 8 * 512, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 512)),

        layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding="same", activation="tanh")
    ])
    return model


# ============== 3️⃣ Build Discriminator Model ==============
def build_discriminator():
    model = keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)  # No activation (Hinge loss)
    ])
    return model


# ============== 4️⃣ Define Loss Functions & Optimizers ==============

# Hinge Loss for better training stability
def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_output))
    fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_output))
    return real_loss + fake_loss


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)  # Maximize fake image confidence


# Optimizers
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# ============== 5️⃣ Train the GAN ==============

generator = build_generator()
discriminator = build_discriminator()

EPOCHS = 1000  # Increase for better quality
SAVE_INTERVAL = 100  # Save every 100 epochs


@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        print(f"Epoch {epoch + 1}/{epochs} - Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")

        # Save model & generate sample images every SAVE_INTERVAL epochs
        if (epoch + 1) % SAVE_INTERVAL == 0:
            generator.save(f"generator_epoch_{epoch + 1}.h5")
            generate_and_save_images(generator, epoch + 1)


# ============== 6️⃣ Generate & Save Images ==============
def generate_and_save_images(model, epoch, num_examples=1):
    noise = tf.random.normal([num_examples, LATENT_DIM])
    generated_images = model(noise, training=False)

    for i in range(num_examples):
        plt.imshow((generated_images[i] + 1) / 2)  # Convert from [-1,1] to [0,1]
        plt.axis("off")
        plt.savefig(f"generated_epoch_{epoch}.png")
        plt.show()


# Run training
train(train_dataset, EPOCHS)


# ============== 7️⃣ Load & Use the Trained Generator ==============

def load_trained_generator(model_path):
    model = keras.models.load_model(model_path)
    return model


def generate_new_image(generator, num_examples=1):
    noise = tf.random.normal([num_examples, LATENT_DIM])
    generated_images = generator(noise, training=False)

    for i in range(num_examples):
        plt.imshow((generated_images[i] + 1) / 2)
        plt.axis("off")
        plt.show()


# Example: Load the trained model and generate an image
trained_generator = load_trained_generator("generator_x3.h5")
generate_new_image(trained_generator)
