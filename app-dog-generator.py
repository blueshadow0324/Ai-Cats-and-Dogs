from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf

generator = load_model("generator_epoch_1000.h5")

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow((generated_image[0] + 1) / 2)  # Convert from [-1,1] to [0,1]
plt.axis("off")
plt.show()