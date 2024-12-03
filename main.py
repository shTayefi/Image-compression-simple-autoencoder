#autoencoder

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, _), (x_test, _) = cifar10.load_data()

# Normalize data to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ensure data is in the correct shape for the model (batch_size, height, width, channels)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Build the Autoencoder
def build_autoencoder(input_shape):
    # Encoder
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

# Define input shape (32x32 RGB images)
input_shape = (32, 32, 3)
autoencoder = build_autoencoder(input_shape)

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the Autoencoder (if model is not already trained)
try:
    # Load the saved model
    autoencoder = load_model("autoencoder_model.h5")
    print("Loaded saved model successfully!")
except:
    print("Training the model...")
    # Train the model and save it
    history = autoencoder.fit(
        x_train, x_train,
        epochs=20,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test)
    )
    # Save the trained model
    autoencoder.save("autoencoder_model.h5")

# Reconstruct images using the trained Autoencoder
decoded_imgs = autoencoder.predict(x_test)

# Display original and reconstructed images
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("Original")
    plt.axis("off")

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstructed")
    plt.axis("off")
plt.tight_layout()
plt.show()
