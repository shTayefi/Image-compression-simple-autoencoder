# Image-compression-simple-autoencoder


### Description
This project demonstrates how to use an Autoencoder for image reconstruction on the CIFAR-10 dataset. It utilizes TensorFlow and Keras to implement the Autoencoder architecture. The project supports saving and reloading the trained model for reuse and has optional quality enhancement techniques outlined in this document.

---

### Features
1. Basic Autoencoder:
   - Compresses input images into a latent space representation.
   - Reconstructs the images using a decoder.

2. Saving and Reloading:
   - Save the trained Autoencoder model to disk for later use.
   - Reload the saved model to reconstruct images without retraining.

3. Optional Quality Enhancements (not implemented in the code):
   - Increased Model Capacity: Add more filters and convolutional layers for better feature extraction.
   - Improved Loss Function: Use `mean_squared_error` (MSE) instead of `binary_crossentropy` for smoother reconstruction.
   - Higher Image Resolution: Resize CIFAR-10 images to higher dimensions (e.g., 128x128).
   - Extended Training: Increase the number of epochs for better learning.
   - Preprocessing Techniques: Apply noise reduction or augmentation to improve input data quality.

---

### Requirements
- Python 3.7+
- TensorFlow 2.x
- Matplotlib
- CIFAR-10 dataset

Install the required dependencies using:
```bash
pip install tensorflow matplotlib
```

---

### Usage

#### 1. Running the Code
Execute the script to:
- Train the Autoencoder on the CIFAR-10 dataset.
- Save the trained model for future use.
- Visualize the original and reconstructed images.

```bash
python autoencoder.py
```

#### 2. Saving and Reloading the Model
- Save the model:
  ```python
  autoencoder.save('autoencoder_model.h5')
  ```
- Reload the model:
  ```python
  from tensorflow.keras.models import load_model
  autoencoder = load_model('autoencoder_model.h5')
  ```

#### 3. Quality Enhancements (Optional)
- Increased Model Capacity: Add more filters (e.g., 64 and 32) and additional convolutional layers in the encoder and decoder.
- Higher Image Resolution: Resize CIFAR-10 images to 128x128 or other resolutions using:
  ```python
  from tensorflow.image import resize
  x_train_resized = tf.image.resize(x_train, [128, 128])
  x_test_resized = tf.image.resize(x_test, [128, 128])
  ```
- Improved Loss Function: Replace `binary_crossentropy` with `mean_squared_error`:
  ```python
  autoencoder.compile(optimizer='adam', loss='mean_squared_error')
  ```
- Extended Training: Train the model for 50 epochs or more for better results.

---

### Expected Improvements
- Better reconstruction quality with less noise in output images.
- Reduced loss values during training.
- Ability to save and reload the model to avoid retraining.

