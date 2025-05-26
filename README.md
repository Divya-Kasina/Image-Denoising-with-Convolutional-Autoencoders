# Image-Denoising-with-Convolutional-Autoencoders

**What is Image Denoising?**

Image denoising is the process of removing noise — random, unwanted variations — from an image.

Noise can be introduced during image capture due to poor lighting, sensor imperfections, or errors during transmission.

The objective is to recover a clean, clear version of the original image from its noisy counterpart.

**What is an Autoencoder?**

An autoencoder is a neural network designed to learn how to copy its input to its output.

It consists of two main parts:

_Encoder:_ Compresses the input image into a smaller, encoded representation (like squeezing information).

_Decoder:_ Reconstructs the image back to its original size from the compressed form.

**What is a Denoising Autoencoder?**

A denoising autoencoder is trained to remove noise from images.

It receives noisy images as input and learns to output the original, clean images.

Over time, it discovers noise patterns and learns how to restore the clean details in the image.

**How Does the Code Work?**

1.Loading and Preparing Data.

2.Loads a dataset of images (MNIST, CIFAR-10).

3.Normalizes images so pixel values lie between 0 and 1.

4.Adds Gaussian noise to create noisy versions of each image.

5.Building the Autoencoder Model

Uses Convolutional Layers to capture spatial features such as edges and textures.

_Encoder:_ Applies convolution + max pooling to progressively compress images.

_Decoder:_ Uses convolution + upsampling to restore images to original size.

6.Includes Batch Normalization to stabilize and accelerate training.

7.The output layer uses sigmoid activation to keep output pixels between 0 and 1.

**Training the Model**

_Loss function:_ Mean Squared Error (MSE) between output (denoised) and clean images.

Minimizes the MSE to improve denoising accuracy.

Employs Early Stopping to prevent overfitting by stopping training when validation performance stops improving.

**Evaluating the Model**

Compares denoised output to original images using:

_PSNR (Peak Signal-to-Noise Ratio):_ Measures pixel-wise similarity — higher values mean better quality.

Sensitive to pixel-level differences, useful for detecting low-level noise.

_SSIM (Structural Similarity Index):_ Measures structural and textural similarity — values closer to 1 indicate better structural fidelity.

Focuses on human perception of image quality, comparing structure, texture, and edges.

_Loss Curves:_ Track training and validation MSE loss over epochs to monitor learning progress.

![image](https://github.com/user-attachments/assets/19b7f9b9-dc48-4c4e-bd47-55dcbe7487c8)
for CIFAR-10 DATASET

![image](https://github.com/user-attachments/assets/803f2c57-a717-4a70-aacd-b0dbf8749eb1)
for MNIST DATASET

_Image Grids:_ Side-by-side comparison of noisy inputs, denoised outputs, and original clean images.

_Histograms:_ Distribution of PSNR and SSIM scores to assess denoising quality across samples.

![image](https://github.com/user-attachments/assets/0343362b-cfa2-4225-a91c-93190ff2ce0b)

for CIFAR-10 DATASET

![image](https://github.com/user-attachments/assets/79fa9ca2-9e13-4eca-b1a6-4ebbd444eb01)

for MNIST DATASET (Average PSNR over 100 test samples: 20.48 dB)

**Gaussian Noise:** Simulating Real-World Distortions

Random noise with values distributed according to a normal (bell curve) distribution.

Real-world images often contain noise due to lighting, sensors, or transmission errors.

Adding Gaussian noise artificially during training simulates these conditions.

Enhances the model’s ability to generalize and effectively denoise unseen noisy images.

**Batch Normalization:** Stabilizing and Speeding Up Training

A technique to normalize inputs of each layer to have mean ≈ 0 and standard deviation ≈ 1 during training.

Prevents "internal covariate shift" where input distributions to layers change during training.

Allows higher learning rates and faster convergence.

Acts as a regularizer reducing overfitting.

**Max Pooling:** Reducing Size and Focusing on Important Features

Downsamples feature maps by selecting the maximum value from small windows (e.g., 2x2 pixels).

Reduces spatial dimensions, lowering computational load and model complexity.

Retains important features while discarding irrelevant details.

Increases invariance to small shifts/distortions in the input.

Helps prevent overfitting by simplifying feature maps.

![image](https://github.com/user-attachments/assets/25665f63-57e4-41b3-aa75-995b9a1b497b)

for CIFAR-10 DATASET

![image](https://github.com/user-attachments/assets/bde1e302-060e-471e-a5a7-bed94dc93065)

for MNIST DATASET

**How These Components Work Together in the Model**

Add Gaussian noise to clean images to create noisy inputs.

Convolutional layers extract spatial features from noisy inputs.

Batch Normalization stabilizes and speeds training after each convolutional layer.

Max Pooling reduces feature map size and focuses on the strongest features.

Decoder reconstructs clean images from compressed noisy features, learning to remove noise.

