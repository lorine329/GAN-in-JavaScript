# AnimeFaceGAN in JavaScript

## Overview
AnimeFaceGAN is a cutting-edge project that implements a Generative Adversarial Network (GAN) entirely in JavaScript using TensorFlow.js. This project has been developed from scratch, providing a transparent view of the entire process—from data preprocessing to model training and image generation. The main goal is to generate high-quality anime faces by leveraging the power of GANs in a browser-friendly environment.

## Features
- /**From-Scratch Development**/: All components of the GAN model are implemented from scratch, offering insights into the inner workings of generative models.
- /**JavaScript Implementation**/: Entirely written in JavaScript, enabling seamless integration with web technologies.
- /**Anime Face Generation**/: Utilizes a custom GAN architecture specifically tailored for generating detailed and diverse anime faces.
- /**Customizable**/: Easily modify model parameters and network architecture to experiment with different artistic styles.
- /**Web Integration**/: Designed to run in web browsers, making it accessible for demos and creative projects.

## Data Source
The training dataset consists of high-quality anime face images sourced from Kaggle. This dataset has been instrumental in training the GAN model to understand and replicate the unique artistic styles found in anime. If you are interested in exploring the dataset further or contributing improvements, you can find it on Kaggle.

## Project Structure and Code Descriptions

### 1. AnimeFaceGAN.js (GAN Model Implementation)
This script contains the full implementation of the GAN model, including:
- **Data Loading**: Reads anime face images from the dataset folder.
- **Preprocessing**: Converts image data to tensors, normalizes pixel values to the [-1, 1] range for stable training, and batches images for efficient processing.
- **Generator Model**:
  - Uses transposed convolutional layers to generate high-resolution anime face images.
  - Applies batch normalization and leaky ReLU activations to improve stability.
  - Outputs a final image using a tanh activation function.
- **Discriminator Model**:
  - A convolutional neural network (CNN) that classifies images as real or fake.
  - Includes dropout layers for regularization and batch normalization to stabilize training.
  - Uses sigmoid activation to output a probability score.
- **Training Loop**:
  - Generates fake images from random noise.
  - Trains the discriminator on real and generated images.
  - Trains the generator using adversarial loss to fool the discriminator.
  - Tracks **generator loss, discriminator loss, and accuracy** across epochs.
- **Model Saving**: Saves the trained generator model after completing training.

### 2. anime_data.js (Data Preprocessing)
This script is responsible for preparing the dataset for training. It includes:
- **Reading Images**: Loads anime face images from a local folder.
- **Image Conversion**: Converts images into tensors using `tfnode.node.decodeImage`.
- **Normalization**: Scales pixel values from [0, 255] to the [-1, 1] range.
- **Batching**: Splits images into batches to optimize training efficiency.

These scripts work together to train and generate anime faces using JavaScript. The GAN model is built entirely from scratch, making it a valuable resource for learning and experimentation.
