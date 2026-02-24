# Road Segmentation using Deep Learning

This project implements a semantic segmentation model to automatically detect road areas from images
The main objective is to build a robust deep learning pipeline for training, evaluation, and inference.

# Model Architecture

The model is based on U-Net with a MobileNetV2 encoder pre-trained on ImageNet.

Architecture: U-Net

Backbone: MobileNetV2

Pretraining: ImageNet weights

Task: Binary semantic segmentation (road vs background)

Using a pre-trained MobileNetV2 backbone improves feature extraction efficiency while keeping the model lightweight and suitable for GPU training environments like Google Colab.

# Dataset

The model is trained on the BDD10K (Berkeley DeepDrive 10K) dataset.

Dataset: BDD10K

Task: Road segmentation

Data type: Urban driving scenes

Annotations: Pixel-wise semantic labels

BDD10K provides diverse real-world driving scenarios including different weather, lighting, and traffic conditions, making it well-suited for robust road segmentation training.

# Example
<p align="center"> <img src="files/demo.gif" width="700"/> </p>
