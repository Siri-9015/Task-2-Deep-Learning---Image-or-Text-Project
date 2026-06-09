# CIFAR-10 Image Classification using MobileNetV2

## Objective

Develop a Deep Learning model using TensorFlow and Transfer Learning for image classification.

## Dataset

CIFAR-10 Dataset with 10 classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

## Technologies Used

* TensorFlow
* Keras
* MobileNetV2
* Transfer Learning
* Data Augmentation

## Features Implemented

* Pretrained MobileNetV2
* Image Augmentation
* Training and Validation Curves
* Confusion Matrix
* Classification Report
* Model Saving and Loading
* User Image Testing

## Project Structure

cifar10-transfer-learning/
│
├── train.py
├── inference.py
├── requirements.txt
├── README.md
│
├── saved_model/
│   └── cifar10_mobilenetv2.keras
│
├── outputs/
│   ├── accuracy_loss.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
└── test_images/
└── sample.jpg

## How to Run

### Install Dependencies

pip install -r requirements.txt

### Train the Model

python train.py

### Run Inference

python inference.py

## Expected Accuracy

Approximately 85% to 90% validation accuracy depending on training epochs and hardware.

## Learning Outcome

This project provided practical experience with transfer learning, image classification, model evaluation, and inference using TensorFlow.
