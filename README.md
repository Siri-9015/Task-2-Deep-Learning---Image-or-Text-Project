# CIFAR-10 Image Classification using MobileNetV2

## 📌 Project Overview

This project implements an Image Classification model using Transfer Learning with MobileNetV2 on the CIFAR-10 dataset. The model is built using TensorFlow and Keras and is capable of classifying images into 10 different categories.

This project was developed as **Task 2** for the AI Internship Program.

---

## 🎯 Objective

Develop a Deep Learning model using TensorFlow and Transfer Learning for image classification with high accuracy and efficient inference.

---

## 📂 Dataset

The project uses the CIFAR-10 dataset containing 60,000 images across 10 classes:

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

---

## 🛠️ Technologies Used

* TensorFlow
* Keras
* MobileNetV2
* Transfer Learning
* Data Augmentation
* Matplotlib
* Scikit-learn

---

## ✨ Features Implemented

* Pretrained MobileNetV2 Model
* Transfer Learning
* Image Augmentation
* Training & Validation Accuracy Graphs
* Confusion Matrix
* Classification Report
* Model Saving and Loading
* User Image Prediction System
* Multi-image Testing Support

---

## 📁 Project Structure

```bash
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
    ├── sample.jpg
    └── sample1.jpg
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python train.py
```

### 3️⃣ Run Inference

```bash
python inference.py
```

---

## 📊 Model Performance

* Validation Accuracy: ~85% to 90%
* Optimized using Transfer Learning and Data Augmentation

---

## 📈 Outputs Generated

* Accuracy & Loss Curves
* Confusion Matrix
* Classification Report
* Saved `.keras` Model

---

## 🎓 Learning Outcome

This project provided practical experience in:

* Deep Learning
* Transfer Learning
* CNN-based Image Classification
* Model Evaluation
* TensorFlow & Keras Workflow
* Real-world AI Project Development

---

## 👩‍💻 Internship Information

Developed as part of the AI Internship Program - Task 2.
* User Image Testing

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
