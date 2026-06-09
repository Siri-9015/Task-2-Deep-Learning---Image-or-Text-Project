import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Class Names
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

IMG_SIZE = 96

# Load Saved Model
model = tf.keras.models.load_model(
    "saved_model/cifar10_mobilenetv2.keras"
)

# List of test images
test_images = [
    "test_images/sample.jpg",
    "test_images/sample1.jpg"
]

for img_path in test_images:

    # Check file exists
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    # Load Image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"\nImage: {img_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")