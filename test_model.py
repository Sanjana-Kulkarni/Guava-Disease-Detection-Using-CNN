import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

def test_model(image_path):
    labels = ['Disease Free', 'Phytopthora', 'Red rust', 'Scab', 'Styler and Root']
    
    # Load the model
    model = load_model("model/vgg_weights.hdf5")
    
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display results
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Display the image with prediction
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")
    image = cv2.resize(image, (300, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_image_path = "scab.jpg"
    if os.path.exists(test_image_path):
        test_model(test_image_path)
    else:
        print(f"Error: The image file {test_image_path} does not exist.")
        print("Please check the file path and try again.")