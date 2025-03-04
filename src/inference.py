import tensorflow as tf
import numpy as np
import os
import argparse
import cv2
from helper import display_sample_predictions
import time
from PIL import Image

def load_trained_model(model_type, model_dir="../models"):
    """
    Loads a pre-trained model based on the model type.
    """

    # Dynamically determine the model name based on the model type
    model_name = f"{model_type}_skin_cancer_model.h5"  # e.g., cnn_skin_cancer_model.h5
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

def predict(model, image_path, IMG_SIZE = (224,224)):
    """Preprocesses the image and makes a prediction."""
    
    if not os.path.exists(image_path):
        print(f"Error: The image path {image_path} does not exist.")
        return
    
    try:
        # Open and resize the image to the required input size
        img = Image.open(image_path).resize((IMG_SIZE[0], IMG_SIZE[1]))
        img_array = np.array(img)
        
        if img_array.shape[-1] != 3:  
            raise ValueError("Image must have 3 channels (RGB).")
        
        img_array = img_array.astype('float32') / 255.0
        
        image = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Make prediction using the model
    start_time = time.time()
    prediction = model.predict(image)
    end_time = time.time()

    # Post-processing of the prediction
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Inference time in milliseconds
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    print(f"Inference Time: {inference_time:.2f} ms")
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

def make_predictions(model, dataset, num_samples=10):
    """
    Makes predictions using the trained model and displays sample results.
    """
    images, labels = next(iter(dataset.take(1)))  # Take a single batch
    predictions = model.predict(images)
    # Display predictions
    display_sample_predictions(images.numpy(), labels.numpy(), predictions, num_samples=num_samples)
    return predictions, labels.numpy()

if __name__ == "__main__":
    # Set up for argument parsing
    parser = argparse.ArgumentParser(description="Run inference on a single image or dataset")
    parser.add_argument("--model", choices=['cnn', 'convnext', 'vit'], required=True, 
                        help="Specify the model type: 'cnn', 'convnext', or 'vit'")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Load the trained model based on the model type
    model = load_trained_model(model_type=args.model)

    # Make prediction on the provided image
    predict(model, args.image_path)
