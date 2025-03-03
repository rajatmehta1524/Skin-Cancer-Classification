import tensorflow as tf
import numpy as np
import os
import argparse
import cv2
from helper import display_sample_predictions, preprocess_image
import time

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

def predict(model, image_path):
    """Preprocesses the image and makes a prediction."""
    image = preprocess_image(image_path)  # Resize, normalize, contrast adjustment
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    start_time = time.time()
    prediction = model.predict(image)
    end_time = time.time()

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

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
