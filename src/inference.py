import tensorflow as tf
import numpy as np
import os
import argparse
import cv2
from data import load_data
from helper import display_sample_predictions,preprocess_image

def load_trained_model(model_path="../models/skin_cancer_model.h5"):
    """
    Loads a pre-trained model.

    Parameters:
        model_path (str): Path to the trained model.

    Returns:
        model (tf.keras.Model): Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

def predict(model, image_path):
    """Preprocesses the image and makes a prediction."""
    image = preprocess_image(image_path)  # Resize & normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

def make_predictions(model, dataset, num_samples=10):
    """
    Makes predictions using the trained model and displays sample results.

    Parameters:
        model (tf.keras.Model): The trained model.
        dataset (tf.data.Dataset): Dataset to run inference on.
        num_samples (int): Number of sample predictions to display.

    Returns:
        predictions (np.ndarray): Predicted class probabilities.
        true_labels (np.ndarray): True labels for evaluation.
    """
    images, labels = next(iter(dataset.take(1)))  # Take a single batch
    predictions = model.predict(images)
    # Display predictions
    display_sample_predictions(images.numpy(), labels.numpy(), predictions, num_samples=num_samples)
    return predictions, labels.numpy()

if __name__ == "__main__":
    # Load test dataset
    # _, _, test_dataset, _, _ = load_data()

    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    # Load trained model
    model = load_trained_model()
    predict(model, args.image_path)

    # # Run inference on test data
    # make_predictions(model, test_dataset)
