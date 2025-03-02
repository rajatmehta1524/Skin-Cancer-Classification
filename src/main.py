import tensorflow as tf
import argparse
from data import load_data
from helper import plot_training_history, evaluate_model, display_sample_predictions
import os

def main(model_type):
    # Load dataset
    train_dataset, val_dataset, test_dataset, input_shape, num_classes = load_data()

    # Load the trained model corresponding to the selected model type
    model_name = f"{model_type}_skin_cancer_model.h5"  # Dynamically set the model name
    model_path = os.path.join("../models", model_name)
    
    # Check if model exists
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded trained model from {model_path}")
    else:
        print(f"Model {model_name} not found. Please train the model first.")
        return

    # Evaluate model on test data
    test_loss, test_acc = evaluate_model(model, test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Display sample predictions
    display_sample_predictions(model, test_dataset)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Select model type for loading and inference.")
    parser.add_argument(
        '--model', 
        choices=['cnn', 'convnext', 'vit'], 
        required=True, 
        help="Specify the model type: 'cnn', 'convnext', or 'vit'"
    )
    args = parser.parse_args()

    # Run the main function with the selected model
    main(args.model)
