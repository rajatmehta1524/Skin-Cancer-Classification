import tensorflow as tf
import argparse
from data import load_data
from model import get_model  # Import get_model to select model type
from helper import plot_training_history
import os

def train_model(model_type, epochs=10, save_model=True, model_dir="../models"):
    """
    Trains the selected model and saves it if required.

    Parameters:
        model_type (str): The type of model to train ('cnn', 'convnext', 'vit').
        epochs (int): Number of epochs for training.
        save_model (bool): Whether to save the trained model.
        model_dir (str): Directory to save the model.
        model_name (str): Name of the saved model file.

    Returns:
        model (tf.keras.Model): The trained model.
        history (tf.keras.callbacks.History): Training history.
    """
    # Load dataset
    train_dataset, val_dataset, _, input_shape, num_classes = load_data()

    # Build model using the get_model function
    model = get_model(model_type, input_shape, num_classes)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs)

    # Plot training history
    plot_training_history(history, model_dir, model_type)

    # Saving the model
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"{model_type}_skin_cancer_model.h5"  # Save the model with a name reflecting the model type
        model_path = os.path.join(model_dir, model_name)
        model.save(model_path)
        print(f"Model saved successfully at: {model_path}")

    return model, history

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Select model type for training.")
    parser.add_argument(
        '--model', 
        choices=['cnn', 'convnext', 'vit'], 
        required=True, 
        help="Specify the model type: 'cnn', 'convnext', or 'vit'"
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10, 
        help="Number of epochs for training (default is 10)"
    )
    parser.add_argument(
        '--save_model', 
        type=bool, 
        default=True, 
        help="Whether to save the trained model (default is True)"
    )
    parser.add_argument(
        '--model_dir', 
        type=str, 
        default="../models", 
        help="Directory to save the model (default is '../models')"
    )

    args = parser.parse_args()

    # Train the selected model
    train_model(
        model_type=args.model, 
        epochs=args.epochs, 
        save_model=args.save_model, 
        model_dir=args.model_dir, 
    )
