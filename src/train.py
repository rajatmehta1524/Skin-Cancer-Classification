import tensorflow as tf
import argparse
from data import load_data
from model import get_model  # Import get_model to select model type
from helper import plot_training_history
import os
import time

def train_model(model_type, epochs=10, save_model=True, model_dir="../models"):
    """
    Trains the selected model and saves it if required.
    """
    # Load dataset
    train_dataset, val_dataset, _, input_shape, num_classes = load_data(model_type)

    # Building model using the get_model function
    model = get_model(model_type, input_shape, num_classes)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    start_time = time.time()

    # Train model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs)

    end_time = time.time()
    training_time = (end_time - start_time) / 60  # Convert to minutes
    
    print(f"Training Time = {training_time:.2f} mins")

    # Plot training history
    plot_training_history(history, model_dir, model_type)

    # Saving the model
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        model_name = f"{model_type}_skin_cancer_model.h5"  # Save the model with a name reflecting the model type
        model_path = os.path.join(model_dir, model_name)
        model.save(model_path)
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB
        print(f"Model Size: {model_size:.2f} MB")

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
