import tensorflow as tf
from data import load_data
from model import build_model
from helper import plot_training_history
import os

def train_model(epochs=10, save_model=True, model_dir="../models", model_name="skin_cancer_model.h5"):
    """
    Trains the model and saves it if required.

    Parameters:
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

    # Build model
    model = build_model(input_shape, num_classes)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs)

    # Plot training history
    plot_training_history(history)

    # Save model if required
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        model.save(model_path)
        print(f"Model saved successfully at: {model_path}")

    return model, history

if __name__ == "__main__":
    train_model()
