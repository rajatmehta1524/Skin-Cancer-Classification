import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

def plot_training_history(history):
    """
    Plots training accuracy and loss curves.

    Args:
        history (tf.keras.callbacks.History): The history object returned from model.fit().
    """
    # Extract metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r*-', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def display_sample_predictions(model, test_data, class_labels, num_samples=5):
    """
    Displays a few sample test images along with their predicted labels.

    Args:
        model (tf.keras.Model): The trained model for making predictions.
        test_data (tf.keras.preprocessing.image.DirectoryIterator): The test dataset.
        class_labels (list): List of class names corresponding to label indices.
        num_samples (int): Number of images to display.
    """
    test_images, test_labels = next(test_data)
    predictions = model.predict(test_images)

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(test_images[i].astype("uint8"))
        predicted_class = class_labels[np.argmax(predictions[i])]
        actual_class = class_labels[np.argmax(test_labels[i])]
        plt.title(f"Pred: {predicted_class}\nActual: {actual_class}")
        plt.axis('off')
    
    plt.show()

def save_model(model, model_path="model.h5"):
    """
    Saves the trained model to a file.

    Args:
        model (tf.keras.Model): The trained model.
        model_path (str): Path to save the model.
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path="model.h5"):
    """
    Loads a trained model from a file.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        tf.keras.Model: The loaded model.
    """
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
