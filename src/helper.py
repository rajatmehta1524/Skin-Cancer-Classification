import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def save_model_info(model, model_type, test_dataset, class_names, model_dir, batch_size=8):
    """
    Saves model information, training parameters, and classification report to a text file.

    Parameters:
        model (tf.keras.Model): The trained model.
        model_type (str): The type of model used (cnn, convnext, vit).
        test_dataset (tf.data.Dataset): The test dataset.
        class_names (list): List of class names.
        model_dir (str): Directory to save model-related files.
        epochs (int): Number of epochs used for training.
        batch_size (int): Batch size used for training.
    """
    # Create model directory if not exists
    os.makedirs(model_dir, exist_ok=True)

    # Get model configuration
    model_info = {
        "Model Type": model_type,
        "Optimizer": str(model.optimizer.get_config()["name"]),
        "Loss Function": str(model.loss),
        "Input Image Shape": str(model.input_shape[1:]),  # Ignore batch size
        "Batch Size": batch_size,
    }

    # Extract true labels and predictions
    test_data_iter = iter(test_dataset)
    test_images, test_labels = next(test_data_iter)
    y_true = np.argmax(test_labels.numpy(), axis=1)
    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)

    # Generate classification report
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Convert classification report to a string
    class_report_str = classification_report(y_true, y_pred, target_names=class_names)

    # Save model info and classification report to a text file
    model_info_path = os.path.join(model_dir, f"{model_type}_model_info.txt")
    with open(model_info_path, "w") as f:
        f.write("Model Information:\n")
        f.write(json.dumps(model_info, indent=4))
        f.write("\n\nClassification Report:\n")
        f.write(class_report_str)

    print(f"Model information saved to {model_info_path}")

    # # Save confusion matrix
    # plot_confusion_matrix(y_true, y_pred, class_names, model_dir, model_type)

def plot_confusion_matrix(y_true, y_pred, class_names, model_dir, model_type):
    """
    Plots and saves the confusion matrix.

    Parameters:
        y_true (array): True class labels.
        y_pred (array): Predicted class labels.
        class_names (list): List of class names.
        model_dir (str): Directory to save the confusion matrix.
        model_type (str): Model type (cnn, convnext, vit).
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_type}")

    cm_path = os.path.join(model_dir, f"{model_type}_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()


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
    # test_data_iter = iter(test_data)
    # test_images, test_labels = next(test_data_iter)
    # predictions = model.predict(test_images)

    # plt.figure(figsize=(12, 6))
    # for i in range(num_samples):
    #     plt.subplot(1, num_samples, i+1)
    #     plt.imshow(test_images[i].astype("uint8"))
    #     predicted_class = class_labels[np.argmax(predictions[i])]
    #     actual_class = class_labels[np.argmax(test_labels[i])]
    #     plt.title(f"Pred: {predicted_class}\nActual: {actual_class}")
    #     plt.axis('off')
    
    # plt.show()

    # Get a batch of test images and labels
    test_data_iter = iter(test_data)  # Create an iterator
    test_images, test_labels = next(test_data_iter)  # Get one batch
    
    # Get predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels.numpy(), axis=1)

    # Display images with predicted labels
    plt.figure(figsize=(10, 5))
    for i in range(min(5, len(test_images))):  # Show up to 5 images
        plt.subplot(1, 5, i + 1)
        plt.imshow(test_images[i])
        plt.title(f"Pred: {class_labels[predicted_classes[i]]}\nTrue: {class_labels[true_classes[i]]}")
        plt.axis("off")
    
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


def evaluate_model(model, test_dataset):
    """
    Evaluates the model on the test dataset.

    Parameters:
        model (tf.keras.Model): The trained model.
        test_dataset (tf.data.Dataset): The test dataset.

    Returns:
        test_loss (float): The loss on the test dataset.
        test_acc (float): The accuracy on the test dataset.
    """
    test_loss, test_acc = model.evaluate(test_dataset)
    return test_loss, test_acc
