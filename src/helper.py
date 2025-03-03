import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def save_model_info(model, model_type, test_dataset, class_names, model_dir, batch_size=8):
    """
        This function is used for saving model information, training parameters, and classification report to a text file.
    """

    os.makedirs(model_dir, exist_ok=True)

    #Model configuration
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
    For Plotting and saving the confusion matrix.
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


def plot_training_history(history, model_dir, model_type):
    """
    For Plotting training accuracy and loss curves.
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

    epoch_acc_path = os.path.join(model_dir, f"{model_type}_Epochs_vs_acc.png")
    plt.savefig(epoch_acc_path)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    train_val_path = os.path.join(model_dir, f"{model_type}_Training_vs_Val_loss.png")
    plt.savefig(train_val_path)

    plt.show()

def display_sample_predictions(model, test_data, class_labels, num_samples=5):
    """
    Displays a few sample test images along with their predicted labels.

    """

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
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path="model.h5"):
    """
    Loads a trained model from a file.
    """
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"No model found at {model_path}")


def evaluate_model(model, test_dataset):
    """
    Evaluates the model on the test dataset.
    """
    test_loss, test_acc = model.evaluate(test_dataset)
    return test_loss, test_acc
