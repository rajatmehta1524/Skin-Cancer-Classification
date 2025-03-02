import tensorflow as tf
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data import load_data
from helper import plot_training_history, evaluate_model, display_sample_predictions,plot_confusion_matrix, save_model_info

def main(model_type):
    # Load dataset
    train_dataset, val_dataset, test_dataset, input_shape, num_classes = load_data()

    class_names = ["Benign", "Malignant"]

    # Load the trained model corresponding to the selected model type
    model_name = f"{model_type}_skin_cancer_model.h5"  # Dynamically set the model name
    model_path = os.path.join("../models", model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found. Please train the model first.")
        return
    
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded trained model from {model_path}")

    # Evaluate model on test data
    test_loss, test_acc = evaluate_model(model, test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model metadata and classification report
    save_model_info(model, model_type, test_dataset, class_names, "../models", batch_size=8)


    # Compute predictions for confusion matrix
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # Save Confusion Matrix separately
    plot_confusion_matrix(y_true, y_pred, class_names, "../models", model_type)

    # # Generate Confusion Matrix
    # cm = confusion_matrix(y_true, y_pred)
    
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.title(f"Confusion Matrix - {model_type.upper()}")
    
    # cm_path = os.path.join("../models", f"{model_type}_confusion_matrix.png")
    # plt.savefig(cm_path)
    # plt.show()
    # print(f"Confusion matrix saved at: {cm_path}")

    # Display Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Display sample predictions
    display_sample_predictions(model, test_dataset, class_names)

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
