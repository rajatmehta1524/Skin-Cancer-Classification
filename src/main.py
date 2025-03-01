import tensorflow as tf
from data import load_data
from helper import plot_training_history, evaluate_model, display_sample_predictions

def main():
    # Load dataset
    train_dataset, val_dataset, test_dataset, input_shape, num_classes = load_data()

    # Load trained model
    model_path = "../models/skin_cancer_model.h5"
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded trained model from {model_path}")

    # Evaluate model on test data
    test_loss, test_acc = evaluate_model(model, test_dataset)

    print(f"Test Accuracy: {test_acc:.4f}")

    # Display sample predictions
    display_sample_predictions(model, test_dataset)

if __name__ == "__main__":
    main()
