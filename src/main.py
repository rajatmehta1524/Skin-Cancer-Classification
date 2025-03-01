import tensorflow as tf
from data import load_data
from model import build_model
from helper import plot_training_history, evaluate_model, display_sample_predictions

def main():
    # Load dataset
    train_dataset, val_dataset, test_dataset, input_shape, num_classes = load_data()

    # Build model
    model = build_model(input_shape, num_classes)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=10)

    # Evaluate model on test data
    test_loss, test_acc = evaluate_model(model, test_dataset)

    # Plot training history
    plot_training_history(history)

    # Save model
    model.save("../models/skin_cancer_model.h5")
    print("Model saved successfully!")

    # Display sample predictions
    display_sample_predictions(model, test_dataset)

if __name__ == "__main__":
    main()
