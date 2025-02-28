#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
# from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import PIL
from PIL import Image
print("Pillow version:", PIL.__version__)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)


# Check for GPU/CPU and Setup TensorFlow
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Running on {device_name}")

#%%
# Set device for TensorFlow
with tf.device(device_name):
    # 1. Data Preparation
    image_size = (224, 224)  # Image dimensions
    
    # Data Augmentation for Training Data
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalizing the images
        shear_range=0.2,  # Random shearing
        zoom_range=0.2,   # Random zoom
        horizontal_flip=True  # Random flipping
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training and testing data
    train_data = train_datagen.flow_from_directory(
        'dataset/train',  # Path to your training data
        target_size=image_size,  # Resize images
        batch_size=4,  # Batch size
        class_mode='categorical'  # For multi-class classification
    )

    test_data = test_datagen.flow_from_directory(
        'dataset/test',  # Path to your testing data
        target_size=image_size,  # Resize images
        batch_size=4,  # Batch size
        class_mode='categorical'  # For multi-class classification
    )


    # 2. Build the Model
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))  # 2 classes: benign, malignant

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # 3. Set Learning Rate Scheduler (optional)
    def lr_scheduler(epoch, lr):
        if epoch % 10 == 0 and epoch != 0:
            lr = lr * 0.1  # Reduce learning rate by 10% every 10 epochs
        return lr

    # 4. Train the Model
    history = model.fit(
        train_data,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        epochs=20,  # Adjust epochs as needed
        validation_data=test_data,
        validation_steps=test_data.samples // test_data.batch_size,
        callbacks=[LearningRateScheduler(lr_scheduler)]
    )

    # 5. Evaluate the Model
    test_loss, test_acc = model.evaluate(test_data, steps=test_data.samples // test_data.batch_size)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # 6. Save the Model
    model.save('skin_cancer_model.h5')  # Model saved in HDF5 format

    # 7. Plot Training History (Loss/Accuracy Curves)
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    # 8. Inference & Evaluation on Test Data
    predictions = model.predict(test_data)
    predictions = np.argmax(predictions, axis=-1)

    # Ground truth labels
    true_labels = test_data.classes

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

# %%
