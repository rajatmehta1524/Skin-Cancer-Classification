import tensorflow as tf
import os

# Define dataset paths
DATASET_DIR = "../dataset"
IMG_SIZE = (224, 224)  # Resize images to 224x224
BATCH_SIZE = 8  # Match batch size from main.py
AUTOTUNE = tf.data.AUTOTUNE

def load_data():
    """
    Loads and preprocesses the dataset.
    Returns:
        train_dataset, val_dataset, test_dataset (tf.data.Dataset)
        input_shape (tuple): Shape of input images
        num_classes (int): Number of classes in dataset
    """
    # Load dataset from directories
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # Extract class names and determine number of classes
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

    # Prefetch to improve performance
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    input_shape = IMG_SIZE + (3,)  # (224, 224, 3) for RGB images

    return train_ds, val_ds, test_ds, input_shape, num_classes
