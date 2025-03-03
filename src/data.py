import tensorflow as tf
import os

DATASET_DIR = "../dataset"
IMG_SIZE = (224, 224)  # Input Image size
BATCH_SIZE = 8  # Batch Size for training purpose
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_image(image, label):

    # Data Augmentation Techniques
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0] + 20, IMG_SIZE[1] + 20)
    # image = tf.image.random_crop(image, size=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Defining the 3x3 Gaussian filter 
    gaussian_filter = tf.constant([[[[1]], [[2]], [[1]]], 
                                   [[[2]], [[4]], [[2]]], 
                                   [[[1]], [[2]], [[1]]]], dtype=tf.float32) / 16.0

    # Noise Removal (Gaussian Blur approximation using depthwise convolution)
    # image = tf.nn.depthwise_conv2d(
    #     image,
    #     # filter=tf.constant([[[[1]], [[2]], [[1]]], [[[2]], [[4]], [[2]]], [[[1]], [[2]], [[1]]]], dtype=tf.float32) / 16.0,
    #     filter = gaussian_filter,
    #     strides=[1, 1, 1, 1], padding='SAME'
    # )[0]
    
    # Contrast Adjustment
    image = tf.image.adjust_contrast(image, contrast_factor=1.5)
    
    return image, label

def load_data():

    # Loading train, val and test datasets from respective directories

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATASET_DIR, "test"),
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
    
    # Normalizing pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    
    # Applying preprocessing
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    
    # Prefetch to improve performance
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    input_shape = IMG_SIZE + (3,)  # (224,224,3) for RGB images

    return train_ds, val_ds, test_ds, input_shape, num_classes
