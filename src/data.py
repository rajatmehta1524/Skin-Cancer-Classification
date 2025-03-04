import tensorflow as tf
from transformers import ViTFeatureExtractor
import os

DATASET_DIR = "../dataset"
IMG_SIZE = (224, 224)  # Input Image size
BATCH_SIZE = 8  # Batch Size for training purpose
AUTOTUNE = tf.data.AUTOTUNE

# Load ViT Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# def preprocess_fn(image):
#     """Function to preprocess input image for ViT"""
#     return feature_extractor(image, return_tensors="tf")["pixel_values"]

def preprocess_fn(image, label):
    """Function to preprocess input image for ViT"""
    image = tf.image.resize(image, [224, 224])  # Ensure correct input size
    image = image.numpy()  # Convert tensor to NumPy array
    image = feature_extractor(image, return_tensors="tf")["pixel_values"]
    return image, label  # Ensure label is returned unchanged

# Ensure TF Dataset uses NumPy conversion
def tf_preprocess_fn(image, label):
    """Wrap preprocess_fn for use in tf.data pipeline"""
    image, label = tf.py_function(preprocess_fn, [image, label], [tf.float32, tf.int64])
    image.set_shape((1, 3, 224, 224))  # ViT expects (batch, channels, height, width)
    return image, label

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

def load_data(model_type):

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

    if model_type != "vit": # Data Preprocessing for CNN and ConvNext Model 
        # Normalizing pixel values
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    
        # Applying preprocessing
        train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)

    else: # Data Preprocessing for Vision Transformer model
        # train_ds = train_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
        # val_ds = val_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
        # test_ds = test_ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.map(tf_preprocess_fn, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(tf_preprocess_fn, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(tf_preprocess_fn, num_parallel_calls=AUTOTUNE)



    # Prefetch to improve performance
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    input_shape = IMG_SIZE + (3,)  # (224,224,3) for RGB images

    return train_ds, val_ds, test_ds, input_shape, num_classes
