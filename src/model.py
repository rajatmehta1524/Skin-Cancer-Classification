import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ConvNeXtTiny
from transformers import ViTForImageClassification, ViTFeatureExtractor, TFAutoModelForImageClassification
import os

# CNN Model
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        # layers.Dense(128, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Assuming 'num_classes' output classes
    ])
    return model

# ConvNeXt Model
def build_convnext_model(input_shape, num_classes):
    base_model = ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Vision Transformer (ViT) Model
def build_vit_model(input_shape, num_classes):
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    model = TFAutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",         
        num_labels=num_classes, 
        ignore_mismatched_sizes=True
    )

    # model = ViTForImageClassification.from_pretrained(
    #     'google/vit-base-patch16-224-in21k', 
    #     num_labels=num_classes, 
    #     ignore_mismatched_sizes=True
    # )
    
    return model

# Function to get the model based on the type
def get_model(model_type, input_shape, num_classes):
    if model_type == "cnn":
        return build_cnn_model(input_shape, num_classes)
    elif model_type == "convnext":
        return build_convnext_model(input_shape, num_classes)
    elif model_type == "vit":
        return build_vit_model(input_shape, num_classes)
    else:
        raise ValueError("Invalid model type. Choose from: 'cnn', 'convnext', 'vit'")
