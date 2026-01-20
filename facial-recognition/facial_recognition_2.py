import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def create_arcface_model(input_shape=(112, 112, 3), num_classes=8631):
    """
    Create the ArcFace model architecture using a ResNet50 backbone.
    
    Args:
    - input_shape: Input shape of the images (height, width, channels).
    - num_classes: The number of classes (identities in the dataset).
    
    Returns:
    - A compiled Keras model ready for training.
    """
    
    # Input layer
    input_layer = layers.Input(shape=input_shape)
    
    # ResNet50 backbone for feature extraction (without top classification layer)
    backbone = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Freeze the ResNet layers (optional for fine-tuning)
    backbone.trainable = False
    
    # Add the backbone output to the model
    x = backbone(input_layer)  # Ensure the input is passed through ResNet50
    
    # Global Average Pooling to reduce the feature map to a 1D vector
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add a dense layer for feature embedding
    x = layers.Dense(512, activation='relu')(x)
    
    # Output layer: Linear activation for face classification (no activation here, softmax in ArcFace)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs=input_layer, outputs=output)
    
    return model

# Example: Create the model
model = create_arcface_model()
model.summary()

def arcface_loss(y_true, y_pred, s=64.0, m=0.5):
    """
    ArcFace loss implementation. This loss function maximizes the angular margin between features.

    Args:
    - y_true: The true labels (one-hot encoded).
    - y_pred: The predicted embeddings (raw output of the network).
    - s: The scale factor (typically 64.0).
    - m: The angular margin (typically 0.5).

    Returns:
    - A tensor representing the ArcFace loss.
    """
    # Normalize the embeddings and labels
    y_pred = K.l2_normalize(y_pred, axis=-1)  # L2 normalize predictions (embedding)
    y_true = K.l2_normalize(y_true, axis=-1)  # L2 normalize labels (embedding)

    # Compute cosine similarity between the embeddings
    cosine_similarity = K.sum(y_true * y_pred, axis=-1)

    # Apply the angular margin
    theta = K.acos(K.clip(cosine_similarity, -1.0, 1.0))  # Ensure the value is within bounds of acos
    theta_m = theta + m
    cosine_m = K.cos(theta_m)

    # Compute the loss
    loss = K.mean(K.maximum(0.0, 1.0 - cosine_m))

    return loss
    
#define training dataset
dataset_dir = '/home/facial-recognition/training-dataset'

#make image processor
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#load training data and preprocess. each subfolder in dataset_dir will have all of its images loaded and given a label unique to the subdfolder
train_gen = datagen.flow_from_directory(
    dataset_dir,              # The main directory containing subfolders
    target_size=(112, 112),    # Resize all images to 112x112 (to match the model input)
    batch_size=32,             # Number of images per batch
    class_mode='categorical',  # Label the images with one-hot encoding (categorical)
    subset='training',         # Use 80% of data for training
    shuffle=True               # Shuffle the images for better randomness
)

# Load validation data from the same directory (20% of the data)
valid_gen = datagen.flow_from_directory(
    dataset_dir,              # Same directory as above
    target_size=(112, 112),    # Resize all images to 112x112
    batch_size=32,             # Number of images per batch
    class_mode='categorical',  # One-hot encoded labels
    subset='validation',       # Use 20% of data for validation
    shuffle=True               # Shuffle the images
)

#check images and labels in a batch
images, labels = next(train_gen)

# Check the shape of the images and labels
print("Images shape: ", images.shape)  # (batch_size, height, width, channels)
print("Labels shape: ", labels.shape)  # (batch_size, num_classes)

# Check the first image and its label
print("First image shape: ", images[0].shape)
print("First label: ", labels[0])

#Todo: add code for training the model and inference

