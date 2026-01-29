import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import numpy as np
from math import pi
import matplotlib.pyplot as plt

def create_arcface_model(input_shape=(112, 112, 3), num_classes=2):
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
	for layer in backbone.layers[-10:]:
		layer.trainable = True

	# Global Average Pooling to reduce the feature map to a 1D vector
	x = layers.GlobalAveragePooling2D()(x)

	# Add a dense layer for feature embedding
	x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(5e-4))(x)
	x = layers.Dropout(0.5)(x)  # Add dropout before/after dense layers

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
	# theta = K.acos(K.clip(cosine_similarity, -1.0, 1.0))  # Ensure the value is within bounds of acos
	theta = tf.acos(tf.clip_by_value(cosine_similarity, -1.0, 1.0))
	theta_m = theta + m
	cosine_m = K.cos(theta_m)

	# Compute the loss
	loss = K.mean(K.maximum(0.0, 1.0 - cosine_m))

	return loss


#define training dataset
dataset_dir = '/workspace/facial-recognition/faces_dataset'

# Data augmentation for TRAINING only 
train_datagen = ImageDataGenerator(
	rescale=1./255,
	validation_split=0.2,
	rotation_range=10,          # random rotation (-10..+10 degrees)
	width_shift_range=0.05,     # random horizontal shift (±5%)
	height_shift_range=0.05,    # random vertical shift (±5%)
	zoom_range=0.10,            # random zoom in/out (±10%)
	horizontal_flip=True,       # 50% chance mirror flip
	brightness_range=(0.8, 1.2) # random brightness
)

# NO augmentation for validation (only rescale) 
valid_datagen = ImageDataGenerator(
	rescale=1./255,
	validation_split=0.2
)

# Train generator uses augmented datagen 
train_gen = train_datagen.flow_from_directory(
	dataset_dir,
	target_size=(112, 112),
	batch_size=8,
	class_mode='categorical',
	subset='training',
	shuffle=True
)

# Validation generator uses non-augmented datagen
valid_gen = valid_datagen.flow_from_directory(
	dataset_dir,
	target_size=(112, 112),
	batch_size=8,
	class_mode='categorical',
	subset='validation',
	shuffle=False  # [EDITED] optional: keep validation deterministic
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
# Compile the model with ArcFace loss and optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=arcface_loss, metrics=['accuracy'])
#model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4), loss=arcface_loss, metrics=['accuracy'])

# Store training and validation logs
history = model.fit(train_gen, validation_data=valid_gen, epochs=20)

# Save the model if needed
model.save('arcface_model.h5')

print("train class_indices:", train_gen.class_indices)
print("val class_indices:", valid_gen.class_indices)

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()
