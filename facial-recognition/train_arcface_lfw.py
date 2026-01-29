import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# ArcFace Head
# ----------------------------
class ArcFace(layers.Layer):
    def __init__(self, num_classes, s=64.0, m=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.s = s
        self.m = m

    def build(self, input_shape):
        emb_dim = int(input_shape[0][-1])
        self.W = self.add_weight(
            name="W",
            shape=(emb_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        embeddings, labels = inputs
        
        x = tf.nn.l2_normalize(embeddings, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        cos_t = tf.matmul(x, W)
        cos_t = tf.clip_by_value(cos_t, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Get one-hot labels
        labels = tf.cast(labels, tf.int32)
        one_hot = tf.one_hot(labels, depth=self.num_classes)
        
        # DEBUG: Print which class is the target for first sample
        #tf.print("Target class for sample 0:", labels[0])
        #tf.print("One-hot for sample 0 (first 10 positions):", one_hot[0, :10])
        
        # Compute theta and add margin
        theta = tf.acos(cos_t)
        cos_t_m = tf.cos(theta + self.m)
        
        # DEBUG: Check if margin is being applied
        #tf.print("BEFORE margin - cos_t[0] at target class:", tf.gather(cos_t[0], labels[0]))
        #tf.print("AFTER margin - cos_t_m[0] at target class:", tf.gather(cos_t_m[0], labels[0]))
        
        # Apply margin only to target class
        final_cos_t = tf.where(tf.cast(one_hot, dtype=tf.bool), cos_t_m, cos_t)
        
        logits = final_cos_t * self.s
        
        return logits

# ----------------------------
# Build model: image -> embedding -> ArcFace logits
# ----------------------------
def build_arcface_model(num_classes, input_shape=(112,112,3), emb_dim=512, train_backbone=True):
    image_in = layers.Input(shape=input_shape, name="image")
    label_in = layers.Input(shape=(), dtype=tf.int32, name="label")

    backbone = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    backbone.trainable = train_backbone

    x = backbone(image_in, training=train_backbone)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(emb_dim, use_bias=False, name="emb_dense")(x)
    x = layers.BatchNormalization(name="emb_bn")(x)

    logits = ArcFace(num_classes=num_classes, s=64.0, m=0.5, name="arcface")([x, label_in])
    return models.Model(inputs={"image": image_in, "label": label_in}, outputs=logits)

# ----------------------------
# Data
# ----------------------------
base = "/workspace/facial-recognition/faces_dataset_lfw"
train_dir = os.path.join(base, "Training")
val_dir   = os.path.join(base, "Validation")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(112,112), batch_size=32, class_mode="sparse", shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(112,112), batch_size=32, class_mode="sparse", shuffle=False
)

if train_gen.class_indices != val_gen.class_indices:
    raise ValueError("Train/Val class folders do not match exactly.")

num_classes = train_gen.num_classes
print("num_classes:", num_classes)

# tf.data wrapper so labels can be used BOTH as model input and y_true
def arcface_ds(gen):
    while True:
        x, y = next(gen)
        y = tf.cast(y, tf.int32)
        yield ({"image": x, "label": y}, y)

train_ds = tf.data.Dataset.from_generator(
    lambda: arcface_ds(train_gen),
    output_signature=(
        {"image": tf.TensorSpec((None,112,112,3), tf.float32),
         "label": tf.TensorSpec((None,), tf.int32)},
        tf.TensorSpec((None,), tf.int32),
    )
).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: arcface_ds(val_gen),
    output_signature=(
        {"image": tf.TensorSpec((None,112,112,3), tf.float32),
         "label": tf.TensorSpec((None,), tf.int32)},
        tf.TensorSpec((None,), tf.int32),
    )
).prefetch(tf.data.AUTOTUNE)

# ----------------------------
# Train
# ----------------------------
model = build_arcface_model(num_classes=num_classes, train_backbone=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen),
)

model.save("/workspace/facial-recognition/arcface_lfw_model.keras")
print("Saved: arcface_lfw_model.keras")

import matplotlib.pyplot as plt

# ---- PLOT TRAINING CURVES ----
hist = history.history

# Keys will typically be: 'loss', 'acc', 'val_loss', 'val_acc'
print("History keys:", list(hist.keys()))

epochs = range(1, len(hist["loss"]) + 1)

# Loss plot
plt.figure()
plt.plot(epochs, hist["loss"], label="Train Loss")
plt.plot(epochs, hist["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/workspace/facial-recognition/lfw_loss.png", dpi=200)
plt.close()

# Accuracy plot
plt.figure()
plt.plot(epochs, hist["acc"], label="Train Acc")
plt.plot(epochs, hist["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/workspace/facial-recognition/lfw_accuracy.png", dpi=200)
plt.close()

print("Saved plots:")
print(" - /workspace/facial-recognition/lfw_loss.png")
print(" - /workspace/facial-recognition/lfw_accuracy.png")

