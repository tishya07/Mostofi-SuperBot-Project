"""
Simplified ArcFace Face Recognition with OAK-D
==============================================
Usage:
1. First run: Creates embeddings from dataset
2. Later runs: Auto-loads saved embeddings
"""

import cv2
import numpy as np
import depthai as dai
import tensorflow as tf
import pickle
import os
from glob import glob
from tensorflow.keras import layers

# ----------------------------
# Enable GPU memory growth FIRST
# ----------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "/workspace/facial-recognition/arcface_lfw_model.keras"
REGISTERED_FACES_PATH = "/workspace/facial-recognition/registered_faces.pkl"
THRESHOLD = 0.6

# TODO: Set your dataset path here
# Dataset should have structure: dataset_path/person_name/*.jpg
DATASET_PATH = "/workspace/facial-recognition/faces_dataset"

# ----------------------------
# ArcFace Class Definition
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
        
        # Compute theta and add margin
        theta = tf.acos(cos_t)
        cos_t_m = tf.cos(theta + self.m)
        
        # Apply margin only to target class
        final_cos_t = tf.where(tf.cast(one_hot, dtype=tf.bool), cos_t_m, cos_t)
        
        logits = final_cos_t * self.s
        
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "s": self.s,
            "m": self.m
        })
        return config

# ----------------------------
# Load Model
# ----------------------------
print("Loading model...")
full_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'ArcFace': ArcFace}
)

embedding_model = tf.keras.models.Model(
    inputs=full_model.get_layer("image").input,
    outputs=full_model.get_layer("emb_bn").output
)
print("✓ Model loaded")

# Face detector
face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# ----------------------------
# Helper Functions
# ----------------------------
def get_embedding(face_img):
    """Get normalized embedding from face image"""
    face_img = cv2.resize(face_img, (112, 112))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    
    embedding = embedding_model.predict(face_img, verbose=0)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def match_face(test_embedding, registered_embeddings, registered_names):
    """Find best match for test embedding"""
    best_match = "Unknown"
    best_score = -1
    
    for name, ref_embedding in zip(registered_names, registered_embeddings):
        similarity = np.dot(test_embedding.flatten(), ref_embedding.flatten())
        if similarity > best_score:
            best_score = similarity
            best_match = name
    
    if best_score < THRESHOLD:
        best_match = "Unknown"
    
    return best_match, best_score

def load_embeddings_from_dataset(dataset_path):
    """Load images from dataset and create embeddings"""
    registered_names = []
    registered_embeddings = []
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path not found: {dataset_path}")
        return registered_names, registered_embeddings
    
    # Get all person folders
    person_folders = [f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))]
    
    if len(person_folders) == 0:
        print(f"✗ No person folders found in {dataset_path}")
        return registered_names, registered_embeddings
    
    print(f"\nCreating embeddings from dataset...")
    print(f"Found {len(person_folders)} person(s)")
    
    for person_name in person_folders:
        person_path = os.path.join(dataset_path, person_name)
        image_files = glob(os.path.join(person_path, "*.jpg")) + \
                     glob(os.path.join(person_path, "*.png")) + \
                     glob(os.path.join(person_path, "*.jpeg"))
        
        print(f"\n  Processing {person_name}: {len(image_files)} images")
        
        person_embeddings = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Get embedding directly from image
            embedding = get_embedding(img)
            person_embeddings.append(embedding)
        
        if len(person_embeddings) > 0:
            # Average all embeddings for this person
            avg_embedding = np.mean(person_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            
            registered_names.append(person_name)
            registered_embeddings.append(avg_embedding)
            print(f"    ✓ Created embedding from {len(person_embeddings)} images")
    
    print(f"\n✓ Created {len(registered_names)} embeddings from dataset")
    return registered_names, registered_embeddings

# ----------------------------
# Load or Create Registered Faces
# ----------------------------
if os.path.exists(REGISTERED_FACES_PATH):
    with open(REGISTERED_FACES_PATH, 'rb') as f:
        data = pickle.load(f)
        registered_names = data['names']
        registered_embeddings = data['embeddings']
    print(f"✓ Loaded {len(registered_names)} registered faces from file")
else:
    # Try to load from dataset
    registered_names, registered_embeddings = load_embeddings_from_dataset(DATASET_PATH)
    
    if len(registered_embeddings) == 0:
        print("No registered faces found. Will start registration mode.")

# ----------------------------
# Setup OAK-D Camera
# ----------------------------
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480)
cam.setInterleaved(False)
cam.setFps(30)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam.preview.link(xout.input)

device = dai.Device(pipeline)
queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

print("\n" + "="*60)
print("CONTROLS:")
print("  'r' - Register new face")
print("  's' - Save registered faces")
print("  'q' - Quit")
print("="*60 + "\n")

# ----------------------------
# Main Loop
# ----------------------------
while True:
    frame = queue.get().getCvFrame()
    
    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Process each face
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        # Get embedding
        embedding = get_embedding(face_roi)
        
        # Match
        if len(registered_embeddings) > 0:
            name, score = match_face(embedding, registered_embeddings, registered_names)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({score:.2f})"
        else:
            color = (255, 255, 0)
            label = "Press 'r' to register"
        
        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Info
    cv2.putText(frame, f"Registered: {len(registered_names)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Face Recognition", frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r') and len(faces) > 0:
        # Register face
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        embedding = get_embedding(face_roi)
        
        name = input("\nEnter name: ")
        registered_names.append(name)
        registered_embeddings.append(embedding)
        print(f"✓ Registered: {name}")
        
    elif key == ord('s'):
        # Save
        with open(REGISTERED_FACES_PATH, 'wb') as f:
            pickle.dump({
                'names': registered_names,
                'embeddings': registered_embeddings
            }, f)
        print(f"✓ Saved {len(registered_names)} faces to {REGISTERED_FACES_PATH}")
        
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
device.close()
print("\n✓ Done")
