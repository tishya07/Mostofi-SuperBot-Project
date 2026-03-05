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

# Dataset should have structure: dataset_path/person_name/*.jpg
DATASET_PATH = "/workspace/facial-recognition/faces_dataset/Training"

# Validation folder (Option 4)
VAL_PATH = "/workspace/facial-recognition/faces_dataset/Validation"

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

        labels = tf.cast(labels, tf.int32)
        one_hot = tf.one_hot(labels, depth=self.num_classes)

        theta = tf.acos(cos_t)
        cos_t_m = tf.cos(theta + self.m)
        final_cos_t = tf.where(tf.cast(one_hot, dtype=tf.bool), cos_t_m, cos_t)

        logits = final_cos_t * self.s
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "s": self.s, "m": self.m})
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

# ----------------------------
# Load SSD Face Detector
# ----------------------------
print("Loading SSD face detector...")
net = cv2.dnn.readNetFromCaffe(
    "/workspace/facial-recognition/deploy.prototxt",
    "/workspace/facial-recognition/res10_300x300_ssd_iter_140000.caffemodel"
)
print("✓ SSD detector loaded")

# ----------------------------
# Helper Functions
# ----------------------------
def detect_and_crop_face(img, confidence_threshold=0.5):
    """Detect face using SSD and return largest face ROI, or None if no face found"""
    h, w = img.shape[:2]
    
    # Prepare blob for DNN
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    best_confidence = 0
    best_box = None
    
    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            
            # Ensure box is within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_box = (x1, y1, x2, y2)
    
    if best_box is None:
        return None
    
    x1, y1, x2, y2 = best_box
    face_roi = img[y1:y2, x1:x2]
    return face_roi

def detect_all_faces(img, confidence_threshold=0.5):
    """Detect all faces using SSD and return list of (x1, y1, x2, y2, confidence)"""
    h, w = img.shape[:2]
    
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            
            # Ensure box is within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            faces.append((x1, y1, x2, y2, confidence))
    
    return faces

def get_embedding(face_img):
    """Get L2-normalized embedding from face image (shape: (D,))"""
    if face_img.size == 0:
        return None
        
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (112, 112))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    emb = embedding_model.predict(face_img, verbose=0)[0]   # (D,)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb

def match_face(test_emb, registered_embeddings, registered_names):
    """Cosine similarity with normalized embeddings"""
    best_match = "Unknown"
    best_score = -1.0

    for name, ref_emb in zip(registered_names, registered_embeddings):
        score = float(np.dot(test_emb, ref_emb))
        if score > best_score:
            best_score = score
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
                
            face_roi = detect_and_crop_face(img)
            if face_roi is None:
                print(f"    ⚠ No face detected in {os.path.basename(img_path)}")
                continue

            embedding = get_embedding(face_roi)
            if embedding is not None:
                person_embeddings.append(embedding)

        if len(person_embeddings) > 0:
            avg_embedding = np.mean(np.stack(person_embeddings, axis=0), axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-12)

            registered_names.append(person_name)
            registered_embeddings.append(avg_embedding)
            print(f"    ✓ Created embedding from {len(person_embeddings)} images")
        else:
            print(f"    ✗ No valid faces found for {person_name}")

    print(f"\n✓ Created {len(registered_names)} embeddings from dataset")
    return registered_names, registered_embeddings

# ----------------------------
# Validation Function
# ----------------------------
def run_validation(val_path):
    image_paths = glob(os.path.join(val_path, "*", "*.jpg")) + \
                  glob(os.path.join(val_path, "*", "*.png")) + \
                  glob(os.path.join(val_path, "*", "*.jpeg"))

    if len(image_paths) == 0:
        return False

    if len(registered_embeddings) == 0:
        print("\n✗ Cannot validate: no registered embeddings available.")
        return True

    correct = 0
    total = 0

    print("\n" + "="*60)
    print(f"VALIDATION MODE: {val_path}")
    print(f"Images found: {len(image_paths)} | Threshold: {THRESHOLD}")
    print("="*60)

    for img_path in sorted(image_paths):
        true_label = os.path.basename(os.path.dirname(img_path))

        img = cv2.imread(img_path)
        if img is None:
            continue

        face_roi = detect_and_crop_face(img)
        if face_roi is None:
            print(f"{os.path.relpath(img_path, val_path)} | GT={true_label:<15} ⚠ No face detected")
            continue

        emb = get_embedding(face_roi)
        if emb is None:
            continue
            
        pred, score = match_face(emb, registered_embeddings, registered_names)

        total += 1
        is_correct = (pred == true_label)
        correct += int(is_correct)

        print(f"{os.path.relpath(img_path, val_path)} | GT={true_label:<15} Pred={pred:<15} sim={score:.3f} {'✓' if is_correct else '✗'}")

    acc = (correct / total) if total > 0 else 0.0
    print("\n" + "-"*60)
    print(f"Accuracy: {correct}/{total} = {acc:.2%}")
    print("-"*60 + "\n")
    return True

# ----------------------------
# Load or Create Registered Faces
# ----------------------------
registered_names, registered_embeddings = [], []

if os.path.exists(REGISTERED_FACES_PATH):
    with open(REGISTERED_FACES_PATH, 'rb') as f:
        data = pickle.load(f)
        registered_names = data['names']
        registered_embeddings = data['embeddings']
    print(f"✓ Loaded {len(registered_names)} registered faces from file")
else:
    registered_names, registered_embeddings = load_embeddings_from_dataset(DATASET_PATH)
    if len(registered_embeddings) == 0:
        print("⚠ No registered faces found. Press 'r' to register faces.")
    else:
        # ← ADD THIS: auto-save after generating from dataset
        with open(REGISTERED_FACES_PATH, 'wb') as f:
            pickle.dump({
                'names': registered_names,
                'embeddings': registered_embeddings
            }, f)
        print(f"✓ Auto-saved {len(registered_names)} embeddings to {REGISTERED_FACES_PATH}")

# ----------------------------
# Check if validation mode
# ----------------------------
if os.path.exists(VAL_PATH):
    val_images = glob(os.path.join(VAL_PATH, "*", "*.jpg")) + \
                 glob(os.path.join(VAL_PATH, "*", "*.png")) + \
                 glob(os.path.join(VAL_PATH, "*", "*.jpeg"))
    
    if len(val_images) > 0:
        print("\n" + "="*60)
        print("VALIDATION MODE DETECTED")
        print("="*60)
        response = input("\nRun validation? (y/n): ").lower()
        
        if response == 'y':
            run_validation(VAL_PATH)
            print("\n✓ Validation complete. Starting live detection...\n")
