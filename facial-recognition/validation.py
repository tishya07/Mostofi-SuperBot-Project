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

# ----------------------------
# NEW: Validation folder (Option 4)
# Structure: VAL_PATH/person_name/*.jpg
# ----------------------------
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

# Face detector
face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# ----------------------------
# Helper Functions
# ----------------------------
def get_embedding(face_img):
    """Get L2-normalized embedding from face image (shape: (D,))"""
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

    # test_emb is (D,)
    for name, ref_emb in zip(registered_names, registered_embeddings):
        # ref_emb should also be (D,)
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

            embedding = get_embedding(img)
            person_embeddings.append(embedding)

        if len(person_embeddings) > 0:
            avg_embedding = np.mean(np.stack(person_embeddings, axis=0), axis=0)  # (D,)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-12)


            registered_names.append(person_name)
            registered_embeddings.append(avg_embedding)
            print(f"    ✓ Created embedding from {len(person_embeddings)} images")

    print(f"\n✓ Created {len(registered_names)} embeddings from dataset")
    return registered_names, registered_embeddings

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
        print("No registered faces found. Will start registration mode.")

# ----------------------------
# NEW: Option 4 — Validation accuracy on folder and EXIT
# ----------------------------
def run_validation(val_path):
    image_paths = glob(os.path.join(val_path, "*", "*.jpg")) + \
                  glob(os.path.join(val_path, "*", "*.png")) + \
                  glob(os.path.join(val_path, "*", "*.jpeg"))

    if len(image_paths) == 0:
        return False  # nothing to validate

    if len(registered_embeddings) == 0:
        print("\n✗ Cannot validate: no registered embeddings available.")
        print("  Create registered embeddings first (dataset or registered_faces.pkl).")
        return True  # we did attempt validation, stop

    correct = 0
    total = 0

    print("\n" + "="*60)
    print(f"VALIDATION MODE (Option 4): {val_path}")
    print(f"Images found: {len(image_paths)} | Threshold: {THRESHOLD}")
    print("="*60)

    for img_path in sorted(image_paths):
        true_label = os.path.basename(os.path.dirname(img_path))

        img = cv2.imread(img_path)
        if img is None:
            continue

        # OPTIONAL: detect face and crop first; keep super simple:
        # If no face found, just embed whole image (works if your val images are already face crops)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
            roi = img[y:y+h, x:x+w]
        else:
            roi = img

        emb = get_embedding(roi)
        pred, score = match_face(emb, registered_embeddings, registered_names)

        total += 1
        is_correct = (pred == true_label)
        correct += int(is_correct)

        print(f"{os.path.relpath(img_path, val_path)} | GT={true_label:<15} Pred={pred:<15} sim={score:.3f} {'✓' if is_correct else '✗'}")

    acc = (correct / total) if total > 0 else 0.0
    print("\n" + "-"*60)
    print(f"Accuracy: {correct}/{total} = {acc:.2%}")
    print("-"*60 + "\n")
    return True  # validation ran

# If validation folder has images, run it and exit (Option 4)
if os.path.exists(VAL_PATH) and run_validation(VAL_PATH):
    print("✓ Done (validation mode)")
    raise SystemExit(0)
