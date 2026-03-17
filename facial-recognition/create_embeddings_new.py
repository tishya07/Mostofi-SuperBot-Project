import cv2
import numpy as np
import pickle
import os
from glob import glob
import insightface
from insightface.app import FaceAnalysis

# ----------------------------
# Config
# ----------------------------
REGISTERED_FACES_PATH = "/workspace/facial-recognition/registered_faces.pkl"
DATASET_PATH = "/workspace/facial-recognition/faces_dataset/Training"
VAL_PATH = "/workspace/facial-recognition/faces_dataset/Validation"
THRESHOLD = 0.35  # InsightFace similarities are lower than before, adjust threshold

# ----------------------------
# Load InsightFace
# ----------------------------
print("Loading InsightFace model...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("✓ InsightFace loaded")

# ----------------------------
# Helper Functions
# ----------------------------
def get_embedding(img):
    """Get normalized embedding from full image using InsightFace"""
    faces = app.get(img)
    if not faces:
        return None
    # Use the largest face detected
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    emb = face.embedding
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb

def match_face(test_emb, registered_embeddings, registered_names):
    best_match, best_score = "Unknown", -1.0
    for name, ref_emb in zip(registered_names, registered_embeddings):
        score = float(np.dot(test_emb, ref_emb))
        if score > best_score:
            best_score, best_match = score, name
    if best_score < THRESHOLD:
        best_match = "Unknown"
    return best_match, best_score

def load_embeddings_from_dataset(dataset_path):
    registered_names, registered_embeddings = [], []

    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path not found: {dataset_path}")
        return registered_names, registered_embeddings

    person_folders = [f for f in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, f))]

    if not person_folders:
        print(f"✗ No person folders found in {dataset_path}")
        return registered_names, registered_embeddings

    print(f"\nCreating embeddings from dataset...")
    print(f"Found {len(person_folders)} person(s)")

    for person_name in sorted(person_folders):
        person_path = os.path.join(dataset_path, person_name)
        image_files = (glob(os.path.join(person_path, "*.jpg")) +
                       glob(os.path.join(person_path, "*.jpeg")) +
                       glob(os.path.join(person_path, "*.png")))

        print(f"\n  Processing {person_name}: {len(image_files)} images")

        person_embeddings = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            emb = get_embedding(img)
            if emb is not None:
                person_embeddings.append(emb)
            else:
                print(f"    ⚠ No face: {os.path.basename(img_path)}")

        if person_embeddings:
            avg = np.mean(np.stack(person_embeddings, axis=0), axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-12)
            registered_names.append(person_name)
            registered_embeddings.append(avg)
            print(f"    ✓ Created embedding from {len(person_embeddings)} images")
        else:
            print(f"    ✗ No valid faces found for {person_name}")

    print(f"\n✓ Created {len(registered_names)} embeddings")
    return registered_names, registered_embeddings

# ----------------------------
# Validation Function
# ----------------------------
def run_validation(val_path, registered_names, registered_embeddings):
    image_paths = (glob(os.path.join(val_path, "*", "*.jpg")) +
                   glob(os.path.join(val_path, "*", "*.jpeg")) +
                   glob(os.path.join(val_path, "*", "*.png")))

    if not image_paths:
        print("No validation images found.")
        return

    correct, total = 0, 0
    print("\n" + "="*60)
    print(f"VALIDATION: {val_path}")
    print(f"Images: {len(image_paths)} | Threshold: {THRESHOLD}")
    print("="*60)

    for img_path in sorted(image_paths):
        true_label = os.path.basename(os.path.dirname(img_path))
        img = cv2.imread(img_path)
        if img is None:
            continue

        emb = get_embedding(img)
        if emb is None:
            print(f"{os.path.relpath(img_path, val_path)} | GT={true_label:<15} ⚠ No face")
            continue

        pred, score = match_face(emb, registered_embeddings, registered_names)
        total += 1
        is_correct = (pred == true_label)
        correct += int(is_correct)
        print(f"{os.path.relpath(img_path, val_path)} | GT={true_label:<15} Pred={pred:<15} sim={score:.3f} {'✓' if is_correct else '✗'}")

    acc = correct / total if total > 0 else 0.0
    print("\n" + "-"*60)
    print(f"Accuracy: {correct}/{total} = {acc:.2%}")
    print("-"*60 + "\n")

# ----------------------------
# Load or Create Embeddings
# ----------------------------
registered_names, registered_embeddings = [], []

pkl_exists = os.path.exists(REGISTERED_FACES_PATH)
dataset_exists = os.path.exists(DATASET_PATH)

if pkl_exists and dataset_exists:
    pkl_mtime = os.path.getmtime(REGISTERED_FACES_PATH)
    dataset_mtime = max(
        os.path.getmtime(os.path.join(dp, f))
        for dp, _, filenames in os.walk(DATASET_PATH)
        for f in filenames
    ) if any(os.scandir(DATASET_PATH)) else 0
    if dataset_mtime > pkl_mtime:
        print("⚠ Dataset newer than saved embeddings, regenerating...")
        pkl_exists = False

if pkl_exists:
    with open(REGISTERED_FACES_PATH, 'rb') as f:
        data = pickle.load(f)
        registered_names = data['names']
        registered_embeddings = data['embeddings']
    print(f"✓ Loaded {len(registered_names)} registered faces from file")
else:
    registered_names, registered_embeddings = load_embeddings_from_dataset(DATASET_PATH)
    if registered_embeddings:
        with open(REGISTERED_FACES_PATH, 'wb') as f:
            pickle.dump({'names': registered_names, 'embeddings': registered_embeddings}, f)
        print(f"✓ Saved {len(registered_names)} embeddings to {REGISTERED_FACES_PATH}")
    else:
        print("⚠ No embeddings created.")

# ----------------------------
# Validation
# ----------------------------
if os.path.exists(VAL_PATH):
    val_images = (glob(os.path.join(VAL_PATH, "*", "*.jpg")) +
                  glob(os.path.join(VAL_PATH, "*", "*.png")))
    if val_images:
        print("\n" + "="*60)
        print("VALIDATION MODE DETECTED")
        print("="*60)
        response = input("\nRun validation? (y/n): ").lower()
        if response == 'y':
            run_validation(VAL_PATH, registered_names, registered_embeddings)
            print("✓ Validation complete")
