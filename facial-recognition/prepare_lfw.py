import os
import random
from PIL import Image
from tqdm import tqdm
import tensorflow_datasets as tfds

# ================= CONFIG =================
OUT_BASE = "/workspace/facial-recognition/faces_dataset_lfw"
TRAIN_DIR = os.path.join(OUT_BASE, "Training")
VAL_DIR   = os.path.join(OUT_BASE, "Validation")

MIN_PER_PERSON = 10
VAL_FRAC = 0.2
SEED = 123

random.seed(SEED)

# ================= CLEAN OUTPUT DIRS =================
for d in [TRAIN_DIR, VAL_DIR]:
    if os.path.isdir(d):
        # comment this out if you don't want auto-delete
        import shutil
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# ================= LOAD LFW =================
print("[INFO] Loading LFW from TFDS")
ds = tfds.load("lfw", split="train", as_supervised=True)

# ================= GROUP BY PERSON =================
people = {}

for name_bytes, img in tfds.as_numpy(ds):
    person = name_bytes.decode("utf-8")
    people.setdefault(person, []).append(img)

# ================= SPLIT + SAVE =================
kept = 0
total_imgs = 0

print("[INFO] Writing Training / Validation folders")

for person, imgs in tqdm(sorted(people.items()), desc="Identities"):
    if len(imgs) < MIN_PER_PERSON:
        continue

    random.shuffle(imgs)
    split = max(1, int(len(imgs) * (1 - VAL_FRAC)))

    train_imgs = imgs[:split]
    val_imgs   = imgs[split:]

    train_pdir = os.path.join(TRAIN_DIR, person)
    val_pdir   = os.path.join(VAL_DIR, person)
    os.makedirs(train_pdir, exist_ok=True)
    os.makedirs(val_pdir, exist_ok=True)

    for i, img in enumerate(train_imgs):
        Image.fromarray(img).save(
            os.path.join(train_pdir, f"{i:04d}.jpg"),
            quality=95
        )

    for i, img in enumerate(val_imgs):
        Image.fromarray(img).save(
            os.path.join(val_pdir, f"{i:04d}.jpg"),
            quality=95
        )

    kept += 1
    total_imgs += len(imgs)

print(f"[DONE] Kept identities: {kept}")
print(f"[DONE] Total images across kept identities: {total_imgs}")
print(f"[DONE] Output written to: {OUT_BASE}")

