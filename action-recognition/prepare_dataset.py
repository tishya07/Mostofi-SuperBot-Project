import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.expanduser("~/activity_dataset")
CLIP_LEN     = 5       # frames per clip fed to X3D-XS
FRAME_SIZE   = 182     # X3D-XS native input size
BATCH_SIZE   = 4       # keep small for Jetson memory
NUM_WORKERS  = 2

CLASSES = ["falling", "running", "sitting", "walking"]   # sorted = label 0,1,2
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45],
                         std=[0.225, 0.225, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45],
                         std=[0.225, 0.225, 0.225]),
])

# ─── DATASET ──────────────────────────────────────────────────────────────────
class ActivityDataset(Dataset):
    """
    Loads .mp4 clips from:
        DATASET_DIR/
            train/
                walking/  *.mp4
                running/  *.mp4
                falling/  *.mp4
            val/
                ...

    Returns (clip_tensor, label) where clip_tensor is (C, T, H, W).
    """

    def __init__(self, root, split="train", transform=None, clip_len=CLIP_LEN):
        self.clip_len  = clip_len
        self.transform = transform
        self.samples   = []

        split_dir = os.path.join(root, split)
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"  [WARN] Missing folder: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith(".mp4"):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), CLASS_TO_IDX[cls])
                    )

        print(f"[{split}] {len(self.samples)} clips across {len(CLASSES)} classes")

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < self.clip_len:
            indices = list(range(total)) + [total - 1] * (self.clip_len - total)
        else:
            indices = np.linspace(0, total - 1, self.clip_len, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                frames.append(frames[-1] if frames else np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_frames(path)
        tensors = [self.transform(f) for f in frames]
        clip    = torch.stack(tensors, dim=1)
        return clip, label


# ─── DATALOADERS ──────────────────────────────────────────────────────────────
def get_dataloaders(dataset_dir=DATASET_DIR,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS):

    train_ds = ActivityDataset(dataset_dir, split="train", transform=train_transform)
    val_ds   = ActivityDataset(dataset_dir, split="val",   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader


# ─── QUICK SANITY CHECK ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders()

    clips, labels = next(iter(train_loader))
    print(f"Clip batch shape : {clips.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Label values     : {labels.tolist()}")
    print(f"Class mapping    : {CLASS_TO_IDX}")
    print("Dataset looks good!")
