import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.models.hub import x3d_xs
import cv2
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# ---------- CONFIG ----------
DATASET_DIR = os.path.expanduser("~/activity_dataset")
CLASSES = ["falling", "running", "sitting", "walking"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

CLIP_FRAMES = 5       # X3D-XS expects 5 frames
FRAME_SIZE = 182      # X3D-XS input size
BATCH_SIZE = 2        # Reduced for Jetson memory
EPOCHS = 15
LR = 1e-4
SAVE_PATH = os.path.expanduser("~/activity_detection/x3d_activity.pth")

# ---------- DATASET ----------
class VideoClipDataset(Dataset):
    def __init__(self, root, split="train"):
        self.samples = []
        split_dir = os.path.join(root, split)
        for label in CLASSES:
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.endswith(".mp4"):
                    self.samples.append((
                        os.path.join(label_dir, fname),
                        CLASS_TO_IDX[label]
                    ))
        print(f"[{split}] Loaded {len(self.samples)} clips")

    def __len__(self):
        return len(self.samples)

    def load_clip(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < CLIP_FRAMES:
            indices = list(range(total)) + [total - 1] * (CLIP_FRAMES - total)
        else:
            indices = np.linspace(0, total - 1, CLIP_FRAMES, dtype=int)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Stack and normalize: (T, H, W, C) -> (C, T, H, W)
        clip = np.stack(frames).astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45])
        std  = np.array([0.225, 0.225, 0.225])
        clip = (clip - mean) / std
        clip = clip.transpose(3, 0, 1, 2)  # (C, T, H, W)
        return torch.tensor(clip, dtype=torch.float32)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = self.load_clip(path)
        return clip, label


# ---------- MAIN ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clear any leftover GPU memory
    torch.cuda.empty_cache()

    # Datasets
    train_dataset = VideoClipDataset(DATASET_DIR, "train")
    val_dataset   = VideoClipDataset(DATASET_DIR, "val")

    # num_workers=0 to avoid memory issues on Jetson
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("Loading X3D-XS pretrained model...")
    model = x3d_xs(pretrained=True)

    # Freeze ALL pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classification head with NUM_CLASSES outputs
    model.blocks[5].proj = nn.Linear(
        model.blocks[5].proj.in_features, NUM_CLASSES
    )

    # Only train the new head
    for param in model.blocks[5].proj.parameters():
        param.requires_grad = True

    torch.cuda.empty_cache()
    model = model.to(device)

    # Loss, optimizer, scheduler - only pass trainable params
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for clips, labels in train_loader:
            clips  = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # --- VAL ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips  = clips.to(device)
                labels = labels.to(device)
                outputs = model(clips)
                preds   = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100
        avg_loss  = train_loss    / len(train_loader)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": CLASSES,
                "class_to_idx": CLASS_TO_IDX,
            }, SAVE_PATH)
            print(f"  --> Saved best model (val_acc={val_acc:.1f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
