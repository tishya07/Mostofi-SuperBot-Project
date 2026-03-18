import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import depthai as dai
from pytorchvideo.models.hub import x3d_xs

# ---------- CONFIG ----------
MODEL_PATH   = os.path.expanduser("~/activity_detection/x3d_activity.pth")
CLASSES = ["falling", "running", "sitting", "walking"]
CLIP_FRAMES  = 5
FRAME_SIZE   = 182
CLASS_COLORS = {
    "falling": (0, 0, 255),
    "running": (0, 255, 0),
    "sitting": (255, 0, 255),   # Purple
    "walking": (255, 165, 0),
}

# ---------- LOAD MODEL ----------
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = x3d_xs(pretrained=False)
    model.blocks[5].proj = nn.Linear(
        model.blocks[5].proj.in_features, len(CLASSES)  # automatically 4 now
)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    print("Model loaded successfully")
    return model, device

# ---------- PREPROCESS ----------
def preprocess_frames(frames):
    clip = np.stack(frames).astype(np.float32) / 255.0
    mean = np.array([0.45, 0.45, 0.45])
    std  = np.array([0.225, 0.225, 0.225])
    clip = (clip - mean) / std
    clip = clip.transpose(3, 0, 1, 2)
    return torch.tensor(clip, dtype=torch.float32).unsqueeze(0)

# ---------- MAIN ----------
def main():
    model, device = load_model(MODEL_PATH)
    frame_buffer  = deque(maxlen=CLIP_FRAMES)
    current_label = "Initializing..."
    current_conf  = 0.0
    current_color = (255, 255, 255)
    inference_every = 15
    frame_count = 0

    # DepthAI v3 pipeline
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.Camera).build()
    preview = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p)
    q = preview.createOutputQueue()

    pipeline.start()
    print("Starting live activity detection. Press 'q' to quit.")

    while pipeline.isRunning():
        in_frame = q.tryGet()
        if in_frame is None:
            continue

        frame = in_frame.getCvFrame()
        frame_count += 1

        resized = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb)

        if len(frame_buffer) == CLIP_FRAMES and frame_count % inference_every == 0:
            with torch.no_grad():
                clip   = preprocess_frames(list(frame_buffer)).to(device)
                output = model(clip)
                probs  = torch.softmax(output, dim=1)[0]
                conf, pred = probs.max(dim=0)
                current_label = CLASSES[pred.item()]
                current_conf  = conf.item()
                current_color = CLASS_COLORS.get(current_label, (255, 255, 255))

        cv2.putText(
            frame,
            f"{current_label.upper()}  {current_conf:.2f}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            current_color,
            2,
        )
        cv2.imshow("SuperBots Activity Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Exited cleanly.")

if __name__ == "__main__":
    main()
