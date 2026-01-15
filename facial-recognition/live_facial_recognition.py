# live_oakd_face_demo.py
# Live OAK-D Lite feed -> Haar face detect -> LBPH recognize -> overlay label
# Press 'q' to quit

import os
import cv2
import numpy as np
import depthai as dai

# ---------- CONFIG ----------
# This will look for: ./faces_dataset/0/*.jpg and ./faces_dataset/1/*.jpg
DATASET_PATH = "./faces_dataset"

# LBPH expects consistent sizing for training and prediction
FACE_SIZE = (120, 120)

# Rename these to match your demo people
LABEL_NAMES = {0: "Person0", 1: "Person1"}

# LBPH: LOWER confidence = better match.
# If strangers get mislabeled -> LOWER this (e.g., 60-70)
# If it never recognizes anyone -> RAISE this (e.g., 90-120)
UNKNOWN_THRESHOLD = 80


# ---------- HELPERS ----------
def load_dataset(dataset_path: str):
    images, labels = [], []
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}\n"
            "Expected structure: faces_dataset/0/*.jpg and faces_dataset/1/*.jpg"
        )

    for label_dir in os.listdir(dataset_path):
        if label_dir.startswith("."):
            continue
        full_dir = os.path.join(dataset_path, label_dir)
        if not os.path.isdir(full_dir):
            continue

        for fname in os.listdir(full_dir):
            fpath = os.path.join(full_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, FACE_SIZE)
            images.append(img)
            labels.append(int(label_dir))

    return np.array(images), np.array(labels)


def get_name(label: int) -> str:
    return LABEL_NAMES.get(label, "Unknown")


def oakd_queue(preview_size=(640, 480), fps=30):
    """
    Creates a DepthAI pipeline and returns (device, output_queue)
    output_queue returns live frames from the OAK-D Lite.
    """
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setPreviewSize(*preview_size)
    cam.setInterleaved(False)
    cam.setFps(fps)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    device = dai.Device(pipeline)
    q = device.getOutputQueue("rgb", maxSize=4, blocking=True)
    return device, q


# ---------- MAIN ----------
if __name__ == "__main__":
    # 1) Load training data
    images, labels = load_dataset(DATASET_PATH)
    if len(images) < 10:
        raise RuntimeError(
            "Not enough training images.\n"
            "Collect ~30-60 images per person into faces_dataset/0 and faces_dataset/1."
        )

    # 2) Train LBPH recognizer (requires opencv-contrib)
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face not found. Install opencv-contrib-python:\n"
            "  pip3 install opencv-contrib-python"
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)

    # 3) Face detector (simple & demo-friendly)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade face detector.")

    # 4) Start LIVE OAK-D FEED
    device, q_rgb = oakd_queue(preview_size=(640, 480), fps=30)

    print("[INFO] Live OAK-D Face Demo started.")
    print("[INFO] Press 'q' to quit.")
    print("[INFO] Tune UNKNOWN_THRESHOLD if needed.")

    try:
        while True:
            frame = q_rgb.get().getCvFrame()  # LIVE frame from OAK-D
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                if face_roi.size == 0:
                    continue
                face_roi = cv2.resize(face_roi, FACE_SIZE)

                label, conf = recognizer.predict(face_roi)

                name = "Unknown" if conf > UNKNOWN_THRESHOLD else get_name(label)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{name}  LBPH:{conf:.1f}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("LIVE OAK-D Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        device.close()
        print("[INFO] Exited cleanly.")
