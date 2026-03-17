import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis
import depthai as dai

# ----------------------------
# Config
# ----------------------------
REGISTERED_FACES_PATH = "/home/superbots/facial-recognition/registered_faces.pkl"
THRESHOLD = 0.35

# ----------------------------
# Load InsightFace
# ----------------------------
print("Loading InsightFace model...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("✓ InsightFace loaded")

# ----------------------------
# Load Embeddings
# ----------------------------
with open(REGISTERED_FACES_PATH, 'rb') as f:
    data = pickle.load(f)
    registered_names = data['names']
    registered_embeddings = data['embeddings']
print(f"✓ Loaded {len(registered_names)} registered faces: {registered_names}")

# ----------------------------
# Helper Functions
# ----------------------------
def get_embedding(img):
    faces = app.get(img)
    if not faces:
        return None, None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    emb = face.embedding
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    bbox = face.bbox.astype(int)
    return emb, bbox

def match_face(test_emb):
    best_match, best_score = "Unknown", -1.0
    for name, ref_emb in zip(registered_names, registered_embeddings):
        score = float(np.dot(test_emb, ref_emb))
        if score > best_score:
            best_score, best_match = score, name
    if best_score < THRESHOLD:
        best_match = "Unknown"
    return best_match, best_score

# ----------------------------
# Setup OAK-D Camera
# ----------------------------
print("\nInitializing OAK-D camera...")
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.Camera).build()
preview = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.BGR888p, fps=30)
queue = preview.createOutputQueue()
pipeline.start()
print("✓ Camera initialized")

print("\n" + "="*60)
print("LIVE FACE RECOGNITION")
print("  'q' - Quit")
print("="*60 + "\n")

# ----------------------------
# Main Loop
# ----------------------------
try:
    while True:
        frame = queue.get().getCvFrame()

        emb, bbox = get_embedding(frame)

        if emb is not None and bbox is not None:
            name, score = match_face(emb)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({score*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Registered: {len(registered_names)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n✓ Interrupted")
finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    print("✓ Done")
