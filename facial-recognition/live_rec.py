import cv2
import numpy as np
import pickle
import depthai as dai
import onnxruntime as ort

sess = ort.InferenceSession("arcface_embedding.onnx")

# ----------------------------
# Config
# ----------------------------
REGISTERED_FACES_PATH = "registered_faces.pkl"
THRESHOLD = 0.6

# ----------------------------
# Load SSD Face Detector
# ----------------------------
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# ----------------------------
# Load Embeddings
# ----------------------------
with open(REGISTERED_FACES_PATH, "rb") as f:
    data = pickle.load(f)
    registered_names = data['names']
    registered_embeddings = data['embeddings']
print(f"✓ Loaded {len(registered_names)} registered faces")

# ----------------------------
# Helper Functions
# ----------------------------
def detect_all_faces(img, confidence_threshold=0.5):
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
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            faces.append((x1, y1, x2, y2, confidence))
    return faces

def get_embedding_simple(face_img):
    """Lightweight embedding using histogram (no model needed on host)"""
    if face_img.size == 0:
        return None
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (112, 112))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    emb = sess.run(None, {"image": face_img})[0][0]
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb

def match_face(embedding):
    best_match, best_score = "Unknown", -1.0
    for name, stored_emb in zip(registered_names, registered_embeddings):
        score = float(np.dot(embedding, stored_emb))
        if score > best_score:
            best_score, best_match = score, name
    return (best_match if best_score > THRESHOLD else "Unknown"), best_score

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
print("LIVE FACE RECOGNITION - CONTROLS:")
print("  'q' - Quit")
print("="*60 + "\n")

# ----------------------------
# Main Loop
# ----------------------------
try:
    while True:
        frame = queue.get().getCvFrame()
        faces = detect_all_faces(frame, confidence_threshold=0.5)

        for (x1, y1, x2, y2, conf) in faces:
            face_roi = frame[y1:y2, x1:x2]
            embedding = get_embedding_simple(face_roi)
            if embedding is None:
                continue

            name, score = match_face(embedding)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Registered: {len(registered_names)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n✓ Interrupted by user")

finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    print("✓ Done\n")
