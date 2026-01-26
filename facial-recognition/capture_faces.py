import os
import cv2
import depthai as dai

# ================= CONFIG =================
PERSON_LABEL = 0           # change to 1 for second person
FACE_SIZE = (120, 120)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "faces_dataset", str(PERSON_LABEL))
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= OAK-D CAMERA =================
def make_oakd_rgb_queue(preview_size=(640, 480), fps=30):
    pipeline = dai.Pipeline()

    # ColorCamera still works, but deprecated (warning is OK)
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(*preview_size)
    cam.setInterleaved(False)
    cam.setFps(fps)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # DepthAI v3: no explicit XLinkOut. Create queue directly from output.
    q = cam.preview.createOutputQueue(maxSize=4, blocking=False)

    # Start pipeline (replaces dai.Device(pipeline))
    pipeline.start()

    return pipeline, q

# ================= FACE DETECTOR =================
face_front = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


# ================= MAIN =================
pipeline, q_rgb = make_oakd_rgb_queue()

save_count = 0
last_face = None  # <-- KEY FIX

print(f"[INFO] Saving faces to: {SAVE_DIR}")
print("[INFO] Press 's' to save, 'q' to quit")
print("[INFO] CLICK THE VIDEO WINDOW BEFORE PRESSING KEYS")

while True:
    in_rgb = q_rgb.tryGet()
    if in_rgb is None:
        continue
    frame = in_rgb.getCvFrame()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------- FRONT FACE ----------
    faces_f = face_front.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    # ---------- SIDE PROFILE ----------
    faces_p = face_profile.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80)
    )

    # ---------- MIRRORED PROFILE ----------
    gray_flip = cv2.flip(gray, 1)
    faces_p_flip = face_profile.detectMultiScale(
        gray_flip, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80)
    )

    # Convert flipped coords back
    faces_pf = []
    for (x, y, w, h) in faces_p_flip:
        faces_pf.append((gray.shape[1] - x - w, y, w, h))

    # Combine all detections
    faces = list(faces_f) + list(faces_p) + faces_pf

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # take first detected face
        last_face = gray[y:y+h, x:x+w]
        last_face = cv2.resize(last_face, FACE_SIZE)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "FACE DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO FACE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Collect Faces (OAK-D)", frame)

    key = cv2.waitKey(1) & 0xFF

    # ===== SAVE FACE =====
    if key == ord('s'):
        if last_face is not None:
            out_path = os.path.join(SAVE_DIR, f"{save_count:04d}.jpg")
            cv2.imwrite(out_path, last_face)
            print("[SAVED]", out_path)
            save_count += 1
        else:
            print("[WARN] No face detected yet, nothing to save")

    # ===== QUIT =====
    if key == ord('q'):
        break

cv2.destroyAllWindows()
pipeline.stop()
print("[INFO] Done.")
