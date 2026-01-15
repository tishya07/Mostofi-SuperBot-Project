import os
import cv2
import depthai as dai

# ================= CONFIG =================
PERSON_LABEL = 0  # change to 1 for second person
FACE_SIZE = (120, 120)

# Get directory where THIS script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "faces_dataset", str(PERSON_LABEL))
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= OAK-D CAMERA =================
def make_oakd_rgb_queue(preview_size=(640, 480), fps=30):
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

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= MAIN =================
device, q_rgb = make_oakd_rgb_queue()
count = 0

print(f"[INFO] Saving face images to: {SAVE_DIR}")

while True:
    frame = q_rgb.get().getCvFrame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        face = gray[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face = cv2.resize(face, FACE_SIZE)

        cv2.putText(
            frame,
            "Press 's' to save face",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.imshow("Collect Faces (LIVE OAK-D)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('s') and len(faces) > 0:
        (x, y, w, h) = faces[0]  # save first detected face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, FACE_SIZE)

        out_path = os.path.join(SAVE_DIR, f"{count:04d}.jpg")
        cv2.imwrite(out_path, face)
        print("[SAVED]", out_path)
        count += 1

cv2.destroyAllWindows()
device.close()
