"""
context_inferencer.py

Combines:
  - YOLOv8 spatial detection (DepthAI blob) for objects
  - InsightFace for facial recognition against registered embeddings
  - LSTM (custom fine-tuned) for activity classification

Outputs per-person natural language sentences, e.g.:
  "Alice is sitting on chair"
  "Bob is standing with Alice"
  "Unknown is holding cell phone"
"""

import os
import cv2
import json
import time
import uuid
import pickle
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from collections import deque, Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import onnxruntime  # noqa: F401 — must import before depthai
from ultralytics import YOLO
import depthai as dai
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

REGISTERED_FACES_PATH = "/home/vis-navteam/facial-recognition/registered_faces.pkl"
YOLO_BLOB_PATH        = "/home/vis-navteam/object-detection/yolov8n_coco_640x352_openvino_2022_1_6shave.blob"
LOG_PATH              = "logs/context_inference_log.jsonl"

FACE_THRESHOLD  = 0.35
OBJ_THRESHOLD   = 0.50

WINDOW_SIZE       = 15   # was 30 — halves scene graph lag
INFERENCE_EVERY_N = 5    # was 10 — infer context twice as often

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

LSTM_MODEL_PATH        = os.path.expanduser("~/activity_detection/lstm_v3_activity.pth")
LSTM_CLASSES           = ["falling", "running", "sitting", "walking"]
LSTM_WINDOW            = 16   # was 20 — fewer frames needed before first prediction
LSTM_FEATURES          = 92
LSTM_HIDDEN            = 128
LSTM_LAYERS            = 1
LSTM_IMG_SIZE          = 640
LSTM_CONF_THRESHOLD    = 0.75
LSTM_UNKNOWN_THRESHOLD = 0.60
LSTM_VOTE_WINDOW       = 6    # was 8 — shorter voting window
LSTM_MAJORITY_NEEDED   = 4    # was 4 — easier majority
LSTM_STABLE_SECONDS    = 1.5  # was 3.0 — clear stale labels faster
LSTM_KP_CONF_THRESH    = 0.3

LSTM_HIP_L, LSTM_HIP_R           = 11, 12
LSTM_KNEE_L, LSTM_KNEE_R         = 13, 14
LSTM_ANKLE_L, LSTM_ANKLE_R       = 15, 16
LSTM_SHOULDER_L, LSTM_SHOULDER_R = 5, 6
LSTM_ELBOW_L, LSTM_ELBOW_R       = 7, 8
LSTM_WRIST_L, LSTM_WRIST_R       = 9, 10

person_states = {}

# ---------------------------------------------------------------------------
# LSTM MODEL
# ---------------------------------------------------------------------------

class ActivityLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(LSTM_FEATURES, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, len(LSTM_CLASSES))
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x.reshape(B * T, -1)).reshape(B, T, -1)
        out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(out), dim=1)
        context = (attn * out).sum(dim=1)
        return self.classifier(context)


def clean_keypoints(kps, conf, prev_kps=None):
    cleaned = kps.copy()
    for i in range(17):
        if conf[i] < LSTM_KP_CONF_THRESH:
            if prev_kps is not None and (prev_kps[i, 0] > 0 or prev_kps[i, 1] > 0):
                cleaned[i] = prev_kps[i]
    return cleaned


def compute_angle(a, b, c):
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle)) / 180.0


def extract_features(kps):
    hip_valid = kps[LSTM_HIP_L, 0] > 0 or kps[LSTM_HIP_L, 1] > 0
    if hip_valid:
        hip_center = (kps[LSTM_HIP_L] + kps[LSTM_HIP_R]) / 2.0
    else:
        hip_center = (kps[LSTM_SHOULDER_L] + kps[LSTM_SHOULDER_R]) / 2.0

    shoulder_center = (kps[LSTM_SHOULDER_L] + kps[LSTM_SHOULDER_R]) / 2.0
    torso_height    = np.linalg.norm(shoulder_center - hip_center) + 1e-6

    rel_kps   = (kps - hip_center) / torso_height
    positions = rel_kps.flatten()  # 34

    angles = np.array([
        compute_angle(kps[LSTM_HIP_L],      kps[LSTM_KNEE_L],     kps[LSTM_ANKLE_L]),
        compute_angle(kps[LSTM_HIP_R],      kps[LSTM_KNEE_R],     kps[LSTM_ANKLE_R]),
        compute_angle(kps[LSTM_SHOULDER_L], kps[LSTM_HIP_L],      kps[LSTM_KNEE_L]),
        compute_angle(kps[LSTM_SHOULDER_R], kps[LSTM_HIP_R],      kps[LSTM_KNEE_R]),
        compute_angle(kps[LSTM_SHOULDER_L], kps[LSTM_ELBOW_L],    kps[LSTM_WRIST_L]),
        compute_angle(kps[LSTM_SHOULDER_R], kps[LSTM_ELBOW_R],    kps[LSTM_WRIST_R]),
        compute_angle(kps[LSTM_ELBOW_L],    kps[LSTM_SHOULDER_L], kps[LSTM_HIP_L]),
        compute_angle(kps[LSTM_ELBOW_R],    kps[LSTM_SHOULDER_R], kps[LSTM_HIP_R]),
        compute_angle(kps[LSTM_KNEE_L],     kps[LSTM_ANKLE_L],    kps[LSTM_HIP_L]),
        compute_angle(kps[LSTM_KNEE_R],     kps[LSTM_ANKLE_R],    kps[LSTM_HIP_R]),
        compute_angle(kps[LSTM_SHOULDER_L], kps[LSTM_SHOULDER_R], kps[LSTM_HIP_R]),
        compute_angle(kps[LSTM_HIP_L],      kps[LSTM_HIP_R],      kps[LSTM_KNEE_R]),
    ])  # 12

    return np.concatenate([positions, angles])  # 46

# ---------------------------------------------------------------------------
# 1. DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def distance_to(self, other: "BoundingBox") -> float:
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class ObjectDetection:
    label: str
    confidence: float
    bbox: BoundingBox


@dataclass
class FaceDetection:
    identity: str
    confidence: float
    bbox: BoundingBox
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ActivityDetection:
    action: str
    confidence: float


@dataclass
class FrameDetections:
    frame_id: int
    timestamp: float
    objects: list[ObjectDetection]
    faces: list[FaceDetection]
    activities: list[ActivityDetection]
    frame_hw: tuple[int, int]


# ---------------------------------------------------------------------------
# 2. MODEL LOADERS
# ---------------------------------------------------------------------------

def load_lstm(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ActivityLSTM().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[LSTM] Loaded on {device}")
    return model, device


def load_insightface(registered_faces_path: str):
    # GPU provider selection
    available = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "gpu_mem_limit": 1 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider",
        ]
        ctx_id = 0
        print("[InsightFace] Using CUDAExecutionProvider")
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        print("[InsightFace] Using CPUExecutionProvider")

    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(320, 320))

    with open(registered_faces_path, "rb") as f:
        data = pickle.load(f)
    names = data["names"]
    raw_embeddings = data["embeddings"]

    # Pre-normalize embedding matrix for vectorized matching
    emb_matrix = np.array(raw_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12
    emb_matrix = emb_matrix / norms

    print(f"[InsightFace] Loaded {len(names)} registered faces: {names}")
    return app, names, emb_matrix


def build_depthai_pipeline(blob_path: str):
    """DepthAI 2.x pipeline — returns device + queues."""
    pipeline = dai.Pipeline()

    # Camera
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 352)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    # RGB output stream
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    # YOLO neural network
    yolo = pipeline.create(dai.node.NeuralNetwork)
    yolo.setBlobPath(blob_path)
    yolo.setNumInferenceThreads(2)
    yolo.setNumNCEPerInferenceThread(1)
    yolo.input.setBlocking(False)   # drop frames if YOLO is busy
    cam.preview.link(yolo.input)

    # Detection output stream
    xout_det = pipeline.create(dai.node.XLinkOut)
    xout_det.setStreamName("det")
    yolo.out.link(xout_det.input)

    device = dai.Device(pipeline)
    rgb_q  = device.getOutputQueue("rgb", maxSize=1, blocking=False)
    det_q  = device.getOutputQueue("det", maxSize=1, blocking=False)

    return device, rgb_q, det_q


# ---------------------------------------------------------------------------
# 3. DETECTION COLLECTOR
# ---------------------------------------------------------------------------

class DetectionCollector:
    def __init__(
        self,
        lstm_model,
        lstm_device,
        insightface_app,
        pose_model,
        registered_names: list,
        registered_emb_matrix: np.ndarray,   # pre-normalized (N, 512)
    ):
        self.lstm_model        = lstm_model
        self.lstm_device       = lstm_device
        self.face_app          = insightface_app
        self.pose_model        = pose_model
        self.registered_names  = registered_names
        self.emb_matrix        = registered_emb_matrix  # vectorized matching

        self._frame_id = 0

    # ── YOLO via DepthAI ──────────────────────────────────────────────────

    def _parse_yolo(self, nn_data) -> list[ObjectDetection]:
        results = []
        try:
            raw   = nn_data.getLayerFp16(nn_data.getAllLayerNames()[0])
            layer = np.array(raw).reshape(-1)
            n     = len(layer) // 6
            for i in range(n):
                x1, y1, x2, y2, conf, cls_id = layer[i*6:(i+1)*6]
                if conf < OBJ_THRESHOLD:
                    continue
                label = COCO_LABELS[int(cls_id)] if int(cls_id) < len(COCO_LABELS) else str(int(cls_id))
                results.append(ObjectDetection(
                    label=label,
                    confidence=float(conf),
                    bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                ))
        except Exception as e:
            print(f"[YOLO parse error] {e}")
        return results

    # ── InsightFace ───────────────────────────────────────────────────────

    def _run_insightface(self, frame: np.ndarray) -> list[FaceDetection]:
        h, w = frame.shape[:2]
        faces_raw = self.face_app.get(frame)
        results = []
        for face in faces_raw:
            if float(face.det_score) < 0.5:
                continue
            emb = face.embedding
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            identity, score = self._match_face(emb)
            x1, y1, x2, y2 = face.bbox.astype(float)
            results.append(FaceDetection(
                identity=identity,
                confidence=score,
                bbox=BoundingBox(x1/w, y1/h, x2/w, y2/h),
                embedding=emb,
            ))
        return results

    def _match_face(self, test_emb: np.ndarray) -> tuple[str, float]:
        # Vectorized matmul — no Python loop
        scores     = self.emb_matrix @ test_emb
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        name = self.registered_names[best_idx] if best_score >= FACE_THRESHOLD else "Unknown"
        return name, best_score

    # ── LSTM ──────────────────────────────────────────────────────────────

    def _run_lstm(self, frame: np.ndarray) -> list[ActivityDetection]:
        h, w    = frame.shape[:2]
        results = self.pose_model(frame, verbose=False, imgsz=LSTM_IMG_SIZE)

        detected_ids = set()
        # FIX: use self.lstm_device not bare lstm_device (was NameError)
        probs = torch.zeros(len(LSTM_CLASSES), device=self.lstm_device)

        if (results[0].keypoints is not None and
            len(results[0].keypoints.xy) > 0 and
            results[0].boxes is not None and
            results[0].keypoints.conf is not None):

            kps_all  = results[0].keypoints.xy.cpu().numpy()
            conf_all = results[0].keypoints.conf.cpu().numpy()
            boxes    = results[0].boxes.xyxy.cpu().numpy()

            for person_id, (kps_raw, kp_conf, box) in enumerate(
                    zip(kps_all, conf_all, boxes)):

                detected_ids.add(person_id)

                if person_id not in person_states:
                    person_states[person_id] = {
                        "buffer":        deque(maxlen=LSTM_WINDOW),
                        "prev_kps":      None,
                        "prev_features": None,
                        "vote_history":  deque(maxlen=LSTM_VOTE_WINDOW),
                        "label":         "...",
                        "conf":          0.0,
                        "last_above":    time.time(),
                    }

                state = person_states[person_id]

                kps = kps_raw.copy()
                kps[:, 0] /= w
                kps[:, 1] /= h

                kps_clean         = clean_keypoints(kps, kp_conf, state["prev_kps"])
                state["prev_kps"] = kps_clean.copy()

                features = extract_features(kps_clean)

                velocity = (features - state["prev_features"]
                            if state["prev_features"] is not None
                            else np.zeros(46))
                state["prev_features"] = features.copy()

                full_features = np.concatenate([features, velocity])
                state["buffer"].append(full_features)

                if len(state["buffer"]) == LSTM_WINDOW:
                    seq = np.array(state["buffer"], dtype=np.float32)
                    # FIX: use torch.from_numpy (avoids extra CPU copy)
                    x = torch.from_numpy(seq).unsqueeze(0).to(self.lstm_device, non_blocking=True)

                    with torch.no_grad():
                        out   = self.lstm_model(x)
                        probs = torch.softmax(out, dim=1)[0]
                        conf, pred = probs.max(dim=0)
                        conf = conf.item()
                        pred = pred.item()

                    if conf >= LSTM_CONF_THRESHOLD:
                        state["vote_history"].append((pred, conf))
                        state["last_above"] = time.time()

                    if len(state["vote_history"]) >= LSTM_MAJORITY_NEEDED:
                        counts = Counter(p for p, c in state["vote_history"])
                        top_pred, top_count = counts.most_common(1)[0]
                        if top_count >= LSTM_MAJORITY_NEEDED:
                            avg_conf       = np.mean([c for p, c in state["vote_history"] if p == top_pred])
                            state["label"] = LSTM_CLASSES[top_pred]
                            state["conf"]  = avg_conf

                if time.time() - state["last_above"] > LSTM_STABLE_SECONDS:
                    state["label"] = "..."
                    state["conf"]  = 0.0
                    state["vote_history"].clear()

                if state["conf"] < LSTM_UNKNOWN_THRESHOLD and state["label"] not in ("...",):
                    state["label"] = "UNKNOWN"
                    state["conf"]  = 0.0

            for pid in list(person_states.keys()):
                if pid not in detected_ids:
                    del person_states[pid]

        return [
            ActivityDetection(action=LSTM_CLASSES[i], confidence=float(c))
            for i, c in enumerate(probs.tolist())
            if c >= LSTM_UNKNOWN_THRESHOLD
        ]

    # ── Combined ──────────────────────────────────────────────────────────

    def collect(self, frame: np.ndarray, nn_data) -> FrameDetections:
        h, w = frame.shape[:2]
        self._frame_id += 1
        return FrameDetections(
            frame_id=self._frame_id,
            timestamp=time.time(),
            objects=self._parse_yolo(nn_data),
            faces=self._run_insightface(frame),
            activities=self._run_lstm(frame),
            frame_hw=(h, w),
        )


# ---------------------------------------------------------------------------
# 4. SCENE GRAPH
# ---------------------------------------------------------------------------

@dataclass
class SceneGraph:
    frame_id: int
    timestamp: float
    objects: list[ObjectDetection]
    faces: list[FaceDetection]
    activities: list[ActivityDetection]
    object_labels: set            = field(default_factory=set)
    identities_present: set       = field(default_factory=set)
    action_labels: set            = field(default_factory=set)
    person_object_proximity: dict = field(default_factory=dict)
    person_person_proximity: dict = field(default_factory=dict)


def build_scene_graph(det: FrameDetections) -> SceneGraph:
    g = SceneGraph(
        frame_id=det.frame_id,
        timestamp=det.timestamp,
        objects=det.objects,
        faces=det.faces,
        activities=det.activities,
    )
    g.object_labels      = {o.label for o in det.objects}
    g.identities_present = {f.identity for f in det.faces}
    g.action_labels      = {a.action for a in det.activities}

    for face in det.faces:
        for obj in det.objects:
            key  = (face.identity, obj.label)
            dist = face.bbox.distance_to(obj.bbox)
            if key not in g.person_object_proximity or dist < g.person_object_proximity[key]:
                g.person_object_proximity[key] = round(dist, 4)

    for i, fa in enumerate(det.faces):
        for fb in det.faces[i+1:]:
            key = tuple(sorted([fa.identity, fb.identity]))
            g.person_person_proximity[key] = round(fa.bbox.distance_to(fb.bbox), 4)

    return g


# ---------------------------------------------------------------------------
# 5. SLIDING WINDOW BUFFER
# ---------------------------------------------------------------------------

class SlidingWindowBuffer:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._buffer: deque[SceneGraph] = deque(maxlen=window_size)

    def push(self, graph: SceneGraph):
        self._buffer.append(graph)

    @property
    def frames(self) -> list[SceneGraph]:
        return list(self._buffer)

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) >= max(1, self.window_size // 2)

    def aggregate_features(self) -> dict:
        frames = self.frames
        if not frames:
            return {}
        n = len(frames)

        label_counts = defaultdict(int)
        for g in frames:
            for lbl in g.object_labels:
                label_counts[lbl] += 1
        object_presence_rate = {k: v / n for k, v in label_counts.items()}

        id_counts = defaultdict(int)
        for g in frames:
            for ident in g.identities_present:
                id_counts[ident] += 1
        identity_presence_rate = {k: v / n for k, v in id_counts.items()}

        act_counts = defaultdict(int)
        for g in frames:
            for act in g.action_labels:
                act_counts[act] += 1
        action_presence_rate = {k: v / n for k, v in act_counts.items()}

        prox_accum = defaultdict(list)
        for g in frames:
            for (ident, obj), dist in g.person_object_proximity.items():
                prox_accum[(ident, obj)].append(dist)
        mean_proximity = {
            f"{ident}__{obj}": round(sum(v) / len(v), 4)
            for (ident, obj), v in prox_accum.items()
        }

        pp_accum = defaultdict(list)
        for g in frames:
            for pair, dist in g.person_person_proximity.items():
                pp_accum[pair].append(dist)
        mean_person_proximity = {
            f"{a}__{b}": round(sum(v) / len(v), 4)
            for (a, b), v in pp_accum.items()
        }

        return {
            "object_presence_rate":   object_presence_rate,
            "identity_presence_rate": identity_presence_rate,
            "action_presence_rate":   action_presence_rate,
            "mean_proximity":         mean_proximity,
            "mean_person_proximity":  mean_person_proximity,
            "all_identities":         list(id_counts.keys()),
            "n_frames":               n,
            "timestamp":              frames[-1].timestamp,
        }


# ---------------------------------------------------------------------------
# 6. CONTEXT INFERENCER
# ---------------------------------------------------------------------------

@dataclass
class PersonSentence:
    identity: str
    sentence: str
    confidence: float
    evidence: list[str]


@dataclass
class ContextResult:
    sentences: list[PersonSentence]
    inferencer_type: str
    timestamp: float = field(default_factory=time.time)

    def display_lines(self) -> list[str]:
        return [s.sentence for s in self.sentences] or ["No persons detected"]


class ContextInferencer(ABC):
    @abstractmethod
    def infer(self, features: dict) -> ContextResult:
        ...


# ---------------------------------------------------------------------------
# 7. RULE-BASED INFERENCER
# ---------------------------------------------------------------------------

class RuleBasedInferencer(ContextInferencer):
    PRESENT   = 0.5
    SOMETIMES = 0.25
    NEAR      = 0.20
    TOGETHER  = 0.25

    def infer(self, features: dict) -> ContextResult:
        sentences = [
            PersonSentence(*self._infer_person(ident, features))
            for ident in features.get("all_identities", [])
        ]
        return ContextResult(sentences=sentences, inferencer_type="rule_based",
                             timestamp=features.get("timestamp", time.time()))

    def _act(self, f, action):
        return f.get("action_presence_rate", {}).get(action, 0.0)

    def _obj(self, f, label):
        return f.get("object_presence_rate", {}).get(label, 0.0)

    def _prox(self, f, identity, obj_label):
        return f.get("mean_proximity", {}).get(f"{identity}__{obj_label}", 999.0)

    def _person_prox(self, f, id_a, id_b):
        pp = f.get("mean_person_proximity", {})
        return min(
            pp.get(f"{id_a}__{id_b}", 999.0),
            pp.get(f"{id_b}__{id_a}", 999.0),
        )

    def _nearest_other(self, f, identity):
        others = [i for i in f.get("all_identities", []) if i != identity]
        best, best_dist = None, 999.0
        for other in others:
            dist = self._person_prox(f, identity, other)
            if dist < best_dist:
                best_dist, best = dist, other
        return best, best_dist

    def _infer_person(self, identity: str, f: dict) -> tuple[str, str, float, list]:
        name = identity

        if self._act(f, "falling") > self.SOMETIMES:
            return identity, f"{name} is falling", 0.92, ["falling detected by LSTM"]
        if self._act(f, "running") > self.PRESENT:
            return identity, f"{name} is running", 0.88, ["running detected by LSTM"]
        if (self._obj(f, "cell phone") > self.SOMETIMES and
                self._prox(f, identity, "cell phone") < self.NEAR):
            return identity, f"{name} is holding cell phone", 0.85, ["cell phone nearby"]
        if (self._obj(f, "laptop") > self.SOMETIMES and
                self._prox(f, identity, "laptop") < self.NEAR):
            return identity, f"{name} is working on laptop", 0.85, ["laptop nearby"]
        if (self._act(f, "sitting") > self.PRESENT and
                self._obj(f, "chair") > self.SOMETIMES and
                self._prox(f, identity, "chair") < self.NEAR):
            return identity, f"{name} is sitting on chair", 0.87, ["sitting + chair nearby"]
        if self._act(f, "sitting") > self.PRESENT:
            return identity, f"{name} is sitting", 0.75, ["sitting detected by LSTM"]
        other, dist = self._nearest_other(f, identity)
        if other is not None and dist < self.TOGETHER:
            return identity, f"{name} is standing with {other}", 0.78, [f"near {other} ({dist:.2f})"]
        if self._act(f, "walking") > self.PRESENT:
            return identity, f"{name} is walking", 0.82, ["walking detected by LSTM"]

        return identity, f"{name} is present", 0.40, ["no strong rule matched"]


# ---------------------------------------------------------------------------
# 8. LEARNED INFERENCER — placeholder
# ---------------------------------------------------------------------------

class LearnedInferencer(ContextInferencer):
    def __init__(self, model_path: str):
        raise NotImplementedError("Train on JSONL logs first, then implement.")

    def infer(self, features: dict) -> ContextResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 9. CONTEXT LOGGER
# ---------------------------------------------------------------------------

class ContextLogger:
    def __init__(self, log_path: str = LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, features: dict, result: ContextResult):
        record = {
            "id":              str(uuid.uuid4()),
            "timestamp":       features.get("timestamp", time.time()),
            "inferencer_type": result.inferencer_type,
            "labels": [
                {"identity": s.identity, "sentence": s.sentence, "confidence": s.confidence}
                for s in result.sentences
            ],
            "features": {
                "object_presence_rate":   features.get("object_presence_rate", {}),
                "action_presence_rate":   features.get("action_presence_rate", {}),
                "identity_presence_rate": features.get("identity_presence_rate", {}),
                "mean_proximity":         features.get("mean_proximity", {}),
                "mean_person_proximity":  features.get("mean_person_proximity", {}),
                "all_identities":         features.get("all_identities", []),
                "n_frames":               features.get("n_frames", 0),
            },
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# 10. MAIN PIPELINE
# ---------------------------------------------------------------------------

class ContextPipeline:
    def __init__(
        self,
        collector: DetectionCollector,
        inferencer: Optional[ContextInferencer] = None,
        log_path: str = LOG_PATH,
        window_size: int = WINDOW_SIZE,
        inference_every_n: int = INFERENCE_EVERY_N,
    ):
        self.collector         = collector
        self.buffer            = SlidingWindowBuffer(window_size)
        self.inferencer        = inferencer or RuleBasedInferencer()
        self.logger            = ContextLogger(log_path)
        self.inference_every_n = inference_every_n
        self._frame_count      = 0
        self._last_result: Optional[ContextResult] = None

    def process_frame(self, frame: np.ndarray, dai_detections) -> Optional[ContextResult]:
        t0 = time.perf_counter()
        self._frame_count += 1

        t_collect = time.perf_counter()
        det = self.collector.collect(frame, dai_detections)
        t_collect = (time.perf_counter() - t_collect) * 1000

        graph = build_scene_graph(det)
        self.buffer.push(graph)

        if self._frame_count % self.inference_every_n != 0:
            return self._last_result
        if not self.buffer.is_ready:
            return None

        t_infer = time.perf_counter()
        features = self.buffer.aggregate_features()
        result   = self.inferencer.infer(features)
        t_infer  = (time.perf_counter() - t_infer) * 1000

        self.logger.log(features, result)
        self._last_result = result

        t_total = (time.perf_counter() - t0) * 1000
        result._latency = {"collect_ms": t_collect, "infer_ms": t_infer, "total_ms": t_total}
        return result

    def run(self, display: bool = True):
        # FIX: use device from build_depthai_pipeline, not dai_pipeline
        dai_device, rgb_q, det_q = build_depthai_pipeline(YOLO_BLOB_PATH)

        print(f"[ContextPipeline] Running on OAK-D. Logs → {self.logger.log_path}")
        print("Press 'q' or ESC to quit.")

        try:
            frame_times = []
            while True:
                t_loop = time.perf_counter()

                in_rgb = rgb_q.tryGet()
                in_det = det_q.tryGet()

                if in_rgb is None or in_det is None:
                    continue

                frame  = in_rgb.getCvFrame()
                result = self.process_frame(frame, in_det)

                if result is not None:
                    for line in result.display_lines():
                        print(f"[{result.inferencer_type}] {line}")
                    lat = getattr(result, "_latency", None)
                    if lat:
                        print(f"  latency → collect:{lat['collect_ms']:.1f}ms  "
                              f"infer:{lat['infer_ms']:.1f}ms  "
                              f"total:{lat['total_ms']:.1f}ms")

                # FPS
                frame_times.append(time.perf_counter() - t_loop)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

                if display:
                    overlay = frame.copy()
                    y = 30
                    lines = result.display_lines() if result else ["Initializing..."]
                    for line in lines:
                        color = (0, 255, 0) if result else (200, 200, 200)
                        cv2.putText(overlay, line, (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                        y += 32

                    # Latency overlay
                    lat = getattr(result, "_latency", None) if result else None
                    if lat:
                        cv2.putText(overlay,
                                    f"collect:{lat['collect_ms']:.0f}ms  infer:{lat['infer_ms']:.0f}ms  total:{lat['total_ms']:.0f}ms",
                                    (10, overlay.shape[0] - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(overlay, f"FPS:{fps:.1f}", (10, overlay.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow("Context Inference", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        print("\n[ContextPipeline] Quit.")
                        break

        finally:
            if display:
                cv2.destroyAllWindows()
            dai_device.close()  # FIX: close device, not pipeline.stop()
            print("[ContextPipeline] Stopped.")


# ---------------------------------------------------------------------------
# 11. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # let CUDA optimize kernel selection

    lstm_model, lstm_device = load_lstm(LSTM_MODEL_PATH)
    print(f"LSTM using device: {lstm_device}")

    yolo_pose_model = YOLO("yolov8n-pose.pt")
    yolo_pose_model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("YOLOv8 Pose loaded")

    # FIX: load_insightface now returns emb_matrix (pre-normalized), not raw embeddings list
    face_app, reg_names, emb_matrix = load_insightface(REGISTERED_FACES_PATH)

    collector = DetectionCollector(
        lstm_model=lstm_model,
        lstm_device=lstm_device,
        insightface_app=face_app,
        pose_model=yolo_pose_model,
        registered_names=reg_names,
        registered_emb_matrix=emb_matrix,  # FIX: renamed param
    )

    pipeline = ContextPipeline(
        collector=collector,
        inferencer=RuleBasedInferencer(),
        log_path=LOG_PATH,
        window_size=WINDOW_SIZE,
        inference_every_n=INFERENCE_EVERY_N,
    )

    pipeline.run(display=True)
