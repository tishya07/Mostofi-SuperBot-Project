"""
context_inferencer.py

Combines:
  - YOLOv8 spatial detection (DepthAI blob) for objects
  - InsightFace for facial recognition against registered embeddings
  - LSTM (custom fine-tuned) for activity classification
  - StereoDepth + SpatialLocationCalculator for real-world XYZ coordinates
  - Interactive search mode: find a person, object, or action by name

Outputs per-person natural language sentences with bounding boxes + depth, e.g.:
  "Alice is sitting on chair [(1.2m, 0.1m, 2.5m)]"
  "Bob is standing with Alice"
  "Unknown is holding cell phone"

GPU optimizations:
  - extract_features / compute_angle run on GPU (torch tensors end-to-end)
  - LSTM sequence buffer stored as GPU tensors — no from_numpy().to(device) per frame
  - InsightFace embedding matching via torch.mv() on GPU
  - Eliminated 3x .cpu().numpy() round-trips per person per frame
  - InsightFace ONNX detection still on CPU (no torch backend available), cached every N frames
  - TF32 tensor cores enabled for faster matmul
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

start_time = time.time()

IMG_SIZE = (640, 352)
REGISTERED_FACES_PATH = "/home/vis-navteam/facial-recognition/registered_faces.pkl"
YOLO_BLOB_PATH = "/home/vis-navteam/object-detection/yolov8n_coco_640x352_openvino_2022_1_6shave.blob"
LOG_PATH = "logs/context_inference_log.jsonl"

FACE_THRESHOLD = 0.35
OBJ_THRESHOLD = 0.50
WINDOW_SIZE = 15
INFERENCE_EVERY_N = 5
FACE_RUN_EVERY_N = 3  # InsightFace runs every N frames; result cached between

# Stereo depth thresholds (mm)
SPATIAL_LOWER_THRESH = 100
SPATIAL_UPPER_THRESH = 10000

COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

LSTM_MODEL_PATH = os.path.expanduser("~/activity_detection/lstm_v3_activity.pth")
LSTM_CLASSES = ["falling", "running", "sitting", "walking"]
LSTM_WINDOW = 16
LSTM_FEATURES = 92
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
LSTM_IMG_SIZE = 480
LSTM_CONF_THRESHOLD = 0.75
LSTM_UNKNOWN_THRESHOLD = 0.60
LSTM_VOTE_WINDOW = 6
LSTM_MAJORITY_NEEDED = 4
LSTM_STABLE_SECONDS = 1.5
LSTM_KP_CONF_THRESH = 0.3

LSTM_HIP_L, LSTM_HIP_R = 11, 12
LSTM_KNEE_L, LSTM_KNEE_R = 13, 14
LSTM_ANKLE_L, LSTM_ANKLE_R = 15, 16
LSTM_SHOULDER_L, LSTM_SHOULDER_R = 5, 6
LSTM_ELBOW_L, LSTM_ELBOW_R = 7, 8
LSTM_WRIST_L, LSTM_WRIST_R = 9, 10

person_states = {}

# ---------------------------------------------------------------------------
# GPU-NATIVE FEATURE EXTRACTION
# ---------------------------------------------------------------------------


def clean_keypoints_gpu(
    kps: torch.Tensor, conf: torch.Tensor, prev_kps: Optional[torch.Tensor] = None
) -> torch.Tensor:
    cleaned = kps.clone()
    low_conf = conf < LSTM_KP_CONF_THRESH
    if prev_kps is not None:
        has_prev = (prev_kps[:, 0] != 0) | (prev_kps[:, 1] != 0)
        use_prev = low_conf & has_prev
        cleaned[use_prev] = prev_kps[use_prev]
    return cleaned


def compute_angle_gpu(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    ba = a - b
    bc = c - b
    dot = (ba * bc).sum(dim=-1)
    norm = ba.norm(dim=-1) * bc.norm(dim=-1)
    cos = torch.clamp(dot / (norm + 1e-6), -1.0, 1.0)
    return torch.acos(cos) * (180.0 / torch.pi) / 180.0


def extract_features_gpu(kps: torch.Tensor) -> torch.Tensor:
    """kps: (17, 2) normalized GPU tensor → (46,) feature tensor on GPU."""
    hip_l, hip_r = kps[LSTM_HIP_L], kps[LSTM_HIP_R]
    sh_l, sh_r = kps[LSTM_SHOULDER_L], kps[LSTM_SHOULDER_R]

    hip_valid = (hip_l[0] != 0) or (hip_l[1] != 0)
    hip_center = (hip_l + hip_r) / 2.0 if hip_valid else (sh_l + sh_r) / 2.0

    shoulder_center = (sh_l + sh_r) / 2.0
    torso_height = (shoulder_center - hip_center).norm() + 1e-6

    positions = ((kps - hip_center) / torso_height).flatten()  # (34,)

    A = torch.stack(
        [
            kps[LSTM_HIP_L],
            kps[LSTM_HIP_R],
            kps[LSTM_SHOULDER_L],
            kps[LSTM_SHOULDER_R],
            kps[LSTM_SHOULDER_L],
            kps[LSTM_SHOULDER_R],
            kps[LSTM_ELBOW_L],
            kps[LSTM_ELBOW_R],
            kps[LSTM_KNEE_L],
            kps[LSTM_KNEE_R],
            kps[LSTM_SHOULDER_L],
            kps[LSTM_HIP_L],
        ]
    )
    B = torch.stack(
        [
            kps[LSTM_KNEE_L],
            kps[LSTM_KNEE_R],
            kps[LSTM_HIP_L],
            kps[LSTM_HIP_R],
            kps[LSTM_ELBOW_L],
            kps[LSTM_ELBOW_R],
            kps[LSTM_SHOULDER_L],
            kps[LSTM_SHOULDER_R],
            kps[LSTM_ANKLE_L],
            kps[LSTM_ANKLE_R],
            kps[LSTM_SHOULDER_L],
            kps[LSTM_SHOULDER_R],
        ]
    )
    C = torch.stack(
        [
            kps[LSTM_ANKLE_L],
            kps[LSTM_ANKLE_R],
            kps[LSTM_KNEE_L],
            kps[LSTM_KNEE_R],
            kps[LSTM_WRIST_L],
            kps[LSTM_WRIST_R],
            kps[LSTM_HIP_L],
            kps[LSTM_HIP_R],
            kps[LSTM_HIP_L],
            kps[LSTM_HIP_R],
            kps[LSTM_HIP_R],
            kps[LSTM_KNEE_R],
        ]
    )
    angles = compute_angle_gpu(A, B, C)  # (12,)

    return torch.cat([positions, angles])  # (46,)


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
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, len(LSTM_CLASSES)),
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x.reshape(B * T, -1)).reshape(B, T, -1)
        out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(out), dim=1)
        context = (attn * out).sum(dim=1)
        return self.classifier(context)


# ---------------------------------------------------------------------------
# 1. DATA STRUCTURES
# ---------------------------------------------------------------------------


@dataclass
class SpatialCoord:
    x: float  # mm, positive = right
    y: float  # mm, positive = down
    z: float  # mm, positive = away from camera

    def __str__(self):
        return f"({self.x/1000:.2f}m, {self.y/1000:.2f}m, {self.z/1000:.2f}m)"


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

    def to_pixel(self, w: int, h: int) -> tuple[int, int, int, int]:
        return (int(self.x1 * w), int(self.y1 * h), int(self.x2 * w), int(self.y2 * h))


@dataclass
class ObjectDetection:
    label: str
    confidence: float
    bbox: BoundingBox
    spatial: Optional[SpatialCoord] = None


@dataclass
class FaceDetection:
    identity: str
    confidence: float
    bbox: BoundingBox
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    spatial: Optional[SpatialCoord] = None


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


def load_lstm(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActivityLSTM().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[LSTM] Loaded on {device}")
    return model, device


def load_insightface(registered_faces_path: str, device: torch.device):
    available = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "gpu_mem_limit": 1 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
        ctx_id = 0
        print("[InsightFace] Using CUDAExecutionProvider")
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        print("[InsightFace] Using CPUExecutionProvider")

    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(160, 160))

    with open(registered_faces_path, "rb") as f:
        data = pickle.load(f)
    names = data["names"]
    raw_embeddings = data["embeddings"]

    emb_np = np.array(raw_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-12
    emb_np = emb_np / norms
    emb_gpu = torch.from_numpy(emb_np).to(device)  # (N, 512) on GPU

    print(f"[InsightFace] Loaded {len(names)} registered faces on {device}: {names}")
    return app, names, emb_gpu


def build_depthai_pipeline(blob_path: str):
    """
    DepthAI pipeline with:
      RGB camera → preview → XLinkOut + YOLO NN
      Mono L/R   → StereoDepth → XLinkOut (depth)
      StereoDepth → SpatialLocationCalculator → XLinkOut (spatial_data)
      XLinkIn    → SpatialLocationCalculator config
    """
    pipeline = dai.Pipeline()

    # ── RGB ─────────────────────────────────────────────────────────────
    cam = pipeline.create(dai.node.ColorCamera)
    # cam.setVideoSize(IMG_SIZE[0], IMG_SIZE[1])
    cam.setPreviewSize(640, 352)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    # ── Mono cameras ─────────────────────────────────────────────────────
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # ── Stereo depth ──────────────────────────────────────────────────────
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align to RGB

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    # ── Spatial location calculator ───────────────────────────────────────
    spatial_calc = pipeline.create(dai.node.SpatialLocationCalculator)
    spatial_calc.inputDepth.setBlocking(False)
    spatial_calc.setWaitForConfigInput(True)
    stereo.depth.link(spatial_calc.inputDepth)

    xin_spatial_cfg = pipeline.create(dai.node.XLinkIn)
    xin_spatial_cfg.setStreamName("spatial_cfg")
    xin_spatial_cfg.out.link(spatial_calc.inputConfig)

    xout_spatial = pipeline.create(dai.node.XLinkOut)
    xout_spatial.setStreamName("spatial_data")
    spatial_calc.out.link(xout_spatial.input)

    # ── YOLO NN ───────────────────────────────────────────────────────────
    yolo = pipeline.create(dai.node.NeuralNetwork)
    yolo.setBlobPath(blob_path)
    yolo.setNumInferenceThreads(2)
    yolo.setNumNCEPerInferenceThread(1)
    yolo.input.setBlocking(False)
    cam.preview.link(yolo.input)

    xout_det = pipeline.create(dai.node.XLinkOut)
    xout_det.setStreamName("det")
    yolo.out.link(xout_det.input)

    # ── Start device ───────────────────────────────────────────────────────
    device = dai.Device(pipeline)
    rgb_q = device.getOutputQueue("rgb", maxSize=1, blocking=False)
    det_q = device.getOutputQueue("det", maxSize=1, blocking=False)
    depth_q = device.getOutputQueue("depth", maxSize=1, blocking=False)
    spatial_q = device.getOutputQueue("spatial_data", maxSize=4, blocking=False)
    spatial_cfg_q = device.getInputQueue("spatial_cfg")

    return device, rgb_q, det_q, depth_q, spatial_q, spatial_cfg_q


# ---------------------------------------------------------------------------
# SPATIAL HELPERS
# ---------------------------------------------------------------------------


def send_spatial_rois(
    bboxes_norm: list[tuple], spatial_cfg_q, max_rois: int = 4
) -> None:
    """
    Hard cap at max_rois — each ROI adds ~10-15KB to the XLink message.
    The device limit is 51200B, so 4 ROIs is the safe ceiling.
    """
    if not bboxes_norm:
        return
    cfg = dai.SpatialLocationCalculatorConfig()
    for x1, y1, x2, y2 in bboxes_norm[:max_rois]:  # ← cap here
        x1 = max(0.001, min(0.999, x1))
        y1 = max(0.001, min(0.999, y1))
        x2 = max(0.001, min(0.999, x2))
        y2 = max(0.001, min(0.999, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = dai.SpatialLocationCalculatorConfigData()
        roi.roi = dai.Rect(dai.Point2f(x1, y1), dai.Point2f(x2, y2))
        roi.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
        roi.depthThresholds.lowerThreshold = SPATIAL_LOWER_THRESH
        roi.depthThresholds.upperThreshold = SPATIAL_UPPER_THRESH
        cfg.addROI(roi)
    spatial_cfg_q.send(cfg)


def read_spatial_results(spatial_q) -> list[SpatialCoord]:
    """
    Drain the spatial output queue and return all results as SpatialCoord.
    Returns empty list if nothing is ready yet.
    """
    in_spatial = spatial_q.tryGet()
    if in_spatial is None:
        return []
    coords = []
    for loc in in_spatial.getSpatialLocations():
        c = loc.spatialCoordinates
        coords.append(SpatialCoord(float(c.x), float(c.y), float(c.z)))
    return coords


# ---------------------------------------------------------------------------
# 3. DETECTION COLLECTOR
# ---------------------------------------------------------------------------


class DetectionCollector:
    def __init__(
        self,
        lstm_model,
        lstm_device: torch.device,
        insightface_app,
        pose_model,
        registered_names: list,
        emb_tensor: torch.Tensor,  # (N, 512) pre-normalized, on GPU
    ):
        self.lstm_model = lstm_model
        self.lstm_device = lstm_device
        self.face_app = insightface_app
        self.pose_model = pose_model
        self.registered_names = registered_names
        self.emb_tensor = emb_tensor

        self._frame_id = 0
        self._last_faces: list[FaceDetection] = []

    # ── YOLO via DepthAI ──────────────────────────────────────────────────

    def _parse_yolo(self, nn_data, frame_w: int, frame_h: int) -> list[ObjectDetection]:
        results = []
        try:
            raw = nn_data.getLayerFp16(nn_data.getAllLayerNames()[0])
            layer = np.array(raw).reshape(-1)
            n = len(layer) // 6
            for i in range(n):
                x1, y1, x2, y2, conf, cls_id = layer[i * 6 : (i + 1) * 6]
                if conf < OBJ_THRESHOLD:
                    continue
                label = (
                    COCO_LABELS[int(cls_id)]
                    if int(cls_id) < len(COCO_LABELS)
                    else str(int(cls_id))
                )
                results.append(
                    ObjectDetection(
                        label=label,
                        confidence=float(conf),
                        # YOLO blob outputs pixel coords — normalize here
                        bbox=BoundingBox(
                            float(x1) / frame_w,
                            float(y1) / frame_h,
                            float(x2) / frame_w,
                            float(y2) / frame_h,
                        ),
                    )
                )
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
            emb_np = face.embedding.astype(np.float32)
            emb_t = torch.from_numpy(emb_np).to(self.lstm_device)
            emb_t = emb_t / (emb_t.norm() + 1e-12)
            identity, score = self._match_face_gpu(emb_t)
            x1, y1, x2, y2 = face.bbox.astype(float)
            results.append(
                FaceDetection(
                    identity=identity,
                    confidence=score,
                    bbox=BoundingBox(x1 / w, y1 / h, x2 / w, y2 / h),
                    embedding=emb_np,
                )
            )
        return results

    def _match_face_gpu(self, emb_t: torch.Tensor) -> tuple[str, float]:
        scores = self.emb_tensor @ emb_t  # (N,) on GPU
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        name = (
            self.registered_names[best_idx]
            if best_score >= FACE_THRESHOLD
            else "Unknown"
        )
        return name, best_score

    # ── LSTM (fully GPU) ──────────────────────────────────────────────────

    def _run_lstm(self, frame: np.ndarray) -> list[ActivityDetection]:
        h, w = frame.shape[:2]
        results = self.pose_model(frame, verbose=False, imgsz=LSTM_IMG_SIZE)

        detected_ids = set()
        probs = torch.zeros(len(LSTM_CLASSES), device=self.lstm_device)

        if (
            results[0].keypoints is not None
            and len(results[0].keypoints.xy) > 0
            and results[0].boxes is not None
            and results[0].keypoints.conf is not None
        ):

            kps_all = results[0].keypoints.xy  # (P, 17, 2) on GPU
            conf_all = results[0].keypoints.conf  # (P, 17)    on GPU
            wh = torch.tensor([w, h], dtype=torch.float32, device=self.lstm_device)
            kps_norm = kps_all / wh  # normalize on GPU

            for person_id in range(len(kps_norm)):
                detected_ids.add(person_id)

                if person_id not in person_states:
                    person_states[person_id] = {
                        "buffer": deque(maxlen=LSTM_WINDOW),
                        "prev_kps": None,
                        "prev_features": None,
                        "vote_history": deque(maxlen=LSTM_VOTE_WINDOW),
                        "label": "...",
                        "conf": 0.0,
                        "last_above": time.time(),
                    }

                state = person_states[person_id]
                kps = kps_norm[person_id]  # (17, 2) GPU tensor
                kp_conf = conf_all[person_id]  # (17,)   GPU tensor

                kps_clean = clean_keypoints_gpu(kps, kp_conf, state["prev_kps"])
                state["prev_kps"] = kps_clean

                features = extract_features_gpu(kps_clean)  # (46,) GPU
                velocity = (
                    features - state["prev_features"]
                    if state["prev_features"] is not None
                    else torch.zeros(46, device=self.lstm_device)
                )
                state["prev_features"] = features

                full_features = torch.cat([features, velocity])  # (92,) GPU
                state["buffer"].append(full_features)

                if len(state["buffer"]) == LSTM_WINDOW:
                    seq = torch.stack(list(state["buffer"])).unsqueeze(
                        0
                    )  # (1,T,92) GPU

                    with torch.no_grad():
                        out = self.lstm_model(seq)
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
                            avg_conf = np.mean(
                                [c for p, c in state["vote_history"] if p == top_pred]
                            )
                            state["label"] = LSTM_CLASSES[top_pred]
                            state["conf"] = avg_conf

                if time.time() - state["last_above"] > LSTM_STABLE_SECONDS:
                    state["label"] = "..."
                    state["conf"] = 0.0
                    state["vote_history"].clear()

                if state["conf"] < LSTM_UNKNOWN_THRESHOLD and state["label"] not in (
                    "...",
                ):
                    state["label"] = "UNKNOWN"
                    state["conf"] = 0.0

            for pid in list(person_states.keys()):
                if pid not in detected_ids:
                    del person_states[pid]

        return [
            ActivityDetection(action=LSTM_CLASSES[i], confidence=float(c))
            for i, c in enumerate(probs.tolist())
            if c >= LSTM_UNKNOWN_THRESHOLD
        ]

    # ── Combined ──────────────────────────────────────────────────────────

    def collect(
        self, frame: np.ndarray, nn_data, spatial_cfg_q=None, spatial_q=None
    ) -> FrameDetections:
        h, w = frame.shape[:2]
        self._frame_id += 1

        # InsightFace — cached every FACE_RUN_EVERY_N frames
        if self._frame_id % FACE_RUN_EVERY_N == 0:
            self._last_faces = self._run_insightface(frame)

        objects = self._parse_yolo(nn_data, w, h)
        activities = self._run_lstm(frame)
        faces = self._last_faces

        # ── Spatial: send ROIs for all faces + objects, read results ──────
        if spatial_cfg_q is not None and spatial_q is not None:
            # Cap at 4 ROIs max — each ROI adds ~10-15KB of metadata
            roi_order: list[str] = []
            bboxes: list[tuple] = []

            for face in faces[:4]:  # max 4 faces
                bb = face.bbox
                bboxes.append((bb.x1, bb.y1, bb.x2, bb.y2))
                roi_order.append(f"face:{face.identity}")

            send_spatial_rois(bboxes, spatial_cfg_q)
            time.sleep(0.005)
            coords = read_spatial_results(spatial_q)

            for i, key in enumerate(roi_order):
                if i >= len(coords):
                    break
                coord = coords[i]
                if coord.z <= 0:
                    continue
                ident = key[5:]
                for face in faces:
                    if face.identity == ident and face.spatial is None:
                        face.spatial = coord
                        break

        return FrameDetections(
            frame_id=self._frame_id,
            timestamp=time.time() - start_time,
            objects=objects,
            faces=faces,
            activities=activities,
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
    object_labels: set = field(default_factory=set)
    identities_present: set = field(default_factory=set)
    action_labels: set = field(default_factory=set)
    person_object_proximity: dict = field(default_factory=dict)
    person_person_proximity: dict = field(default_factory=dict)
    person_latest_bbox: dict = field(default_factory=dict)  # identity → BoundingBox
    spatial_map: dict = field(default_factory=dict)  # key → SpatialCoord


def build_scene_graph(det: FrameDetections) -> SceneGraph:
    g = SceneGraph(
        frame_id=det.frame_id,
        timestamp=det.timestamp,
        objects=det.objects,
        faces=det.faces,
        activities=det.activities,
    )
    g.object_labels = {o.label for o in det.objects}
    g.identities_present = {f.identity for f in det.faces}
    g.action_labels = {a.action for a in det.activities}

    # Proximity
    for face in det.faces:
        for obj in det.objects:
            key = (face.identity, obj.label)
            dist = face.bbox.distance_to(obj.bbox)
            if (
                key not in g.person_object_proximity
                or dist < g.person_object_proximity[key]
            ):
                g.person_object_proximity[key] = round(dist, 4)

    for i, fa in enumerate(det.faces):
        for fb in det.faces[i + 1 :]:
            key = tuple(sorted([fa.identity, fb.identity]))
            g.person_person_proximity[key] = round(fa.bbox.distance_to(fb.bbox), 4)

    # Latest bbox per identity (highest confidence face in this frame)
    best_conf: dict[str, float] = {}
    for face in det.faces:
        if face.identity not in best_conf or face.confidence > best_conf[face.identity]:
            best_conf[face.identity] = face.confidence
            g.person_latest_bbox[face.identity] = face.bbox

    # Spatial map
    for face in det.faces:
        if face.spatial:
            g.spatial_map[f"face:{face.identity}"] = face.spatial
    for obj in det.objects:
        if obj.spatial:
            g.spatial_map[f"obj:{obj.label}"] = obj.spatial

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

    def latest_person_bboxes(self) -> dict[str, BoundingBox]:
        result: dict[str, BoundingBox] = {}
        for graph in reversed(self.frames):
            for ident, bbox in graph.person_latest_bbox.items():
                if ident not in result:
                    result[ident] = bbox
        return result

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
            f"{a}__{b}": round(sum(v) / len(v), 4) for (a, b), v in pp_accum.items()
        }

        # Most recent valid spatial reading per entity
        spatial_map: dict[str, SpatialCoord] = {}
        for g in reversed(frames):
            for key, coord in g.spatial_map.items():
                if key not in spatial_map and coord.z > 0:
                    spatial_map[key] = coord

        return {
            "object_presence_rate": object_presence_rate,
            "identity_presence_rate": identity_presence_rate,
            "action_presence_rate": action_presence_rate,
            "mean_proximity": mean_proximity,
            "mean_person_proximity": mean_person_proximity,
            "all_identities": list(id_counts.keys()),
            "spatial_map": spatial_map,
            "n_frames": n,
            "timestamp": frames[-1].timestamp,
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
    bbox: Optional[BoundingBox] = None  # normalized [0,1]
    spatial: Optional[SpatialCoord] = None


@dataclass
class ContextResult:
    sentences: list[PersonSentence]
    inferencer_type: str
    timestamp: float = field(default_factory=time.time)
    # Search match: set by SearchContextPipeline when a target is found
    search_target: Optional[PersonSentence] = None

    def display_lines(self) -> list[str]:
        lines = []
        for s in self.sentences:
            line = s.sentence
            if s.spatial and s.spatial.z > 0:
                line += f"  [{s.spatial}]"
            lines.append(line)
        return lines or ["No persons detected"]


class ContextInferencer(ABC):
    @abstractmethod
    def infer(
        self, features: dict, person_bboxes: Optional[dict[str, BoundingBox]] = None
    ) -> ContextResult: ...


# ---------------------------------------------------------------------------
# 7. RULE-BASED INFERENCER
# ---------------------------------------------------------------------------


class RuleBasedInferencer(ContextInferencer):
    PRESENT = 0.5
    SOMETIMES = 0.25
    NEAR = 0.20
    TOGETHER = 0.25

    def infer(
        self, features: dict, person_bboxes: Optional[dict[str, BoundingBox]] = None
    ) -> ContextResult:
        person_bboxes = person_bboxes or {}
        spatial_map = features.get("spatial_map", {})
        sentences = []
        for ident in features.get("all_identities", []):
            sentence, confidence, evidence = self._infer_person(ident, features)
            sentences.append(
                PersonSentence(
                    identity=ident,
                    sentence=sentence,
                    confidence=confidence,
                    evidence=evidence,
                    bbox=person_bboxes.get(ident),
                    spatial=spatial_map.get(f"face:{ident}"),
                )
            )
        return ContextResult(
            sentences=sentences,
            inferencer_type="rule_based",
            timestamp=features.get("timestamp", time.time()),
        )

    def _act(self, f, action):
        return f.get("action_presence_rate", {}).get(action, 0.0)

    def _obj(self, f, label):
        return f.get("object_presence_rate", {}).get(label, 0.0)

    def _prox(self, f, identity, obj_label):
        return f.get("mean_proximity", {}).get(f"{identity}__{obj_label}", 999.0)

    def _person_prox(self, f, id_a, id_b):
        pp = f.get("mean_person_proximity", {})
        return min(pp.get(f"{id_a}__{id_b}", 999.0), pp.get(f"{id_b}__{id_a}", 999.0))

    def _nearest_other(self, f, identity):
        others = [i for i in f.get("all_identities", []) if i != identity]
        best, best_dist = None, 999.0
        for other in others:
            dist = self._person_prox(f, identity, other)
            if dist < best_dist:
                best_dist, best = dist, other
        return best, best_dist

    def _infer_person(self, identity: str, f: dict) -> tuple[str, float, list]:
        name = identity

        if self._act(f, "falling") > self.SOMETIMES:
            return f"{name} is falling", 0.92, ["falling detected by LSTM"]
        if self._act(f, "running") > self.PRESENT:
            return f"{name} is running", 0.88, ["running detected by LSTM"]
        if (
            self._obj(f, "cell phone") > self.SOMETIMES
            and self._prox(f, identity, "cell phone") < self.NEAR
        ):
            return f"{name} is holding cell phone", 0.85, ["cell phone nearby"]
        if (
            self._obj(f, "laptop") > self.SOMETIMES
            and self._prox(f, identity, "laptop") < self.NEAR
        ):
            return f"{name} is working on laptop", 0.85, ["laptop nearby"]
        if (
            self._act(f, "sitting") > self.PRESENT
            and self._obj(f, "chair") > self.SOMETIMES
            and self._prox(f, identity, "chair") < self.NEAR
        ):
            return f"{name} is sitting on chair", 0.87, ["sitting + chair nearby"]
        if self._act(f, "sitting") > self.PRESENT:
            return f"{name} is sitting", 0.75, ["sitting detected by LSTM"]
        other, dist = self._nearest_other(f, identity)
        if other is not None and dist < self.TOGETHER:
            return (
                f"{name} is standing with {other}",
                0.78,
                [f"near {other} ({dist:.2f})"],
            )
        if self._act(f, "walking") > self.PRESENT:
            return f"{name} is walking", 0.82, ["walking detected by LSTM"]

        return f"{name} is present", 0.40, ["no strong rule matched"]


# ---------------------------------------------------------------------------
# 8. CONTEXT LOGGER
# ---------------------------------------------------------------------------


class ContextLogger:
    def __init__(self, log_path: str = LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, features: dict, result: ContextResult):
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": features.get("timestamp", time.time()),
            "inferencer_type": result.inferencer_type,
            "labels": [
                {
                    "identity": s.identity,
                    "sentence": s.sentence,
                    "confidence": s.confidence,
                    "spatial": str(s.spatial) if s.spatial else None,
                }
                for s in result.sentences
            ],
            "features": {
                "object_presence_rate": features.get("object_presence_rate", {}),
                "action_presence_rate": features.get("action_presence_rate", {}),
                "identity_presence_rate": features.get("identity_presence_rate", {}),
                "mean_proximity": features.get("mean_proximity", {}),
                "mean_person_proximity": features.get("mean_person_proximity", {}),
                "all_identities": features.get("all_identities", []),
                "n_frames": features.get("n_frames", 0),
            },
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# 9. BASE CONTEXT PIPELINE
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
        self.collector = collector
        self.buffer = SlidingWindowBuffer(window_size)
        self.inferencer = inferencer or RuleBasedInferencer()
        self.logger = ContextLogger(log_path)
        self.inference_every_n = inference_every_n
        self._frame_count = 0
        self._last_result: Optional[ContextResult] = None

    def process_frame(
        self, frame: np.ndarray, nn_data, spatial_cfg_q=None, spatial_q=None
    ) -> Optional[ContextResult]:
        t0 = time.perf_counter()
        self._frame_count += 1

        t_collect = time.perf_counter()
        det = self.collector.collect(frame, nn_data, spatial_cfg_q, spatial_q)
        t_collect = (time.perf_counter() - t_collect) * 1000

        graph = build_scene_graph(det)
        self.buffer.push(graph)

        if self._frame_count % self.inference_every_n != 0:
            return self._last_result
        if not self.buffer.is_ready:
            return None

        t_infer = time.perf_counter()
        features = self.buffer.aggregate_features()
        person_bboxes = self.buffer.latest_person_bboxes()
        result = self.inferencer.infer(features, person_bboxes)
        t_infer = (time.perf_counter() - t_infer) * 1000

        self.logger.log(features, result)
        self._last_result = result

        t_total = (time.perf_counter() - t0) * 1000
        result._latency = {
            "collect_ms": t_collect,
            "infer_ms": t_infer,
            "total_ms": t_total,
        }
        return result

    # ── Display helpers ───────────────────────────────────────────────────

    @staticmethod
    def _draw_result(
        frame: np.ndarray,
        last_result: Optional[ContextResult],
        fps: float,
        highlight_identity: Optional[str] = None,
    ) -> None:
        h, w = frame.shape[:2]

        if last_result is not None:
            for sentence in last_result.sentences:
                if sentence.bbox is None:
                    continue
                x1, y1, x2, y2 = sentence.bbox.to_pixel(w, h)
                is_target = (
                    highlight_identity is not None
                    and sentence.identity == highlight_identity
                )
                color = (0, 0, 255) if is_target else (0, 255, 0)
                thickness = 3 if is_target else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Identity + depth above box
                label = sentence.identity
                if sentence.spatial and sentence.spatial.z > 0:
                    label += f" {sentence.spatial}"
                label_y = max(y1 - 8, 14)
                cv2.putText(
                    frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Sentence overlay top-left
        y_text = 30
        lines = last_result.display_lines() if last_result else ["Initializing..."]
        for line in lines:
            color = (0, 255, 0) if last_result else (200, 200, 200)
            cv2.putText(
                frame,
                line,
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            y_text += 30

        # Latency + FPS bottom
        lat = getattr(last_result, "_latency", None) if last_result else None
        if lat:
            cv2.putText(
                frame,
                f"collect:{lat['collect_ms']:.0f}ms  "
                f"infer:{lat['infer_ms']:.0f}ms  "
                f"total:{lat['total_ms']:.0f}ms",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"FPS:{fps:.1f}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def run(self, display: bool = True):
        dai_device, rgb_q, det_q, depth_q, spatial_q, spatial_cfg_q = (
            build_depthai_pipeline(YOLO_BLOB_PATH)
        )

        print(f"[ContextPipeline] Running. Logs → {self.logger.log_path}")
        print("Press 'q' or ESC to quit.")

        try:
            frame_times = []
            last_result = None

            while True:
                t_loop = time.perf_counter()

                in_rgb = rgb_q.tryGet()
                in_det = det_q.tryGet()
                if in_rgb is None or in_det is None:
                    time.sleep(0.001)
                    continue

                frame = in_rgb.getCvFrame()
                result = self.process_frame(frame, in_det, spatial_cfg_q, spatial_q)

                if result is not None and result is not last_result:
                    last_result = result
                    for line in result.display_lines():
                        print(f"[{result.inferencer_type}] {line}")
                    lat = getattr(result, "_latency", None)
                    if lat:
                        print(
                            f"  latency → collect:{lat['collect_ms']:.1f}ms  "
                            f"infer:{lat['infer_ms']:.1f}ms  "
                            f"total:{lat['total_ms']:.1f}ms"
                        )

                frame_times.append(time.perf_counter() - t_loop)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

                if display:
                    self._draw_result(frame, last_result, fps)
                    cv2.imshow("Context Inference", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("\n[ContextPipeline] Quit.")
                        break

        finally:
            if display:
                cv2.destroyAllWindows()
            dai_device.close()
            print("[ContextPipeline] Stopped.")


# ---------------------------------------------------------------------------
# 10. SEARCH CONTEXT PIPELINE
# Wraps ContextPipeline and highlights targets matching the search query.
# Modes:
#   "person" — match by face identity name (e.g. "Alice")
#   "object" — match by COCO label (e.g. "laptop")
#   "action" — match by LSTM class (e.g. "sitting")
# ---------------------------------------------------------------------------


class SearchContextPipeline(ContextPipeline):

    VALID_MODES = ("person", "object", "action")

    def __init__(
        self,
        collector: DetectionCollector,
        inferencer: Optional[ContextInferencer] = None,
        log_path: str = LOG_PATH,
        window_size: int = WINDOW_SIZE,
        inference_every_n: int = INFERENCE_EVERY_N,
        search_mode: str = "person",  # "person" | "object" | "action"
        search_query: str = "",
    ):
        super().__init__(
            collector, inferencer, log_path, window_size, inference_every_n
        )
        self.search_mode = search_mode.lower().strip()
        self.search_query = search_query.lower().strip()
        assert (
            self.search_mode in self.VALID_MODES
        ), f"search_mode must be one of {self.VALID_MODES}"
        print(f"[Search] mode={self.search_mode!r}  query={self.search_query!r}")

    def _find_target(
        self, result: ContextResult, features: dict
    ) -> Optional[PersonSentence]:
        """
        Return the PersonSentence that best matches the search query, or None.
        """
        q = self.search_query

        if self.search_mode == "person":
            for s in result.sentences:
                if s.identity.lower() == q:
                    return s

        elif self.search_mode == "object":
            # Find which person is closest to the searched object
            best_s, best_dist = None, 999.0
            for s in result.sentences:
                dist = features.get("mean_proximity", {}).get(
                    f"{s.identity}__{q}", 999.0
                )
                if dist < best_dist:
                    best_dist, best_s = dist, s
            # Only return if object is actually present in scene
            if (
                best_s is not None
                and features.get("object_presence_rate", {}).get(q, 0) > 0
            ):
                return best_s

        elif self.search_mode == "action":
            # Return the person most confidently performing the action
            best_s, best_rate = None, 0.0
            rate = features.get("action_presence_rate", {}).get(q, 0.0)
            if rate > 0:
                for s in result.sentences:
                    if q in s.sentence.lower():
                        return s
                # Fallback: just return first person if action is happening
                if result.sentences:
                    return result.sentences[0]

        return None

    def process_frame(
        self, frame: np.ndarray, nn_data, spatial_cfg_q=None, spatial_q=None
    ) -> Optional[ContextResult]:
        result = super().process_frame(frame, nn_data, spatial_cfg_q, spatial_q)
        if result is None:
            return None

        # Attach search target to result
        features = self.buffer.aggregate_features()
        result.search_target = self._find_target(result, features)

        if result.search_target is not None:
            t = result.search_target
            spatial_str = f"  {t.spatial}" if t.spatial and t.spatial.z > 0 else ""
            print(f"[FOUND] {t.sentence}{spatial_str}")

        return result

    def run(self, display: bool = True):
        dai_device, rgb_q, det_q, depth_q, spatial_q, spatial_cfg_q = (
            build_depthai_pipeline(YOLO_BLOB_PATH)
        )

        mode_label = f"Searching [{self.search_mode}]: {self.search_query}"
        print(f"[SearchContextPipeline] {mode_label}")
        print(f"  Logs → {self.logger.log_path}")
        print("Press 'q' or ESC to quit.")

        # Pre-open window so it's visible during first inference warmup
        if display:
            cv2.namedWindow("Context Inference — Search", cv2.WINDOW_NORMAL)

        try:
            frame_times = []
            last_result = None

            while True:
                t_loop = time.perf_counter()

                in_rgb = rgb_q.tryGet()
                in_det = det_q.tryGet()
                if in_rgb is None or in_det is None:
                    if display:
                        placeholder = np.zeros((352, 640, 3), dtype=np.uint8)
                        cv2.putText(
                            placeholder,
                            "Initializing...",
                            (200, 176),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (200, 200, 200),
                            2,
                        )
                        cv2.imshow("Context Inference — Search", placeholder)
                        cv2.waitKey(1)
                    time.sleep(0.001)
                    continue

                frame = in_rgb.getCvFrame()
                result = self.process_frame(frame, in_det, spatial_cfg_q, spatial_q)

                if result is not None and result is not last_result:
                    last_result = result

                frame_times.append(time.perf_counter() - t_loop)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

                if display:
                    h, w = frame.shape[:2]
                    highlight = (
                        last_result.search_target.identity
                        if last_result and last_result.search_target
                        else None
                    )

                    # Bounding boxes + identity labels
                    if last_result is not None:
                        for sentence in last_result.sentences:
                            if sentence.bbox is None:
                                continue
                            x1, y1, x2, y2 = sentence.bbox.to_pixel(w, h)
                            is_target = (
                                highlight is not None and sentence.identity == highlight
                            )
                            color = (0, 0, 255) if is_target else (0, 255, 0)
                            thickness = 3 if is_target else 2
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                            label = sentence.identity
                            if sentence.spatial and sentence.spatial.z > 0:
                                label += f" {sentence.spatial}"
                            label_y = max(y1 - 8, 14)
                            cv2.putText(
                                frame,
                                label,
                                (x1, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                color,
                                2,
                                cv2.LINE_AA,
                            )

                    # Sentence overlay top-left
                    y_text = 30
                    lines = (
                        last_result.display_lines()
                        if last_result
                        else ["Initializing..."]
                    )
                    for line in lines:
                        color = (0, 255, 0) if last_result else (200, 200, 200)
                        cv2.putText(
                            frame,
                            line,
                            (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                        y_text += 30

                    # FPS bottom
                    cv2.putText(
                        frame,
                        f"FPS:{fps:.1f}",
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # Search banner bottom
                    found = (
                        last_result is not None
                        and last_result.search_target is not None
                    )
                    banner_color = (0, 0, 255) if found else (100, 100, 100)
                    banner_text = (
                        f"FOUND: {last_result.search_target.sentence}"
                        if found
                        else f"Searching: {self.search_query}"
                    )
                    if found and last_result.search_target.spatial:
                        banner_text += f"  {last_result.search_target.spatial}"
                    cv2.putText(
                        frame,
                        banner_text,
                        (10, h - 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        banner_color,
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow("Context Inference — Search", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("\n[SearchContextPipeline] Quit.")
                        break

        finally:
            if display:
                cv2.destroyAllWindows()
            dai_device.close()
            print("[SearchContextPipeline] Stopped.")


# ---------------------------------------------------------------------------
# 11. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # enable TF32 tensor cores

    # ── Search query ──────────────────────────────────────────────────────
    print("Search mode — what are you looking for?")
    print("  Options: person, object, action")
    search_mode = input("Type: ").strip().lower()
    while search_mode not in SearchContextPipeline.VALID_MODES:
        print(f"  Invalid. Choose from: {SearchContextPipeline.VALID_MODES}")
        search_mode = input("Type: ").strip().lower()

    search_query = input(f"Search query ({search_mode}): ").strip()
    print()

    # ── Load models ───────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    lstm_model, lstm_device = load_lstm(LSTM_MODEL_PATH)
    print(f"LSTM using device: {lstm_device}")

    torch.cuda.empty_cache()
    yolo_pose_model = YOLO("yolov8n-pose.pt")
    yolo_pose_model.to(lstm_device)
    print(f"YOLOv8 Pose loaded on {lstm_device}")

    torch.cuda.empty_cache()
    face_app, reg_names, emb_tensor = load_insightface(
        REGISTERED_FACES_PATH, lstm_device
    )

    collector = DetectionCollector(
        lstm_model=lstm_model,
        lstm_device=lstm_device,
        insightface_app=face_app,
        pose_model=yolo_pose_model,
        registered_names=reg_names,
        emb_tensor=emb_tensor,
    )

    pipeline = SearchContextPipeline(
        collector=collector,
        inferencer=RuleBasedInferencer(),
        log_path=LOG_PATH,
        window_size=WINDOW_SIZE,
        inference_every_n=INFERENCE_EVERY_N,
        search_mode=search_mode,
        search_query=search_query,
    )

    pipeline.run(display=True)
