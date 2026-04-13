"""
VISHESH SOMPURA - Vision Language Model Backend
Zero-shot object detection using OWLv2 (Google)
"""

import io
import os
import base64
import tempfile
import logging
import uuid
import asyncio
from typing import List
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import Owlv2Processor, Owlv2ForObjectDetection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vishesh-vlm")

# ── device selection ─────────────────────────────────────────
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = pick_device()
logger.info(f"Running on device: {DEVICE}")

# ── model cache ──────────────────────────────────────────────
MODEL_ID = "google/owlv2-base-patch16-ensemble"
_processor = None
_model = None

def load_model():
    global _processor, _model
    if _model is not None:
        return _processor, _model
    logger.info(f"Loading {MODEL_ID} ...")
    _processor = Owlv2Processor.from_pretrained(MODEL_ID, use_fast=False)
    _model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).to(DEVICE).eval()
    logger.info("Model ready.")
    return _processor, _model

# ── lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="VISHESH SOMPURA - VLM API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── local file storage ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

# ── helpers ──────────────────────────────────────────────────
CONF_THRESHOLD = 0.25
NMS_IOU = 0.3

def nms(boxes, scores, iou_thresh=NMS_IOU):
    if len(boxes) == 0:
        return []
    order = np.argsort(-np.array(scores))
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = compute_iou(boxes[i], boxes[rest])
        mask = ious < iou_thresh
        order = rest[mask]
    return keep

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)

def detect(image: Image.Image, queries: List[str], threshold: float):
    proc, model = load_model()
    texts = [queries]
    inputs = proc(text=texts, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
    results = proc.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]
    boxes_np = results["boxes"].cpu().numpy()
    scores_np = results["scores"].cpu().numpy()
    labels_np = results["labels"].cpu().numpy()
    if len(boxes_np) == 0:
        return []
    keep = nms(boxes_np, scores_np)
    detections = []
    w, h = image.size
    for idx in keep:
        x1, y1, x2, y2 = boxes_np[idx]
        detections.append({
            "label": queries[int(labels_np[idx])],
            "confidence": round(float(scores_np[idx]), 3),
            "box": {
                "x": round(float(x1) / w, 4),
                "y": round(float(y1) / h, 4),
                "w": round(float(x2 - x1) / w, 4),
                "h": round(float(y2 - y1) / h, 4),
            },
        })
    return detections


def extract_frames(video_bytes: bytes, max_frames=30, target_fps=2):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes)
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total < 1:
        cap.release()
        os.unlink(tmp.name)
        raise HTTPException(400, "Could not read any frames from the video.")

    step = max(1, int(fps / target_fps))
    frames = []

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        # Read POS_MSEC *after* cap.read() — OpenCV only updates the
        # position property once a frame has actually been decoded.
        # Reading it before gives stale (often 0.0) values.
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((Image.fromarray(rgb), timestamp))
        if len(frames) >= max_frames:
            break

    cap.release()
    os.unlink(tmp.name)
    return frames


def render_video_with_boxes(video_bytes: bytes, labels, threshold: float):
    """
    Run detection on sampled frames, then re-encode the full video with
    bounding boxes burned onto every frame. Returns (url_path, all_hits).
    """
    tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_in.write(video_bytes)
    tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total < 1:
        cap.release()
        os.unlink(tmp_in.name)
        raise HTTPException(400, "Could not read any frames from the video.")

    TARGET_FPS  = 2
    step        = max(1, int(fps / TARGET_FPS))
    total_samples = len(range(0, total, step))

    # Pass 1: sample frames across the ENTIRE video and run detection
    sampled_dets = []   # list of (frame_index, timestamp, [dets])
    all_hits = []
    sample_num = 0
    for frame_idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not ok:
            continue
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dets = detect(Image.fromarray(rgb), labels, threshold)
        for d in dets:
            d["frame"]     = sample_num
            d["timestamp"] = round(ts, 3)
            all_hits.append(d)
        sampled_dets.append((frame_idx, ts, dets))
        sample_num += 1
        if sample_num % 5 == 0 or sample_num == total_samples:
            logger.info(f"  Detection pass: {sample_num}/{total_samples} keyframes processed")

    # Pass 2: re-encode every frame with boxes burned in
    out_name = f"{uuid.uuid4().hex}_result.mp4"
    out_path = os.path.join(UPLOAD_DIR, out_name)
    fourcc   = cv2.VideoWriter_fourcc(*"avc1")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    COLORS = [
        (0, 255, 200), (80, 80, 255), (80, 180, 255),
        (0, 200, 255), (180, 80, 255), (80, 255, 80),
    ]

    # Only apply detections if the current frame is close to a sampled
    # keyframe.  Beyond half-a-step away the boxes would be stale.
    max_dist = step // 2 + 1   # frames

    def nearest_dets(fi):
        if not sampled_dets:
            return []
        best = min(sampled_dets, key=lambda t: abs(t[0] - fi))
        if abs(best[0] - fi) > max_dist:
            return []          # too far from any keyframe — skip boxes
        return best[2]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fi in range(total):
        ok, bgr = cap.read()
        if not ok:
            break
        for det in nearest_dets(fi):
            bx  = det["box"]
            x1  = int(bx["x"] * width)
            y1  = int(bx["y"] * height)
            x2  = int((bx["x"] + bx["w"]) * width)
            y2  = int((bx["y"] + bx["h"]) * height)
            color = COLORS[hash(det["label"]) % len(COLORS)]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det['label']} {int(det['confidence']*100)}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(bgr, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        writer.write(bgr)

    cap.release()
    writer.release()
    os.unlink(tmp_in.name)
    return f"/files/{out_name}", all_hits


# ── WebSocket live inference ─────────────────────────────────
# Semaphore so only one inference runs at a time per connection
# (model is single-threaded anyway; concurrent calls just serialize)

@app.websocket("/ws/live")
async def live_inference(websocket: WebSocket):
    """
    Live frame-by-frame inference over WebSocket.

    Client sends JSON:
      { "frame": "<base64 JPEG>", "queries": ["person", "car"], "threshold": 0.25 }

    Server replies with JSON:
      { "detections": [ { "label": ..., "confidence": ..., "box": {...} }, ... ] }

    The client should wait for a reply before sending the next frame to avoid
    queuing up more work than the GPU can handle.
    """
    await websocket.accept()
    logger.info("Live WS connection opened")
    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except Exception:
                break

            frame_b64 = data.get("frame", "")
            queries = [q.strip() for q in data.get("queries", []) if str(q).strip()]
            threshold = float(data.get("threshold", CONF_THRESHOLD))

            if not frame_b64 or not queries:
                await websocket.send_json({"detections": []})
                continue

            try:
                img_bytes = base64.b64decode(frame_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                await websocket.send_json({"error": f"Bad frame: {e}", "detections": []})
                continue

            # Run blocking inference in the thread-pool so the event loop stays free
            try:
                detections = await loop.run_in_executor(None, detect, img, queries, threshold)
                await websocket.send_json({"detections": detections})
            except Exception as e:
                logger.error(f"Inference error: {e}")
                await websocket.send_json({"error": str(e), "detections": []})

    except WebSocketDisconnect:
        logger.info("Live WS connection closed")

# ── REST endpoints (unchanged) ───────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not ext:
        if file.content_type and file.content_type.startswith("image/"):
            ext = ".png"
        elif file.content_type and file.content_type.startswith("video/"):
            ext = ".mp4"

    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext or '(no extension)'}")

    file_id = uuid.uuid4().hex
    saved_name = f"{file_id}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    data = await file.read()
    with open(saved_path, "wb") as f:
        f.write(data)

    media_type = "video" if (ext in VIDEO_EXTS or (file.content_type or "").startswith("video/")) else "image"
    return {
        "file_id": file_id,
        "type": media_type,
        "filename": saved_name,
        "url": f"/files/{saved_name}",
    }

@app.post("/detect")
async def run_detection(
    file: UploadFile = File(None),
    file_id: str = Form(None),
    queries: str = Form(...),
    threshold: float = Form(CONF_THRESHOLD),
):
    labels = [q.strip() for q in queries.split(",") if q.strip()]
    if not labels:
        raise HTTPException(400, "Provide at least one object query.")

    data = None
    ext = ""
    content_type = ""

    if file is not None:
        ext = os.path.splitext(file.filename or "")[1].lower()
        content_type = file.content_type or ""
        data = await file.read()
    elif file_id:
        matches = []
        for name in os.listdir(UPLOAD_DIR):
            if name.startswith(file_id):
                matches.append(name)
        if not matches:
            raise HTTPException(404, "Uploaded file not found. Please upload again.")
        matches.sort()
        saved_name = matches[0]
        ext = os.path.splitext(saved_name)[1].lower()
        saved_path = os.path.join(UPLOAD_DIR, saved_name)
        with open(saved_path, "rb") as f:
            data = f.read()
        content_type = "video" if ext in VIDEO_EXTS else "image"
    else:
        raise HTTPException(400, "Provide either a file upload or a file_id.")

    if ext in IMAGE_EXTS or (content_type.startswith("image")):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        hits = detect(img, labels, threshold)
        return {"type": "image", "detections": hits}

    if ext in VIDEO_EXTS or (content_type.startswith("video")):
        result_url, all_hits = render_video_with_boxes(data, labels, threshold)
        return {
            "type": "video",
            "frame_count": len(set(d["frame"] for d in all_hits)),
            "detections": all_hits,
            "result_url": result_url,
        }

    raise HTTPException(400, f"Unsupported file type: {ext}")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
