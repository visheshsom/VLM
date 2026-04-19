# OpenVision Engine - Vision Language Model Agent

Zero-shot object detection in images, videos, and live camera feeds using **Google OWLv2**.

![App Screenshot](./screenshot.png)

---

## Installation

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```bash
# From the project root
npm install
```

---

## Configuration

| Parameter | File | Default | Description |
|---|---|---|---|
| `CONF_THRESHOLD` | `backend/server.py` | `0.25` | Minimum confidence score to show a detection |
| `NMS_IOU` | `backend/server.py` | `0.3` | IoU threshold for Non-Maximum Suppression |
| `TARGET_FPS` | `backend/server.py` | `2` | Frame sampling rate for video inference |
| `CAPTURE_INTERVAL_MS` | `src/components/LiveCamera.jsx` | `600` | Webcam frame send interval (ms) |

The backend auto-selects the best available compute device: **CUDA → Apple MPS → CPU**.

---

## Usage

**1. Start the backend** (model downloads ~1.5 GB on first run):

```bash
cd backend
source venv/bin/activate
python server.py
# Runs at http://localhost:8000
```

**2. Start the frontend:**

```bash
npm run dev
# Runs at http://localhost:5173
```

**3. Open http://localhost:5173 in your browser.**

### Upload Mode
- Drag & drop an image or video into the upload panel.
- Type the objects to detect in the chat (e.g. `person, car, drone`) and hit **Send**.
- Results appear as bounding-box overlays (image) or a re-encoded video with boxes burned in.

### Live Camera Mode
- Switch to the **📷 Live Camera** tab.
- Enter objects to detect, then click **▶ Start Camera**.
- Real-time detections stream via WebSocket and are overlaid on the camera feed.

## 👤 Author

**Vishesh Sompura**

## 📄 License

Private
