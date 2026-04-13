import { useState, useRef, useEffect, useCallback } from 'react';

const WS_URL = 'ws://localhost:8000/ws/live';
const CAPTURE_INTERVAL_MS = 600;   // send a frame every 600 ms
const JPEG_QUALITY = 0.72;          // balance speed vs accuracy

// colour palette for bounding boxes
const COLOURS = ['#66fcf1', '#45a29e', '#f8c471', '#e74c3c', '#a29bfe', '#fd79a8', '#55efc4'];
const labelColour = (label) => COLOURS[Math.abs([...label].reduce((h, c) => (h << 5) - h + c.charCodeAt(0), 0)) % COLOURS.length];

export default function LiveCamera({ queries, isActive, onStatusChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);          // hidden – captures frames
  const overlayCanvasRef = useRef(null);   // visible overlay for boxes
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const pendingRef = useRef(false);        // true while server hasn't replied yet

  const [status, setStatus] = useState('idle');   // idle | connecting | live | error
  const [detections, setDetections] = useState([]);
  const [fps, setFps] = useState(0);
  const fpsCountRef = useRef(0);
  const fpsTimerRef = useRef(null);

  // ── draw boxes on the overlay canvas ──────────────────────
  const drawBoxes = useCallback((dets) => {
    const overlay = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;

    const vw = video.videoWidth  || video.clientWidth;
    const vh = video.videoHeight || video.clientHeight;
    overlay.width  = vw;
    overlay.height = vh;

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, vw, vh);

    dets.forEach((det) => {
      const { x, y, w, h } = det.box;
      const px = x * vw, py = y * vh, pw = w * vw, ph = h * vh;
      const colour = labelColour(det.label);

      ctx.strokeStyle = colour;
      ctx.lineWidth   = 2.5;
      ctx.strokeRect(px, py, pw, ph);

      const label = `${det.label} ${Math.round(det.confidence * 100)}%`;
      ctx.font      = 'bold 13px Outfit, sans-serif';
      const tw      = ctx.measureText(label).width + 10;
      const th      = 20;
      ctx.fillStyle = colour;
      ctx.fillRect(px, py - th, tw, th);
      ctx.fillStyle = '#0b0c10';
      ctx.fillText(label, px + 5, py - 5);
    });
  }, []);

  // ── capture one frame and send via WebSocket ──────────────
  const sendFrame = useCallback(() => {
    const ws      = wsRef.current;
    const video   = videoRef.current;
    const canvas  = canvasRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video || video.readyState < 2) return;   // not enough data yet
    if (pendingRef.current) return;               // still waiting for last reply

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;

    canvas.width  = w;
    canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);
    const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
    const b64     = dataUrl.split(',')[1];

    pendingRef.current = true;
    ws.send(JSON.stringify({
      frame:     b64,
      queries:   queries,
      threshold: 0.25,
    }));
  }, [queries]);

  // ── WebSocket lifecycle ────────────────────────────────────
  const connectWS = useCallback(() => {
    if (wsRef.current) wsRef.current.close();

    setStatus('connecting');
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('live');
      onStatusChange?.('live');
      intervalRef.current = setInterval(sendFrame, CAPTURE_INTERVAL_MS);

      // FPS ticker
      fpsTimerRef.current = setInterval(() => {
        setFps(fpsCountRef.current);
        fpsCountRef.current = 0;
      }, 1000);
    };

    ws.onmessage = (ev) => {
      pendingRef.current = false;
      try {
        const msg = JSON.parse(ev.data);
        const dets = msg.detections ?? [];
        setDetections(dets);
        drawBoxes(dets);
        fpsCountRef.current += 1;
      } catch { /* ignore */ }
    };

    ws.onerror = () => {
      setStatus('error');
      onStatusChange?.('error');
    };

    ws.onclose = () => {
      if (status !== 'idle') {
        setStatus('idle');
        onStatusChange?.('idle');
      }
    };
  }, [sendFrame, drawBoxes, onStatusChange, status]);

  // ── start / stop ──────────────────────────────────────────
  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      connectWS();
    } catch (err) {
      setStatus('error');
      onStatusChange?.('error');
      console.error('Camera error:', err);
    }
  }, [connectWS, onStatusChange]);

  const stop = useCallback(() => {
    clearInterval(intervalRef.current);
    clearInterval(fpsTimerRef.current);
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    pendingRef.current = false;
    setStatus('idle');
    setDetections([]);
    setFps(0);
    // clear overlay
    const overlay = overlayCanvasRef.current;
    if (overlay) overlay.getContext('2d').clearRect(0, 0, overlay.width, overlay.height);
    onStatusChange?.('idle');
  }, [onStatusChange]);

  // respond to parent toggling `isActive`
  useEffect(() => {
    if (isActive) start();
    else stop();
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  // re-send when queries change (but only if already live)
  useEffect(() => {
    // nothing needed; sendFrame already closes over queries
  }, [queries]);

  // ── render ────────────────────────────────────────────────
  const statusDot = {
    idle:        { colour: '#888',    label: 'Idle' },
    connecting:  { colour: '#f8c471', label: 'Connecting…' },
    live:        { colour: '#66fcf1', label: `Live  ${fps} fps` },
    error:       { colour: '#e74c3c', label: 'Error – check backend & camera' },
  }[status];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* status bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '8px',
        padding: '0.5rem 1rem', borderRadius: '8px',
        background: 'rgba(255,255,255,0.04)', fontSize: '0.85rem',
        color: 'var(--text-secondary)',
      }}>
        <span style={{ width: 10, height: 10, borderRadius: '50%', background: statusDot.colour,
          boxShadow: status === 'live' ? `0 0 6px ${statusDot.colour}` : 'none',
          flexShrink: 0 }} />
        {statusDot.label}
        {status === 'live' && detections.length > 0 && (
          <span style={{ marginLeft: 'auto', color: 'var(--accent-color)' }}>
            {detections.length} detection{detections.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* video + overlay */}
      <div style={{ position: 'relative', background: '#000', borderRadius: '12px', overflow: 'hidden', minHeight: 300 }}>
        <video
          ref={videoRef}
          muted
          playsInline
          style={{ width: '100%', height: '100%', display: 'block', objectFit: 'contain' }}
        />
        {/* overlay canvas – absolutely positioned over the video */}
        <canvas
          ref={overlayCanvasRef}
          style={{
            position: 'absolute', inset: 0,
            width: '100%', height: '100%',
            pointerEvents: 'none',
          }}
        />
        {/* hidden capture canvas */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {/* placeholder when camera is off */}
        {status === 'idle' && (
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            color: 'var(--text-secondary)', gap: '0.75rem',
          }}>
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
            <p style={{ fontSize: '0.9rem' }}>Camera off</p>
          </div>
        )}
      </div>

      {/* detection chips */}
      {detections.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
          {detections.map((det, i) => (
            <span key={i} style={{
              padding: '3px 10px', borderRadius: '999px', fontSize: '0.78rem', fontWeight: 600,
              background: labelColour(det.label) + '22',
              border: `1px solid ${labelColour(det.label)}`,
              color: labelColour(det.label),
            }}>
              {det.label} {Math.round(det.confidence * 100)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
