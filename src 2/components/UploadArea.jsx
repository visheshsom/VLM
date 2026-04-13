import { useState, useRef, useEffect } from 'react';

function UploadArea({ filePreviewUrl, isVideo, detections, onUpload, onClear, isUploading }) {
  const [dragActive, setDragActive] = useState(false);
  const [videoTime, setVideoTime] = useState(0);
  const [mediaRect, setMediaRect] = useState(null);
  const inputRef = useRef(null);
  const mediaRef = useRef(null);

  // Show the result panel only after inference has completed (detections is an array, even empty)
  // and there is no ongoing upload.
  const showResultPanel =
    filePreviewUrl &&
    !isUploading &&
    detections !== null &&
    detections !== undefined;

  // ── frame-accurate bounding box filtering ────────────────────
  // Backend samples at ~2 fps (step = fps / target_fps = fps / 2).
  // Half-window of 0.25 s ensures only the closest sampled frame's
  // boxes are visible while that frame is playing.
  const FRAME_HALF_WINDOW = 0.25;

  let activeDetections = [];
  if (!isVideo) {
    activeDetections = detections || [];
  } else if (detections && detections.length > 0) {
    let closestTimestamp = detections[0].timestamp;
    let minDiff = Math.abs(detections[0].timestamp - videoTime);

    for (const det of detections) {
      const diff = Math.abs(det.timestamp - videoTime);
      if (diff < minDiff) {
        minDiff = diff;
        closestTimestamp = det.timestamp;
      }
    }

    if (minDiff <= FRAME_HALF_WINDOW) {
      activeDetections = detections.filter(
        (det) => det.timestamp === closestTimestamp
      );
    }
  }

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  const handleTimeUpdate = (e) => {
    if (isVideo) setVideoTime(e.target.currentTime);
  };

  // Keep media element rect in sync for pixel-accurate box placement
  useEffect(() => {
    if (!mediaRef.current || !detections || detections.length === 0) return;

    const updateRect = () => {
      if (mediaRef.current)
        setMediaRect(mediaRef.current.getBoundingClientRect());
    };

    updateRect();
    window.addEventListener('resize', updateRect);
    const interval = setInterval(updateRect, 500);
    return () => {
      window.removeEventListener('resize', updateRect);
      clearInterval(interval);
    };
  }, [filePreviewUrl, detections]);

  const renderBox = (det, idx) => {
    const { x, y, w, h } = det.box;
    let finalStyle = {
      left: `${x * 100}%`,
      top: `${y * 100}%`,
      width: `${w * 100}%`,
      height: `${h * 100}%`,
    };

    if (mediaRect && mediaRef.current) {
      const containerRect =
        mediaRef.current.parentElement.getBoundingClientRect();
      finalStyle = {
        left: `${mediaRect.left - containerRect.left + x * mediaRect.width}px`,
        top: `${mediaRect.top - containerRect.top + y * mediaRect.height}px`,
        width: `${w * mediaRect.width}px`,
        height: `${h * mediaRect.height}px`,
      };
    }

    return (
      <div key={idx} className="bounding-box" style={finalStyle}>
        <div className="bounding-box-label">
          {det.label} ({Math.round(det.confidence * 100)}%)
        </div>
      </div>
    );
  };

  return (
    <div
      className="upload-container"
      style={{
        gap: '1.5rem',
        background: 'transparent',
        border: 'none',
        boxShadow: 'none',
        padding: '0',
      }}
    >
      {/* ── Drop Zone ── */}
      <div className="glass-panel" style={{ padding: '1.5rem', flexShrink: 0 }}>
        <label
          className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          style={{
            minHeight: '120px',
            padding: '1rem',
            border: '2px dashed var(--panel-border)',
            borderRadius: '16px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: isUploading ? 'not-allowed' : 'pointer',
            opacity: isUploading ? 0.6 : 1,
            transition: 'opacity 0.2s',
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*,video/*"
            style={{ display: 'none' }}
            onChange={handleChange}
            disabled={isUploading}
          />
          <div
            style={{
              fontSize: '2rem',
              marginBottom: '0.5rem',
              color: 'var(--accent-color)',
            }}
          >
            <svg
              width="48"
              height="48"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <h3 style={{ fontSize: '1.1rem' }}>
            {isUploading
              ? 'Uploading…'
              : filePreviewUrl
              ? 'Upload a different Image or Video'
              : 'Upload Image or Video'}
          </h3>
          <p
            style={{
              color: 'var(--text-secondary)',
              marginTop: '4px',
              fontSize: '0.9rem',
            }}
          >
            {isUploading
              ? 'Please wait while your file is being uploaded'
              : filePreviewUrl
              ? 'New file will replace current selection'
              : 'Drag and drop or click to select'}
          </p>
        </label>

        {/* ── Indeterminate progress bar while uploading ── */}
        {isUploading && (
          <>
            <div
              style={{
                marginTop: '1rem',
                background: 'rgba(255,255,255,0.08)',
                borderRadius: '999px',
                overflow: 'hidden',
                height: '6px',
              }}
            >
              <div
                style={{
                  height: '100%',
                  background: 'var(--accent-color)',
                  borderRadius: '999px',
                  animation: 'upload-indeterminate 1.4s ease-in-out infinite',
                }}
              />
            </div>
            <style>{`
              @keyframes upload-indeterminate {
                0%   { transform: translateX(-100%) scaleX(0.35); }
                50%  { transform: translateX(60%)   scaleX(0.55); }
                100% { transform: translateX(200%)  scaleX(0.35); }
              }
            `}</style>
          </>
        )}
      </div>

      {/* ── Inference Result Panel ──
          Rendered only once inference is complete.
          Videos start paused — no autoPlay. */}
      {showResultPanel && (
        <div
          className="glass-panel"
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              padding: '1rem 1.5rem',
              borderBottom: '1px solid var(--panel-border)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <h3 style={{ fontSize: '1.1rem', fontWeight: '600' }}>
              Inference Result
            </h3>
            <button
              className="clear-btn"
              onClick={onClear}
              title="Remove file"
              style={{ position: 'static', width: '30px', height: '30px' }}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>

          <div
            className="media-preview-container"
            style={{
              position: 'relative',
              flex: 1,
              background: '#000',
              minHeight: '300px',
            }}
          >
            {isVideo ? (
              <video
                ref={mediaRef}
                src={filePreviewUrl}
                className="media-preview"
                controls
                loop
                muted
                /* autoPlay intentionally removed */
                onTimeUpdate={handleTimeUpdate}
                onLoadedData={() => setVideoTime(0)}
              />
            ) : (
              <img
                ref={mediaRef}
                src={filePreviewUrl}
                alt="Inference Result"
                className="media-preview"
                onLoad={() => setVideoTime(0)}
              />
            )}

            {activeDetections.map(renderBox)}
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadArea;
