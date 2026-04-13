import { useState, useRef } from 'react';
import './index.css';
import UploadArea from './components/UploadArea';
import ChatBox from './components/ChatBox';
import LiveCamera from './components/LiveCamera';

// ── tab button ────────────────────────────────────────────────
function TabBtn({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        padding: '0.6rem 1.2rem',
        borderRadius: '10px',
        border: 'none',
        cursor: 'pointer',
        fontFamily: 'inherit',
        fontWeight: 600,
        fontSize: '0.9rem',
        transition: 'all 0.2s',
        background: active ? 'var(--accent-color)' : 'rgba(255,255,255,0.06)',
        color:      active ? '#0b0c10'             : 'var(--text-secondary)',
        boxShadow:  active ? '0 0 14px rgba(102,252,241,0.35)' : 'none',
      }}
    >
      {children}
    </button>
  );
}

// ── live query bar ────────────────────────────────────────────
function LiveQueryBar({ queries, onChange }) {
  const [raw, setRaw] = useState(queries.join(', '));

  const commit = () => {
    const parsed = raw.split(',').map(s => s.trim()).filter(Boolean);
    onChange(parsed.length ? parsed : ['person']);
  };

  return (
    <div className="glass-panel" style={{ padding: '1rem 1.5rem', flexShrink: 0 }}>
      <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
        Objects to detect (comma-separated)
      </label>
      <div style={{ display: 'flex', gap: '8px' }}>
        <input
          type="text"
          value={raw}
          onChange={e => setRaw(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && commit()}
          placeholder="person, car, dog, bottle…"
          style={{
            flex: 1, background: 'rgba(255,255,255,0.06)',
            border: '1px solid var(--panel-border)', borderRadius: '8px',
            color: 'var(--text-primary)', padding: '0.5rem 0.9rem',
            fontFamily: 'inherit', fontSize: '0.9rem', outline: 'none',
          }}
        />
        <button
          onClick={commit}
          style={{
            padding: '0.5rem 1.1rem', borderRadius: '8px', border: 'none',
            background: 'var(--accent-color)', color: '#0b0c10',
            fontWeight: 700, cursor: 'pointer', fontFamily: 'inherit', fontSize: '0.9rem',
          }}
        >
          Apply
        </button>
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────
export default function App() {
  const [mode, setMode] = useState('upload');        // 'upload' | 'live'
  const [liveActive, setLiveActive] = useState(false);
  const [liveQueries, setLiveQueries] = useState(['person']);

  // ── upload-mode state (unchanged) ──────────────────────────
  const [file, setFile] = useState(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState(null);
  const [uploadedFileId, setUploadedFileId] = useState(null);
  const [isVideo, setIsVideo] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'agent', content: 'Hello! Please upload an image or video, then ask me to find any object. (e.g. "find a person, car, dog")' }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [detections, setDetections] = useState(null); // null = no inference run yet; [] = ran but found nothing

  const handleFileUpload = async (selectedFile) => {
    setFile(selectedFile);
    setDetections(null);  // reset: inference not yet run
    setUploadedFileId(null);
    setFilePreviewUrl(null);
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const response = await fetch('http://localhost:8000/upload', { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();
      setUploadedFileId(data.file_id);
      setIsVideo(data.type === 'video');
      setFilePreviewUrl(`http://localhost:8000${data.url}`);
      setMessages(prev => [...prev, { role: 'agent', content: `Upload complete. What would you like me to detect in the ${data.type}?` }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'agent', content: `Error: ${error.message}. Is the backend running?` }]);
      setFile(null); setUploadedFileId(null); setIsVideo(false); setFilePreviewUrl(null);
    } finally {
      setIsUploading(false);
    }
  };

  const handleClear = () => {
    setFile(null); setFilePreviewUrl(null); setUploadedFileId(null);
    setDetections(null); setIsVideo(false);
  };

  const handleSendMessage = async (text) => {
    if (!text.trim()) return;
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    if (!uploadedFileId) {
      setMessages(prev => [...prev, { role: 'agent', content: 'Please upload an image or video first before asking for detections.' }]);
      return;
    }
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file_id', uploadedFileId);
      formData.append('queries', text);
      const response = await fetch('http://localhost:8000/detect', { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Analysis failed');
      const data = await response.json();
      let replyText = '';
      if (data.detections && data.detections.length > 0) {
        setDetections(data.detections);
        const counts = data.detections.reduce((acc, d) => { acc[d.label] = (acc[d.label] || 0) + 1; return acc; }, {});
        const summary = Object.entries(counts).map(([k, v]) => `${v} ${k}(s)`).join(', ');
        replyText = `Detection complete! Found: ${summary}. Check the highlighted boxes.`;
      } else {
        setDetections([]);
        replyText = `I couldn't find anything matching "${text}" with high confidence.`;
      }
      setMessages(prev => [...prev, { role: 'agent', content: replyText }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'agent', content: `Error: ${error.message}. Is the backend running?` }]);
    } finally {
      setIsLoading(false);
    }
  };

  // switch mode: stop live camera when switching away
  const switchMode = (m) => {
    if (m === 'upload') setLiveActive(false);
    setMode(m);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>VISHESH SOMPURA</h1>
        <p>Vision Language Model Agent</p>
      </header>

      {/* ── mode tabs ── */}
      <div style={{
        display: 'flex', gap: '8px', marginBottom: '1.5rem',
        padding: '6px', background: 'rgba(255,255,255,0.04)',
        borderRadius: '14px', border: '1px solid var(--panel-border)',
        maxWidth: 360, alignSelf: 'center', width: '100%',
      }}>
        <TabBtn active={mode === 'upload'} onClick={() => switchMode('upload')}>
          📁 Upload
        </TabBtn>
        <TabBtn active={mode === 'live'} onClick={() => switchMode('live')}>
          📷 Live Camera
        </TabBtn>
      </div>

      <main className="main-content">
        {mode === 'upload' ? (
          <>
            <UploadArea
              filePreviewUrl={filePreviewUrl}
              isVideo={isVideo}
              detections={detections}
              onUpload={handleFileUpload}
              onClear={handleClear}
              isUploading={isUploading}
            />
            <ChatBox
              messages={messages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              disabled={!uploadedFileId || isUploading}
            />
          </>
        ) : (
          /* ── Live Camera panel ── */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', flex: 1, minWidth: 0 }}>
            {/* query config */}
            <LiveQueryBar queries={liveQueries} onChange={setLiveQueries} />

            {/* camera panel */}
            <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div style={{
                padding: '1rem 1.5rem', borderBottom: '1px solid var(--panel-border)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Live Inference</h3>
                <button
                  onClick={() => setLiveActive(a => !a)}
                  style={{
                    padding: '0.45rem 1.1rem', borderRadius: '8px', border: 'none',
                    background: liveActive ? '#e74c3c' : 'var(--accent-color)',
                    color: liveActive ? '#fff' : '#0b0c10',
                    fontWeight: 700, cursor: 'pointer', fontFamily: 'inherit', fontSize: '0.85rem',
                    transition: 'all 0.2s',
                  }}
                >
                  {liveActive ? '⏹ Stop' : '▶ Start Camera'}
                </button>
              </div>

              <div style={{ padding: '1.5rem', flex: 1, overflow: 'auto' }}>
                <LiveCamera
                  queries={liveQueries}
                  isActive={liveActive}
                  onStatusChange={(s) => { if (s !== 'live') setLiveActive(false); }}
                />
              </div>

              {!liveActive && (
                <p style={{
                  textAlign: 'center', padding: '0 1.5rem 1.5rem',
                  color: 'var(--text-secondary)', fontSize: '0.85rem',
                }}>
                  Enter the objects you want to detect above, then click <strong>Start Camera</strong>.
                  Inference runs on every frame sent to your local backend.
                </p>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
