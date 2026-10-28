import React, { useState, useEffect, useRef } from 'react';
import {
  Camera,
  CameraOff,
  Sparkles,
  Volume2,
  VolumeX,
  Upload,
  CheckCircle2,
  AlertTriangle,
  MinusCircle,
  PlusCircle,
  Package,
  Layers,
  ArrowRight,
  RefreshCw,
  X,
  Image as ImageIcon,
  Zap,
  Sliders,
  ChevronDown,
  ChevronUp,
  Trash2,
  Clock,
  Eye,
  Check,
  Smartphone,
  RotateCcw,
  QrCode,
  Copy,
  Laptop,
  ExternalLink,
} from 'lucide-react';
import confetti from 'canvas-confetti';

const STYLE_OPTIONS = [
  { label: 'Circular / Ellipse (e.g. Patches, Coins)', value: 'ellipse' },
  { label: 'Translucent Box Fill', value: 'fill' },
  { label: 'Spotlight (Dim Background)', value: 'spotlight' },
  { label: 'Object Contour (Silhouetted)', value: 'contour' },
  { label: 'Outlines & Badges Only', value: 'outline_only' },
];

const PALETTE_OPTIONS = [
  { label: 'Vibrant Rainbow', value: 'rainbow' },
  { label: 'Cyber Cyan', value: 'cyber_cyan' },
  { label: 'Emerald Green', value: 'emerald' },
  { label: 'Sunset Amber', value: 'sunset_amber' },
  { label: 'High Contrast Neon', value: 'high_contrast' },
];

function PhoneCameraStreamer() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const sendIntervalRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [facingMode, setFacingMode] = useState('environment');
  const [statusInfo, setStatusInfo] = useState({ count: 0, status: 'WAITING', target: 20 });
  const [torchOn, setTorchOn] = useState(false);
  const [hasTorch, setHasTorch] = useState(false);

  useEffect(() => {
    let wakeLock = null;
    const requestWakeLock = async () => {
      try {
        if ('wakeLock' in navigator) {
          wakeLock = await navigator.wakeLock.request('screen');
        }
      } catch (e) {
        console.warn('WakeLock error:', e);
      }
    };
    requestWakeLock();
    return () => {
      if (wakeLock) wakeLock.release().catch(() => {});
    };
  }, []);

  const startStream = async (facing) => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      const activeFacing = facing || facingMode;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: activeFacing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      const track = stream.getVideoTracks()[0];
      const capabilities = track.getCapabilities ? track.getCapabilities() : {};
      if (capabilities.torch) {
        setHasTorch(true);
      }

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/remote-camera/live`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        ws.send(JSON.stringify({ type: 'join', role: 'phone' }));

        if (!canvasRef.current) {
          canvasRef.current = document.createElement('canvas');
        }
        if (sendIntervalRef.current) clearInterval(sendIntervalRef.current);

        sendIntervalRef.current = setInterval(() => {
          if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
          const v = videoRef.current;
          if (v.readyState < 2) return;

          const c = canvasRef.current;
          c.width = 640;
          c.height = 360;
          const ctx = c.getContext('2d');
          ctx.drawImage(v, 0, 0, c.width, c.height);

          const base64 = c.toDataURL('image/jpeg', 0.65);
          wsRef.current.send(JSON.stringify({ type: 'frame', frame: base64 }));
        }, 33);
      };

      ws.onmessage = (event) => {
        try {
          const res = JSON.parse(event.data);
          if (res.event === 'ack') {
            setStatusInfo({ count: res.count, status: res.status, target: res.target || 20 });
          }
        } catch (e) {}
      };

      ws.onclose = () => {
        setConnected(false);
      };
    } catch (err) {
      alert(`Could not access mobile camera: ${err.message}. Please allow camera access in browser permissions.`);
    }
  };

  useEffect(() => {
    startStream(facingMode);
    return () => {
      if (sendIntervalRef.current) clearInterval(sendIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const handleFlip = () => {
    const nextFacing = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextFacing);
    startStream(nextFacing);
  };

  const handleTorch = async () => {
    if (!streamRef.current) return;
    const track = streamRef.current.getVideoTracks()[0];
    if (track && track.applyConstraints) {
      try {
        await track.applyConstraints({ advanced: [{ torch: !torchOn }] });
        setTorchOn(!torchOn);
      } catch (e) {
        console.warn('Torch error:', e);
      }
    }
  };

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#000', color: '#fff', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10, padding: '14px 18px', background: 'linear-gradient(to bottom, rgba(0,0,0,0.85), transparent)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 800, color: '#38bdf8', letterSpacing: 0.5 }}>
            📱 MOBILE CAMERA STREAMER
          </div>
          <div style={{ fontSize: 11, color: connected ? '#34d399' : '#f59e0b', display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: connected ? '#34d399' : '#f59e0b', display: 'inline-block' }} />
            {connected ? 'Streaming live to Laptop Dashboard' : 'Connecting to Laptop...'}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          {hasTorch && (
            <button onClick={handleTorch} className="btn btn-secondary" style={{ padding: '8px 12px', fontSize: 12 }}>
              <Zap size={15} color={torchOn ? '#facc15' : '#fff'} />
            </button>
          )}
          <button onClick={handleFlip} className="btn btn-secondary" style={{ padding: '8px 12px', fontSize: 12 }}>
            <RotateCcw size={15} /> Flip
          </button>
        </div>
      </div>

      <video ref={videoRef} playsInline autoPlay muted style={{ width: '100%', height: '100%', objectFit: 'cover' }} />

      <div style={{ position: 'absolute', bottom: 16, left: 16, right: 16, zIndex: 10, background: 'rgba(15, 23, 42, 0.9)', backdropFilter: 'blur(10px)', border: '1px solid rgba(56, 189, 248, 0.3)', borderRadius: 14, padding: '14px 18px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 700 }}>AI COUNT ON LAPTOP</div>
            <div style={{ fontSize: 24, fontWeight: 900, color: '#ffffff' }}>
              {statusInfo.count} <span style={{ fontSize: 14, color: '#38bdf8' }}>/ {statusInfo.target} items</span>
            </div>
          </div>
          <div style={{ padding: '6px 14px', borderRadius: 8, fontSize: 12, fontWeight: 800, background: statusInfo.status === 'MATCH' ? '#059669' : statusInfo.status === 'OVER' ? '#dc2626' : '#0284c7', color: '#fff' }}>
            {statusInfo.status}
          </div>
        </div>
        <div style={{ fontSize: 11.5, color: '#94a3b8', marginTop: 8, textAlign: 'center' }}>
          💻 All packing controls, target settings & audit records are running on your laptop screen.
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const isCamMode =
    typeof window !== 'undefined' &&
    (window.location.pathname.startsWith('/cam') ||
      window.location.hash === '#cam' ||
      window.location.search.includes('cam='));

  if (isCamMode) {
    return <PhoneCameraStreamer />;
  }

  // Navigation
  const [activeTab, setActiveTab] = useState('single'); // 'single' | 'live' | 'history'
  const [soundEnabled, setSoundEnabled] = useState(true);

  // Common Settings
  const [target, setTarget] = useState(20);

  // ----------------------------------------------------
  // TAB 1: LIVE PACKING ASSISTANT
  // ----------------------------------------------------
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraSourceType, setCameraSourceType] = useState('local'); // 'local' | 'remote'
  const [remotePhoneConnected, setRemotePhoneConnected] = useState(false);
  const [copiedLink, setCopiedLink] = useState(false);
  const remoteWsRef = useRef(null);
  const [liveData, setLiveData] = useState({
    count: 0,
    target: 20,
    status: 'EMPTY',
    needed: 20,
    excess: 0,
    foreign_count: 0,
    foreign_items: [],
    latency_ms: 0,
    annotated_frame: null,
  });
  const [liveParam2, setLiveParam2] = useState(26);
  const [liveMinR, setLiveMinR] = useState(15);
  const [showLiveTuning, setShowLiveTuning] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const [videoDevices, setVideoDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [facingMode, setFacingMode] = useState('environment'); // 'environment' (back) or 'user' (front)
  const [showPhoneGuide, setShowPhoneGuide] = useState(false);

  // ----------------------------------------------------
  // TAB 2: SINGLE IMAGE COUNT & MASK
  // ----------------------------------------------------
  const [batchImageFile, setBatchImageFile] = useState(null);
  const [batchImagePreview, setBatchImagePreview] = useState(null);
  const [refImageFile, setRefImageFile] = useState(null);
  const [refImagePreview, setRefImagePreview] = useState(null);
  const [productName, setProductName] = useState('');
  
  // Tuning & Appearance
  const [showTuning, setShowTuning] = useState(true);
  const [overlapThreshold, setOverlapThreshold] = useState(0.50);
  const [maskStyle, setMaskStyle] = useState('ellipse');
  const [maskOpacity, setMaskOpacity] = useState(0.45);
  const [colorPalette, setColorPalette] = useState('rainbow');
  const [showBadges, setShowBadges] = useState(true);
  const [showBorders, setShowBorders] = useState(true);
  const [showCentroids, setShowCentroids] = useState(true);

  // Single Image Output
  const [countLoading, setCountLoading] = useState(false);
  const [countResult, setCountResult] = useState(null);
  const [outputTab, setOutputTab] = useState('masked'); // 'masked' | 'sidebyside' | 'mask_only' | 'table'

  // ----------------------------------------------------
  // TAB 3: STORED BATCHES & HISTORY
  // ----------------------------------------------------
  const [batches, setBatches] = useState([]);
  const [selectedBatch, setSelectedBatch] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyViewerTab, setHistoryViewerTab] = useState('masked'); // 'masked' | 'orig'

  // Refs for WebRTC & Audio
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const sendIntervalRef = useRef(null);
  const prevStatusRef = useRef(null);

  // Play sound for packing events
  const playSound = (type) => {
    if (!soundEnabled) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);

      if (type === 'match') {
        osc.frequency.setValueAtTime(587.33, ctx.currentTime);
        osc.frequency.setValueAtTime(880.0, ctx.currentTime + 0.12);
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
        osc.start();
        osc.stop(ctx.currentTime + 0.35);
      } else if (type === 'alert') {
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(220, ctx.currentTime);
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
        osc.start();
        osc.stop(ctx.currentTime + 0.3);
      }
    } catch {
      // Audio context may require user interaction
    }
  };

  // Initial Load: Fetch default reference image & batches
  useEffect(() => {
    fetch('/api/reference')
      .then((res) => res.json())
      .then((data) => {
        if (data.exists && data.image) {
          setRefImagePreview(data.image);
        }
      })
      .catch((err) => console.error('Failed to load default reference:', err));

    loadBatches();
  }, []);

  // Fetch all batches
  const loadBatches = async () => {
    setHistoryLoading(true);
    try {
      const res = await fetch('/api/batches');
      const data = await res.json();
      setBatches(data);
      if (data.length > 0 && !selectedBatch) {
        setSelectedBatch(data[0]);
      }
    } catch (err) {
      console.error('Failed to load batches:', err);
    } finally {
      setHistoryLoading(false);
    }
  };

  // Confetti & Sound on Target Match in Live Mode
  useEffect(() => {
    if (liveData.status === 'MATCH' && prevStatusRef.current !== 'MATCH') {
      playSound('match');
      confetti({
        particleCount: 50,
        spread: 60,
        origin: { y: 0.6 },
        colors: ['#10b981', '#34d399', '#38bdf8', '#fbbf24'],
      });
    }
    prevStatusRef.current = liveData.status;
  }, [liveData.status]);

  // Update target live calculation immediately when target changes
  const handleTargetChange = (newTarget) => {
    const t = Math.max(1, parseInt(newTarget) || 1);
    setTarget(t);
    setLiveData((prev) => {
      const count = prev.count || 0;
      const diff = count - t;
      const status = count === 0 ? 'EMPTY' : diff === 0 ? 'MATCH' : diff < 0 ? 'ADD' : 'REMOVE';
      return {
        ...prev,
        target: t,
        needed: Math.max(0, t - count),
        excess: Math.max(0, count - t),
        status,
      };
    });
    if (remoteWsRef.current && remoteWsRef.current.readyState === WebSocket.OPEN) {
      remoteWsRef.current.send(JSON.stringify({ type: 'target_update', target: t }));
    }
  };

  // Remote Phone Camera WebSocket Stream listener
  useEffect(() => {
    if (cameraSourceType !== 'remote' || activeTab !== 'live') {
      if (remoteWsRef.current) {
        remoteWsRef.current.close();
        remoteWsRef.current = null;
      }
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/remote-camera/live`;
    const ws = new WebSocket(wsUrl);
    remoteWsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'join', role: 'dashboard', target }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event === 'phone_connected') {
          setRemotePhoneConnected(true);
        } else if (msg.event === 'phone_disconnected') {
          setRemotePhoneConnected(false);
        } else if (msg.event === 'status') {
          setRemotePhoneConnected(!!msg.phone_connected);
        } else if (msg.event === 'live_data') {
          setRemotePhoneConnected(true);
          setLiveData(msg);
        }
      } catch (e) {
        console.error('Remote WS parse error:', e);
      }
    };

    ws.onclose = () => {
      setRemotePhoneConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [cameraSourceType, activeTab]);

  // ----------------------------------------------------
  // LIVE WEBCAM & WEBSOCKET ENGINE
  // ----------------------------------------------------
  const updateDeviceList = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === 'videoinput');
      setVideoDevices(videoInputs);
    } catch (err) {
      console.warn('Could not enumerate media devices:', err);
    }
  };

  useEffect(() => {
    updateDeviceList();
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
      navigator.mediaDevices.addEventListener('devicechange', updateDeviceList);
      return () => navigator.mediaDevices.removeEventListener('devicechange', updateDeviceList);
    }
  }, []);

  const toggleCamera = async () => {
    if (cameraActive) {
      stopCamera();
    } else {
      await startCamera();
    }
  };

  const startCamera = async (overrideDeviceId, overrideFacing) => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }

      const activeDevId = overrideDeviceId !== undefined ? overrideDeviceId : selectedDeviceId;
      const activeFacing = overrideFacing !== undefined ? overrideFacing : facingMode;

      const videoConstraints = {
        width: { ideal: 1280 },
        height: { ideal: 720 },
      };

      if (activeDevId) {
        videoConstraints.deviceId = { exact: activeDevId };
      } else {
        videoConstraints.facingMode = { ideal: activeFacing };
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      // Re-populate device labels with readable names after permission is granted
      await updateDeviceList();

      // Connect WebSocket if not active
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/live`;
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onmessage = (event) => {
          try {
            const res = JSON.parse(event.data);
            setLiveData(res);
          } catch (e) {
            console.error('WS parse error:', e);
          }
        };

        ws.onopen = () => {
          setCameraActive(true);
          // Offscreen canvas to encode JPEG frames at 30 FPS
          if (!canvasRef.current) {
            canvasRef.current = document.createElement('canvas');
          }
          if (sendIntervalRef.current) clearInterval(sendIntervalRef.current);
          sendIntervalRef.current = setInterval(() => {
            if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
            const v = videoRef.current;
            if (v.readyState < 2) return;

            const c = canvasRef.current;
            c.width = 640;
            c.height = 360;
            const ctx = c.getContext('2d');
            ctx.drawImage(v, 0, 0, c.width, c.height);

            const base64 = c.toDataURL('image/jpeg', 0.65);
            wsRef.current.send(
              JSON.stringify({
                frame: base64,
                target: target,
                product_type: 'tyre_patches',
              })
            );
          }, 33); // ~30 FPS
        };

        ws.onclose = () => {
          if (cameraActive) setCameraActive(false);
        };
      } else {
        setCameraActive(true);
      }
    } catch (err) {
      alert(`Could not start camera: ${err.message}. Please allow camera permissions.`);
      setCameraActive(false);
    }
  };

  const handleDeviceChange = async (e) => {
    const devId = e.target.value;
    setSelectedDeviceId(devId);
    if (cameraActive) {
      await startCamera(devId);
    }
  };

  const handleFlipCamera = async () => {
    const nextFacing = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextFacing);
    setSelectedDeviceId(''); // clear exact device id so facingMode takes precedence
    if (cameraActive) {
      await startCamera('', nextFacing);
    }
  };

  const stopCamera = () => {
    if (sendIntervalRef.current) clearInterval(sendIntervalRef.current);
    if (wsRef.current) wsRef.current.close();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
    }
    setCameraActive(false);
    setLiveData((prev) => ({ ...prev, status: 'EMPTY', count: 0, annotated_frame: null }));
  };

  const handleVerifyAiSnapshot = async () => {
    if (!videoRef.current) return;
    setAiLoading(true);
    try {
      const c = document.createElement('canvas');
      c.width = videoRef.current.videoWidth || 1280;
      c.height = videoRef.current.videoHeight || 720;
      c.getContext('2d').drawImage(videoRef.current, 0, 0, c.width, c.height);

      const blob = await new Promise((resolve) => c.toBlob(resolve, 'image/jpeg', 0.95));
      const formData = new FormData();
      formData.append('file', blob, 'snapshot.jpg');
      formData.append('target', target);

      const res = await fetch('/api/verify_ai', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setAiResult(data);
    } catch (err) {
      alert(`AI verification error: ${err.message}`);
    } finally {
      setAiLoading(false);
    }
  };

  // ----------------------------------------------------
  // SINGLE IMAGE COUNT & MASK HANDLERS
  // ----------------------------------------------------
  const handleBatchImageSelect = (file) => {
    if (!file) return;
    setBatchImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setBatchImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleRefImageSelect = (file) => {
    if (!file) return;
    setRefImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setRefImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleLoadSample = async () => {
    try {
      const res = await fetch('/api/sample');
      const data = await res.json();
      if (data.exists && data.image) {
        setBatchImagePreview(data.image);
        // Convert base64 to file
        const blob = await (await fetch(data.image)).blob();
        setBatchImageFile(new File([blob], 'sample_patches.png', { type: 'image/png' }));
      }
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  };

  const handleCountAndMask = async () => {
    if (!batchImageFile) {
      alert('Please upload a batch image first!');
      return;
    }

    setCountLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', batchImageFile);
      formData.append('use_reference', Boolean(refImagePreview));
      if (refImageFile && refImagePreview) {
        formData.append('ref_file', refImageFile);
      }
      if (productName.trim()) {
        formData.append('product_name', productName.trim());
      }
      formData.append('target', target);
      formData.append('overlap_thr', overlapThreshold);
      formData.append('mask_style', maskStyle);
      formData.append('opacity', maskOpacity);
      formData.append('palette', colorPalette);
      formData.append('show_badges', showBadges);
      formData.append('show_borders', showBorders);
      formData.append('show_centroids', showCentroids);

      const res = await fetch('/api/count_single', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Count failed');
      }

      const data = await res.json();
      setCountResult(data);
      // Auto reload batches list so it appears in history immediately
      loadBatches();
    } catch (err) {
      alert(`Detection failed: ${err.message}`);
    } finally {
      setCountLoading(false);
    }
  };

  const handleClearSingle = () => {
    setBatchImageFile(null);
    setBatchImagePreview(null);
    setCountResult(null);
  };

  const handleDeleteBatch = async (batchId) => {
    if (!window.confirm(`Delete batch ${batchId}?`)) return;
    try {
      await fetch(`/api/batches/${batchId}`, { method: 'DELETE' });
      const updated = batches.filter((b) => b.id !== batchId);
      setBatches(updated);
      if (selectedBatch && selectedBatch.id === batchId) {
        setSelectedBatch(updated[0] || null);
      }
    } catch (err) {
      alert(`Failed to delete batch: ${err.message}`);
    }
  };

  const handleClearAllBatches = async () => {
    if (!window.confirm('Are you sure you want to delete ALL stored batches? This cannot be undone.')) return;
    try {
      await fetch('/api/batches', { method: 'DELETE' });
      setBatches([]);
      setSelectedBatch(null);
    } catch (err) {
      alert(`Failed to clear batches: ${err.message}`);
    }
  };

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', padding: '16px 20px', minHeight: '100vh' }}>
      {/* HEADER */}
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 16,
          padding: '16px 22px',
          borderRadius: 16,
          background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          marginBottom: 18,
          boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: 12,
              background: 'linear-gradient(135deg, #0284c7 0%, #2563eb 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 12px rgba(37,99,235,0.4)',
            }}
          >
            <Package size={24} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: 22, fontWeight: 900, color: '#ffffff', letterSpacing: '-0.3px', margin: 0 }}>
              AI Product Counter & Vision Masking
            </h1>
            <p style={{ fontSize: 13, color: '#94a3b8', margin: '3px 0 0 0' }}>
              Built with React.js & Gemini Multimodal AI • Continuous 30+ FPS Guidance & Audit Archive
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'rgba(16,185,129,0.15)',
              border: '1px solid rgba(16,185,129,0.3)',
              borderRadius: 20,
              padding: '6px 14px',
              color: '#34d399',
              fontSize: 12,
              fontWeight: 800,
            }}
          >
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#34d399' }} />
            React App Live
          </div>

          <button
            onClick={() => setSoundEnabled(!soundEnabled)}
            className="btn btn-secondary"
            style={{ padding: '8px 14px', fontSize: 13, borderRadius: 20 }}
          >
            {soundEnabled ? <Volume2 size={16} color="#38bdf8" /> : <VolumeX size={16} color="#94a3b8" />}
            {soundEnabled ? 'Audio On' : 'Muted'}
          </button>
        </div>
      </header>

      {/* TOP NAVIGATION TABS */}
      <nav
        style={{
          display: 'flex',
          gap: 8,
          background: 'rgba(15, 23, 42, 0.8)',
          padding: 6,
          borderRadius: 12,
          border: '1px solid rgba(255,255,255,0.08)',
          marginBottom: 20,
        }}
      >
        <button
          onClick={() => setActiveTab('single')}
          style={{
            flex: 1,
            padding: '12px 18px',
            borderRadius: 8,
            fontSize: 14,
            fontWeight: 800,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            cursor: 'pointer',
            border: 'none',
            transition: 'all 0.2s',
            background: activeTab === 'single' ? 'linear-gradient(135deg, #0284c7 0%, #2563eb 100%)' : 'transparent',
            color: activeTab === 'single' ? '#ffffff' : '#94a3b8',
            boxShadow: activeTab === 'single' ? '0 4px 14px rgba(37,99,235,0.35)' : 'none',
          }}
        >
          <ImageIcon size={18} />
          📸 High-Precision AI Mask & Count (Single Image)
        </button>

        <button
          onClick={() => setActiveTab('live')}
          style={{
            flex: 1,
            padding: '12px 18px',
            borderRadius: 8,
            fontSize: 14,
            fontWeight: 800,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            cursor: 'pointer',
            border: 'none',
            transition: 'all 0.2s',
            background: activeTab === 'live' ? 'linear-gradient(135deg, #0284c7 0%, #2563eb 100%)' : 'transparent',
            color: activeTab === 'live' ? '#ffffff' : '#94a3b8',
            boxShadow: activeTab === 'live' ? '0 4px 14px rgba(37,99,235,0.35)' : 'none',
          }}
        >
          <Camera size={18} />
          🎯 Live Camera & Target Packing Assistant
        </button>

        <button
          onClick={() => {
            setActiveTab('history');
            loadBatches();
          }}
          style={{
            flex: 1,
            padding: '12px 18px',
            borderRadius: 8,
            fontSize: 14,
            fontWeight: 800,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            cursor: 'pointer',
            border: 'none',
            transition: 'all 0.2s',
            background: activeTab === 'history' ? 'linear-gradient(135deg, #0284c7 0%, #2563eb 100%)' : 'transparent',
            color: activeTab === 'history' ? '#ffffff' : '#94a3b8',
            boxShadow: activeTab === 'history' ? '0 4px 14px rgba(37,99,235,0.35)' : 'none',
          }}
        >
          <Clock size={18} />
          📦 Stored Batches & History ({batches.length})
        </button>
      </nav>

      {/* ========================================================================= */}
      {/* TAB 1: SINGLE IMAGE COUNT & MASK                                           */}
      {/* ========================================================================= */}
      {activeTab === 'single' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(380px, 4.5fr) 6.5fr', gap: 20 }}>
          {/* LEFT COLUMN: Input & Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Step 1: Upload Batch Image */}
            <div className="glass-card" style={{ padding: '18px 20px' }}>
              <div style={{ fontSize: 14, fontWeight: 800, color: '#f1f5f9', marginBottom: 10 }}>
                1. Upload Batch Image
              </div>
              <label
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '2px dashed rgba(56, 189, 248, 0.3)',
                  borderRadius: 12,
                  padding: '24px 16px',
                  cursor: 'pointer',
                  background: 'rgba(0,0,0,0.2)',
                  textAlign: 'center',
                }}
              >
                {batchImagePreview ? (
                  <img
                    src={batchImagePreview}
                    alt="Batch Preview"
                    style={{ maxHeight: 180, maxWidth: '100%', objectFit: 'contain', borderRadius: 8 }}
                  />
                ) : (
                  <>
                    <Upload size={32} color="#38bdf8" style={{ marginBottom: 8 }} />
                    <div style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0' }}>Click or Drag Image Here</div>
                    <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>JPEG, PNG, WEBP</div>
                  </>
                )}
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleBatchImageSelect(e.target.files[0])}
                  style={{ display: 'none' }}
                />
              </label>

              <button
                onClick={handleLoadSample}
                className="btn btn-secondary"
                style={{ width: '100%', padding: '8px 12px', marginTop: 10, fontSize: 12.5 }}
              >
                <ImageIcon size={15} color="#38bdf8" />
                Load Sample Batch Image
              </button>
            </div>

            {/* Step 2: Upload Reference Golden Product Image */}
            <div className="glass-card" style={{ padding: '18px 20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <span style={{ fontSize: 14, fontWeight: 800, color: '#f1f5f9' }}>
                  2. Upload Reference Golden Product Image
                </span>
                {refImagePreview && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setRefImageFile(null);
                      setRefImagePreview(null);
                    }}
                    style={{
                      background: 'rgba(239, 68, 68, 0.2)',
                      border: '1px solid rgba(239, 68, 68, 0.4)',
                      borderRadius: '50%',
                      width: 26,
                      height: 26,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      color: '#fca5a5',
                      transition: 'all 0.15s',
                    }}
                    title="Remove reference image"
                  >
                    <X size={15} />
                  </button>
                )}
              </div>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 14,
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 10,
                  padding: '10px 14px',
                  cursor: 'pointer',
                  background: 'rgba(0,0,0,0.25)',
                }}
              >
                {refImagePreview ? (
                  <img
                    src={refImagePreview}
                    alt="Reference Preview"
                    style={{ width: 70, height: 70, objectFit: 'cover', borderRadius: 8, border: '1px solid #38bdf8' }}
                  />
                ) : (
                  <div
                    style={{
                      width: 70,
                      height: 70,
                      borderRadius: 8,
                      background: '#1e293b',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <Package size={24} color="#64748b" />
                  </div>
                )}
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9' }}>
                    {refImagePreview ? 'Golden Reference Sample Loaded' : 'No Reference (Count All Items)'}
                  </div>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
                    {refImagePreview
                      ? 'Only products matching this image will be counted. Foreign items are filtered out.'
                      : 'Click to upload a reference image, or proceed to count all visible products.'}
                  </div>
                </div>

                {refImagePreview && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setRefImageFile(null);
                      setRefImagePreview(null);
                    }}
                    className="btn btn-secondary"
                    style={{
                      padding: '6px 10px',
                      fontSize: 12,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      color: '#fca5a5',
                      border: '1px solid rgba(239,68,68,0.4)',
                    }}
                    title="Cancel reference image"
                  >
                    <X size={14} />
                    Cancel
                  </button>
                )}

                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleRefImageSelect(e.target.files[0])}
                  style={{ display: 'none' }}
                />
              </label>
            </div>

            {/* Step 3: Product Name (Optional) */}
            <div className="glass-card" style={{ padding: '16px 20px' }}>
              <label style={{ fontSize: 13.5, fontWeight: 800, color: '#f1f5f9', display: 'block', marginBottom: 6 }}>
                Product Name (Optional)
              </label>
              <input
                type="text"
                placeholder="e.g. tyre repair patches, bottles, pills, cans..."
                value={productName}
                onChange={(e) => setProductName(e.target.value)}
                style={{
                  width: '100%',
                  background: '#0f172a',
                  border: '1px solid rgba(255,255,255,0.15)',
                  borderRadius: 8,
                  padding: '10px 14px',
                  color: '#ffffff',
                  fontSize: 13.5,
                  outline: 'none',
                }}
              />
            </div>

            {/* Step 4: Detection Tuning & Mask Appearance Accordion */}
            <div className="glass-card" style={{ padding: '16px 20px' }}>
              <div
                onClick={() => setShowTuning(!showTuning)}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  cursor: 'pointer',
                  userSelect: 'none',
                }}
              >
                <div style={{ fontSize: 14, fontWeight: 800, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Sliders size={16} />
                  ⚙️ Detection Tuning & Mask Appearance
                </div>
                {showTuning ? <ChevronUp size={18} color="#94a3b8" /> : <ChevronDown size={18} color="#94a3b8" />}
              </div>

              {showTuning && (
                <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, fontWeight: 700, color: '#cbd5e1', marginBottom: 4 }}>
                      <span>Overlap Threshold (Higher = Keep stacked items)</span>
                      <span style={{ color: '#38bdf8' }}>{overlapThreshold}</span>
                    </div>
                    <input
                      type="range"
                      min="0.30"
                      max="0.75"
                      step="0.05"
                      value={overlapThreshold}
                      onChange={(e) => setOverlapThreshold(parseFloat(e.target.value))}
                      style={{ width: '100%', accentColor: '#0284c7' }}
                    />
                  </div>

                  <div>
                    <label style={{ fontSize: 12.5, fontWeight: 700, color: '#cbd5e1', display: 'block', marginBottom: 4 }}>
                      Mask Overlay Style
                    </label>
                    <select
                      value={maskStyle}
                      onChange={(e) => setMaskStyle(e.target.value)}
                      style={{
                        width: '100%',
                        background: '#0f172a',
                        border: '1px solid rgba(255,255,255,0.15)',
                        borderRadius: 8,
                        padding: '8px 12px',
                        color: '#ffffff',
                        fontSize: 13,
                        outline: 'none',
                      }}
                    >
                      {STYLE_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 700, color: '#cbd5e1', marginBottom: 4 }}>
                        <span>Mask Opacity</span>
                        <span style={{ color: '#38bdf8' }}>{Math.round(maskOpacity * 100)}%</span>
                      </div>
                      <input
                        type="range"
                        min="0.10"
                        max="0.90"
                        step="0.05"
                        value={maskOpacity}
                        onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                        style={{ width: '100%', accentColor: '#0284c7' }}
                      />
                    </div>

                    <div>
                      <label style={{ fontSize: 12, fontWeight: 700, color: '#cbd5e1', display: 'block', marginBottom: 4 }}>
                        Color Palette
                      </label>
                      <select
                        value={colorPalette}
                        onChange={(e) => setColorPalette(e.target.value)}
                        style={{
                          width: '100%',
                          background: '#0f172a',
                          border: '1px solid rgba(255,255,255,0.15)',
                          borderRadius: 8,
                          padding: '7px 10px',
                          color: '#ffffff',
                          fontSize: 12.5,
                          outline: 'none',
                        }}
                      >
                        {PALETTE_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 16, marginTop: 4 }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5, color: '#cbd5e1', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={showBadges}
                        onChange={(e) => setShowBadges(e.target.checked)}
                        style={{ accentColor: '#0284c7' }}
                      />
                      Numbers (#1..#N)
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5, color: '#cbd5e1', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={showBorders}
                        onChange={(e) => setShowBorders(e.target.checked)}
                        style={{ accentColor: '#0284c7' }}
                      />
                      Outlines
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5, color: '#cbd5e1', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={showCentroids}
                        onChange={(e) => setShowCentroids(e.target.checked)}
                        style={{ accentColor: '#0284c7' }}
                      />
                      Center Dots
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Step 5: Action Buttons */}
            <div style={{ display: 'flex', gap: 10 }}>
              <button
                onClick={handleCountAndMask}
                disabled={countLoading}
                className="btn btn-primary"
                style={{ flex: 3, padding: '14px 20px', fontSize: 15 }}
              >
                {countLoading ? (
                  <>
                    <RefreshCw size={18} className="animate-spin" />
                    Counting & Masking...
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    🚀 Count & Mask Products
                  </>
                )}
              </button>

              <button
                onClick={handleClearSingle}
                className="btn btn-secondary"
                style={{ flex: 1, padding: '14px', fontSize: 14 }}
              >
                Clear
              </button>
            </div>
          </div>

          {/* RIGHT COLUMN: Results & Visualizations */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {countResult ? (
              <>
                {/* Contamination Alert (if foreign items present) */}
                {countResult.foreign_count > 0 && (
                  <div
                    style={{
                      background: 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)',
                      border: '2px solid #ef4444',
                      borderRadius: 12,
                      padding: '14px 18px',
                      color: '#ffffff',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <AlertTriangle size={24} color="#fef08a" />
                      <div>
                        <div style={{ fontSize: 15, fontWeight: 800, color: '#ffffff' }}>
                          DIFFERENT PRODUCT DETECTED — REMOVE FROM BATCH!
                        </div>
                        <div style={{ fontSize: 13, color: '#fee2e2', marginTop: 2 }}>
                          Found <b style={{ color: '#fef08a' }}>{countResult.foreign_count} non-matching foreign item(s)</b>. These are flagged in RED and strictly <b style={{ color: '#fef08a', textDecoration: 'underline' }}>EXCLUDED</b> from the verified batch count.
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Primary KPI Card */}
                <div
                  style={{
                    background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
                    border: '1px solid rgba(56, 189, 248, 0.35)',
                    borderRadius: 14,
                    padding: '20px 24px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: 16,
                  }}
                >
                  <div>
                    <div style={{ fontSize: 11.5, fontWeight: 800, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                      VERIFIED PRODUCT COUNT
                    </div>
                    <div style={{ fontSize: 44, fontWeight: 900, color: '#ffffff', lineHeight: 1.1, marginTop: 4 }}>
                      {countResult.count} <span style={{ fontSize: 18, color: '#94a3b8', fontWeight: 600 }}>items</span>
                    </div>
                    <div style={{ fontSize: 12.5, color: '#64748b', marginTop: 4 }}>
                      Batch ID: <b style={{ color: '#cbd5e1' }}>{countResult.batch_id}</b> • Latency: <b style={{ color: '#34d399' }}>{countResult.latency_ms}ms</b>
                    </div>
                  </div>
                </div>

                {/* Result Tabs & Visualizations */}
                <div className="glass-card" style={{ padding: '16px' }}>
                  <div style={{ display: 'flex', gap: 8, borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: 12, marginBottom: 14 }}>
                    <button
                      onClick={() => setOutputTab('masked')}
                      className={`btn ${outputTab === 'masked' ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '8px 16px', fontSize: 13 }}
                    >
                      ✨ Masked Image
                    </button>
                    <button
                      onClick={() => setOutputTab('sidebyside')}
                      className={`btn ${outputTab === 'sidebyside' ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '8px 16px', fontSize: 13 }}
                    >
                      ↔️ Side-by-Side
                    </button>
                    <button
                      onClick={() => setOutputTab('mask_only')}
                      className={`btn ${outputTab === 'mask_only' ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '8px 16px', fontSize: 13 }}
                    >
                      ⬛ Mask Only
                    </button>
                  </div>

                  {outputTab === 'masked' && (
                    <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
                      <img src={countResult.annotated_image} alt="Masked Output" style={{ width: '100%', display: 'block' }} />
                    </div>
                  )}

                  {outputTab === 'sidebyside' && (
                    <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
                      <img src={countResult.side_by_side} alt="Side by Side" style={{ width: '100%', display: 'block' }} />
                    </div>
                  )}

                  {outputTab === 'mask_only' && (
                    <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
                      <img src={countResult.mask_only} alt="Mask Only" style={{ width: '100%', display: 'block' }} />
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div
                className="glass-card"
                style={{
                  minHeight: 460,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: 30,
                  textAlign: 'center',
                }}
              >
                <Layers size={52} color="#475569" style={{ marginBottom: 14 }} />
                <div style={{ fontSize: 16, fontWeight: 700, color: '#cbd5e1' }}>Ready to Count & Mask</div>
                <div style={{ fontSize: 13, color: '#64748b', marginTop: 6, maxWidth: 360 }}>
                  Upload a batch image on the left and click "🚀 Count & Mask Products" to see AI segmentation and enumeration.
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: LIVE CAMERA & TARGET PACKING ASSISTANT                              */}
      {/* ========================================================================= */}
      {activeTab === 'live' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(340px, 4fr) 8fr', gap: 20 }}>
          {/* LEFT: Controls & Presets */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Target Slider & Presets */}
            <div className="glass-card" style={{ padding: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <span style={{ fontSize: 14, fontWeight: 800, color: '#f1f5f9' }}>🎯 Target Product Count</span>
                <span style={{ fontSize: 20, fontWeight: 900, color: '#38bdf8' }}>{target}</span>
              </div>

              <input
                type="range"
                min="1"
                max="100"
                value={target}
                onChange={(e) => handleTargetChange(e.target.value)}
                style={{ width: '100%', accentColor: '#0284c7', marginBottom: 14 }}
              />

              <div style={{ fontSize: 11, fontWeight: 800, color: '#94a3b8', textTransform: 'uppercase', marginBottom: 8 }}>
                QUICK TARGET PRESETS:
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 6 }}>
                {[10, 15, 20, 25, 30, 50].map((num) => (
                  <button
                    key={num}
                    onClick={() => handleTargetChange(num)}
                    className="btn btn-secondary"
                    style={{
                      padding: '6px 0',
                      fontSize: 12.5,
                      borderRadius: 8,
                      background: target === num ? '#0284c7' : '#1e293b',
                      color: target === num ? '#ffffff' : '#cbd5e1',
                      border: target === num ? '1px solid #38bdf8' : '1px solid rgba(255,255,255,0.08)',
                    }}
                  >
                    {num}
                  </button>
                ))}
              </div>
            </div>

            {/* Camera Action & Options */}
            <div className="glass-card" style={{ padding: '20px' }}>
              {/* Camera Mode Tabs: Laptop Webcam vs Phone Camera */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 14 }}>
                <button
                  onClick={() => setCameraSourceType('local')}
                  className={`btn ${cameraSourceType === 'local' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '8px 10px', fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}
                >
                  <Laptop size={14} /> Laptop Webcam
                </button>
                <button
                  onClick={() => {
                    if (cameraActive) stopCamera();
                    setCameraSourceType('remote');
                  }}
                  className={`btn ${cameraSourceType === 'remote' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ padding: '8px 10px', fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}
                >
                  <Smartphone size={14} /> Phone Camera
                </button>
              </div>

              {cameraSourceType === 'local' ? (
                <>
                  {/* Camera Source Selector & Flip */}
                  <div style={{ marginBottom: 14 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <label style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8' }}>
                        📷 Camera Source / Input
                      </label>
                      <button
                        onClick={handleFlipCamera}
                        className="btn btn-secondary"
                        title="Flip camera (Back / Front / Alternate)"
                        style={{ padding: '4px 8px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}
                      >
                        <RotateCcw size={12} /> Flip
                      </button>
                    </div>
                    <select
                      value={selectedDeviceId}
                      onChange={handleDeviceChange}
                      style={{
                        width: '100%',
                        background: '#0f172a',
                        border: '1px solid rgba(255,255,255,0.15)',
                        borderRadius: 8,
                        padding: '8px 12px',
                        color: '#ffffff',
                        fontSize: 12.5,
                        outline: 'none',
                      }}
                    >
                      <option value="">
                        {facingMode === 'environment' ? '📱 Mobile Back Camera / Default' : '🤳 Mobile Front Camera / Default'}
                      </option>
                      {videoDevices.map((dev, idx) => (
                        <option key={dev.deviceId || idx} value={dev.deviceId}>
                          {dev.label || `Camera ${idx + 1}`}
                        </option>
                      ))}
                    </select>
                  </div>

                  <button
                    onClick={toggleCamera}
                    className={`btn ${cameraActive ? 'btn-danger' : 'btn-emerald'}`}
                    style={{ width: '100%', padding: '14px 20px', fontSize: 15, marginBottom: 12 }}
                  >
                    {cameraActive ? <CameraOff size={20} /> : <Camera size={20} />}
                    {cameraActive ? 'Stop Live Camera' : 'Start Live Camera'}
                  </button>

                  {cameraActive && (
                    <button
                      onClick={handleVerifyAiSnapshot}
                      disabled={aiLoading}
                      className="btn btn-primary"
                      style={{ width: '100%', padding: '12px 18px', fontSize: 13.5, marginBottom: 12 }}
                    >
                      <Sparkles size={16} />
                      {aiLoading ? 'Verifying...' : '🧠 AI Double-Check Snapshot'}
                    </button>
                  )}
                </>
              ) : (
                <>
                  {/* Remote Phone Camera Status Box */}
                  <div style={{ padding: '14px', background: 'rgba(15, 23, 42, 0.7)', borderRadius: 10, border: '1px solid rgba(56, 189, 248, 0.25)', marginBottom: 14 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <span style={{ fontSize: 12.5, fontWeight: 800, color: '#38bdf8' }}>📱 Phone Streamer</span>
                      <span style={{
                        padding: '3px 8px',
                        borderRadius: 6,
                        fontSize: 11,
                        fontWeight: 700,
                        background: remotePhoneConnected ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                        color: remotePhoneConnected ? '#34d399' : '#f59e0b',
                        border: `1px solid ${remotePhoneConnected ? '#10b981' : '#f59e0b'}`,
                      }}>
                        {remotePhoneConnected ? '● Connected' : '○ Waiting'}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>
                      {remotePhoneConnected
                        ? 'Your mobile camera is streaming live to this laptop! Full AI counting is running.'
                        : 'Open the QR code or link on the right in your phone’s Chrome browser. Zero apps needed!'}
                    </div>
                  </div>

                  {remotePhoneConnected && (
                    <button
                      onClick={handleVerifyAiSnapshot}
                      disabled={aiLoading}
                      className="btn btn-primary"
                      style={{ width: '100%', padding: '12px 18px', fontSize: 13.5, marginBottom: 12 }}
                    >
                      <Sparkles size={16} />
                      {aiLoading ? 'Verifying...' : '🧠 AI Double-Check Snapshot'}
                    </button>
                  )}
                </>
              )}
            </div>

            {/* Sensitivity Tuning (Collapsible) */}
            <div className="glass-card" style={{ padding: '16px 20px' }}>
              <div
                onClick={() => setShowLiveTuning(!showLiveTuning)}
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
              >
                <span style={{ fontSize: 13, fontWeight: 800, color: '#38bdf8' }}>⚙️ Sensitivity & Scale Tuning</span>
                {showLiveTuning ? <ChevronUp size={16} color="#94a3b8" /> : <ChevronDown size={16} color="#94a3b8" />}
              </div>

              {showLiveTuning && (
                <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>
                      <span>Sensitivity (param2)</span>
                      <span style={{ color: '#38bdf8' }}>{liveParam2}</span>
                    </div>
                    <input
                      type="range"
                      min="15"
                      max="40"
                      value={liveParam2}
                      onChange={(e) => setLiveParam2(parseInt(e.target.value))}
                      style={{ width: '100%', accentColor: '#0284c7' }}
                    />
                  </div>

                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>
                      <span>Min Radius (px)</span>
                      <span style={{ color: '#38bdf8' }}>{liveMinR}</span>
                    </div>
                    <input
                      type="range"
                      min="8"
                      max="60"
                      value={liveMinR}
                      onChange={(e) => setLiveMinR(parseInt(e.target.value))}
                      style={{ width: '100%', accentColor: '#0284c7' }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT: Live Guidance HUD & Camera Feed */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Live Guidance Card */}
            <div
              style={{
                borderRadius: 14,
                padding: '18px 22px',
                background:
                  liveData.status === 'MATCH'
                    ? 'linear-gradient(135deg, #064e3b 0%, #065f46 100%)'
                    : liveData.status === 'ADD'
                    ? 'linear-gradient(135deg, #78350f 0%, #92400e 100%)'
                    : liveData.status === 'REMOVE'
                    ? 'linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)'
                    : 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
                border:
                  liveData.status === 'MATCH'
                    ? '2px solid #10b981'
                    : liveData.status === 'ADD'
                    ? '2px solid #f59e0b'
                    : liveData.status === 'REMOVE'
                    ? '2px solid #ef4444'
                    : '1px solid rgba(255,255,255,0.1)',
                boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 900,
                    textTransform: 'uppercase',
                    letterSpacing: 0.5,
                    color:
                      liveData.status === 'MATCH'
                        ? '#34d399'
                        : liveData.status === 'ADD'
                        ? '#fbbf24'
                        : liveData.status === 'REMOVE'
                        ? '#fca5a5'
                        : '#94a3b8',
                  }}
                >
                  {liveData.status === 'MATCH'
                    ? '✅ TARGET MATCHED'
                    : liveData.status === 'ADD'
                    ? '➕ ACTION REQUIRED'
                    : liveData.status === 'REMOVE'
                    ? '➖ ACTION REQUIRED'
                    : '📹 LIVE FEED READY'}
                </div>

                <div style={{ fontSize: 11, fontWeight: 700, color: '#38bdf8' }}>
                  ⚡ {liveData.latency_ms || 12} ms (30 FPS)
                </div>
              </div>

              <div style={{ fontSize: 32, fontWeight: 900, color: '#ffffff', margin: '6px 0 2px 0' }}>
                {liveData.status === 'MATCH'
                  ? `EXACT ${target} ITEMS MATCHED!`
                  : liveData.status === 'ADD'
                  ? `ADD ${liveData.needed} MORE TO REACH ${target}`
                  : liveData.status === 'REMOVE'
                  ? `REMOVE ${liveData.excess} PRODUCTS`
                  : `0 / ${target} DETECTED`}
              </div>

              <div style={{ fontSize: 13, color: '#e2e8f0' }}>
                {liveData.status === 'EMPTY'
                  ? `Spread products across the table in front of your camera (Target: ${target} items).`
                  : `Current count: ${liveData.count} | Target: ${target}`}
              </div>

              {/* Progress bar */}
              <div style={{ width: '100%', height: 7, background: 'rgba(0,0,0,0.3)', borderRadius: 4, overflow: 'hidden', marginTop: 12 }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.min(100, Math.round((liveData.count / target) * 100))}%`,
                    background:
                      liveData.status === 'MATCH' ? '#34d399' : liveData.status === 'ADD' ? '#fbbf24' : '#ef4444',
                    transition: 'width 0.2s',
                  }}
                />
              </div>
            </div>

            {/* Video View */}
            <div
              style={{
                position: 'relative',
                borderRadius: 14,
                overflow: 'hidden',
                background: '#090d16',
                border: '1px solid rgba(255,255,255,0.1)',
                minHeight: 460,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <video ref={videoRef} playsInline muted style={{ display: 'none' }} />

              {((cameraActive && cameraSourceType === 'local') || (cameraSourceType === 'remote' && remotePhoneConnected)) && liveData.annotated_frame ? (
                <img
                  src={liveData.annotated_frame}
                  alt="Live Camera Feed"
                  style={{ width: '100%', height: 'auto', display: 'block', maxHeight: 560, objectFit: 'contain' }}
                />
              ) : cameraSourceType === 'remote' ? (
                <div style={{ textAlign: 'center', padding: '36px 20px', maxWidth: 460 }}>
                  <div style={{ display: 'inline-flex', padding: 12, background: 'rgba(56, 189, 248, 0.1)', borderRadius: 14, border: '1px solid rgba(56, 189, 248, 0.25)', marginBottom: 14 }}>
                    <Smartphone size={32} color="#38bdf8" />
                  </div>
                  <div style={{ fontSize: 18, fontWeight: 900, color: '#f8fafc', marginBottom: 4 }}>
                    Connect Mobile Camera (Zero Apps)
                  </div>
                  <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.5, marginBottom: 16 }}>
                    Scan with your phone camera or open the link below in <b>Chrome</b> on your phone:
                  </div>

                  <div style={{ display: 'inline-block', background: '#0f172a', padding: 10, borderRadius: 12, border: '1px solid rgba(56, 189, 248, 0.3)', marginBottom: 14 }}>
                    <img
                      src={`https://api.qrserver.com/v1/create-qr-code/?size=180x180&color=38-189-248&bgcolor=15-23-42&data=${encodeURIComponent(`${window.location.origin}/cam`)}`}
                      alt="Phone Camera QR Code"
                      style={{ width: 170, height: 170, display: 'block', borderRadius: 6 }}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', background: '#0f172a', padding: '8px 12px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)', marginBottom: 12 }}>
                    <span style={{ fontSize: 12.5, color: '#38bdf8', flex: 1, textAlign: 'left', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: 'monospace' }}>
                      {`${window.location.origin}/cam`}
                    </span>
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(`${window.location.origin}/cam`);
                        setCopiedLink(true);
                        setTimeout(() => setCopiedLink(false), 2500);
                      }}
                      className="btn btn-secondary"
                      style={{ padding: '6px 12px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}
                    >
                      {copiedLink ? <Check size={13} color="#34d399" /> : <Copy size={13} />}
                      {copiedLink ? 'Copied' : 'Copy Link'}
                    </button>
                  </div>

                  <div style={{ fontSize: 12, color: '#64748b' }}>
                    💡 <b>Works via USB or Wi-Fi</b>: Connect phone with USB cable (enable USB tethering) or Wi-Fi. The phone acts as the camera; this laptop stays as your main dashboard!
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '50px 20px', color: '#64748b' }}>
                  <Camera size={52} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
                  <div style={{ fontSize: 16, fontWeight: 700, color: '#cbd5e1' }}>Camera is Currently Off</div>
                  <div style={{ fontSize: 13, marginTop: 4, color: '#64748b' }}>
                    Click "Start Live Camera" on the left to begin continuous packaging verification.
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: STORED BATCHES & HISTORY                                           */}
      {/* ========================================================================= */}
      {activeTab === 'history' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Summary Banner */}
          <div
            style={{
              background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
              border: '1px solid rgba(56, 189, 248, 0.35)',
              borderRadius: 14,
              padding: '18px 24px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 16,
            }}
          >
            <div>
              <div style={{ fontSize: 16, fontWeight: 800, color: '#38bdf8' }}>📦 Batch Audit Log & Count Archive</div>
              <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 2 }}>
                Every uploaded batch image and its AI-verified count are securely archived on disk. Review any batch below.
              </div>
            </div>

            <div style={{ display: 'flex', gap: 12 }}>
              <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(255,255,255,0.1)', padding: '8px 16px', borderRadius: 10, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 700 }}>TOTAL BATCHES</div>
                <div style={{ fontSize: 20, fontWeight: 900, color: '#38bdf8', marginTop: 2 }}>{batches.length}</div>
              </div>
              <div style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid rgba(255,255,255,0.1)', padding: '8px 16px', borderRadius: 10, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 700 }}>TOTAL ITEMS</div>
                <div style={{ fontSize: 20, fontWeight: 900, color: '#34d399', marginTop: 2 }}>
                  {batches.reduce((sum, b) => sum + (b.count || 0), 0)}
                </div>
              </div>
            </div>
          </div>

          {/* Main History Layout */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(360px, 5fr) 7fr', gap: 20 }}>
            {/* Left: Table & Controls */}
            <div className="glass-card" style={{ padding: '18px 20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <span style={{ fontSize: 14, fontWeight: 800, color: '#f1f5f9' }}>Saved Batch Records</span>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button onClick={loadBatches} className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: 12 }}>
                    <RefreshCw size={14} /> Refresh
                  </button>
                  <button onClick={handleClearAllBatches} className="btn btn-danger" style={{ padding: '6px 12px', fontSize: 12 }}>
                    <Trash2 size={14} /> Clear All
                  </button>
                </div>
              </div>

              {batches.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '40px 10px', color: '#64748b' }}>
                  <Clock size={36} style={{ margin: '0 auto 8px', opacity: 0.4 }} />
                  <div style={{ fontSize: 14, color: '#cbd5e1' }}>No batches stored yet</div>
                  <div style={{ fontSize: 12, marginTop: 4 }}>Count an image in the Single Image tab to automatically save your first batch.</div>
                </div>
              ) : (
                <div style={{ maxHeight: 520, overflowY: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5, textAlign: 'left' }}>
                    <thead>
                      <tr style={{ background: '#0f172a', color: '#94a3b8', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                        <th style={{ padding: '8px 10px' }}>Batch ID</th>
                        <th style={{ padding: '8px 10px' }}>Date</th>
                        <th style={{ padding: '8px 10px' }}>Count</th>
                        <th style={{ padding: '8px 10px' }}>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batches.map((b) => (
                        <tr
                          key={b.id}
                          onClick={() => setSelectedBatch(b)}
                          style={{
                            borderBottom: '1px solid rgba(255,255,255,0.05)',
                            cursor: 'pointer',
                            background: selectedBatch?.id === b.id ? 'rgba(56, 189, 248, 0.15)' : 'transparent',
                          }}
                        >
                          <td style={{ padding: '10px', fontWeight: 800, color: selectedBatch?.id === b.id ? '#38bdf8' : '#e2e8f0' }}>
                            {b.id}
                          </td>
                          <td style={{ padding: '10px', color: '#94a3b8' }}>{b.timestamp?.split(' ')[0]}</td>
                          <td style={{ padding: '10px', fontWeight: 800, color: '#34d399' }}>{b.count}</td>
                          <td style={{ padding: '10px' }}>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteBatch(b.id);
                              }}
                              style={{ background: 'transparent', border: 'none', color: '#f87171', cursor: 'pointer', padding: 4 }}
                            >
                              <Trash2 size={14} />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Right: Selected Batch Viewer */}
            <div className="glass-card" style={{ padding: '20px' }}>
              {selectedBatch ? (
                <>
                  <div
                    style={{
                      background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
                      border: '1px solid rgba(56, 189, 248, 0.35)',
                      borderRadius: 12,
                      padding: '16px 20px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: 16,
                    }}
                  >
                    <div>
                      <div style={{ fontSize: 11, fontWeight: 800, color: '#38bdf8', textTransform: 'uppercase' }}>
                        STORED BATCH DETAILS
                      </div>
                      <div style={{ fontSize: 20, fontWeight: 900, color: '#ffffff', marginTop: 2 }}>{selectedBatch.id}</div>
                      <div style={{ fontSize: 12.5, color: '#94a3b8', marginTop: 2 }}>
                        Recorded: {selectedBatch.timestamp} • Product: {selectedBatch.product}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: 11, color: '#a5b4fc', fontWeight: 800 }}>CALCULATED COUNT</div>
                      <div style={{ fontSize: 32, fontWeight: 900, color: '#38bdf8', lineHeight: 1.1 }}>
                        {selectedBatch.count} <span style={{ fontSize: 14, color: '#94a3b8' }}>items</span>
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                    <button
                      onClick={() => setHistoryViewerTab('masked')}
                      className={`btn ${historyViewerTab === 'masked' ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '6px 14px', fontSize: 12.5 }}
                    >
                      ✨ Masked Image
                    </button>
                    <button
                      onClick={() => setHistoryViewerTab('orig')}
                      className={`btn ${historyViewerTab === 'orig' ? 'btn-primary' : 'btn-secondary'}`}
                      style={{ padding: '6px 14px', fontSize: 12.5 }}
                    >
                      Original Batch Photo
                    </button>
                  </div>

                  <div
                    style={{
                      borderRadius: 10,
                      overflow: 'hidden',
                      border: '1px solid rgba(255,255,255,0.1)',
                      background: '#0a0f1d',
                      minHeight: 280,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <img
                      key={`${selectedBatch.id}-${historyViewerTab}`}
                      src={`/api/batches/${selectedBatch.id}/image/${historyViewerTab === 'masked' ? 'masked' : 'orig'}`}
                      alt="Batch Viewer"
                      style={{ width: '100%', display: 'block', maxHeight: 480, objectFit: 'contain' }}
                      onError={(e) => {
                        e.target.style.display = 'none';
                        const fallback = document.getElementById(`batch-err-${selectedBatch.id}`);
                        if (fallback) fallback.style.display = 'flex';
                      }}
                    />
                    <div
                      id={`batch-err-${selectedBatch.id}`}
                      style={{
                        display: 'none',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '40px 20px',
                        color: '#64748b',
                        textAlign: 'center',
                      }}
                    >
                      <AlertTriangle size={32} color="#f59e0b" style={{ marginBottom: 8 }} />
                      <div style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0' }}>Image Not Found on Server</div>
                      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
                        The image file for this batch could not be found.
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: '60px 20px', color: '#64748b' }}>
                  <Package size={48} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
                  <div style={{ fontSize: 15, color: '#cbd5e1' }}>Select a batch from the list on the left</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* AI Double-Check Modal Overlay */}
      {aiResult && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.85)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: 20,
          }}
        >
          <div
            className="glass-card"
            style={{
              maxWidth: 750,
              width: '100%',
              padding: '24px',
              border: '1px solid rgba(56, 189, 248, 0.4)',
              boxShadow: '0 20px 50px rgba(0,0,0,0.8)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <Sparkles size={22} color="#38bdf8" />
                <h3 style={{ fontSize: 18, fontWeight: 900, color: '#ffffff', margin: 0 }}>
                  Gemini Deep AI Verification Result
                </h3>
              </div>
              <button
                onClick={() => setAiResult(null)}
                style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '14px 18px',
                borderRadius: 10,
                background: aiResult.status === 'MATCH' ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)',
                border: `1px solid ${aiResult.status === 'MATCH' ? '#10b981' : '#ef4444'}`,
                marginBottom: 16,
              }}
            >
              <div>
                <div style={{ fontSize: 12, fontWeight: 800, color: aiResult.status === 'MATCH' ? '#34d399' : '#fca5a5' }}>
                  {aiResult.status === 'MATCH' ? 'EXACT TARGET MATCH' : 'DISCREPANCY FOUND'}
                </div>
                <div style={{ fontSize: 24, fontWeight: 900, color: '#ffffff' }}>
                  {aiResult.count} items <span style={{ fontSize: 14, color: '#94a3b8' }}>/ Target {target}</span>
                </div>
              </div>
            </div>

            {aiResult.annotated_image && (
              <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
                <img src={aiResult.annotated_image} alt="AI Verification" style={{ width: '100%', display: 'block' }} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
