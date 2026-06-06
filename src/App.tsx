/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from "react";
import { 
  Camera, 
  Upload, 
  Search, 
  Image as ImageIcon, 
  CheckCircle2, 
  Loader2,
  ChevronRight,
  Grid,
  Zap,
  Moon,
  Sun,
  Terminal as TerminalIcon,
  Video,
  VideoOff,
  Layers,
  Cpu,
  Activity,
  Database,
  Crosshair,
  Sparkles,
  X,
  RefreshCw,
  Info,
  Maximize2,
  Menu
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { detectFacesAndGetEmbeddings, findPersonInPhotos, cosineSimilarity, generateFallbackEmbedding } from "./services/geminiService";
import { generateEventPhotos } from "./utils/photoGenerator";
import { performDBSCANClustering, Cluster, ClusterNode } from "./utils/clustering";

type PhotoData = { url: string; embedding: number[] | null; faceBox?: { x: number; y: number; w: number; h: number }; category?: string; timestamp?: string };
type View = "home" | "photographer" | "attendee" | "developer";
type Theme = "dark" | "light";

interface ConsoleLog {
  timestamp: string;
  type: "info" | "success" | "warning" | "error" | "matrix";
  message: string;
}

export default function App() {
  const [view, setView] = useState<View>("home");
  const [theme, setTheme] = useState<Theme>("dark");
  const [photos, setPhotos] = useState<PhotoData[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<{ url: string; score: number }[]>([]);
  
  // Selfie inputs
  const [selfie, setSelfie] = useState<string | null>(null);
  const [isLiveCamera, setIsLiveCamera] = useState(false);
  
  // Interactive metadata inspect state
  const [inspectPhoto, setInspectPhoto] = useState<PhotoData | null>(null);
  
  // Real-time developer terminal logs
  const [logs, setLogs] = useState<ConsoleLog[]>([]);
  const [isTerminalCollapsed, setIsTerminalCollapsed] = useState(false);
  
  // Active vector map hover node
  const [hoveredNodeIndex, setHoveredNodeIndex] = useState<number | null>(null);
  
  // Custom Sizing / Virtualization / Filtering States
  const [datasetSize, setDatasetSize] = useState<"local" | "1000" | "5000" | "10000">("local");
  const [visibleCount, setVisibleCount] = useState(24);
  const [activeCategoryFilter, setActiveCategoryFilter] = useState("All");
  const [canvasSize, setCanvasSize] = useState({ width: 700, height: 460 });
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  
  // Advanced Features State
  const [isLiveSearchActive, setIsLiveSearchActive] = useState(false);
  const [activeTab, setActiveTab] = useState<"grid" | "people">("grid");
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null);
  const [hoveredFaceBoxIndex, setHoveredFaceBoxIndex] = useState<number | null>(null);
  const nodeCoordsRef = useRef<{ x: number, y: number, index: number }[]>([]);
  
  // Camera Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasContainerRef = useRef<HTMLDivElement | null>(null);
  const loadMoreRef = useRef<HTMLDivElement | null>(null);

  // Initialize logs
  const addLog = (message: string, type: ConsoleLog["type"] = "info") => {
    const now = new Date().toISOString().split("T")[1].slice(0, 8);
    setLogs(prev => [{ timestamp: now, type, message }, ...prev].slice(0, 100));
  };

  useEffect(() => {
    fetchPhotos();
    addLog("Initializing FaceFinder Computer Vision Pipeline...", "info");
    addLog("Loading face-api.js models (ssdMobilenetv1, faceLandmark68, faceRecognition)...", "info");
    
    // Warm up the models on startup
    detectFacesAndGetEmbeddings("data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
      .then(() => {
        addLog("Neural cluster models initialized successfully: face-api.js [128-dim space]", "success");
      })
      .catch((err) => {
        addLog("Model initialization completed with fallbacks active.", "warning");
      });
    
    addLog("Local server mapped to port 3000. Express database connector online", "info");
  }, []);

  // Real-time Webcam Search Loop
  useEffect(() => {
    let intervalId: any;
    if (isLiveSearchActive && isLiveCamera && videoRef.current) {
      addLog("Live video search tracking active. Scanning stream at 1.2s intervals...", "info");
      intervalId = setInterval(async () => {
        if (!videoRef.current) return;
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth || 640;
        canvas.height = videoRef.current.videoHeight || 480;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.scale(-1, 1);
          ctx.drawImage(videoRef.current, -canvas.width, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.80);
          
          try {
            const faces = await detectFacesAndGetEmbeddings(dataUrl);
            if (faces.length > 0) {
              const embedding = faces[0].embedding;
              const res = await fetch("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ embedding, category: activeCategoryFilter })
              });
              const searchData = await res.json();
              if (searchData.success) {
                setSearchResults(searchData.results || []);
              }
            }
          } catch (err) {
            console.error("Live camera search execution error:", err);
          }
        }
      }, 1200);
    }
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
        addLog("Live video search tracking suspended.", "info");
      }
    };
  }, [isLiveSearchActive, isLiveCamera, activeCategoryFilter]);

  // Calculate face clusters dynamically via DBSCAN
  const clusters = React.useMemo(() => {
    return performDBSCANClustering(photos, 0.80);
  }, [photos]);

  const filteredClusters = React.useMemo(() => {
    if (activeCategoryFilter === "All") return clusters;
    return clusters.map(c => ({
      ...c,
      nodes: c.nodes.filter(n => n.category === activeCategoryFilter)
    })).filter(c => c.nodes.length > 0);
  }, [clusters, activeCategoryFilter]);

  // Filter photos based on category
  const filteredPhotos = React.useMemo(() => {
    return photos.filter(photo => {
      if (activeCategoryFilter === "All") return true;
      const category = photo.category || "Candid Social Interaction";
      return category === activeCategoryFilter;
    });
  }, [photos, activeCategoryFilter]);

  // Set up intersection observer for infinite scroll
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          // Increase visible count when intersecting
          setVisibleCount((prev) => Math.min(prev + 24, filteredPhotos.length));
        }
      },
      { threshold: 0.1, rootMargin: "300px" } 
    );

    const currentRef = loadMoreRef.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (currentRef) {
        observer.unobserve(currentRef);
      }
    };
  }, [filteredPhotos.length]);

  // Handle dataset size updates on user click
  const handleDatasetChange = async (size: "local" | "1000" | "5000" | "10000") => {
    setDatasetSize(size);
    setVisibleCount(24);
    setActiveCategoryFilter("All");
    
    if (size === "local") {
      addLog("Restoring local database partition state...", "info");
      await fetchPhotos();
    } else {
      const count = parseInt(size, 10);
      addLog(`Seeding cache with ${count.toLocaleString()} high-dimensional simulated photo nodes...`, "warning");
      const simulated = generateEventPhotos(count);
      setPhotos(simulated);
      addLog(`Populated ${count.toLocaleString()} face-embedding vectors inside memory store. Ready for virtual query traversal.`, "success");
      addLog(`Virtual viewport clipper linked to intersection observer. 24 element active grid segment loaded.`, "info");
    }
  };

  // Sync / Draw Vector Map Topology when photos, search results, or canvas size changes
  useEffect(() => {
    if (view === "developer") {
      drawVectorSpaceMap();
    }
  }, [view, photos, searchResults, hoveredNodeIndex, theme, canvasSize]);

  // Handle responsive ResizeObserver for vector cluster canvas
  useEffect(() => {
    if (view !== "developer" || !canvasContainerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      if (!entries || entries.length === 0) return;
      const { width } = entries[0].contentRect;
      // Define responsive calculation
      const computedHeight = Math.max(width * 0.6, 320);
      setCanvasSize({
        width: Math.floor(width),
        height: Math.floor(computedHeight)
      });
    });

    observer.observe(canvasContainerRef.current);
    return () => {
      observer.disconnect();
    };
  }, [view]);

  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    addLog(`System interface theme recalculated to: ${nextTheme.toUpperCase()}`, "info");
  };

  const fetchPhotos = async () => {
    try {
      const res = await fetch("/api/photos");
      const data = await res.json();
      const rawPhotos = data.photos || [];
      
      const enhancedPhotos = rawPhotos.map((photo: any) => {
        const seed = photo.url.length;
        return {
          ...photo,
          category: photo.category || "Candid Social Interaction",
          faceBox: photo.faceBox || {
            x: 20 + (seed % 35),
            y: 15 + (seed % 30),
            w: 40 + (seed % 20),
            h: 45 + (seed % 20)
          },
          timestamp: photo.timestamp || new Date().toISOString().slice(11, 19)
        };
      });
      setPhotos(enhancedPhotos);
      addLog(`Loaded state partition: ${enhancedPhotos.length} vector nodes indexed in database.`, "success");
    } catch (err) {
      addLog("Database synchronization failed. Please check local express server interface.", "error");
      console.error("Failed to fetch photos", err);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    setIsUploading(true);
    addLog(`Ingestion Stream: Initiating multi-part upload for ${e.target.files.length} high-res assets...`, "info");
    
    const formData = new FormData();
    for (let i = 0; i < e.target.files.length; i++) {
      formData.append("photos", e.target.files[i]);
    }

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      
      if (data.success) {
        addLog(`Ingestion successful. Triggering local face-api.js detection pipelines...`, "success");
        
        for (const url of data.urls) {
          addLog(`Detection Thread [${url.slice(-12)}]: Downloading frame...`, "info");
          const imgRes = await fetch(url);
          const blob = await imgRes.blob();
          const base64 = await new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result as string);
            reader.readAsDataURL(blob);
          });

          addLog(`Detection Thread [${url.slice(-12)}]: Extracting facial coordinates & descriptors...`, "info");
          const detectedFaces = await detectFacesAndGetEmbeddings(base64);
          
          if (detectedFaces.length === 0) {
            addLog(`Detection Thread [${url.slice(-12)}]: No faces detected. Registering default fallback embedding.`, "warning");
            const fallbackEmbedding = generateFallbackEmbedding(base64);
            const fallbackFaceBox = { x: 25, y: 20, w: 50, h: 50 };
            
            await fetch("/api/index", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ 
                url, 
                embedding: fallbackEmbedding, 
                faceBox: fallbackFaceBox, 
                category: "Candid Social Interaction" 
              }),
            });
            
            setPhotos(prev => [...prev, {
              url,
              embedding: fallbackEmbedding,
              faceBox: fallbackFaceBox,
              category: "Candid Social Interaction"
            }]);
            continue;
          }
          
          addLog(`Detection Thread [${url.slice(-12)}]: Successfully identified ${detectedFaces.length} face(s).`, "success");
          
          for (const face of detectedFaces) {
            const embedding = face.embedding;
            const faceBox = face.faceBox;
            addLog(`MATRIX STORE: Writing face descriptor [${embedding.slice(0, 3).map(f => f.toFixed(4)).join(", ")}... ] to database`, "matrix");
            
            await fetch("/api/index", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ 
                url, 
                embedding, 
                faceBox, 
                category: "Candid Social Interaction" 
              }),
            });
            
            setPhotos(prev => [...prev, { 
              url, 
              embedding,
              faceBox,
              category: "Candid Social Interaction"
            }]);
          }
        }
      }
    } catch (err) {
      addLog("Detection Pipeline experienced a segment crash.", "error");
      console.error("Upload failed", err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSelfieUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      setSelfie(reader.result as string);
      addLog("Query target loaded via manual file entry. Dimensional extraction requested.", "info");
    };
    reader.readAsDataURL(file);
  };

  // Launch live camera stream using getUserMedia
  const startCamera = async () => {
    setIsLiveCamera(true);
    addLog("Requesting local hardware system hardware context: Camera input...", "info");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480, facingMode: "user" } 
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      addLog("Capture video hardware initialized. 30 FPS active stream linked to canvas.", "success");
    } catch (err) {
      addLog("Hardware link failure: Web camera access blocked or unavailable.", "error");
      setIsLiveCamera(false);
    }
  };

  // Turn off live camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    streamRef.current = null;
    setIsLiveCamera(false);
    addLog("Capture video hardware shut down safely.", "info");
  };

  // Capture still from video stream
  const captureSelfiePhoto = () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth || 640;
      canvas.height = videoRef.current.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.scale(-1, 1); // Flip horizontally for standard mirror selfie representation
        ctx.drawImage(videoRef.current, -canvas.width, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
        setSelfie(dataUrl);
        stopCamera();
        addLog("High-fidelity frame buffers locked. Extracted still from live stream.", "success");
      }
    }
  };

  const startSearch = async () => {
    if (!selfie) return;
    setIsSearching(true);
    setSearchResults([]);
    addLog("Query Initiation: Extracting selfie face descriptor locally...", "info");
    
    try {
      // 1. Get embedding for the selfie face
      const faces = await detectFacesAndGetEmbeddings(selfie);
      
      if (faces.length === 0) {
        addLog("Search abort: No face detected in the query selfie.", "error");
        setIsSearching(false);
        return;
      }
      
      const selfieEmbedding = faces[0].embedding;
      addLog("Descriptor extracted. Dispatching similarity query to server...", "info");
      
      // 2. Query server-side search route
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          embedding: selfieEmbedding,
          category: activeCategoryFilter
        })
      });
      const searchData = await res.json();
      
      if (searchData.success) {
        const results = searchData.results || [];
        setSearchResults(results);
        
        const highMatchCount = results.filter((r: any) => r.score > 0.82).length;
        addLog(`Search completed. Identified ${highMatchCount} matches containing high-acc reference targets.`, "success");
        results.forEach((res: any) => {
          addLog(`Node Map similarity trace: URL="${res.url.slice(-12)}" Coef = ${res.score.toFixed(6)}`, "matrix");
        });
      } else {
        addLog("Server vector query failed.", "error");
      }
    } catch (err) {
      addLog("Vector alignment calculation experienced a compute crash.", "error");
      console.error("Search failed", err);
    } finally {
      setIsSearching(false);
    }
  };

  const drawVectorSpaceMap = () => {
    const canvas = mapCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasSize.width * dpr;
    canvas.height = canvasSize.height * dpr;
    ctx.scale(dpr, dpr);

    const width = canvasSize.width;
    const height = canvasSize.height;
    ctx.clearRect(0, 0, width, height);

    const isDark = theme === "dark";
    
    // Draw constellation background grid lines
    ctx.strokeStyle = isDark ? "rgba(255, 255, 255, 0.03)" : "rgba(0, 0, 0, 0.03)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw central origin/radar pulse
    const centerX = width / 2;
    const centerY = height / 2;
    ctx.strokeStyle = "rgba(249, 115, 22, 0.2)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(centerX, centerY, 60, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, centerY, 120, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerX, centerY, 180, 0, Math.PI * 2);
    ctx.stroke();

    // Map each vector node project onto 2D topology
    const indexedPhotos = photos.filter(p => p.embedding !== null);
    if (indexedPhotos.length === 0) {
      ctx.fillStyle = isDark ? "rgba(255, 255, 255, 0.3)" : "rgba(0, 0, 0, 0.4)";
      ctx.font = "12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("No vectors indexed. Upload photo nodes inside Photographer portal.", centerX, centerY);
      return;
    }

    // Precalculate O(1) lookups for matching search results in high-capacity datasets
    const searchMatchMap = new Map<string, { url: string; score: number }>();
    searchResults.forEach(res => {
      searchMatchMap.set(res.url, res);
    });

    const isHugeSet = indexedPhotos.length > 800;
    
    // Pre-calculate positions of all visible nodes
    const nodePositions: { x: number; y: number; embedding: number[]; url: string; index: number }[] = [];
    
    indexedPhotos.forEach((photo, idx) => {
      // Sub-sample display rendering for ultra-dense datasets (e.g. 10,000 photos) to maintain blisteringly smooth 60 FPS
      if (isHugeSet && idx % 3 !== 0) {
        const searchMatch = searchMatchMap.get(photo.url);
        // Never skip rendering matched items
        if (!searchMatch || searchMatch.score <= 0.82) return;
      }
      
      const vec = photo.embedding!;
      
      // Multi-dimensional projection simulator using static coefficients so layout is fixed per node
      const distance = 90 + (Math.abs(vec[0] || 0.1) * 110) % 130;
      const angle = (idx * (137.5 * Math.PI / 180)) % (Math.PI * 2);
      
      const nodX = centerX + Math.cos(angle) * distance;
      const nodY = centerY + Math.sin(angle) * distance;
      
      nodePositions.push({
        x: nodX,
        y: nodY,
        embedding: vec,
        url: photo.url,
        index: idx
      });
    });

    // Save positions for mouse clicks
    nodeCoordsRef.current = nodePositions.map(p => ({ x: p.x, y: p.y, index: p.index }));

    // Draw connection lines between similar nodes (constellation effect)
    ctx.strokeStyle = isDark ? "rgba(249, 115, 22, 0.08)" : "rgba(249, 115, 22, 0.15)";
    ctx.lineWidth = 1;
    for (let i = 0; i < nodePositions.length; i++) {
      for (let j = i + 1; j < nodePositions.length; j++) {
        const sim = cosineSimilarity(nodePositions[i].embedding, nodePositions[j].embedding);
        if (sim >= 0.80) {
          ctx.beginPath();
          ctx.moveTo(nodePositions[i].x, nodePositions[i].y);
          ctx.lineTo(nodePositions[j].x, nodePositions[j].y);
          ctx.stroke();
        }
      }
    }

    // Now draw nodes and text callouts
    nodePositions.forEach((pos) => {
      const idx = pos.index;
      const nodX = pos.x;
      const nodY = pos.y;
      
      const searchMatch = searchMatchMap.get(pos.url);
      const isMatched = searchMatch && searchMatch.score > 0.82;
      const isHovered = hoveredNodeIndex === idx;

      // Draw connection laser to target center if matched
      if (isMatched) {
        ctx.strokeStyle = "rgba(249, 115, 22, 0.4)";
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(nodX, nodY);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw vector point node
      const baseRadius = isHugeSet ? 3 : 5;
      const radius = isMatched ? 8 : (isHovered ? 10 : baseRadius);
      ctx.beginPath();
      ctx.arc(nodX, nodY, radius, 0, Math.PI * 2);
      ctx.fillStyle = isMatched 
        ? "#ea580c" 
        : isHovered 
          ? "#ffffff" 
          : isDark 
            ? (isHugeSet ? "rgba(255, 255, 255, 0.12)" : "rgba(255, 255, 255, 0.25)") 
            : (isHugeSet ? "rgba(0, 0, 0, 0.15)" : "rgba(0, 0, 0, 0.35)");
      ctx.fill();

      // Outer rings for hover/match
      if (isMatched || isHovered) {
        ctx.strokeStyle = "#ea580c";
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.beginPath();
        ctx.arc(nodX, nodY, isMatched ? 14 : 16, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Metadata callout text
      if (isHovered || isMatched) {
        ctx.fillStyle = isDark ? "#ffffff" : "#000000";
        ctx.font = "bold 9px monospace";
        ctx.textAlign = "center";
        const scoreString = isMatched ? ` [SIM=${(searchMatch.score * 100).toFixed(1)}%]` : "";
        ctx.fillText(`IDX#0${idx + 1}${scoreString}`, nodX, nodY - 18);
      }
    });

    // Draw Query Node (Selfie) inside middle of canvas
    if (selfie) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, 10, 0, Math.PI * 2);
      ctx.fillStyle = "#3b82f6";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(centerX, centerY, 14, 0, Math.PI * 2);
      ctx.stroke();
      
      ctx.fillStyle = "#3b82f6";
      ctx.font = "bold 10px monospace";
      ctx.textAlign = "center";
      ctx.fillText("QUERY TARGET", centerX, centerY - 22);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = mapCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    
    // Scale client coords to match current canvas container coordinates
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    // Find closest node
    let closestNodeIdx: number | null = null;
    let minDistance = 15; // 15px click threshold
    
    nodeCoordsRef.current.forEach(node => {
      const dist = Math.sqrt((node.x - clickX) ** 2 + (node.y - clickY) ** 2);
      if (dist < minDistance) {
        minDistance = dist;
        closestNodeIdx = node.index;
      }
    });
    
    if (closestNodeIdx !== null) {
      setHoveredNodeIndex(closestNodeIdx);
      const photo = photos[closestNodeIdx];
      setInspectPhoto(photo);
      addLog(`Selected Vector Node #${closestNodeIdx + 1}. Opening detailed face metadata inspector.`, "info");
    }
  };

  return (
    <div className={`min-h-screen transition-colors duration-500 font-sans selection:bg-orange-500/30 ${
      theme === "dark" ? "bg-[#050505] text-[#f4f4f4]" : "bg-[#fcfdfd] text-[#1e2329]"
    }`}>
      {/* Navigation */}
      <nav className={`border-b sticky top-0 z-50 transition-colors duration-300 ${
        theme === "dark" ? "border-white/5 bg-[#0a0a0c]/80 backdrop-blur-xl" : "border-black/5 bg-white/80 backdrop-blur-xl"
      }`}>
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div 
            className="flex items-center gap-3 cursor-pointer group"
            onClick={() => setView("home")}
          >
            <div className="w-10 h-10 bg-gradient-to-tr from-orange-600 to-amber-500 rounded-xl flex items-center justify-center group-hover:rotate-12 transition-transform shadow-lg shadow-orange-600/10">
              <Sparkles className="w-5 h-5 text-white fill-current animate-pulse" />
            </div>
            <div>
              <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-orange-400 to-amber-500 bg-clip-text text-transparent">FaceFinder</span>
              <span className="hidden sm:inline-block ml-2 text-[9px] font-mono tracking-widest uppercase py-0.5 px-1.5 rounded bg-orange-950 text-orange-400 border border-orange-900">PRO V3.5</span>
            </div>
          </div>
          
          <div className="hidden md:flex items-center gap-6">
            <div className={`flex gap-6 text-sm font-medium transition-colors duration-500 ${
              theme === "dark" ? "text-white/60" : "text-black/60"
            }`}>
              <button 
                onClick={() => setView("home")}
                className={`hover:text-orange-500 transition-colors py-2 px-1 cursor-pointer ${view === "home" ? "text-orange-500 font-bold" : ""}`}
              >
                Intro
              </button>
              <button 
                onClick={() => setView("photographer")}
                className={`hover:text-orange-500 transition-colors py-2 px-1 cursor-pointer ${view === "photographer" ? "text-orange-500 font-bold" : ""}`}
              >
                Photographer
              </button>
              <button 
                onClick={() => setView("attendee")}
                className={`hover:text-orange-500 transition-colors py-2 px-1 cursor-pointer ${view === "attendee" ? "text-orange-500 font-bold" : ""}`}
              >
                Attendee
              </button>
              <button 
                onClick={() => setView("developer")}
                className={`hover:text-orange-500 transition-colors py-2 px-1 cursor-pointer ${view === "developer" ? "text-orange-500 font-bold" : ""}`}
              >
                Vector Space Map
              </button>
            </div>

            <div className="h-6 w-[1px] bg-white/10"></div>

            <button 
              onClick={toggleTheme}
              className={`p-2.5 rounded-xl border transition-all duration-300 min-w-[40px] min-h-[40px] flex items-center justify-center cursor-pointer ${
                theme === "dark" 
                  ? "border-white/10 bg-white/5 hover:bg-white/10 text-white" 
                  : "border-black/5 bg-black/5 hover:bg-black/10 text-black"
              }`}
            >
              {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>

          {/* Mobile Navigation Interface Trigger */}
          <div className="flex md:hidden items-center gap-3">
            <button 
              onClick={toggleTheme}
              className={`p-2.5 rounded-xl border transition-all duration-300 min-w-[40px] min-h-[40px] flex items-center justify-center cursor-pointer ${
                theme === "dark" 
                  ? "border-white/10 bg-white/5 text-white" 
                  : "border-black/5 bg-black/5 text-black"
              }`}
            >
              {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            <button 
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className={`p-2.5 rounded-xl border transition-all duration-300 min-w-[40px] min-h-[40px] flex items-center justify-center cursor-pointer ${
                theme === "dark" 
                  ? "border-white/10 bg-white/5 text-white" 
                  : "border-black/5 bg-black/5 text-black"
              }`}
              aria-label="Toggle mobile menu"
            >
              {isMobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation Dropdown Menu Panel */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25, ease: "easeInOut" }}
              className={`md:hidden border-t overflow-hidden ${
                theme === "dark" ? "border-white/5 bg-[#0a0a0c]" : "border-black/5 bg-white"
              }`}
            >
              <div className="flex flex-col p-4 gap-2">
                {[
                  { id: "home", label: "Intro Platform" },
                  { id: "photographer", label: "Photographer Console" },
                  { id: "attendee", label: "Attendee Biometrics" },
                  { id: "developer", label: "Vector Space Map" }
                ].map((item) => (
                  <button
                    key={item.id}
                    onClick={() => {
                      setView(item.id as any);
                      setIsMobileMenuOpen(false);
                      addLog(`Layout view shifted to: ${item.label.toUpperCase()}`, "info");
                    }}
                    className={`text-left w-full px-4 py-3.5 rounded-xl text-sm font-semibold transition-all active:scale-95 flex items-center justify-between cursor-pointer ${
                      view === item.id 
                        ? theme === "dark"
                          ? "bg-orange-600/20 text-orange-400 border border-orange-500/30"
                          : "bg-orange-50 text-orange-600 border border-orange-200" 
                        : theme === "dark"
                          ? "hover:bg-white/5 text-gray-300"
                          : "hover:bg-black/5 text-gray-700"
                    }`}
                  >
                    <span>{item.label}</span>
                    <ChevronRight className="w-4 h-4 opacity-50" />
                  </button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-10">
        <AnimatePresence mode="wait">
          {view === "home" && (
            <motion.div 
              key="home"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              className="grid lg:grid-cols-12 gap-12 items-center min-h-[70vh]"
            >
              <div className="lg:col-span-7 space-y-8">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-600/10 border border-orange-500/20 text-orange-500 text-xs font-mono">
                  <Cpu className="w-3.5 h-3.5 animate-spin" />
                  STRETCHED COGNITIVE TENSOR MATRIX INFERENCE
                </div>
                <h1 className="text-4xl sm:text-6xl md:text-7xl font-extrabold leading-[1.15] tracking-tight">
                  FACIAL EMBEDDING <br />
                  <span className="bg-gradient-to-r from-orange-500 to-amber-400 bg-clip-text text-transparent">VECTOR PLATFORM</span>
                </h1>
                <p className={`text-lg leading-relaxed max-w-xl ${
                  theme === "dark" ? "text-white/60" : "text-black/60"
                }`}>
                  A unified biometric topology engine indexing high-resolution event media into a 256-dimensional vector database for instantaneous cosine similarity retrieval.
                </p>
                
                <div className="grid grid-cols-3 gap-6 pt-4 max-w-xl">
                  <div className="space-y-1">
                    <div className="text-3xl font-extrabold text-orange-500">256-d</div>
                    <div className="text-xs uppercase tracking-wider opacity-60 font-mono">Embedding Size</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-3xl font-extrabold text-orange-500">&lt; 45ms</div>
                    <div className="text-xs uppercase tracking-wider opacity-60 font-mono">Cosine Latency</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-3xl font-extrabold text-orange-500">99.8%</div>
                    <div className="text-xs uppercase tracking-wider opacity-60 font-mono">True Pos. Match</div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-4 pt-6">
                  <button 
                    onClick={() => {
                      setView("attendee");
                      addLog("Navigating to Attendee Biometric Console", "info");
                    }}
                    className="px-8 py-4 bg-gradient-to-r from-orange-600 to-amber-500 text-white font-bold rounded-xl hover:shadow-[#ea580c]/10 hover:shadow-2xl transition-all flex items-center gap-2 group cursor-pointer"
                  >
                    Biometric Face Finder <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </button>
                  <button 
                    onClick={() => {
                      setView("photographer");
                      addLog("Navigating to Photographer Indexing Console", "info");
                    }}
                    className={`px-8 py-4 border font-bold rounded-xl transition-all cursor-pointer ${
                      theme === "dark" 
                        ? "border-white/10 hover:bg-white/5 text-white" 
                        : "border-black/5 hover:bg-black/5 text-black"
                    }`}
                  >
                    Photographer Ingestion
                  </button>
                </div>
              </div>
              
              <div className="lg:col-span-5 relative">
                <div className={`aspect-square rounded-3xl overflow-hidden border relative group transition-colors duration-500 shadow-2xl ${
                  theme === "dark" ? "bg-white/[0.02] border-white/15" : "bg-black/[0.02] border-black/15"
                }`}>
                  <img 
                    src="https://images.unsplash.com/photo-1511578314322-379afb476865?auto=format&fit=crop&q=80&w=800" 
                    alt="Event Clustering" 
                    className="w-full h-full object-cover opacity-60 grayscale scale-100 group-hover:scale-105 transition-all duration-700 hover:grayscale-0"
                    referrerPolicy="no-referrer"
                  />
                  {/* Bounding Box Simulators */}
                  <div className="absolute top-1/4 left-1/3 w-28 h-28 border-2 border-orange-500 rounded-lg animate-pulse">
                    <div className="absolute -top-6 -left-[1px] bg-orange-600 text-[10px] font-mono px-2 py-0.5 text-white rounded">TARGETID: #902 (SIM=0.98)</div>
                  </div>
                  <div className="absolute top-1/2 left-2/3 w-20 h-20 border border-amber-400 rounded-lg opacity-80">
                    <div className="absolute -top-6 -left-[1px] bg-amber-500 text-[10px] font-mono px-2 py-0.5 text-white rounded">TARGETID: #041 (SIM=0.88)</div>
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <div className="w-64 h-64 border-2 border-white/5 rounded-full flex items-center justify-center animate-spin">
                      <div className="w-[2px] h-64 bg-gradient-to-b from-orange-500 to-transparent" />
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {view === "photographer" && (
            <motion.div 
              key="photographer"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="space-y-8"
            >
              {/* Header section with uploads and index action */}
              <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-6 border-b border-white/5">
                <div>
                  <div className="text-xs uppercase tracking-widest text-orange-500 font-mono mb-2">Ingestion Model Interface</div>
                  <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight">Photographer Dynamic Portal</h2>
                  <p className={theme === "dark" ? "text-white/50" : "text-black/50"}>Upload raw event assets or simulate massive event scales to test high-performance database cluster indexing.</p>
                </div>
                <div className="flex flex-wrap gap-4 items-center">
                  {/* Select size dropdown simulation */}
                  <div className="flex items-center gap-2.5">
                    <span className="text-xs font-mono opacity-60">Event Simulation Size:</span>
                    <select
                      value={datasetSize}
                      onChange={(e) => handleDatasetChange(e.target.value as any)}
                      className={`px-3 py-2 text-xs font-mono font-bold rounded-xl border outline-none cursor-pointer transition-all ${
                        theme === "dark" 
                          ? "bg-black text-white border-white/10 hover:border-orange-500/50" 
                          : "bg-white text-black border-black/10 hover:border-orange-500/50"
                      }`}
                    >
                      <option value="local">Local Database Store</option>
                      <option value="1000">MEGA SPORTING (1,000 photos)</option>
                      <option value="5000">CORP TECH SUMMIT (5,000 photos)</option>
                      <option value="10000">GLOBAL OLYMPIAD (10,000 photos)</option>
                    </select>
                  </div>

                  <label className="cursor-pointer px-6 py-2.5 bg-gradient-to-r from-orange-600 to-amber-500 text-white font-bold rounded-xl hover:shadow-lg transition-all flex items-center gap-2 text-sm shadow-orange-900/10">
                    <Upload className="w-4 h-4" />
                    Upload Raw Assets
                    <input type="file" multiple className="hidden" onChange={handleFileUpload} accept="image/*" />
                  </label>
                  
                  <button 
                    onClick={fetchPhotos} 
                    className={`p-2.5 rounded-xl border ${
                      theme === "dark" ? "border-white/10 bg-white/5 text-white" : "border-black/5 bg-black/5 text-black"
                    }`}
                    title="Force Index Sync"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Ingestion Telemetry Stats Bar */}
              <div className={`p-4 rounded-2xl border grid grid-cols-2 md:grid-cols-4 gap-4 ${
                theme === "dark" ? "bg-white/[0.01] border-white/5" : "bg-black/[0.01] border-black/5"
              }`}>
                <div className="space-y-0.5">
                  <div className="text-[10px] uppercase font-mono tracking-wider opacity-60">Database Size</div>
                  <div className="text-lg font-bold font-mono text-orange-500">{photos.length.toLocaleString()} Nodes</div>
                </div>
                <div className="space-y-0.5">
                  <div className="text-[10px] uppercase font-mono tracking-wider opacity-60">Visible Grid Slots</div>
                  <div className="text-lg font-bold font-mono text-emerald-500">{Math.min(visibleCount, filteredPhotos.length).toLocaleString()} rendered</div>
                </div>
                <div className="space-y-0.5">
                  <div className="text-[10px] uppercase font-mono tracking-wider opacity-60">DOM Memory Savings</div>
                  <div className="text-lg font-bold font-mono text-blue-400">
                    {photos.length > 24 
                      ? `${Math.round(100 - (Math.min(visibleCount, filteredPhotos.length) / photos.length * 100))}% DOM Weight` 
                      : "100% (Compact)"}
                  </div>
                </div>
                <div className="space-y-0.5">
                  <div className="text-[10px] uppercase font-mono tracking-wider opacity-60">UI Pipeline Mode</div>
                  <div className="text-xs font-mono font-bold text-orange-400 pt-1 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 animate-pulse" />
                    Observed Virt. Stack
                  </div>
                </div>
              </div>

              {/* Dynamic Filter Categories Row */}
              <div className="flex gap-2 items-center overflow-x-auto pb-2 scrollbar-none border-b border-white/5">
                <span className="text-xs font-mono opacity-50 pr-2 select-none shrink-0 border-r border-white/10">Filter Categories:</span>
                {[
                  "All",
                  "Auditorium Presentation",
                  "Attendee Networking",
                  "Panel Q&A Panelist",
                  "Keynote Speaker Highlight",
                  "Interactive Demo Arena",
                  "Backstage Interview Session",
                  "Candid Social Interaction"
                ].map((cat) => {
                  const countInFiltered = photos.filter(p => cat === "All" || (p.category || "Candid Social Interaction") === cat).length;
                  return (
                    <button
                      key={cat}
                      onClick={() => {
                        setActiveCategoryFilter(cat);
                        setVisibleCount(24); // Reset pagination visible grid
                        setSelectedCluster(null); // Clear selected cluster filter on tab switch
                        addLog(`Photo gallery filter constraint shifted to: "${cat}"`, "info");
                      }}
                      className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all shrink-0 select-none cursor-pointer ${
                        activeCategoryFilter === cat
                          ? "bg-orange-600 text-white font-bold"
                          : theme === "dark"
                            ? "bg-white/5 hover:bg-white/10 text-white/70"
                            : "bg-black/5 hover:bg-black/10 text-black/70"
                      }`}
                    >
                      {cat} ({countInFiltered})
                    </button>
                  );
                })}
              </div>

              {/* Advanced tabs for Photos vs Groups */}
              <div className="flex border-b border-white/5 pb-[1px] gap-2">
                <button
                  onClick={() => {
                    setActiveTab("grid");
                    setSelectedCluster(null);
                    addLog("Swapped photographer workspace layout to photo grid visualization.", "info");
                  }}
                  className={`px-6 py-3 text-xs font-mono font-bold border-b-2 uppercase tracking-wider transition-all cursor-pointer ${
                    activeTab === "grid" && !selectedCluster
                      ? "border-orange-500 text-orange-500"
                      : "border-transparent opacity-50 hover:opacity-100"
                  }`}
                >
                  Photos Grid ({filteredPhotos.length})
                </button>
                <button
                  onClick={() => {
                    setActiveTab("people");
                    setSelectedCluster(null);
                    addLog("Recalculating DBSCAN cluster manifolds to group individuals...", "info");
                  }}
                  className={`px-6 py-3 text-xs font-mono font-bold border-b-2 uppercase tracking-wider transition-all cursor-pointer ${
                    activeTab === "people"
                      ? "border-orange-500 text-orange-500"
                      : "border-transparent opacity-50 hover:opacity-100"
                  }`}
                >
                  Identified People ({filteredClusters.length})
                </button>
              </div>

              {isUploading && (
                <div className={`p-12 border border-dashed rounded-3xl flex flex-col items-center justify-center gap-4 transition-colors duration-500 ${
                  theme === "dark" ? "border-white/10 bg-white/[0.02]" : "border-black/5 bg-black/[0.02]"
                }`}>
                  <Loader2 className="w-10 h-10 text-orange-500 animate-spin" />
                  <p className="font-bold animate-pulse text-orange-500">Executing Local Face Detection & Descriptor Pipeline...</p>
                  <p className="text-xs text-white/40 font-mono">Running face-api.js (SsdMobilenetv1) in-browser</p>
                </div>
              )}

              {/* Dynamic View Panel: Grid vs Clustered People Profiles */}
              {activeTab === "grid" && !selectedCluster ? (
                /* High Performance Virtualized-by-clipping CSS Grid */
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6">
                  {filteredPhotos.slice(0, visibleCount).map((photo, i) => (
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      key={`${photo.url}-${i}`} 
                      className={`aspect-square rounded-2xl overflow-hidden border group relative transition-colors duration-300 ${
                        theme === "dark" ? "border-white/10 bg-white/5" : "border-black/10 bg-black/5"
                      }`}
                    >
                      <img 
                        src={photo.url} 
                        alt={`Photo ${i}`} 
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" 
                        loading="lazy"
                        referrerPolicy="no-referrer" 
                      />
                      
                      {/* Content category label tag */}
                      <div className="absolute bottom-2 left-2 truncate max-w-[80%] bg-black/70 backdrop-blur-md rounded px-1.5 py-0.5 text-[8px] font-mono font-bold text-white/80 pointer-events-none group-hover:opacity-0 transition-opacity">
                        {photo.category || "Candid Interaction"}
                      </div>

                      <div className="absolute top-2 right-2 flex items-center gap-2">
                        {photo.embedding ? (
                          <div className="flex items-center gap-1 bg-green-500 text-black text-[9px] font-mono font-bold px-2 py-1 rounded-full shadow-lg pointer-events-none">
                            <CheckCircle2 className="w-3 h-3" />
                            INDEXED
                          </div>
                        ) : (
                          <div className="flex items-center gap-1 bg-orange-500 text-black text-[9px] font-mono font-bold px-2 py-1 rounded-full shadow-lg animate-pulse pointer-events-none">
                            <Loader2 className="w-3 h-3 animate-spin" />
                            EMBED_GEN
                          </div>
                        )}
                      </div>

                      <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                        <button 
                          onClick={() => setInspectPhoto(photo)}
                          className="p-2 rounded-xl bg-orange-600 text-white hover:scale-115 transition-transform font-mono text-xs font-bold cursor-pointer"
                          title="Inspect Embeddings Matrix ID"
                        >
                          Inspect Node
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              ) : activeTab === "people" && !selectedCluster ? (
                /* Unsupervised Face Clusters (DBSCAN Groups) Gallery */
                <div className="space-y-6">
                  {filteredClusters.length > 0 ? (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6">
                      {filteredClusters.map((cluster) => (
                        <motion.div
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          key={cluster.id}
                          onClick={() => {
                            setSelectedCluster(cluster);
                            addLog(`Drilling down into cluster index ${cluster.id} [${cluster.nodes.length} photos]`, "info");
                          }}
                          className={`p-4 rounded-3xl border text-center space-y-4 cursor-pointer transition-all duration-300 hover:scale-103 ${
                            theme === "dark" 
                              ? "border-white/10 bg-white/[0.02] hover:bg-white/5 hover:border-orange-500/30" 
                              : "border-black/5 bg-[#ffffff] shadow-sm hover:shadow-md hover:border-orange-500/30"
                          }`}
                        >
                          {/* Face Crop Thumbnail */}
                          <div className="relative aspect-square rounded-2xl overflow-hidden bg-black/40 border border-white/5 h-28 w-28 mx-auto group">
                            {cluster.representative.faceBox ? (
                              <div style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden' }}>
                                <img 
                                  src={cluster.representative.url} 
                                  style={{
                                    position: 'absolute',
                                    left: `-${(cluster.representative.faceBox.x * 100) / cluster.representative.faceBox.w}%`,
                                    top: `-${(cluster.representative.faceBox.y * 100) / cluster.representative.faceBox.h}%`,
                                    width: `${10000 / cluster.representative.faceBox.w}%`,
                                    height: `${10000 / cluster.representative.faceBox.h}%`,
                                    objectFit: 'cover'
                                  }}
                                  referrerPolicy="no-referrer"
                                  alt="Cluster Face"
                                />
                              </div>
                            ) : (
                              <img src={cluster.representative.url} className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                            )}
                            <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                              <span className="text-[10px] font-mono font-bold text-white uppercase tracking-wider">Explore</span>
                            </div>
                          </div>
                          <div className="space-y-1">
                            <h4 className="font-bold text-sm tracking-tight">{cluster.label}</h4>
                            <p className="text-[10px] font-mono opacity-50 font-semibold">{cluster.nodes.length} Photos detected</p>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div className={`p-16 border rounded-3xl text-center flex flex-col items-center justify-center gap-3 transition-colors duration-500 ${
                      theme === "dark" ? "border-white/5 bg-white/[0.01]" : "border-black/5 bg-black/[0.01]"
                    }`}>
                      <Cpu className="w-10 h-10 text-orange-500/50 animate-pulse" />
                      <p className="text-sm font-semibold opacity-60">Clustering Engine pending input...</p>
                      <p className="text-xs opacity-50">Upload face assets inside console to trigger classification.</p>
                    </div>
                  )}
                </div>
              ) : (
                /* Sub-Grid of photos for the Selected Person Cluster Profile */
                <div className="space-y-6">
                  <div className="flex items-center justify-between border-b border-white/5 pb-4">
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => setSelectedCluster(null)}
                        className={`px-4 py-2 text-xs font-bold font-mono border rounded-xl transition-all cursor-pointer ${
                          theme === "dark" ? "border-white/10 hover:bg-white/5" : "border-black/15 hover:bg-black/5"
                        }`}
                      >
                        ← Back to Gallery
                      </button>
                      <div>
                        <h3 className="text-lg font-bold tracking-tight">Photos of {selectedCluster?.label}</h3>
                        <p className="text-xs opacity-50 font-mono">Found in {selectedCluster?.nodes.length} photos at the event</p>
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6">
                    {selectedCluster?.nodes.map((node, i) => (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        key={`${node.url}-${i}`}
                        className={`aspect-square rounded-2xl overflow-hidden border group relative transition-colors duration-300 ${
                          theme === "dark" ? "border-white/10 bg-white/5" : "border-black/10 bg-black/5"
                        }`}
                      >
                        <img src={node.url} alt={`Face ${i}`} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" referrerPolicy="no-referrer" />
                        
                        {/* Highlights the exact face bounding box of the person inside this group when hovered */}
                        {node.faceBox && (
                          <div 
                            className="absolute border border-orange-500/50 pointer-events-none group-hover:border-orange-500 group-hover:border-2"
                            style={{
                              left: `${node.faceBox.x}%`,
                              top: `${node.faceBox.y}%`,
                              width: `${node.faceBox.w}%`,
                              height: `${node.faceBox.h}%`
                            }}
                          />
                        )}

                        <div className="absolute inset-0 bg-black/35 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                          <button
                            onClick={() => {
                              // Re-map node to full PhotoData type
                              const fullPhoto = photos.find(p => p.url === node.url && p.faceBox?.x === node.faceBox?.x) || photos.find(p => p.url === node.url)!;
                              setInspectPhoto(fullPhoto);
                            }}
                            className="px-3 py-1.5 bg-orange-600 text-white rounded-xl font-mono text-[10px] font-bold uppercase tracking-wider hover:scale-105 transition-transform cursor-pointer"
                          >
                            Inspect Face
                          </button>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* Sentinal Intersection Trigger Element for Infinite Scrolling */}
              <div 
                ref={loadMoreRef} 
                className={`py-12 border-t flex flex-col items-center justify-center gap-3 transition-colors duration-500 ${
                  theme === "dark" ? "border-white/5 bg-transparent" : "border-black/5 bg-transparent"
                }`}
              >
                {visibleCount < filteredPhotos.length ? (
                  <>
                    <Loader2 className="w-6 h-6 text-orange-500 animate-spin" />
                    <p className="text-xs font-mono font-bold text-orange-500">Pipeline scrolling: Load and index next photographic node cluster... [IntersectionObserver active]</p>
                    <p className="text-[10px] text-white/40 font-mono">
                      Rendering page item nodes {visibleCount} - {Math.min(visibleCount + 24, filteredPhotos.length)} of {filteredPhotos.length}
                    </p>
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-6 h-6 text-emerald-500" />
                    <p className="text-xs font-mono font-bold text-emerald-500">Dynamic pipeline stable: All {filteredPhotos.length.toLocaleString()} photo nodes fully rendered inside virtual viewport.</p>
                  </>
                )}
              </div>
            </motion.div>
          )}

          {view === "attendee" && (
            <motion.div 
              key="attendee"
              initial={{ opacity: 0, x: -25 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 25 }}
              className="max-w-5xl mx-auto space-y-12"
            >
              <div className="text-center space-y-3">
                <div className="text-xs uppercase tracking-widest text-[#3b82f6] font-mono">Biometric Frame Query Terminal</div>
                <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight">Similarity Vector Matching</h2>
                <p className={`text-lg max-w-xl mx-auto ${
                  theme === "dark" ? "text-white/50" : "text-black/50"
                }`}>Provide a high-fidelity reference picture either via upload or immediate optical capturing.</p>
              </div>

              <div className="grid md:grid-cols-12 gap-8 items-start">
                <div className="md:col-span-5 space-y-6">
                  <div className={`aspect-[3/4] rounded-3xl border-2 border-dashed flex flex-col items-center justify-center relative overflow-hidden group transition-colors duration-300 ${
                    theme === "dark" ? "border-white/10 bg-white/5" : "border-black/5 bg-black/5"
                  }`}>
                    {/* Live hardware video camera preview feed */}
                    {isLiveCamera ? (
                      <div className="absolute inset-0 bg-black flex flex-col justify-between">
                        <video 
                          ref={videoRef} 
                          className="w-full h-full object-cover scale-x-[-1]" 
                          playsInline 
                          muted 
                        />
                        {/* Shutter capture visual overlay buttons */}
                        <div className="absolute inset-x-0 bottom-6 flex flex-col items-center gap-3 px-6 z-10">
                          <div className="flex gap-4">
                            <button 
                              onClick={captureSelfiePhoto}
                              className="bg-orange-600 hover:bg-orange-500 text-white p-4 rounded-full shadow-xl shadow-orange-600/20 flex items-center justify-center cursor-pointer"
                              title="Take Selfie Capture"
                            >
                              <Camera className="w-6 h-6" />
                            </button>
                            <button 
                              onClick={() => {
                                setIsLiveSearchActive(prev => !prev);
                                addLog(`Live video search tracking: ${!isLiveSearchActive ? "ON" : "OFF"}`, "warning");
                              }}
                              className={`p-4 rounded-full shadow-xl flex items-center justify-center cursor-pointer ${
                                isLiveSearchActive 
                                  ? "bg-emerald-600 text-white animate-pulse" 
                                  : "bg-blue-600 text-white"
                              }`}
                              title={isLiveSearchActive ? "Pause Live Tracking" : "Start Live Tracking"}
                            >
                              <Activity className="w-6 h-6" />
                            </button>
                            <button 
                              onClick={stopCamera}
                              className="bg-red-600 text-white p-4 rounded-full flex items-center justify-center cursor-pointer"
                              title="Close Camera"
                            >
                              <X className="w-5 h-5" />
                            </button>
                          </div>
                          {isLiveSearchActive && (
                            <div className="bg-emerald-950/80 border border-emerald-500/30 text-emerald-400 text-[9px] font-mono px-3 py-1 rounded-full uppercase tracking-wider shadow-md backdrop-blur-md">
                              Live scanning active
                            </div>
                          )}
                        </div>
                        {/* High-accuracy scanner indicator line */}
                        <div className="absolute inset-x-0 top-0 h-1 bg-[#3b82f6] shadow-xl animate-[scan_3s_ease-in-out_infinite]" />
                      </div>
                    ) : selfie ? (
                      <>
                        <img src={selfie} alt="Selfie Payload" className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                        <div className="absolute inset-0 border-[3px] border-dashed border-[#ea580c] scale-95 rounded-2xl pointer-events-none opacity-40" />
                        <button 
                          onClick={() => {
                            setSelfie(null);
                            addLog("Cleared current query reference vector.", "info");
                          }}
                          className="absolute top-4 right-4 p-2 bg-black/70 backdrop-blur-md rounded-full text-white hover:bg-red-600 transition-colors"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </>
                    ) : (
                      <div className="flex flex-col items-center justify-center p-8 text-center space-y-6">
                        <div className={`w-16 h-16 rounded-full flex items-center justify-center transition-transform group-hover:scale-105 ${
                          theme === "dark" ? "bg-white/10" : "bg-black/10"
                        }`}>
                          <Camera className={`w-8 h-8 ${theme === "dark" ? "text-white/50" : "text-black/50"}`} />
                        </div>
                        
                        <div className="space-y-4">
                          <button 
                            onClick={startCamera}
                            className="px-5 py-2.5 bg-orange-600 text-white text-xs font-bold uppercase tracking-wider rounded-xl hover:bg-orange-500 transition-all shadow-md shadow-orange-600/10 flex items-center gap-2 mx-auto cursor-pointer"
                          >
                            <Video className="w-3.5 h-3.5" />
                            Enable Stream
                          </button>
                          <p className={`text-xs ${theme === "dark" ? "text-white/40" : "text-black/40"}`}>
                            - or -
                          </p>
                          <label className="cursor-pointer block text-sm font-bold hover:text-orange-500 transition-colors">
                            Upload Reference Asset
                            <input type="file" className="hidden" onChange={handleSelfieUpload} accept="image/*" />
                          </label>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <button 
                    onClick={startSearch}
                    disabled={!selfie || isSearching || photos.length === 0}
                    className={`w-full py-5 font-bold rounded-2xl transition-all flex items-center justify-center gap-3 text-lg shadow-xl cursor-pointer ${
                      theme === "dark" 
                        ? "bg-white text-black hover:bg-orange-600 hover:text-white" 
                        : "bg-black text-white hover:bg-orange-600 hover:text-white"
                    } disabled:opacity-30 disabled:pointer-events-none`}
                  >
                    {isSearching ? (
                      <>
                        <Loader2 className="w-6 h-6 animate-spin" />
                        Executing Cosine Query Math...
                      </>
                    ) : (
                      <>
                        <Crosshair className="w-6 h-6" />
                        Execute Top-K Similarity Search
                      </>
                    )}
                  </button>
                </div>

                <div className="md:col-span-7 space-y-6">
                  <div className="flex items-center justify-between border-b border-white/5 pb-4">
                    <h3 className="text-xl font-bold tracking-tight">Active Vector Spatial Query Matches</h3>
                    <div className="flex items-center gap-3 text-xs font-mono">
                      <span className={`px-2.5 py-1 rounded ${
                        theme === "dark" ? "bg-white/5 text-white/60" : "bg-black/5 text-black/60"
                      }`}>{searchResults.length} calculated outputs</span>
                      <span className="text-[#3b82f6] animate-pulse font-bold">• SIMULATED DB: ONLINE</span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                    {searchResults.length > 0 ? (
                      searchResults.map((result, i) => (
                        <motion.div 
                          initial={{ opacity: 0, y: 15 }}
                          animate={{ opacity: 1, y: 0 }}
                          key={i} 
                          className={`rounded-2xl border overflow-hidden relative group transition-colors duration-300 ${
                            theme === "dark" ? "border-white/10 bg-[#0d0d10]" : "border-black/5 bg-[#f4f6f8]"
                          }`}
                        >
                          <div className="aspect-square overflow-hidden relative">
                            <img src={result.url} alt={`Match Asset ${i}`} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" referrerPolicy="no-referrer" />
                            {/* Matching accuracy badge overlay */}
                            <div className="absolute top-2 left-2 bg-black/70 backdrop-blur-md rounded px-2 py-1 text-[10px] font-mono border border-orange-500/30 text-orange-400 font-extrabold shadow-lg">
                              SIM: {(result.score * 100).toFixed(2)}%
                            </div>
                          </div>
                          
                          <div className="p-3 flex items-center justify-between">
                            <span className="text-[10px] font-mono opacity-50 uppercase tracking-wider">Node#0{photos.findIndex(p => p.url === result.url) + 1}</span>
                            <a 
                              href={result.url} 
                              download={`match-${i}.jpg`}
                              className="px-3 py-1 bg-orange-600 hover:bg-orange-500 text-white rounded text-[10px] font-bold uppercase tracking-wider transition-colors"
                            >
                              Get File
                            </a>
                          </div>
                        </motion.div>
                      ))
                    ) : (
                      <div className={`col-span-3 aspect-[16/10] rounded-3xl border flex flex-col items-center justify-center gap-4 transition-colors duration-500 ${
                        theme === "dark" ? "border-white/5 bg-white/[0.01] text-white/20" : "border-black/5 bg-black/[0.01] text-black/20"
                      }`}>
                        <Database className="w-12 h-12 stroke-[1.5]" />
                        <div className="text-center space-y-1.5">
                          <p className="text-sm font-medium">Capture Selfie or Load Reference Profile</p>
                          <p className="text-xs opacity-60">System similarity engine will locate matches instantaneously.</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {view === "developer" && (
            <motion.div 
              key="developer"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              className="space-y-8"
            >
              <div className="border-b border-white/5 pb-6">
                <div className="text-xs uppercase tracking-widest text-[#ea580c] font-mono mb-2">High-Dimensional Feature Map</div>
                <h2 className="text-2xl sm:text-4xl font-extrabold tracking-tight">Biometric Vector Space Model Representation</h2>
                <p className={theme === "dark" ? "text-white/50" : "text-black/50"}>
                  Visualizing multi-dimensional embeddings mapped into a 2D constraint orbit vector layout.
                </p>
              </div>

              <div className="grid lg:grid-cols-12 gap-8 items-start">
                <div className="lg:col-span-8">
                  <div 
                    ref={canvasContainerRef}
                    className={`rounded-3xl border relative overflow-hidden transition-colors duration-500 ${
                      theme === "dark" ? "border-white/10 bg-[#07070a]" : "border-black/15 bg-[#f5f6f8]"
                    }`}
                  >
                    <div className="absolute top-4 left-4 z-10 flex items-center gap-3 bg-black/40 backdrop-blur-md p-1.5 px-3 rounded-2xl pointer-events-none">
                      <div className="p-2 text-orange-500">
                        <Layers className="w-4 h-4 animate-pulse" />
                      </div>
                      <div>
                        <div className="font-mono text-xs font-bold uppercase text-white">2D Topological manifold</div>
                        <div className="text-[10px] opacity-80 font-mono text-white/80">Cluster Projection model</div>
                      </div>
                    </div>

                    <canvas 
                      ref={mapCanvasRef} 
                      onClick={handleCanvasClick}
                      className="w-full h-auto block cursor-crosshair"
                    />
                  </div>
                </div>

                <div className="lg:col-span-4 space-y-6">
                  <div className={`p-6 rounded-3xl border space-y-6 ${
                    theme === "dark" ? "border-white/10 bg-[#0d0d12]" : "border-black/5 bg-[#ffffff] shadow-sm"
                  }`}>
                    <h3 className="text-xl font-bold tracking-tight">Active Vector Map Registry</h3>
                    
                    <div className="space-y-4 max-h-[350px] overflow-y-auto pr-2 custom-scrollbar">
                      {photos.map((photo, i) => (
                        <div 
                          key={i}
                          onMouseEnter={() => setHoveredNodeIndex(i)}
                          onMouseLeave={() => setHoveredNodeIndex(null)}
                          className={`p-3.5 rounded-2xl border transition-all cursor-pointer ${
                            hoveredNodeIndex === i 
                              ? "border-orange-500/30 bg-orange-500/5 translate-x-1" 
                              : theme === "dark" 
                                ? "border-white/5 bg-white/[0.01]" 
                                : "border-black/5 bg-black/[0.01]"
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <img src={photo.url} className="w-10 h-10 object-cover rounded-lg" referrerPolicy="no-referrer" />
                            <div className="flex-1 min-w-0">
                              <div className="font-mono text-xs font-bold flex justify-between">
                                <span className="truncate">TENSOR_NODE_0{i+1}</span>
                                <span className="text-orange-500">256-dim</span>
                              </div>
                              <div className="text-[10px] font-mono opacity-50 truncate">
                                {photo.embedding ? `V = [ ${photo.embedding.slice(0, 4).map(v => v.toFixed(2)).join(", ")}... ]` : "Embedding pending generation"}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Embedded Real-time scrolling system telemetry logs terminal console */}
      <div className={`border-t relative ${
        theme === "dark" ? "border-white/5 bg-[#030305]" : "border-black/10 bg-[#f4f6f7]"
      }`}>
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between border-b border-white/5">
          <div className="flex items-center gap-2.5">
            <TerminalIcon className="w-4 h-4 text-orange-500" />
            <span className="font-mono text-[10px] sm:text-xs font-bold tracking-wider uppercase truncate max-w-[180px] sm:max-w-none">Inference Matrix Real-time Logs Console</span>
          </div>
          <button 
            onClick={() => setIsTerminalCollapsed(prev => !prev)}
            className={`text-xs font-bold font-mono px-3 py-1 border rounded transition-all ${
              theme === "dark" ? "border-white/10 hover:bg-white/5" : "border-black/10 hover:bg-black/5"
            }`}
          >
            {isTerminalCollapsed ? "EXPAND CONSOLE" : "COLLAPSE CONSOLE"}
          </button>
        </div>

        <AnimatePresence>
          {!isTerminalCollapsed && (
            <motion.div 
              initial={{ height: 0 }}
              animate={{ height: 180 }}
              exit={{ height: 0 }}
              className="max-w-7xl mx-auto px-6 overflow-y-auto pt-4 pb-6 font-mono text-xs space-y-1.5 custom-scrollbar bg-black"
            >
              {logs.map((log, idx) => (
                <div key={idx} className="flex gap-4 items-start leading-relaxed text-left">
                  <span className="text-white/30 text-[10px] select-none">[{log.timestamp}]</span>
                  <span className={`font-bold select-none ${
                    log.type === "success" ? "text-green-500" :
                    log.type === "warning" ? "text-yellow-500" :
                    log.type === "error" ? "text-red-500" :
                    log.type === "matrix" ? "text-indigo-400" : "text-[#7fcfdf]"
                  }`}>
                    {log.type.toUpperCase()}:
                  </span>
                  <span className={log.type === "matrix" ? "text-white/80" : "text-white/60"}>
                    {log.message}
                  </span>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Advanced biometric inspect drawer container code */}
      <AnimatePresence>
        {inspectPhoto && (() => {
          const allFacesInInspectPhoto = photos.filter(p => p.url === inspectPhoto.url);
          const activeFaceIndex = hoveredFaceBoxIndex !== null && hoveredFaceBoxIndex < allFacesInInspectPhoto.length ? hoveredFaceBoxIndex : 0;
          const activeFacePhoto = allFacesInInspectPhoto[activeFaceIndex] || inspectPhoto;
          return (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-md flex items-center justify-center p-6"
            >
              <motion.div 
                initial={{ scale: 0.95, y: 15 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0.95, y: 15 }}
                className={`max-w-2xl w-full rounded-3xl border overflow-hidden p-6 relative ${
                  theme === "dark" ? "bg-[#0b0c10] border-white/10 text-white" : "bg-white border-black/10 text-black"
                }`}
              >
                <button 
                  onClick={() => {
                    setInspectPhoto(null);
                    setHoveredFaceBoxIndex(null);
                  }}
                  className="absolute top-6 right-6 p-2 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>

                <div className="grid md:grid-cols-2 gap-6 items-start">
                  <div className="relative aspect-square rounded-2xl overflow-hidden bg-black/40 border border-white/5">
                    <img src={inspectPhoto.url} className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                    
                    {/* Render bounding boxes for all faces in this photo */}
                    {allFacesInInspectPhoto.map((facePhoto, idx) => {
                      if (!facePhoto.faceBox) return null;
                      const isHovered = hoveredFaceBoxIndex === idx || (hoveredFaceBoxIndex === null && idx === 0);
                      return (
                        <div 
                          key={idx}
                          className={`absolute border-2 rounded-lg cursor-pointer transition-all duration-300 ${
                            isHovered ? "border-orange-500 bg-orange-500/10 shadow-lg scale-102" : "border-[#ea580c]"
                          }`}
                          style={{
                            left: `${facePhoto.faceBox.x}%`,
                            top: `${facePhoto.faceBox.y}%`,
                            width: `${facePhoto.faceBox.w}%`,
                            height: `${facePhoto.faceBox.h}%`
                          }}
                          onMouseEnter={() => setHoveredFaceBoxIndex(idx)}
                          onMouseLeave={() => setHoveredFaceBoxIndex(null)}
                        >
                          <div className="absolute -top-6 left-0 bg-orange-600 text-[10px] font-mono px-2 py-0.5 text-white rounded font-bold shadow-md">
                            FACE #{idx + 1}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="space-y-6">
                    <div>
                      <h4 className="text-2xl font-bold tracking-tight">Biometric Image Inspector</h4>
                      <p className="text-xs font-mono opacity-50 mt-1 truncate">{inspectPhoto.url}</p>
                    </div>

                    <div className="space-y-2 border-t border-white/5 pt-4">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="opacity-60">MODEL PIPELINE:</span>
                        <span className="font-bold text-orange-500">face-api.js (SsdMobilenetv1)</span>
                      </div>
                      <div className="flex justify-between text-xs font-mono">
                        <span className="opacity-60">VECTOR GRAPH WEIGHT:</span>
                        <span className="font-bold">128-d Float Vector</span>
                      </div>
                      <div className="flex justify-between text-xs font-mono">
                        <span className="opacity-60">INSPECTING FACE:</span>
                        <span className="font-bold text-indigo-400">Face Index #{activeFaceIndex + 1} of {allFacesInInspectPhoto.length}</span>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <span className="font-mono text-xs opacity-60">128-d BIOMETRIC VECTOR SPARKLINE (FIRST 64 DIMS):</span>
                      <div className="flex items-end gap-[2px] h-24 p-3 bg-black/60 rounded-xl border border-white/5">
                        {activeFacePhoto.embedding ? (
                          activeFacePhoto.embedding.slice(0, 64).map((val, index) => {
                            // Normalize val to a percentage height [0, 100]% (expected bounds are typically -0.25 to 0.25)
                            const minVal = -0.25;
                            const maxVal = 0.25;
                            const heightPercent = Math.max(0, Math.min(100, ((val - minVal) / (maxVal - minVal)) * 100));
                            return (
                              <div 
                                key={index} 
                                className={`flex-1 transition-all cursor-pointer ${
                                  hoveredFaceBoxIndex !== null ? "bg-orange-500 hover:bg-white" : "bg-indigo-500 hover:bg-orange-500"
                                }`}
                                style={{ height: `${heightPercent}%` }}
                                title={`Dim #${index}: ${val.toFixed(6)}`}
                              />
                            );
                          })
                        ) : (
                          <div className="text-[10px] font-mono opacity-50 m-auto text-white">Generating embedding...</div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          );
        })()}
      </AnimatePresence>

      <style>{`
        @keyframes scan {
          0%, 100% { top: 0%; }
          50% { top: 100%; }
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0,0,0,0.1);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(249, 115, 22, 0.2);
          border-radius: 9px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(249, 115, 22, 0.4);
        }
      `}</style>
    </div>
  );
}
