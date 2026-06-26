"use client";
import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload, Camera, X, Loader2, AlertTriangle,
  CheckCircle2, ChevronDown, ChevronUp, Leaf, Eye,
  Microscope, Sprout, Bug, Apple, Search,
} from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

export type VisionTask =
  | "crop-disease"
  | "breed"
  | "weed"
  | "plant-id"
  | "fruit-grade"
  | "fruit-detect"
  | "detect";

export interface ClassPrediction {
  label: string;
  display_name: string;
  confidence: number;
  rank: number;
}

export interface VisionExplanation {
  text: string;
  language: string;
  recommendations: string[];
  urgency: "Low" | "Medium" | "High" | "Critical";
  treatment_products: string[];
}

export interface VisionMeta {
  width: number;
  height: number;
  processing_time_ms: number;
  model_version: string;
  confidence_threshold: number;
}

export interface VisionResult {
  task: string;
  success: boolean;
  error?: string;
  predictions: ClassPrediction[];
  bounding_boxes?: unknown[];
  explanation?: VisionExplanation;
  metadata?: VisionMeta;
  extra?: Record<string, unknown>;
}

// ── Task Definitions ──────────────────────────────────────────────────────────

const TASKS: { id: VisionTask; label: string; icon: React.ReactNode; endpoint: string; hint: string }[] = [
  {
    id: "crop-disease",
    label: "Crop Disease",
    icon: <Microscope size={15} />,
    endpoint: "/vision/crop-disease",
    hint: "Upload a leaf or plant photo to detect diseases",
  },
  {
    id: "breed",
    label: "Animal Breed",
    icon: <Eye size={15} />,
    endpoint: "/vision/breed",
    hint: "Identify the breed of your cow or buffalo",
  },
  {
    id: "weed",
    label: "Weed Detection",
    icon: <Bug size={15} />,
    endpoint: "/vision/weed",
    hint: "Upload a field photo to identify weeds",
  },
  {
    id: "plant-id",
    label: "Plant ID",
    icon: <Sprout size={15} />,
    endpoint: "/vision/plant-id",
    hint: "Identify any plant or leaf species",
  },
  {
    id: "fruit-grade",
    label: "Fruit Quality",
    icon: <Apple size={15} />,
    endpoint: "/vision/fruit-grade",
    hint: "Grade harvested fruit quality (A/B/C)",
  },
  {
    id: "detect",
    label: "Object Detect",
    icon: <Search size={15} />,
    endpoint: "/vision/detect",
    hint: "Detect objects in a farm scene",
  },
];

// ── Props ─────────────────────────────────────────────────────────────────────

interface VisionUploadProps {
  onResult?: (result: VisionResult, imagePreview: string, task: VisionTask) => void;
  apiKey: string;
  backendUrl: string;
  lang?: string;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function VisionUpload({ onResult, apiKey, backendUrl, lang = "en" }: VisionUploadProps) {
  const [selectedTask, setSelectedTask] = useState<VisionTask>("crop-disease");
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VisionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);

  // Clean up blob URLs on unmount
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  const handleFile = useCallback((f: File) => {
    // Revoke previous preview URL
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);

    const url = URL.createObjectURL(f);
    previewUrlRef.current = url;
    setFile(f);
    setPreview(url);
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type.startsWith("image/")) handleFile(dropped);
    else setError("Please drop an image file (JPEG, PNG, WEBP)");
  }, [handleFile]);

  const handleAnalyse = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const taskDef = TASKS.find(t => t.id === selectedTask)!;
      const form = new FormData();
      form.append("image", file);
      form.append("lang", lang);
      form.append("include_explanation", "true");
      form.append("model_version", "latest");

      const res = await fetch(`${backendUrl}${taskDef.endpoint}`, {
        method: "POST",
        headers: { "X-API-Key": apiKey },
        body: form,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Server error (${res.status})`);
      }

      const data: VisionResult = await res.json();
      setResult(data);
      if (onResult && preview) onResult(data, preview, selectedTask);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [file, selectedTask, lang, apiKey, backendUrl, onResult, preview]);

  const handleClear = () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const activeTask = TASKS.find(t => t.id === selectedTask)!;

  return (
    <div className="vision-upload-container">
      {/* ── Task Selector ── */}
      <div className="vision-task-bar">
        {TASKS.map(t => (
          <button
            key={t.id}
            className={`vision-task-btn${selectedTask === t.id ? " active" : ""}`}
            onClick={() => { setSelectedTask(t.id); setResult(null); setError(null); }}
            title={t.hint}
          >
            {t.icon}
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      {/* ── Drop Zone ── */}
      <div
        className={`vision-dropzone${dragging ? " dragging" : ""}${preview ? " has-preview" : ""}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !preview && fileInputRef.current?.click()}
      >
        {preview ? (
          <div className="vision-preview-wrapper">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="Preview" className="vision-preview-img" />
            <button className="vision-clear-btn" onClick={e => { e.stopPropagation(); handleClear(); }}>
              <X size={14} />
            </button>
          </div>
        ) : (
          <div className="vision-drop-content">
            <div className="vision-drop-icon">
              <Leaf size={36} className="text-green-400" />
            </div>
            <p className="vision-drop-title">{activeTask.hint}</p>
            <p className="vision-drop-sub">Drag & drop · Click to browse · or use camera</p>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />
      </div>

      {/* ── Action Buttons ── */}
      <div className="vision-actions">
        <button
          className="vision-camera-btn"
          onClick={() => cameraInputRef.current?.click()}
          title="Take photo"
        >
          <Camera size={16} />
          Camera
        </button>
        <button
          className="vision-gallery-btn"
          onClick={() => fileInputRef.current?.click()}
          title="Choose from gallery"
        >
          <Upload size={16} />
          Gallery
        </button>
        <button
          className={`vision-analyse-btn${!file || loading ? " disabled" : ""}`}
          disabled={!file || loading}
          onClick={handleAnalyse}
        >
          {loading ? (
            <>
              <Loader2 size={16} className="spin" />
              Analysing…
            </>
          ) : (
            <>
              <Microscope size={16} />
              Analyse
            </>
          )}
        </button>
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="vision-error">
          <AlertTriangle size={15} />
          <span>{error}</span>
        </div>
      )}

      {/* ── Result ── */}
      {result && (
        <PredictionCard
          result={result}
          expanded={expanded}
          onToggle={() => setExpanded(v => !v)}
        />
      )}
    </div>
  );
}

// ── PredictionCard (inline for single-file simplicity) ────────────────────────

function PredictionCard({
  result,
  expanded,
  onToggle,
}: {
  result: VisionResult;
  expanded: boolean;
  onToggle: () => void;
}) {
  if (!result.success) {
    return (
      <div className="pred-card pred-error">
        <AlertTriangle size={16} /> <span>{result.error || "Analysis failed"}</span>
      </div>
    );
  }

  const top = result.predictions[0];
  const urgency = result.explanation?.urgency || "Low";
  const urgencyColor = { Low: "#4ade80", Medium: "#facc15", High: "#f97316", Critical: "#ef4444" }[urgency];

  return (
    <div className="pred-card">
      {/* Header */}
      <div className="pred-header">
        <div className="pred-top-label">
          <CheckCircle2 size={16} style={{ color: urgencyColor }} />
          <span className="pred-name">{top?.display_name || "Unknown"}</span>
          <span className="pred-conf" style={{ color: urgencyColor }}>
            {top ? `${(top.confidence * 100).toFixed(1)}%` : "—"}
          </span>
        </div>
        {result.explanation?.urgency && (
          <span className="pred-urgency" style={{ background: urgencyColor + "22", color: urgencyColor }}>
            {urgency}
          </span>
        )}
        <button className="pred-toggle" onClick={onToggle}>
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {/* Top-3 confidence bars */}
      <div className="pred-bars">
        {result.predictions.slice(0, 3).map((p, i) => (
          <div key={i} className="pred-bar-row">
            <span className="pred-bar-label">{p.display_name}</span>
            <div className="pred-bar-track">
              <div
                className="pred-bar-fill"
                style={{
                  width: `${p.confidence * 100}%`,
                  background: i === 0 ? urgencyColor : "#4ade8044",
                }}
              />
            </div>
            <span className="pred-bar-pct">{(p.confidence * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>

      {/* Quick treatment */}
      {result.extra?.quick_treatment && (
        <div className="pred-quick-treatment">
          <span className="pred-qt-label">Quick Treatment:</span>
          <span>{String(result.extra.quick_treatment)}</span>
        </div>
      )}

      {/* Expanded explanation */}
      {expanded && result.explanation && (
        <div className="pred-explanation">
          <p className="pred-exp-text">{result.explanation.text}</p>
          {result.explanation.recommendations.length > 0 && (
            <div className="pred-recs">
              <strong>Recommendations:</strong>
              <ul>
                {result.explanation.recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
          {result.explanation.treatment_products.length > 0 && (
            <div className="pred-products">
              <strong>Products:</strong>{" "}
              {result.explanation.treatment_products.join(" · ")}
            </div>
          )}
        </div>
      )}

      {/* Metadata footer */}
      {result.metadata && (
        <div className="pred-meta">
          <span>{result.metadata.model_version}</span>
          <span>·</span>
          <span>{result.metadata.width}×{result.metadata.height}px</span>
          <span>·</span>
          <span>{result.metadata.processing_time_ms.toFixed(0)}ms</span>
        </div>
      )}
    </div>
  );
}
