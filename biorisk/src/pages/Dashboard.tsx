import { useState, useRef, useCallback, useEffect } from "react";

// ── Types ────────────────────────────────────────────────────────────────────
interface BBox { x1: number; y1: number; x2: number; y2: number; }
interface Detection {
  label: string;
  confidence: number;
  bbox: BBox;
}
interface DetectResponse {
  total: number;
  image_shape: { height: number; width: number };
  detections: Detection[];
}
interface ModelClass { id: number; name: string; }

// ── Risk config ───────────────────────────────────────────────────────────────
const RISK: Record<string, { level: "critical" | "high" | "medium" | "safe"; color: string; border: string; badge: string; text: string; icon: string; description: string; }> = {
  latrodectus: {
    level: "critical",
    color: "bg-red-950/60",
    border: "border-red-500/70",
    badge: "bg-red-500/20 text-red-300 border border-red-500/40",
    text: "text-red-300",
    icon: "◈",
    description: "Araña viuda negra — neurotóxica, mordedura requiere atención médica urgente.",
  },
  loxoceles: {
    level: "critical",
    color: "bg-red-950/60",
    border: "border-red-500/70",
    badge: "bg-red-500/20 text-red-300 border border-red-500/40",
    text: "text-red-300",
    icon: "◈",
    description: "Araña violinista — veneno necrotizante, consultar médico inmediatamente.",
  },
  crotalus: {
    level: "high",
    color: "bg-orange-950/60",
    border: "border-orange-400/70",
    badge: "bg-orange-500/20 text-orange-300 border border-orange-400/40",
    text: "text-orange-300",
    icon: "◆",
    description: "Serpiente de cascabel — hemotóxica, no manipular, llamar servicios de emergencia.",
  },
  "kissing bug": {
    level: "medium",
    color: "bg-yellow-950/60",
    border: "border-yellow-400/70",
    badge: "bg-yellow-500/20 text-yellow-300 border border-yellow-400/40",
    text: "text-yellow-300",
    icon: "◇",
    description: "Chinche besucona — vector del Mal de Chagas, evitar contacto directo.",
  },
  aedes: {
    level: "medium",
    color: "bg-yellow-950/60",
    border: "border-yellow-400/70",
    badge: "bg-yellow-500/20 text-yellow-300 border border-yellow-400/40",
    text: "text-yellow-300",
    icon: "◇",
    description: "Mosquito Aedes — transmisor de dengue, zika y chikungunya.",
  },
  lampropeltis: {
    level: "safe",
    color: "bg-emerald-950/60",
    border: "border-emerald-500/70",
    badge: "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40",
    text: "text-emerald-300",
    icon: "○",
    description: "Serpiente rey — no venenosa, benéfica para el ecosistema.",
  },
};

const getRisk = (label: string) =>
  RISK[label.toLowerCase()] ?? {
    level: "safe",
    color: "bg-emerald-950/60",
    border: "border-emerald-500/70",
    badge: "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40",
    text: "text-emerald-300",
    icon: "○",
    description: "Organismo identificado.",
  };

const LEVEL_LABEL: Record<string, string> = {
  critical: "PELIGRO CRÍTICO",
  high: "RIESGO ALTO",
  medium: "RIESGO MEDIO",
  safe: "SIN RIESGO",
};

const API_BASE = "http://localhost:8000";

// ── Helpers ──────────────────────────────────────────────────────────────────
function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 85 ? "bg-lime-400" : pct >= 60 ? "bg-yellow-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-2 mt-1">
      <div className="flex-1 h-1 rounded-full bg-white/10">
        <div className={`h-1 rounded-full ${color} transition-all duration-700`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono tabular-nums text-white/60">{pct}%</span>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function DetectionDashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [annotated, setAnnotated] = useState<string | null>(null);
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [classes, setClasses] = useState<ModelClass[]>([]);
  const [dragging, setDragging] = useState(false);
  const [confidence, setConfidence] = useState(0.5);
  const [modelOnline, setModelOnline] = useState<boolean | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Health check cada 2 minutos
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(5000) });
        setModelOnline(r.ok);
      } catch {
        setModelOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 120_000);
    return () => clearInterval(interval);
  }, []);

  // Load model classes on mount
  useEffect(() => {
    fetch(`${API_BASE}/model/classes`)
      .then((r) => r.json())
      .then((data) => {
        const list: ModelClass[] = Object.entries(data.classes ?? {}).map(([id, name]) => ({
          id: Number(id),
          name: name as string,
        }));
        setClasses(list);
      })
      .catch(() => {});
  }, []);

  const handleFile = (f: File) => {
    setFile(f);
    setAnnotated(null);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  }, []);

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setAnnotated(null);

    const fd1 = new FormData();
    fd1.append("file", file);
    if (confidence !== 0.5) fd1.append("confidence", String(confidence));

    const fd2 = new FormData();
    fd2.append("file", file);

    try {
      const [r1, r2] = await Promise.all([
        fetch(`${API_BASE}/detect/`, { method: "POST", body: fd1 }),
        fetch(`${API_BASE}/detect/image`, { method: "POST", body: fd2 }),
      ]);

      if (!r1.ok) throw new Error(`HTTP ${r1.status}`);
      const json: DetectResponse = await r1.json();
      setResult(json);

      if (r2.ok) {
        const blob = await r2.blob();
        setAnnotated(URL.createObjectURL(blob));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error desconocido");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setAnnotated(null);
    setResult(null);
    setError(null);
  };

  const highestRisk = result?.detections.length
    ? result.detections.reduce((a, b) => {
        const ra = getRisk(a.label);
        const rb = getRisk(b.label);
        const order = ["critical", "high", "medium", "safe"];
        return order.indexOf(ra.level) <= order.indexOf(rb.level) ? a : b;
      })
    : null;

  return (
    <div className="min-h-screen bg-[#0a0d12] text-white font-mono">
      {/* ── Header ─────────────────────────────────────────── */}
      <header className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-sm bg-lime-400/20 border border-lime-400/50 flex items-center justify-center">
            <span className="text-lime-400 text-xs font-bold">BR</span>
          </div>
          <span className="text-sm font-semibold tracking-widest uppercase text-white/80">
            BioRisk Detector
          </span>
          <span className="text-white/20 text-xs">v1.0</span>
        </div>
        <div className="flex items-center gap-4">
          {classes.length > 0 && (
            <span className="text-xs text-white/30">
              {classes.length} clases activas
            </span>
          )}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${modelOnline === false ? "bg-red-500" : "bg-lime-400 animate-pulse"}`} />
            <span className={`text-xs ${modelOnline === false ? "text-red-400/70" : "text-lime-400/70"}`}>
              {modelOnline === null ? "verificando..." : modelOnline ? "modelo online" : "modelo fuera de línea"}
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── LEFT: Upload panel ────────────────────────────── */}
        <section className="flex flex-col gap-4">
          {/* Drop zone */}
          <div
            className={`relative rounded-xl border-2 border-dashed transition-colors cursor-pointer
              ${dragging ? "border-lime-400 bg-lime-400/5" : "border-white/15 hover:border-white/30 bg-white/[0.02]"}
              ${preview ? "border-solid border-white/10" : ""}`}
            style={{ minHeight: 280 }}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => !preview && inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />

            {preview ? (
              <div className="relative">
                <img
                  src={annotated ?? preview}
                  alt="preview"
                  className="w-full rounded-xl object-contain"
                  style={{ maxHeight: 380 }}
                />
                {annotated && (
                  <span className="absolute top-2 right-2 text-xs px-2 py-1 rounded bg-lime-400/20 text-lime-400 border border-lime-400/30">
                    anotada
                  </span>
                )}
                <button
                  onClick={(e) => { e.stopPropagation(); reset(); }}
                  className="absolute top-2 left-2 text-xs px-2 py-1 rounded bg-black/60 text-white/50 hover:text-white/90 border border-white/10 transition-colors"
                >
                  ✕ nueva imagen
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 px-8 text-center gap-3 select-none">
                <div className="text-3xl text-white/20">⬆</div>
                <p className="text-sm text-white/40">
                  Arrastra una imagen o haz clic para seleccionar
                </p>
                <p className="text-xs text-white/20">PNG, JPG, WEBP</p>
              </div>
            )}
          </div>

          {/* Confidence slider */}
          <div className="bg-white/[0.03] border border-white/10 rounded-xl px-5 py-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs text-white/40 uppercase tracking-wider">Umbral de confianza</span>
              <span className="text-sm text-lime-400 font-mono">{Math.round(confidence * 100)}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
              className="w-full accent-lime-400"
            />
            <div className="flex justify-between mt-1">
              <span className="text-xs text-white/20">más detecciones</span>
              <span className="text-xs text-white/20">más precisión</span>
            </div>
          </div>

          {/* Analyze button */}
          <button
            disabled={!file || loading}
            onClick={handleAnalyze}
            className={`w-full py-3 rounded-xl text-sm font-semibold uppercase tracking-widest transition-all
              ${file && !loading
                ? "bg-lime-400 text-black hover:bg-lime-300 active:scale-[0.98]"
                : "bg-white/5 text-white/20 cursor-not-allowed"}`}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-3 h-3 border border-black/40 border-t-black rounded-full animate-spin" />
                Analizando…
              </span>
            ) : "Analizar imagen"}
          </button>

          {error && (
            <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3">
              ✕ {error}
            </div>
          )}
        </section>

        {/* ── RIGHT: Results panel ──────────────────────────── */}
        <section className="flex flex-col gap-4">
          {!result && !loading && (
            <div className="flex-1 flex flex-col items-center justify-center text-center py-20 gap-4 border border-dashed border-white/10 rounded-xl">
              <div className="text-4xl opacity-20">◫</div>
              <p className="text-sm text-white/30">
                Los resultados aparecerán aquí después del análisis
              </p>
            </div>
          )}

          {loading && (
            <div className="flex-1 flex flex-col items-center justify-center py-20 gap-4 border border-dashed border-lime-400/20 rounded-xl">
              <div className="w-8 h-8 border-2 border-lime-400/30 border-t-lime-400 rounded-full animate-spin" />
              <p className="text-sm text-lime-400/60 tracking-widest">PROCESANDO</p>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Summary bar */}
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: "Total", value: result.total },
                  { label: "Imagen", value: `${result.image_shape.width}×${result.image_shape.height}` },
                  { label: "Riesgo", value: highestRisk ? LEVEL_LABEL[getRisk(highestRisk.label).level] : "—" },
                ].map((m) => (
                  <div key={m.label} className="bg-white/[0.04] border border-white/10 rounded-lg px-3 py-3">
                    <p className="text-xs text-white/30 uppercase tracking-wider mb-1">{m.label}</p>
                    <p className="text-sm font-semibold text-white truncate">{m.value}</p>
                  </div>
                ))}
              </div>

              {/* Detection cards */}
              {result.detections.length === 0 ? (
                <div className="flex-1 flex items-center justify-center py-12 text-sm text-white/30 border border-white/10 rounded-xl">
                  Sin detecciones con el umbral actual
                </div>
              ) : (
                <div className="flex flex-col gap-3 overflow-y-auto" style={{ maxHeight: 440 }}>
                  {result.detections.map((det, i) => {
                    const r = getRisk(det.label);
                    return (
                      <div key={i} className={`rounded-xl border px-4 py-4 ${r.color} ${r.border}`}>
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-center gap-2">
                            <span className={`text-lg ${r.text}`}>{r.icon}</span>
                            <div>
                              <p className="text-sm font-semibold capitalize text-white">
                                {det.label}
                              </p>
                              <span className={`text-xs px-2 py-0.5 rounded-full ${r.badge} mt-1 inline-block`}>
                                {LEVEL_LABEL[r.level]}
                              </span>
                            </div>
                          </div>
                          <div className="text-right shrink-0">
                            <p className={`text-xs ${r.text} font-mono`}>
                              [{det.bbox.x1},{det.bbox.y1}] → [{det.bbox.x2},{det.bbox.y2}]
                            </p>
                          </div>
                        </div>
                        <ConfidenceBar value={det.confidence} />
                        <p className="text-xs text-white/40 mt-2 leading-relaxed">{r.description}</p>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}