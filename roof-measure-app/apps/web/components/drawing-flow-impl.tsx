"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

import { computeSections, type RoofSection } from "@/lib/api";

type Step = "type" | "polygon" | "ridge" | "valley" | "done";
type RoofType = "satteldach" | "walmdach" | "pultdach" | "flachdach";

type Props = {
  lat: number;
  lon: number;
  address: string;
};

const ROOF_TYPES: {
  id: RoofType;
  name: string;
  description: string;
  ridgeLabel: string;
  ridgeHint: string;
  hasRidge: boolean;
}[] = [
  {
    id: "satteldach",
    name: "Satteldach",
    description: "Zwei geneigte Flächen, First in der Mitte. Häufigste Form bei Einfamilienhäusern.",
    ridgeLabel: "First markieren",
    ridgeHint:
      "Klicke die zwei Endpunkte des Firsts — die oberste Linie, an der sich die zwei Dachseiten treffen. Verläuft parallel zu den Traufen.",
    hasRidge: true,
  },
  {
    id: "walmdach",
    name: "Walmdach",
    description: "Vier geneigte Flächen, kürzere First mittig. Auch die Giebelseiten sind geneigt.",
    ridgeLabel: "First markieren",
    ridgeHint:
      "Klicke die zwei Endpunkte des Firsts. Bei einem Walmdach ist die First kürzer als der Polygon-Umriss und sitzt mittig.",
    hasRidge: true,
  },
  {
    id: "pultdach",
    name: "Pultdach",
    description: "Eine geneigte Fläche. Eine Seite ist hoch (First), die gegenüberliegende ist die Traufe.",
    ridgeLabel: "Hohe Kante markieren",
    ridgeHint:
      "Klicke die zwei Ecken der höchsten Kante (oben). Die gegenüberliegende Seite ist die Traufe.",
    hasRidge: true,
  },
  {
    id: "flachdach",
    name: "Flachdach",
    description: "Neigung unter 10°. Nur der Umriss wird benötigt.",
    ridgeLabel: "",
    ridgeHint: "",
    hasRidge: false,
  },
];

function roofTypeInfo(type: RoofType) {
  return ROOF_TYPES.find((t) => t.id === type)!;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

class DopProxyLayer extends L.TileLayer {
  getTileUrl(coords: L.Coords): string {
    return `${API_URL}/tiles/dop/${coords.z}/${coords.x}/${coords.y}`;
  }
}

function polygonAreaM2(points: L.LatLng[]): number {
  if (points.length < 3) return 0;
  const proj = points.map((p) => L.CRS.EPSG3857.project(p));
  let area = 0;
  for (let i = 0, j = proj.length - 1; i < proj.length; j = i++) {
    area += (proj[j].x + proj[i].x) * (proj[j].y - proj[i].y);
  }
  // Mercator overstates area by 1/cos²(lat). Korrigieren auf geodätische Fläche.
  const meanLat = points.reduce((s, p) => s + p.lat, 0) / points.length;
  const correction = Math.cos((meanLat * Math.PI) / 180) ** 2;
  return (Math.abs(area) / 2) * correction;
}

function lineLengthM(points: L.LatLng[]): number {
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    total += points[i - 1].distanceTo(points[i]);
  }
  return total;
}

// Weiches Snap: nur wenn der Klick innerhalb des Schwellwerts (in Metern) am Polygon-Rand
// liegt, wird auf die nächste Kante gesnappt. Sonst Klick unverändert zurückgeben.
// Damit lassen sich Firsts auch frei IM Polygon platzieren (Walmgiebel, T-Anbau, …).
const SNAP_THRESHOLD_M = 1.5;

function snapToPolygonBoundary(click: L.LatLng, polygon: L.LatLng[]): L.LatLng {
  if (polygon.length < 2) return click;

  const c = L.CRS.EPSG3857.project(click);
  const pts = polygon.map((p) => L.CRS.EPSG3857.project(p));

  let bestX = c.x;
  let bestY = c.y;
  let bestDist = Infinity;

  for (let i = 0; i < pts.length; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % pts.length];
    const abx = b.x - a.x;
    const aby = b.y - a.y;
    const lenSq = abx * abx + aby * aby;
    let t = lenSq === 0 ? 0 : ((c.x - a.x) * abx + (c.y - a.y) * aby) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const sx = a.x + t * abx;
    const sy = a.y + t * aby;
    const dx = sx - c.x;
    const dy = sy - c.y;
    const d = dx * dx + dy * dy;
    if (d < bestDist) {
      bestDist = d;
      bestX = sx;
      bestY = sy;
    }
  }

  // Mercator-Distanz mit Cos(lat)-Korrektur auf echte Meter (genau genug auf <2 m).
  const meanLat = (click.lat * Math.PI) / 180;
  const distM = Math.sqrt(bestDist) * Math.cos(meanLat);
  if (distM > SNAP_THRESHOLD_M) return click;
  return L.CRS.EPSG3857.unproject(L.point(bestX, bestY));
}

export function DrawingFlow({ lat, lon, address }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);

  const polygonLayerRef = useRef<L.Polygon | null>(null);
  const polygonPreviewRef = useRef<L.Polyline | null>(null);
  const polygonVertexLayerRef = useRef<L.LayerGroup | null>(null);
  const ridgesLayerRef = useRef<L.LayerGroup | null>(null);
  const valleysLayerRef = useRef<L.LayerGroup | null>(null);
  const sectionsLayerRef = useRef<L.LayerGroup | null>(null);

  const [step, setStep] = useState<Step>("type");
  const [roofType, setRoofType] = useState<RoofType | null>(null);
  const [polygon, setPolygon] = useState<L.LatLng[]>([]);
  // Mehrere First-Linien (für L/T-Satteldächer). Jede ist ein Paar von Endpunkten.
  const [ridges, setRidges] = useState<L.LatLng[][]>([]);
  const [draftRidge, setDraftRidge] = useState<L.LatLng[]>([]);
  // Manuell gezeichnete Kehlen (Innenecke, wo zwei Dachflächen sich treffen)
  const [valleys, setValleys] = useState<L.LatLng[][]>([]);
  const [draftValley, setDraftValley] = useState<L.LatLng[]>([]);

  const [sections, setSections] = useState<RoofSection[] | null>(null);
  const [sectionsLoading, setSectionsLoading] = useState(false);
  const [sectionsError, setSectionsError] = useState<string | null>(null);

  // Map initialisieren – nur einmal
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const center = L.latLng(lat, lon);
    const map = L.map(containerRef.current, {
      center,
      zoom: 19,
      minZoom: 18,
      maxZoom: 22,
      maxBounds: center.toBounds(300),
      maxBoundsViscosity: 1.0,
      zoomControl: true,
      doubleClickZoom: false, // double-click triggert sonst Zoom statt drawing-Klick
      zoomSnap: 0.5, // erlaubt feinere Zwischenstufen
    });

    new DopProxyLayer("", {
      maxZoom: 22,
      maxNativeZoom: 20, // ab z=21 skaliert Leaflet die 10cm-Tiles weich hoch
      tileSize: 256,
      attribution: "© Geobasis NRW",
    }).addTo(map);

    polygonVertexLayerRef.current = L.layerGroup().addTo(map);
    sectionsLayerRef.current = L.layerGroup().addTo(map);
    valleysLayerRef.current = L.layerGroup().addTo(map);
    ridgesLayerRef.current = L.layerGroup().addTo(map);

    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [lat, lon]);

  // Click-Handler + Cursor-Toggle abhängig vom Schritt
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const container = map.getContainer();
    const isDrawing = step === "polygon" || step === "ridge" || step === "valley";
    container.classList.toggle("drawing-mode", isDrawing);

    const onClick = (e: L.LeafletMouseEvent) => {
      if (step === "polygon") {
        setPolygon((prev) => [...prev, e.latlng]);
      } else if (step === "ridge") {
        if (draftRidge.length === 0) {
          setDraftRidge([e.latlng]);
        } else {
          setRidges((prev) => [...prev, [draftRidge[0], e.latlng]]);
          setDraftRidge([]);
        }
      } else if (step === "valley") {
        if (draftValley.length === 0) {
          setDraftValley([e.latlng]);
        } else {
          setValleys((prev) => [...prev, [draftValley[0], e.latlng]]);
          setDraftValley([]);
        }
      }
    };
    map.on("click", onClick);
    return () => {
      map.off("click", onClick);
    };
  }, [step, polygon, draftRidge, draftValley]);

  // Polygon-Overlay rendern (offene Polyline während Step 1, geschlossenes Polygon danach)
  useEffect(() => {
    const map = mapRef.current;
    const vertexLayer = polygonVertexLayerRef.current;
    if (!map || !vertexLayer) return;

    polygonLayerRef.current?.remove();
    polygonLayerRef.current = null;
    polygonPreviewRef.current?.remove();
    polygonPreviewRef.current = null;
    vertexLayer.clearLayers();

    if (polygon.length >= 3 && step !== "polygon") {
      polygonLayerRef.current = L.polygon(polygon, {
        color: "#d97706",
        weight: 2,
        fillOpacity: 0.18,
        fillColor: "#d97706",
        interactive: false,
      }).addTo(map);
    } else if (polygon.length > 0) {
      polygonPreviewRef.current = L.polyline(polygon, {
        color: "#d97706",
        weight: 2,
        dashArray: "5 5",
        interactive: false,
      }).addTo(map);
    }

    polygon.forEach((latlng) => {
      L.circleMarker(latlng, {
        radius: 5,
        color: "#fff",
        weight: 2,
        fillColor: "#d97706",
        fillOpacity: 1,
        interactive: false,
      }).addTo(vertexLayer);
    });
  }, [polygon, step]);

  // Ridge-Overlay: alle abgeschlossenen Firsts + Draft-Marker
  useEffect(() => {
    const layer = ridgesLayerRef.current;
    if (!layer) return;
    layer.clearLayers();

    ridges.forEach(([a, b]) => {
      L.polyline([a, b], { color: "#dc2626", weight: 3, interactive: false }).addTo(layer);
      [a, b].forEach((p) =>
        L.circleMarker(p, {
          radius: 5,
          color: "#fff",
          weight: 2,
          fillColor: "#dc2626",
          fillOpacity: 1,
          interactive: false,
        }).addTo(layer),
      );
    });

    // Draft: einzelner Marker (heller), wenn erster Punkt einer neuen First gesetzt
    draftRidge.forEach((p) =>
      L.circleMarker(p, {
        radius: 5,
        color: "#fff",
        weight: 2,
        fillColor: "#f87171",
        fillOpacity: 1,
        interactive: false,
      }).addTo(layer),
    );
  }, [ridges, draftRidge]);

  const area = useMemo(() => polygonAreaM2(polygon), [polygon]);
  const ridgeTotalLength = useMemo(
    () => ridges.reduce((sum, r) => sum + lineLengthM(r), 0),
    [ridges],
  );
  const valleyTotalLength = useMemo(
    () => valleys.reduce((sum, v) => sum + lineLengthM(v), 0),
    [valleys],
  );

  // Kehlen-Overlay: blaue Linien, vom User selbst gezeichnet
  useEffect(() => {
    const layer = valleysLayerRef.current;
    if (!layer) return;
    layer.clearLayers();

    valleys.forEach(([a, b]) => {
      L.polyline([a, b], { color: "#2563eb", weight: 2.5, interactive: false }).addTo(layer);
      [a, b].forEach((p) =>
        L.circleMarker(p, {
          radius: 5,
          color: "#fff",
          weight: 2,
          fillColor: "#2563eb",
          fillOpacity: 1,
          interactive: false,
        }).addTo(layer),
      );
    });

    draftValley.forEach((p) =>
      L.circleMarker(p, {
        radius: 5,
        color: "#fff",
        weight: 2,
        fillColor: "#60a5fa",
        fillOpacity: 1,
        interactive: false,
      }).addTo(layer),
    );
  }, [valleys, draftValley]);

  function reset() {
    setStep("type");
    setRoofType(null);
    setPolygon([]);
    setRidges([]);
    setDraftRidge([]);
    setValleys([]);
    setDraftValley([]);
    setSections(null);
    setSectionsError(null);
  }

  // Beim Erreichen von "done": Polygon-Aufteilung in Sektionen vom Backend.
  useEffect(() => {
    if (step !== "done") return;
    let cancelled = false;
    setSectionsLoading(true);
    setSectionsError(null);

    const polyPts = polygon.map((p) => [p.lat, p.lng] as [number, number]);
    const ridgePts = ridges.map(
      (r) => r.map((p) => [p.lat, p.lng] as [number, number]) as [number, number][],
    );
    const valleyPts = valleys.map(
      (v) => v.map((p) => [p.lat, p.lng] as [number, number]) as [number, number][],
    );

    computeSections(polyPts, ridgePts, valleyPts)
      .then((res) => {
        if (!cancelled) setSections(res.sections);
      })
      .catch((err: Error) => {
        if (!cancelled) setSectionsError(err.message);
      })
      .finally(() => {
        if (!cancelled) setSectionsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [step, polygon, ridges, valleys]);

  // Render Sektionen als farbige Polygone (nur in Step "done")
  useEffect(() => {
    const layer = sectionsLayerRef.current;
    if (!layer) return;
    layer.clearLayers();
    if (step !== "done" || !sections) return;

    const palette = ["#10b981", "#3b82f6", "#a855f7", "#ec4899", "#eab308", "#06b6d4", "#f97316", "#84cc16"];
    sections.forEach((sec, i) => {
      const color = palette[i % palette.length];
      L.polygon(sec.points, {
        color,
        weight: 1.5,
        opacity: 0.9,
        fillColor: color,
        fillOpacity: 0.25,
        interactive: false,
      }).addTo(layer);
    });
  }, [sections, step]);

  const typeInfo = roofType ? roofTypeInfo(roofType) : null;
  const hasRidge = typeInfo?.hasRidge ?? false;

  return (
    <div className="h-full flex">
      <div className="flex-1 relative">
        <div ref={containerRef} className="absolute inset-0" />
      </div>

      <aside className="w-96 shrink-0 border-l border-stone-200 bg-white overflow-y-auto">
        <div className="p-6 space-y-5">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">Dach einzeichnen</h2>
            <p className="text-xs text-stone-500 mt-1 truncate" title={address}>
              {address}
            </p>
          </div>

          <StepIndicator step={step} hasRidge={hasRidge} />

          {step === "type" && (
            <TypeStep
              selected={roofType}
              onSelect={(t) => {
                setRoofType(t);
                setStep("polygon");
              }}
            />
          )}

          {step === "polygon" && typeInfo && (
            <PolygonStep
              typeName={typeInfo.name}
              count={polygon.length}
              onBack={() => setStep("type")}
              onReset={() => setPolygon([])}
              onNext={() => setStep(hasRidge ? "ridge" : "done")}
              canContinue={polygon.length >= 3}
              nextLabel={hasRidge ? "Weiter →" : "Fertig →"}
            />
          )}

          {step === "ridge" && typeInfo && (
            <RidgeStep
              label={typeInfo.ridgeLabel}
              hint={typeInfo.ridgeHint}
              ridgeCount={ridges.length}
              draftCount={draftRidge.length}
              onBack={() => setStep("polygon")}
              onUndoLast={() => {
                if (draftRidge.length > 0) setDraftRidge([]);
                else setRidges((rs) => rs.slice(0, -1));
              }}
              onResetAll={() => {
                setRidges([]);
                setDraftRidge([]);
              }}
              onNext={() => {
                setDraftRidge([]);
                setStep("valley");
              }}
              canFinish={ridges.length >= 1}
              canUndo={ridges.length > 0 || draftRidge.length > 0}
            />
          )}

          {step === "valley" && typeInfo && (
            <ValleyStep
              valleyCount={valleys.length}
              draftCount={draftValley.length}
              onBack={() => setStep("ridge")}
              onUndoLast={() => {
                if (draftValley.length > 0) setDraftValley([]);
                else setValleys((vs) => vs.slice(0, -1));
              }}
              onResetAll={() => {
                setValleys([]);
                setDraftValley([]);
              }}
              onDone={() => {
                setDraftValley([]);
                setStep("done");
              }}
              canUndo={valleys.length > 0 || draftValley.length > 0}
            />
          )}

          {step === "done" && typeInfo && (
            <ResultPanel
              typeInfo={typeInfo}
              area={area}
              ridgeTotalLength={ridgeTotalLength}
              ridgeCount={ridges.length}
              valleyCount={valleys.length}
              valleyTotalLength={valleyTotalLength}
              vertexCount={polygon.length}
              sections={sections}
              sectionsLoading={sectionsLoading}
              sectionsError={sectionsError}
              onRestart={reset}
              onEditPolygon={() => setStep("polygon")}
              onEditRidge={() => setStep("ridge")}
              onEditValley={() => setStep("valley")}
            />
          )}
        </div>
      </aside>
    </div>
  );
}

function StepIndicator({ step, hasRidge }: { step: Step; hasRidge: boolean }) {
  const steps: { id: Step; label: string }[] = [
    { id: "type", label: "Typ" },
    { id: "polygon", label: "Umriss" },
    ...(hasRidge
      ? [
          { id: "ridge" as Step, label: "First" },
          { id: "valley" as Step, label: "Kehle" },
        ]
      : []),
    { id: "done", label: "Ergebnis" },
  ];
  const activeIdx = steps.findIndex((s) => s.id === step);
  return (
    <ol className="flex gap-1.5 text-xs">
      {steps.map((s, i) => {
        const isActive = i === activeIdx;
        const isDone = i < activeIdx;
        return (
          <li
            key={s.id}
            className={[
              "flex-1 rounded px-2 py-1.5 border text-center font-medium",
              isActive
                ? "border-amber-600 bg-amber-50 text-amber-900"
                : isDone
                  ? "border-stone-300 bg-stone-100 text-stone-500"
                  : "border-stone-200 bg-white text-stone-400",
            ].join(" ")}
          >
            {i + 1}. {s.label}
          </li>
        );
      })}
    </ol>
  );
}

function RoofIcon({ type }: { type: RoofType }) {
  const common = "w-full h-16";
  const stroke = "currentColor";
  switch (type) {
    case "satteldach":
      // First in der Mitte, zwei Schrägen, Traufenlinie unten
      return (
        <svg viewBox="0 0 64 36" className={common} fill="none" stroke={stroke} strokeWidth="2" strokeLinejoin="round">
          <path d="M4 30 L32 8 L60 30" />
          <line x1="4" y1="30" x2="60" y2="30" />
        </svg>
      );
    case "walmdach":
      // First kürzer, vier Schrägen
      return (
        <svg viewBox="0 0 64 36" className={common} fill="none" stroke={stroke} strokeWidth="2" strokeLinejoin="round">
          <path d="M4 30 L20 8 L44 8 L60 30 Z" />
          <line x1="20" y1="8" x2="44" y2="8" strokeWidth="2.5" />
        </svg>
      );
    case "pultdach":
      // Eine Schräge, links hoch, rechts unten
      return (
        <svg viewBox="0 0 64 36" className={common} fill="none" stroke={stroke} strokeWidth="2" strokeLinejoin="round">
          <path d="M4 8 L60 26 L60 30 L4 30 Z" />
        </svg>
      );
    case "flachdach":
      // Einfaches Rechteck
      return (
        <svg viewBox="0 0 64 36" className={common} fill="none" stroke={stroke} strokeWidth="2" strokeLinejoin="round">
          <rect x="4" y="14" width="56" height="16" />
        </svg>
      );
  }
}

function TypeStep({
  selected,
  onSelect,
}: {
  selected: RoofType | null;
  onSelect: (t: RoofType) => void;
}) {
  return (
    <div className="space-y-3">
      <div className="text-base font-semibold">Dachform wählen</div>
      <p className="text-sm text-stone-600 leading-relaxed">
        Welche Form hat dein Dach? Die Auswahl bestimmt, welche Linien du danach einzeichnest.
      </p>
      <div className="grid grid-cols-2 gap-2">
        {ROOF_TYPES.map((t) => {
          const isSelected = selected === t.id;
          return (
            <button
              key={t.id}
              onClick={() => onSelect(t.id)}
              className={[
                "text-left p-3 rounded border transition-colors",
                isSelected
                  ? "border-amber-600 bg-amber-50 ring-1 ring-amber-600"
                  : "border-stone-200 bg-white hover:border-stone-400 hover:bg-stone-50",
              ].join(" ")}
            >
              <div className={isSelected ? "text-amber-700" : "text-stone-700"}>
                <RoofIcon type={t.id} />
              </div>
              <div className="mt-2 text-sm font-medium">{t.name}</div>
              <div className="text-[11px] text-stone-500 leading-snug mt-0.5">{t.description}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function PolygonStep({
  typeName,
  count,
  onBack,
  onReset,
  onNext,
  canContinue,
  nextLabel,
}: {
  typeName: string;
  count: number;
  onBack: () => void;
  onReset: () => void;
  onNext: () => void;
  canContinue: boolean;
  nextLabel: string;
}) {
  return (
    <div className="space-y-3">
      <div>
        <div className="text-base font-semibold">Dachumriss klicken</div>
        <div className="text-xs text-stone-500">Gewählt: {typeName}</div>
      </div>
      <p className="text-sm text-stone-600 leading-relaxed">
        Klicke nacheinander die Ecken deines Dachs auf dem Luftbild. Bei einem rechteckigen Grundriss
        sind das vier Punkte. Mindestens drei.
      </p>
      <div className="flex items-center justify-between rounded bg-stone-50 border border-stone-200 px-3 py-2">
        <span className="text-xs text-stone-500">Gesetzte Ecken</span>
        <span className="font-mono text-sm">{count}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={onBack}
          className="px-3 py-2 text-sm rounded border border-stone-300 hover:bg-stone-50"
        >
          ← Typ
        </button>
        <button
          onClick={onReset}
          disabled={count === 0}
          className="flex-1 min-w-0 px-3 py-2 text-sm rounded border border-stone-300 hover:bg-stone-50 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Zurücksetzen
        </button>
        <button
          onClick={onNext}
          disabled={!canContinue}
          className="flex-1 min-w-0 px-3 py-2 text-sm font-medium rounded bg-amber-600 text-white hover:bg-amber-700 disabled:bg-stone-300 disabled:cursor-not-allowed"
        >
          {nextLabel}
        </button>
      </div>
    </div>
  );
}

function RidgeStep({
  label,
  hint,
  ridgeCount,
  draftCount,
  onBack,
  onUndoLast,
  onResetAll,
  onNext,
  canFinish,
  canUndo,
}: {
  label: string;
  hint: string;
  ridgeCount: number;
  draftCount: number;
  onBack: () => void;
  onUndoLast: () => void;
  onResetAll: () => void;
  onNext: () => void;
  canFinish: boolean;
  canUndo: boolean;
}) {
  return (
    <div className="space-y-3">
      <div className="text-base font-semibold">{label}</div>
      <p className="text-sm text-stone-600 leading-relaxed">{hint}</p>
      <p className="text-xs text-stone-500 leading-relaxed">
        Bei L- oder T-förmigen Dächern mehrere Firsts zeichnen — je zwei Klicks pro Linie. Punkte
        landen genau wo du klickst, kein Snap an den Umriss.
      </p>
      <div className="rounded bg-stone-50 border border-stone-200 px-3 py-2 space-y-1">
        <div className="flex items-center justify-between text-xs text-stone-500">
          <span>Firsts gezeichnet</span>
          <span className="font-mono text-sm text-stone-800">{ridgeCount}</span>
        </div>
        {draftCount > 0 && (
          <div className="flex items-center justify-between text-xs text-amber-700">
            <span>In Bearbeitung</span>
            <span className="font-mono">noch 1 Klick</span>
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={onBack}
          className="px-3 py-2 text-sm rounded border border-stone-300 hover:bg-stone-50"
        >
          ← Umriss
        </button>
        <button
          onClick={onUndoLast}
          disabled={!canUndo}
          className="flex-1 min-w-0 px-3 py-2 text-sm rounded border border-stone-300 hover:bg-stone-50 disabled:opacity-40 disabled:cursor-not-allowed"
          title="Letzten Punkt bzw. letzte First entfernen"
        >
          ↶ Letzte
        </button>
        <button
          onClick={onNext}
          disabled={!canFinish}
          className="flex-1 min-w-0 px-3 py-2 text-sm font-medium rounded bg-amber-600 text-white hover:bg-amber-700 disabled:bg-stone-300 disabled:cursor-not-allowed"
        >
          Weiter →
        </button>
      </div>
      <button
        onClick={onResetAll}
        disabled={!canUndo}
        className="text-xs text-stone-500 hover:text-stone-700 disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Alle Firsts zurücksetzen
      </button>
    </div>
  );
}

function ValleyStep({
  valleyCount,
  draftCount,
  onBack,
  onUndoLast,
  onResetAll,
  onDone,
  canUndo,
}: {
  valleyCount: number;
  draftCount: number;
  onBack: () => void;
  onUndoLast: () => void;
  onResetAll: () => void;
  onDone: () => void;
  canUndo: boolean;
}) {
  return (
    <div className="space-y-3">
      <div className="text-base font-semibold">Kehlen einzeichnen (optional)</div>
      <p className="text-sm text-stone-600 leading-relaxed">
        Kehlen sind Innenecken, wo zwei Dachflächen konkav zusammentreffen — typisch bei L- oder
        T-förmigen Dächern an der Anbau-Stelle. Je zwei Klicks pro Kehle.
      </p>
      <p className="text-xs text-stone-500 leading-relaxed">
        Keine Kehlen vorhanden? Einfach „Überspringen" klicken.
      </p>
      <div className="rounded bg-stone-50 border border-stone-200 px-3 py-2 space-y-1">
        <div className="flex items-center justify-between text-xs text-stone-500">
          <span>Kehlen gezeichnet</span>
          <span className="font-mono text-sm text-stone-800">{valleyCount}</span>
        </div>
        {draftCount > 0 && (
          <div className="flex items-center justify-between text-xs text-blue-700">
            <span>In Bearbeitung</span>
            <span className="font-mono">noch 1 Klick</span>
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={onBack}
          className="px-3 py-2 text-sm rounded border border-stone-300 hover:bg-stone-50"
        >
          ← First
        </button>
        <button
          onClick={onUndoLast}
          disabled={!canUndo}
          className="flex-1 min-w-0 px-3 py-2 text-sm rounded border border-stone-300 hover:bg-stone-50 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          ↶ Letzte
        </button>
        <button
          onClick={onDone}
          className="flex-1 min-w-0 px-3 py-2 text-sm font-medium rounded bg-amber-600 text-white hover:bg-amber-700"
        >
          {valleyCount === 0 ? "Überspringen →" : "Fertig →"}
        </button>
      </div>
      <button
        onClick={onResetAll}
        disabled={!canUndo}
        className="text-xs text-stone-500 hover:text-stone-700 disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Alle Kehlen zurücksetzen
      </button>
    </div>
  );
}

function ResultPanel({
  typeInfo,
  area,
  ridgeTotalLength,
  ridgeCount,
  valleyCount,
  valleyTotalLength,
  vertexCount,
  sections,
  sectionsLoading,
  sectionsError,
  onRestart,
  onEditPolygon,
  onEditRidge,
  onEditValley,
}: {
  typeInfo: (typeof ROOF_TYPES)[number];
  area: number;
  ridgeTotalLength: number;
  ridgeCount: number;
  valleyCount: number;
  valleyTotalLength: number;
  vertexCount: number;
  sections: RoofSection[] | null;
  sectionsLoading: boolean;
  sectionsError: string | null;
  onRestart: () => void;
  onEditPolygon: () => void;
  onEditRidge: () => void;
  onEditValley: () => void;
}) {
  const isPult = typeInfo.id === "pultdach";
  const ridgeMetricLabel =
    ridgeCount === 1
      ? isPult
        ? "Hohe-Kante-Länge"
        : "First-Länge"
      : isPult
        ? "Kanten gesamt"
        : "Firsts gesamt";

  return (
    <div className="space-y-5">
      <div>
        <div className="text-base font-semibold">Ergebnis</div>
        <div className="text-xs text-stone-500 flex items-center gap-2 mt-1">
          <span className="inline-block w-8 text-stone-400">
            <RoofIcon type={typeInfo.id} />
          </span>
          {typeInfo.name}
        </div>
      </div>

      <dl className="space-y-2">
        <Metric label="Grundfläche (2D)" value={`${area.toFixed(1)} m²`} />
        {typeInfo.hasRidge && ridgeCount > 0 && (
          <>
            <Metric label={ridgeMetricLabel} value={`${ridgeTotalLength.toFixed(2)} m`} />
            {ridgeCount > 1 && (
              <Metric label="Anzahl Firsts" value={`${ridgeCount}`} />
            )}
          </>
        )}
        {valleyCount > 0 && (
          <Metric
            label={valleyCount === 1 ? "Kehle (2D)" : `Kehlen (gesamt, 2D)`}
            value={`${valleyTotalLength.toFixed(2)} m`}
          />
        )}
        <Metric label="Polygon-Ecken" value={`${vertexCount}`} />
      </dl>

      <div className="border-t border-stone-200 pt-3">
        <div className="text-sm font-semibold mb-2">Dachflächen — pro Sektion</div>
        {sectionsLoading && (
          <div className="text-xs text-stone-500">Aufteilung + Plane-Fit läuft…</div>
        )}
        {sectionsError && (
          <div className="text-xs text-red-700">Fehler: {sectionsError}</div>
        )}
        {sections && (
          <div className="space-y-2">
            <div className="text-xs text-stone-500">
              {sections.length} Sektion{sections.length === 1 ? "" : "en"} erkannt
            </div>
            {(() => {
              const totalSlope = sections.reduce(
                (sum, s) => sum + (s.area_m2_slope ?? s.area_m2_2d),
                0,
              );
              return (
                <div className="rounded bg-stone-50 border border-stone-200 px-3 py-2 flex justify-between text-sm">
                  <span className="text-stone-600">Schräge-Fläche gesamt</span>
                  <span className="font-mono font-semibold">{totalSlope.toFixed(1)} m²</span>
                </div>
              );
            })()}
            <ul className="space-y-2">
              {sections.map((s, i) => {
                const palette = ["#10b981", "#3b82f6", "#a855f7", "#ec4899", "#eab308", "#06b6d4", "#f97316", "#84cc16"];
                const c = palette[i % palette.length];
                return (
                  <li
                    key={i}
                    className="rounded border border-stone-200 bg-white p-2 space-y-1 text-xs"
                  >
                    <div className="flex items-center justify-between">
                      <span className="flex items-center gap-2 font-medium text-stone-800">
                        <span
                          className="inline-block w-3 h-3 rounded-sm"
                          style={{ background: c }}
                        />
                        Fläche {i + 1}
                      </span>
                      <span className="font-mono text-stone-500">
                        {s.n_samples} Pkt
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 font-mono text-stone-700">
                      <span className="text-stone-500">2D-Fläche</span>
                      <span className="text-right">{s.area_m2_2d.toFixed(1)} m²</span>
                      {s.area_m2_slope != null && (
                        <>
                          <span className="text-stone-500">Schräg-Fläche</span>
                          <span className="text-right font-semibold">
                            {s.area_m2_slope.toFixed(1)} m²
                          </span>
                        </>
                      )}
                      {s.pitch_deg != null && (
                        <>
                          <span className="text-stone-500">Neigung</span>
                          <span className="text-right">{s.pitch_deg.toFixed(1)}°</span>
                        </>
                      )}
                      {s.aspect_label && (
                        <>
                          <span className="text-stone-500">Ausrichtung</span>
                          <span className="text-right">{s.aspect_label}</span>
                        </>
                      )}
                      {s.mean_height_m != null && (
                        <>
                          <span className="text-stone-500">Höhe (∅)</span>
                          <span className="text-right">{s.mean_height_m.toFixed(1)} m</span>
                        </>
                      )}
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={onEditPolygon}
          className="px-3 py-2 text-sm rounded border border-stone-300 hover:bg-stone-50"
        >
          Umriss bearbeiten
        </button>
        {typeInfo.hasRidge && (
          <button
            onClick={onEditRidge}
            className="px-3 py-2 text-sm rounded border border-stone-300 hover:bg-stone-50"
          >
            {typeInfo.id === "pultdach" ? "Kante bearbeiten" : "First bearbeiten"}
          </button>
        )}
        {typeInfo.hasRidge && (
          <button
            onClick={onEditValley}
            className="px-3 py-2 text-sm rounded border border-stone-300 hover:bg-stone-50"
          >
            Kehlen bearbeiten
          </button>
        )}
      </div>

      <button onClick={onRestart} className="text-sm text-amber-700 hover:underline">
        Neu beginnen
      </button>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-stone-100 pb-1.5">
      <dt className="text-sm text-stone-600">{label}</dt>
      <dd className="font-mono text-base font-medium">{value}</dd>
    </div>
  );
}
