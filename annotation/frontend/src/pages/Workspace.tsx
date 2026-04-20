import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import LayersPanel from "../components/LayersPanel";
import PropertiesPanel from "../components/PropertiesPanel";
import SceneCanvas from "../components/SceneCanvas";
import Toolbar from "../components/Toolbar";
import { useAnnotation } from "../hooks/useAnnotation";
import { useKeyboard } from "../hooks/useKeyboard";
import { useScene } from "../hooks/useScene";
import { api } from "../lib/api";
import { colors } from "../lib/theme";
import type { LaneConnection, LayerKey, LayerVisibility, MouseMode } from "../lib/types";

const DEFAULT_LAYERS: LayerVisibility = {
  images: true,
  roadway: true,
  road_markings: true,
  centerlines: true,
  vehicles: false,
  trajectories: false,
  connections: true,
  texts: false,
};

const LAYER_KEYS: LayerKey[] = [
  "roadway",
  "road_markings",
  "centerlines",
  "vehicles",
  "trajectories",
  "connections",
];

export default function Workspace() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { annotation: initial, loading, error, process } = useScene(id!);

  const {
    annotation,
    selected,
    setSelected,
    confirmElement,
    rejectElement,
    updateLane,
    updateVehicle,
    addConnection,
    deleteConnection,
    hideRoadEdge,
    undo,
    selectNext,
    setAnnotation,
  } = useAnnotation(initial);

  const [mode, setMode] = useState<MouseMode>("select");
  const [layers, setLayers] = useState<LayerVisibility>(DEFAULT_LAYERS);

  // Sync annotation when initial loads
  useEffect(() => {
    if (initial) setAnnotation(initial);
  }, [initial, setAnnotation]);

  const toggleLayer = useCallback(
    (key: LayerKey) => setLayers((prev) => ({ ...prev, [key]: !prev[key] })),
    []
  );

  const handleAddConnection = useCallback(
    async (fromId: string, toId: string, fromEnd: "start" | "end", toEnd: "start" | "end") => {
      if (!annotation) return;
      const conn: LaneConnection = {
        id: `conn_${Date.now()}`,
        from_lane_id: fromId,
        to_lane_id: toId,
        from_end: fromEnd,
        to_end: toEnd,
        connection_type: "through",
        control_points: null,
        status: "auto",
      };
      await addConnection(conn);
    },
    [annotation, addConnection]
  );

  const handleMarkReviewed = useCallback(async () => {
    if (!annotation) return;
    const updated = await api.annotations.update(annotation.case_id, {
      workflow_status: "reviewed",
    });
    setAnnotation(updated);
  }, [annotation, setAnnotation]);

  const handleConfirm = useCallback(async () => {
    if (!selected) return;
    const updated = await confirmElement(selected.kind, selected.id);
    selectNext(updated ?? undefined);
  }, [selected, confirmElement, selectNext]);

  const handleReject = useCallback(async () => {
    if (!selected) return;
    if (selected.kind === "connection") {
      await deleteConnection(selected.id);
      setSelected(null);
    } else if (selected.kind === "roadway") {
      await hideRoadEdge(selected.id);
      setSelected(null);
    } else {
      await rejectElement(selected.kind, selected.id);
    }
  }, [selected, rejectElement, deleteConnection, hideRoadEdge, setSelected]);

  const moveSelectedBy = useCallback(
    (delta: -1 | 1) => {
      if (!annotation || !selected) return;

      const ids =
        selected.kind === "lane"
          ? annotation.lanes.map((lane) => lane.id)
          : selected.kind === "vehicle"
            ? annotation.vehicles.map((veh) => veh.id)
            : annotation.lane_connections.map((conn) => conn.id);

      const currentIndex = ids.indexOf(selected.id);
      if (currentIndex < 0) return;

      const nextIndex = currentIndex + delta;
      if (nextIndex < 0 || nextIndex >= ids.length) return;

      setSelected({ kind: selected.kind, id: ids[nextIndex] });
    },
    [annotation, selected, setSelected]
  );

  useKeyboard({
    onConfirm: handleConfirm,
    onReject: handleReject,
    onUndo: undo,
    onNext: selectNext,
    onEscape: () => setSelected(null),
    onModeSelect: () => setMode("select"),
    onModeConnect: () => setMode("connect"),
    onModeEdit: () => setMode("edit"),
    onToggleLayer: (n) => {
      if (n < LAYER_KEYS.length) toggleLayer(LAYER_KEYS[n]);
    },
    onArrowUp: () => moveSelectedBy(-1),
    onArrowDown: () => moveSelectedBy(1),
  });

  if (loading) {
    return (
      <div style={styles.centered}>
        <span style={styles.mono}>Loading {id}…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.centered}>
        <span style={{ color: colors.accent.danger }}>{error}</span>
        <button onClick={() => navigate("/")} style={styles.backBtn}>
          ← BACK
        </button>
      </div>
    );
  }

  if (!annotation) {
    return (
      <div style={styles.centered}>
        <span style={styles.mono}>Case {id} not yet processed.</span>
        <button onClick={process} style={styles.processBtn}>
          PROCESS NOW
        </button>
        <button onClick={() => navigate("/")} style={styles.backBtn}>
          ← BACK
        </button>
      </div>
    );
  }

  return (
    <div style={styles.page}>
      <div style={styles.topBar}>
        <button onClick={() => navigate("/")} style={styles.backBtn}>
          ←
        </button>
        <Toolbar
          caseId={annotation.case_id}
          filename={annotation.far_filename}
          mode={mode}
          onModeChange={setMode}
          workflowStatus={annotation.workflow_status}
          onMarkReviewed={handleMarkReviewed}
        />
      </div>

      <div style={styles.main}>
        <LayersPanel
          annotation={annotation}
          layers={layers}
          onToggle={toggleLayer}
          selected={selected}
          onSelect={setSelected}
        />

        <div style={styles.canvas}>
          <SceneCanvas
            annotation={annotation}
            layers={layers}
            mode={mode}
            selected={selected}
            onSelect={setSelected}
            onAddConnection={handleAddConnection}
            onHideRoadEdge={hideRoadEdge}
          />
        </div>

        <PropertiesPanel
          annotation={annotation}
          selected={selected}
          onConfirm={confirmElement}
          onReject={async (kind, id) => {
            if (kind === "roadway") { await hideRoadEdge(id); setSelected(null); }
            else await rejectElement(kind, id);
          }}
          onUpdateLane={updateLane}
          onUpdateVehicle={updateVehicle}
          onDeleteConnection={async (id) => { await deleteConnection(id); setSelected(null); }}
        />
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    background: colors.surface.page,
    color: colors.text.primary,
  },
  topBar: {
    display: "flex",
    alignItems: "stretch",
    flexShrink: 0,
    borderBottom: `1px solid ${colors.border.default}`,
  },
  main: {
    flex: 1,
    display: "flex",
    overflow: "hidden",
  },
  canvas: {
    flex: 1,
    overflow: "hidden",
  },
  centered: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    gap: 16,
    color: colors.text.muted,
  },
  mono: {
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 13,
  },
  backBtn: {
    padding: "6px 16px",
    background: colors.util.transparent,
    border: `1px solid ${colors.border.strong}`,
    color: colors.text.muted,
    borderRadius: 3,
    cursor: "pointer",
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 12,
    alignSelf: "center",
    margin: "0 8px",
  },
  processBtn: {
    padding: "8px 20px",
    background: colors.accent.info,
    color: colors.text.onDark,
    border: "none",
    borderRadius: 3,
    cursor: "pointer",
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 12,
    fontWeight: "bold",
  },
};
