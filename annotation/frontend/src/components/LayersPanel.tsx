import { useEffect, useRef } from "react";
import type { CaseAnnotation, LayerKey, LayerVisibility, SelectedElement } from "../lib/types";
import { colors, statusColors } from "../lib/theme";

const LAYER_LABELS: { key: LayerKey; label: string }[] = [
  { key: "images", label: "Road signs" },
  { key: "roadway", label: "Roadway edges [1]" },
  { key: "road_markings", label: "Road markings [2]" },
  { key: "centerlines", label: "Centerlines [3]" },
  { key: "vehicles", label: "Vehicles [4]" },
  { key: "trajectories", label: "Trajectories [5]" },
  { key: "connections", label: "Connections [6]" },
  { key: "texts", label: "Text labels" },
];

interface LayersPanelProps {
  annotation: CaseAnnotation;
  layers: LayerVisibility;
  onToggle: (key: LayerKey) => void;
  selected: SelectedElement;
  onSelect: (sel: SelectedElement) => void;
}

export default function LayersPanel({
  annotation,
  layers,
  onToggle,
  selected,
  onSelect,
}: LayersPanelProps) {
  const itemRefs = useRef<Record<string, HTMLDivElement | null>>({});

  useEffect(() => {
    if (!selected) return;
    const key = `${selected.kind}:${selected.id}`;
    const el = itemRefs.current[key];
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [selected]);

  const reviewed = [
    ...annotation.lanes.filter((l) => l.status !== "auto"),
    ...annotation.vehicles.filter((v) => v.status !== "auto"),
  ].length;
  const total =
    annotation.lanes.length +
    annotation.vehicles.length +
    annotation.lane_connections.length;

  return (
    <div style={styles.panel}>
      <div style={styles.section}>
        <div style={styles.sectionTitle}>LAYERS</div>
        {LAYER_LABELS.map(({ key, label }) => (
          <label key={key} style={styles.layerRow}>
            <input
              type="checkbox"
              checked={layers[key]}
              onChange={() => onToggle(key)}
              style={styles.checkbox}
            />
            <span style={{ color: layers[key] ? colors.text.primary : colors.text.disabled, fontSize: 12 }}>
              {label}
            </span>
          </label>
        ))}
      </div>

      <div style={styles.divider} />

      <div style={styles.section}>
        <div style={styles.sectionTitle}>PROGRESS</div>
        <div style={{ fontSize: 11, color: colors.text.muted, marginBottom: 6 }}>
          {reviewed} / {total} reviewed
        </div>
        <div style={styles.progressBar}>
          <div
            style={{
              ...styles.progressFill,
              width: total > 0 ? `${(reviewed / total) * 100}%` : "0%",
            }}
          />
        </div>
      </div>

      <div style={styles.divider} />

      {/* Lanes list */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>LANES ({annotation.lanes.length})</div>
        <div style={styles.elementList}>
          {annotation.lanes.map((lane) => (
            <div
              key={lane.id}
              ref={(el) => {
                itemRefs.current[`lane:${lane.id}`] = el;
              }}
              style={{
                ...styles.elementRow,
                background:
                  selected?.kind === "lane" && selected.id === lane.id
                    ? colors.surface.selected
                    : colors.util.transparent,
              }}
              onClick={() => onSelect({ kind: "lane", id: lane.id })}
            >
              <span
                style={{
                  ...styles.statusDot,
                  background: statusColors.element[lane.status],
                }}
              />
              <span style={styles.elementId}>{lane.id}</span>
              <span style={styles.elementType}>{lane.lane_type}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={styles.divider} />

      {/* Vehicles list */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>VEHICLES ({annotation.vehicles.length})</div>
        <div style={styles.elementList}>
          {annotation.vehicles.map((veh) => (
            <div
              key={veh.id}
              ref={(el) => {
                itemRefs.current[`vehicle:${veh.id}`] = el;
              }}
              style={{
                ...styles.elementRow,
                background:
                  selected?.kind === "vehicle" && selected.id === veh.id
                    ? colors.surface.selected
                    : colors.util.transparent,
              }}
              onClick={() => onSelect({ kind: "vehicle", id: veh.id })}
            >
              <span
                style={{
                  ...styles.statusDot,
                  background: statusColors.element[veh.status],
                }}
              />
              <span style={styles.elementId}>{veh.id}</span>
              <span style={styles.elementType}>{veh.vehicle_type}</span>
            </div>
          ))}
        </div>
      </div>

      {annotation.lane_connections.length > 0 && (
        <>
          <div style={styles.divider} />
          <div style={styles.section}>
            <div style={styles.sectionTitle}>
              CONNECTIONS ({annotation.lane_connections.length})
            </div>
            <div style={styles.elementList}>
              {annotation.lane_connections.map((conn) => (
                <div
                  key={conn.id}
                  ref={(el) => {
                    itemRefs.current[`connection:${conn.id}`] = el;
                  }}
                  style={{
                    ...styles.elementRow,
                    background:
                      selected?.kind === "connection" && selected.id === conn.id
                        ? colors.surface.selected
                        : colors.util.transparent,
                  }}
                  onClick={() => onSelect({ kind: "connection", id: conn.id })}
                >
                  <span
                    style={{
                      ...styles.statusDot,
                      background: statusColors.element[conn.status],
                    }}
                  />
                  <span style={styles.elementId}>
                    {conn.from_lane_id} → {conn.to_lane_id}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 220,
    minWidth: 220,
    background: colors.surface.panel,
    borderRight: `1px solid ${colors.border.default}`,
    display: "flex",
    flexDirection: "column",
    overflowY: "auto",
    fontSize: 12,
  },
  section: {
    padding: "10px 12px",
  },
  sectionTitle: {
    fontSize: 10,
    fontFamily: "JetBrains Mono, monospace",
    color: colors.accent.info,
    letterSpacing: "0.1em",
    marginBottom: 8,
    textTransform: "uppercase" as const,
  },
  layerRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 5,
    cursor: "pointer",
    userSelect: "none" as const,
  },
  checkbox: {
    accentColor: colors.accent.info,
    cursor: "pointer",
  },
  divider: {
    height: 1,
    background: colors.border.default,
  },
  progressBar: {
    height: 4,
    background: colors.surface.elevated,
    borderRadius: 2,
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    background: colors.accent.info,
    borderRadius: 2,
    transition: "width 0.3s",
  },
  elementList: {
    maxHeight: 200,
    overflowY: "auto" as const,
  },
  elementRow: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "3px 4px",
    borderRadius: 3,
    cursor: "pointer",
    userSelect: "none" as const,
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    flexShrink: 0,
  },
  elementId: {
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 11,
    color: colors.text.muted,
    flexShrink: 0,
  },
  elementType: {
    fontSize: 10,
    color: colors.text.disabled,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap" as const,
  },
};
