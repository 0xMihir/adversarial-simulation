import type {
  CaseAnnotation,
  LaneAnnotation,
  SelectedElement,
  VehicleAnnotation,
} from "../lib/types";
import { colors, statusColors } from "../lib/theme";

const LANE_TYPES = ["driving", "bike", "bus", "shoulder", "parking"];
const BOUNDARY_TYPES = [
  "unknown",
  "road_edge",
  "solid_white",
  "broken_white",
  "solid_yellow",
  "broken_yellow",
  "double_yellow",
  "double_white",
  "virtual",
];
const VEHICLE_TYPES = ["car", "truck", "motorcycle", "pedestrian", "bicycle", "animal"];
const PHASE_TYPES = ["pre_crash", "collision", "post_crash", "final_rest"];

interface PropertiesPanelProps {
  annotation: CaseAnnotation;
  selected: SelectedElement;
  onConfirm: (kind: string, id: string) => void;
  onReject: (kind: string, id: string) => void;
  onUpdateLane: (lane: LaneAnnotation) => void;
  onUpdateVehicle: (vehicle: VehicleAnnotation) => void;
  onDeleteConnection: (id: string) => void;
}

export default function PropertiesPanel({
  annotation,
  selected,
  onConfirm,
  onReject,
  onUpdateLane,
  onUpdateVehicle,
  onDeleteConnection,
}: PropertiesPanelProps) {
  if (!selected) {
    const reviewed =
      annotation.lanes.filter((l) => l.status !== "auto").length +
      annotation.vehicles.filter((v) => v.status !== "auto").length;
    const total = annotation.lanes.length + annotation.vehicles.length;
    const caseId = annotation.case_id.split("_")[0];

    return (
      <div style={styles.panel}>
        <div style={styles.header}>PROPERTIES</div>
        <div style={styles.body}>
          <div style={styles.field}>
            <span style={styles.label}>CASE ID</span>
            <a
              href={`https://crashviewer.nhtsa.dot.gov/ciss/details/${caseId}/crash-summary-document`}
              target="_blank"
              rel="noreferrer"
              style={styles.link}
            >
              {caseId}
            </a>
          </div>
          <div style={styles.field}>
            <span style={styles.label}>FILE</span>
            <span style={styles.value}>{annotation.far_filename}</span>
          </div>
          <div style={styles.field}>
            <span style={styles.label}>STATUS</span>
            <span style={{ ...styles.badge, background: statusColor(annotation.workflow_status) }}>
              {annotation.workflow_status.replace("_", " ").toUpperCase()}
            </span>
          </div>
          <div style={styles.field}>
            <span style={styles.label}>REVIEWED</span>
            <span style={styles.value}>{reviewed} / {total}</span>
          </div>
          {annotation.auto_confidence !== null && (
            <div style={styles.field}>
              <span style={styles.label}>CONFIDENCE</span>
              <span style={styles.value}>
                {((annotation.auto_confidence ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
          )}
          <div style={styles.hint}>
            Click an element to select it.
          </div>
        </div>
      </div>
    );
  }

  if (selected.kind === "lane") {
    const lane = annotation.lanes.find((l) => l.id === selected.id);
    if (!lane) return null;

    return (
      <div style={styles.panel}>
        <div style={styles.header}>LANE</div>
        <div style={styles.body}>
          <div style={styles.field}>
            <span style={styles.label}>ID</span>
            <span style={styles.value}>{lane.id}</span>
          </div>
          <div style={styles.field}>
            <span style={styles.label}>STATUS</span>
            <span style={{ ...styles.badge, background: statusBg(lane.status) }}>
              {lane.status.toUpperCase()}
            </span>
          </div>
          <div style={styles.field}>
            <span style={styles.label}>POINTS</span>
            <span style={styles.value}>{lane.polyline.length}</span>
          </div>

          <div style={styles.divider} />

          <div style={styles.fieldCol}>
            <span style={styles.label}>LANE TYPE</span>
            <select
              value={lane.lane_type}
              onChange={(e) => onUpdateLane({ ...lane, lane_type: e.target.value })}
              style={styles.select}
            >
              {LANE_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div style={styles.fieldCol}>
            <span style={styles.label}>LEFT BOUNDARY</span>
            <select
              value={lane.left_boundary_type}
              onChange={(e) => onUpdateLane({ ...lane, left_boundary_type: e.target.value })}
              style={styles.select}
            >
              {BOUNDARY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div style={styles.fieldCol}>
            <span style={styles.label}>RIGHT BOUNDARY</span>
            <select
              value={lane.right_boundary_type}
              onChange={(e) => onUpdateLane({ ...lane, right_boundary_type: e.target.value })}
              style={styles.select}
            >
              {BOUNDARY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div style={styles.fieldCol}>
            <span style={styles.label}>SPEED LIMIT (MPH)</span>
            <input
              type="number"
              value={lane.speed_limit_mph ?? ""}
              placeholder="—"
              onChange={(e) =>
                onUpdateLane({
                  ...lane,
                  speed_limit_mph: e.target.value ? Number(e.target.value) : null,
                })
              }
              style={styles.input}
            />
          </div>
          <div style={styles.fieldCol}>
            <span style={styles.label}>NOTES</span>
            <input
              type="text"
              value={lane.notes ?? ""}
              onChange={(e) => onUpdateLane({ ...lane, notes: e.target.value || null })}
              style={styles.input}
              placeholder="optional"
            />
          </div>

          <div style={styles.divider} />
          <div style={styles.actions}>
            <button
              onClick={() => onConfirm("lane", lane.id)}
              style={{ ...styles.btn, background: colors.accent.success, color: colors.text.onDark }}
            >
              CONFIRM [Enter]
            </button>
            <button
              onClick={() => onReject("lane", lane.id)}
              style={{ ...styles.btn, background: colors.accent.danger, color: colors.text.white }}
            >
              REJECT [⌫]
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (selected.kind === "vehicle") {
    const veh = annotation.vehicles.find((v) => v.id === selected.id);
    if (!veh) return null;

    return (
      <div style={styles.panel}>
        <div style={styles.header}>VEHICLE</div>
        <div style={styles.body}>
          <div style={styles.field}>
            <span style={styles.label}>ID</span>
            <span style={styles.value}>{veh.id}</span>
          </div>
          <div style={styles.field}>
            <span style={styles.label}>STATUS</span>
            <span style={{ ...styles.badge, background: statusBg(veh.status) }}>
              {veh.status.toUpperCase()}
            </span>
          </div>
          <div style={styles.fieldCol}>
            <span style={styles.label}>TYPE</span>
            <select
              value={veh.vehicle_type}
              onChange={(e) => onUpdateVehicle({ ...veh, vehicle_type: e.target.value })}
              style={styles.select}
            >
              {VEHICLE_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          <div style={styles.divider} />
          <div style={styles.sectionTitle}>WAYPOINTS ({veh.waypoints.length})</div>
          <div style={styles.waypointTable}>
            <div style={styles.waypointHeader}>
              <span style={{ width: 24 }}>#</span>
              <span style={{ flex: 1 }}>X</span>
              <span style={{ flex: 1 }}>Y</span>
              <span style={{ flex: 1 }}>PHASE</span>
            </div>
            {veh.waypoints.slice(0, 20).map((wp, i) => (
              <div key={i} style={styles.waypointRow}>
                <span style={{ width: 24, color: colors.text.disabled }}>{wp.timestamp_index}</span>
                <span style={{ flex: 1 }}>{wp.position.x.toFixed(1)}</span>
                <span style={{ flex: 1 }}>{wp.position.y.toFixed(1)}</span>
                <select
                  value={wp.phase}
                  onChange={(e) => {
                    const newWps = veh.waypoints.map((w, wi) =>
                      wi === i ? { ...w, phase: e.target.value } : w
                    );
                    onUpdateVehicle({ ...veh, waypoints: newWps });
                  }}
                  style={{ ...styles.select, flex: 1, padding: "1px 2px" }}
                >
                  {PHASE_TYPES.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
            ))}
            {veh.waypoints.length > 20 && (
              <div style={{ color: colors.text.disabled, fontSize: 10, padding: "4px 0" }}>
                +{veh.waypoints.length - 20} more…
              </div>
            )}
          </div>

          <div style={styles.divider} />
          <div style={styles.actions}>
            <button
              onClick={() => onConfirm("vehicle", veh.id)}
              style={{ ...styles.btn, background: colors.accent.success, color: colors.text.onDark }}
            >
              CONFIRM [Enter]
            </button>
            <button
              onClick={() => onReject("vehicle", veh.id)}
              style={{ ...styles.btn, background: colors.accent.danger, color: colors.text.white }}
            >
              REJECT [⌫]
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (selected.kind === "roadway") {
    const elem = annotation.scene.elements.find((e) => e.id === selected.id);
    return (
      <div style={styles.panel}>
        <div style={styles.header}>ROAD EDGE</div>
        <div style={styles.body}>
          <div style={styles.field}>
            <span style={styles.label}>ID</span>
            <span style={styles.value}>{selected.id}</span>
          </div>
          {elem && (
            <div style={styles.field}>
              <span style={styles.label}>POINTS</span>
              <span style={styles.value}>{elem.resampled_points.length}</span>
            </div>
          )}
          <div style={styles.divider} />
          <div style={styles.actions}>
            <button
              onClick={() => onReject("roadway", selected.id)}
              style={{ ...styles.btn, background: colors.accent.danger, color: colors.text.white }}
            >
              REMOVE [⌫]
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (selected.kind === "connection") {
    const conn = annotation.lane_connections.find((c) => c.id === selected.id);
    if (!conn) return null;
    return (
      <div style={styles.panel}>
        <div style={styles.header}>CONNECTION</div>
        <div style={styles.body}>
          <div style={styles.field}>
            <span style={styles.label}>FROM</span>
            <span style={styles.value}>{conn.from_lane_id}</span>
          </div>
          <div style={styles.field}>
            <span style={styles.label}>TO</span>
            <span style={styles.value}>{conn.to_lane_id}</span>
          </div>
          <div style={styles.field}>
            <span style={styles.label}>TYPE</span>
            <span style={styles.value}>{conn.connection_type}</span>
          </div>
          <div style={styles.field}>
            <span style={styles.label}>STATUS</span>
            <span style={{ ...styles.badge, background: statusBg(conn.status) }}>
              {conn.status.toUpperCase()}
            </span>
          </div>
          <div style={styles.divider} />
          <div style={styles.actions}>
            <button
              onClick={() => onDeleteConnection(conn.id)}
              style={{ ...styles.btn, background: colors.accent.danger, color: colors.text.white }}
            >
              DELETE [⌫]
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
}

function statusBg(status: string): string {
  if (status in statusColors.element) {
    return statusColors.element[status as keyof typeof statusColors.element];
  }
  return statusColors.element.auto;
}

function statusColor(ws: string): string {
  if (ws in statusColors.workflow) {
    return statusColors.workflow[ws as keyof typeof statusColors.workflow];
  }
  return statusColors.workflow.not_started;
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 240,
    minWidth: 240,
    background: colors.surface.panel,
    borderLeft: `1px solid ${colors.border.default}`,
    display: "flex",
    flexDirection: "column",
    overflowY: "auto",
    fontSize: 12,
  },
  header: {
    padding: "10px 14px",
    fontSize: 10,
    fontFamily: "JetBrains Mono, monospace",
    color: colors.accent.info,
    letterSpacing: "0.1em",
    borderBottom: `1px solid ${colors.border.default}`,
    textTransform: "uppercase" as const,
  },
  body: {
    padding: "10px 14px",
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  field: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 8,
  },
  fieldCol: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 3,
  },
  label: {
    fontSize: 10,
    color: colors.text.disabled,
    fontFamily: "JetBrains Mono, monospace",
    textTransform: "uppercase" as const,
    letterSpacing: "0.08em",
    flexShrink: 0,
  },
  value: {
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 11,
    color: colors.text.primary,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap" as const,
    textAlign: "right" as const,
  },
  link: {
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 11,
    color: colors.accent.info,
    textDecoration: "underline",
    textAlign: "right" as const,
  },
  badge: {
    padding: "2px 6px",
    borderRadius: 3,
    fontSize: 10,
    fontFamily: "JetBrains Mono, monospace",
    color: colors.text.primary,
  },
  sectionTitle: {
    fontSize: 10,
    fontFamily: "JetBrains Mono, monospace",
    color: colors.accent.info,
    letterSpacing: "0.1em",
    textTransform: "uppercase" as const,
    marginBottom: 4,
  },
  divider: { height: 1, background: colors.border.default, margin: "4px 0" },
  actions: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 6,
    marginTop: 4,
  },
  btn: {
    padding: "6px 10px",
    border: "none",
    borderRadius: 3,
    cursor: "pointer",
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 11,
    fontWeight: "bold",
    letterSpacing: "0.05em",
  },
  select: {
    background: colors.surface.elevated,
    color: colors.text.primary,
    border: `1px solid ${colors.border.strong}`,
    borderRadius: 3,
    padding: "3px 6px",
    fontSize: 11,
    fontFamily: "JetBrains Mono, monospace",
    width: "100%",
  },
  input: {
    background: colors.surface.elevated,
    color: colors.text.primary,
    border: `1px solid ${colors.border.strong}`,
    borderRadius: 3,
    padding: "3px 6px",
    fontSize: 11,
    fontFamily: "JetBrains Mono, monospace",
    width: "100%",
  },
  waypointTable: {
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 10,
    maxHeight: 180,
    overflowY: "auto" as const,
  },
  waypointHeader: {
    display: "flex",
    color: colors.text.disabled,
    padding: "2px 0",
    borderBottom: `1px solid ${colors.border.default}`,
    marginBottom: 2,
  },
  waypointRow: {
    display: "flex",
    alignItems: "center",
    color: colors.text.muted,
    padding: "1px 0",
    gap: 4,
  },
  hint: {
    fontSize: 11,
    color: colors.text.disabled,
    marginTop: 12,
  },
};
