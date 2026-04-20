import type { MouseMode } from "../lib/types";
import { colors } from "../lib/theme";

interface ToolbarProps {
  caseId: string;
  filename: string;
  mode: MouseMode;
  onModeChange: (m: MouseMode) => void;
  workflowStatus: string;
  onMarkReviewed: () => void;
}

const MODES: { key: MouseMode; label: string; shortcut: string }[] = [
  { key: "select", label: "SELECT", shortcut: "S" },
  { key: "connect", label: "CONNECT", shortcut: "C" },
  { key: "edit", label: "EDIT", shortcut: "E" },
];

export default function Toolbar({
  caseId,
  filename,
  mode,
  onModeChange,
  workflowStatus,
  onMarkReviewed,
}: ToolbarProps) {
  return (
    <div style={styles.bar}>
      <div style={styles.caseInfo}>
        <span style={styles.caseId}>{caseId}</span>
        <span style={styles.filename}>{filename}</span>
      </div>

      <div style={styles.modes}>
        {MODES.map((m) => (
          <button
            key={m.key}
            onClick={() => onModeChange(m.key)}
            style={{
              ...styles.modeBtn,
              background: mode === m.key ? colors.accent.info : colors.util.transparent,
              color: mode === m.key ? colors.text.onDark : colors.text.muted,
              borderColor: mode === m.key ? colors.accent.info : colors.border.strong,
            }}
          >
            {m.label}
            <span style={styles.shortcut}>[{m.shortcut}]</span>
          </button>
        ))}
      </div>

      <div style={styles.right}>
        <span style={styles.hint}>Enter=confirm · ⌫=reject · N=next · Z=undo</span>
        <button
          onClick={onMarkReviewed}
          disabled={workflowStatus === "reviewed"}
          style={{
            ...styles.reviewBtn,
            opacity: workflowStatus === "reviewed" ? 0.4 : 1,
            cursor: workflowStatus === "reviewed" ? "default" : "pointer",
          }}
        >
          {workflowStatus === "reviewed" ? "✓ REVIEWED" : "MARK REVIEWED"}
        </button>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  bar: {
    height: 44,
    background: colors.surface.panel,
    borderBottom: `1px solid ${colors.border.default}`,
    display: "flex",
    alignItems: "center",
    padding: "0 14px",
    gap: 16,
    flexShrink: 0,
  },
  caseInfo: {
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  caseId: {
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 12,
    color: colors.accent.info,
    fontWeight: "bold",
  },
  filename: {
    fontSize: 11,
    color: colors.text.disabled,
    fontFamily: "JetBrains Mono, monospace",
  },
  modes: {
    display: "flex",
    gap: 4,
  },
  modeBtn: {
    padding: "4px 10px",
    border: "1px solid",
    borderRadius: 3,
    cursor: "pointer",
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 10,
    fontWeight: "bold",
    letterSpacing: "0.05em",
    display: "flex",
    alignItems: "center",
    gap: 4,
    transition: "all 0.1s",
  },
  shortcut: {
    fontSize: 9,
    opacity: 0.6,
  },
  right: {
    marginLeft: "auto",
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  hint: {
    fontSize: 10,
    color: colors.text.hint,
    fontFamily: "JetBrains Mono, monospace",
  },
  reviewBtn: {
    padding: "5px 12px",
    background: colors.util.transparent,
    border: `1px solid ${colors.accent.success}`,
    color: colors.accent.success,
    borderRadius: 3,
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 10,
    fontWeight: "bold",
    letterSpacing: "0.05em",
  },
};
