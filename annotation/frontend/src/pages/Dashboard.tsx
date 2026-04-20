import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { colors, statusColors, withOpacity } from "../lib/theme";
import type { CaseSummary } from "../lib/types";

function workflowColor(status: string): string {
  if (status in statusColors.workflow) {
    return statusColors.workflow[status as keyof typeof statusColors.workflow];
  }
  return statusColors.workflow.not_started;
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadCases = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.cases.list();
      setCases(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load cases");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCases();
  }, [loadCases]);

  const handleProcess = useCallback(
    async (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      setProcessing(id);
      try {
        await api.cases.process(id);
        await loadCases();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Processing failed");
      } finally {
        setProcessing(null);
      }
    },
    [loadCases]
  );

  const handleProcessAll = useCallback(async () => {
    const unprocessed = cases.filter((c) => !c.annotated);
    for (const c of unprocessed) {
      setProcessing(c.id);
      try {
        await api.cases.process(c.id);
      } catch {
        // continue with remaining
      }
    }
    setProcessing(null);
    await loadCases();
  }, [cases, loadCases]);

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <div style={styles.title}>CISS ANNOTATION TOOL</div>
          <div style={styles.subtitle}>NHTSA Crash Injury Surveillance System</div>
        </div>
        <div style={styles.headerActions}>
          <button onClick={loadCases} style={styles.secondaryBtn}>
            REFRESH
          </button>
          <button onClick={handleProcessAll} style={styles.primaryBtn} disabled={!!processing}>
            PROCESS ALL UNPROCESSED
          </button>
          <a
            href="/api/export/jsonl?status=reviewed"
            download="annotations.jsonl"
            style={styles.exportBtn}
          >
            EXPORT JSONL
          </a>
        </div>
      </div>

      {error && (
        <div style={styles.errorBanner}>{error}</div>
      )}

      {loading ? (
        <div style={styles.loading}>Loading cases…</div>
      ) : (
        <div style={styles.tableWrapper}>
          <table style={styles.table}>
            <thead>
              <tr>
                {["CASE ID", "FILENAME", "STATUS", "CONFIDENCE", "LANES", "VEHICLES", "UPDATED", ""].map((h) => (
                  <th key={h} style={styles.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cases.length === 0 ? (
                <tr>
                  <td colSpan={8} style={{ ...styles.td, textAlign: "center", color: colors.text.disabled, padding: 24 }}>
                    No .far files found. Check data/nhtsa-ciss/data/output/
                  </td>
                </tr>
              ) : (
                cases.map((c) => (
                  <tr
                    key={c.id}
                    style={styles.row}
                    onClick={() => navigate(`/annotate/${c.id}`)}
                  >
                    <td style={{ ...styles.td, ...styles.mono, color: colors.accent.info }}>
                      {c.id}
                    </td>
                    <td style={{ ...styles.td, ...styles.mono, color: colors.text.muted }}>
                      {c.filename}
                    </td>
                    <td style={styles.td}>
                      {(() => {
                        const color = workflowColor(c.workflow_status);
                        return (
                      <span
                        style={{
                          ...styles.badge,
                          background: withOpacity(color, "22"),
                          color,
                          border: `1px solid ${withOpacity(color, "44")}`,
                        }}
                      >
                        {c.workflow_status.replace("_", " ").toUpperCase()}
                      </span>
                        );
                      })()}
                    </td>
                    <td style={{ ...styles.td, ...styles.mono }}>
                      {c.auto_confidence !== null
                        ? `${((c.auto_confidence ?? 0) * 100).toFixed(0)}%`
                        : "—"}
                    </td>
                    <td style={{ ...styles.td, ...styles.mono }}>{c.lane_count || "—"}</td>
                    <td style={{ ...styles.td, ...styles.mono }}>{c.vehicle_count || "—"}</td>
                    <td style={{ ...styles.td, color: colors.text.disabled, fontSize: 10 }}>
                      {c.updated_at
                        ? new Date(c.updated_at).toLocaleDateString()
                        : "—"}
                    </td>
                    <td style={styles.td} onClick={(e) => e.stopPropagation()}>
                      <button
                        style={styles.processBtn}
                        onClick={(e) => handleProcess(c.id, e)}
                        disabled={processing === c.id}
                      >
                        {processing === c.id ? "…" : c.annotated ? "REPROCESS" : "PROCESS"}
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
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
    fontFamily: "Inter, system-ui, sans-serif",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "20px 28px",
    borderBottom: `1px solid ${colors.border.default}`,
    background: colors.surface.panel,
    flexShrink: 0,
  },
  title: {
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 16,
    fontWeight: "bold",
    color: colors.accent.info,
    letterSpacing: "0.08em",
  },
  subtitle: {
    fontSize: 11,
    color: colors.text.disabled,
    marginTop: 3,
  },
  headerActions: {
    display: "flex",
    gap: 10,
    alignItems: "center",
  },
  primaryBtn: {
    padding: "8px 16px",
    background: colors.accent.info,
    color: colors.text.onDark,
    border: "none",
    borderRadius: 3,
    cursor: "pointer",
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 11,
    fontWeight: "bold",
  },
  secondaryBtn: {
    padding: "8px 14px",
    background: colors.util.transparent,
    color: colors.text.muted,
    border: `1px solid ${colors.border.strong}`,
    borderRadius: 3,
    cursor: "pointer",
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 11,
  },
  exportBtn: {
    padding: "8px 14px",
    background: colors.util.transparent,
    color: colors.accent.success,
    border: `1px solid ${colors.accent.success}`,
    borderRadius: 3,
    cursor: "pointer",
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 11,
    textDecoration: "none",
    display: "inline-block",
  },
  errorBanner: {
    background: colors.surface.error,
    color: colors.text.error,
    padding: "8px 28px",
    fontSize: 12,
    fontFamily: "JetBrains Mono, monospace",
  },
  loading: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
    color: colors.text.disabled,
    fontFamily: "JetBrains Mono, monospace",
  },
  tableWrapper: {
    flex: 1,
    overflowY: "auto",
    padding: "0 28px 28px",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    marginTop: 16,
  },
  th: {
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 10,
    color: colors.text.disabled,
    letterSpacing: "0.1em",
    textAlign: "left" as const,
    padding: "8px 12px",
    borderBottom: `1px solid ${colors.border.default}`,
    textTransform: "uppercase" as const,
    position: "sticky" as const,
    top: 0,
    background: colors.surface.page,
  },
  row: {
    cursor: "pointer",
    borderBottom: `1px solid ${colors.border.subtle}`,
    transition: "background 0.1s",
  },
  td: {
    padding: "10px 12px",
    fontSize: 12,
    color: colors.text.primary,
    verticalAlign: "middle",
  },
  mono: {
    fontFamily: "JetBrains Mono, monospace",
  },
  badge: {
    padding: "2px 7px",
    borderRadius: 3,
    fontSize: 10,
    fontFamily: "JetBrains Mono, monospace",
    whiteSpace: "nowrap" as const,
  },
  processBtn: {
    padding: "4px 10px",
    background: colors.util.transparent,
    border: `1px solid ${colors.border.strong}`,
    color: colors.text.muted,
    borderRadius: 3,
    cursor: "pointer",
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 10,
    fontWeight: "bold",
  },
};
