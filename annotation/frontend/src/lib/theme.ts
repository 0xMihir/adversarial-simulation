export const colors = {
  surface: {
    page: "#0a0a0a",
    panel: "#0d1117",
    elevated: "#1e293b",
    selected: "#1e293b",
    error: "#7f1d1d",
  },
  border: {
    subtle: "#0f172a",
    default: "#1e293b",
    strong: "#334155",
  },
  text: {
    primary: "#e6f1ff",
    muted: "#94a3b8",
    disabled: "#4a5568",
    hint: "#334155",
    error: "#fca5a5",
    onDark: "#0a0a0a",
    white: "#ffffff",
    black: "#000000",
  },
  accent: {
    info: "#00e5ff",
    selected: "#f59e0b",
    connection: "#00b4d8",
    success: "#00e5a0",
    danger: "#ff4444",
  },
  status: {
    element: {
      auto: "#f59e0b",
      confirmed: "#00e5a0",
      corrected: "#00b4d8",
      rejected: "#ff4444",
    },
    workflow: {
      not_started: "#4a5568",
      in_progress: "#d97706",
      reviewed: "#059669",
    },
  },
  canvas: {
    grid: "#1a2a3a",
    roadway: "#334155",
    roadMarking: "#e2e8f0",
    trajectoryCollision: "#ef4444",
    endpoint: "#ff6b35",
    endpointHover: "#ffff00",
    labelBackground: "#0a0a0a",
  },
  util: {
    transparent: "transparent",
  },
} as const;

export const statusColors = {
  element: colors.status.element,
  workflow: colors.status.workflow,
} as const;

export function withOpacity(color: string, alphaHex: string): string {
  return `${color}${alphaHex}`;
}
