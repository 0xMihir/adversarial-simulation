import type {
  CaseAnnotation,
  CaseAnnotationUpdate,
  CaseSummary,
  ParsedScene,
} from "./types";

const BASE = "/api";

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  cases: {
    list: () => request<CaseSummary[]>("/cases"),
    process: (id: string) =>
      request<CaseAnnotation>(`/cases/${id}/process`, { method: "POST" }),
    scene: (id: string) => request<ParsedScene>(`/cases/${id}/scene`),
  },

  annotations: {
    get: (id: string) => request<CaseAnnotation>(`/annotations/${id}`),
    update: (id: string, update: CaseAnnotationUpdate) =>
      request<CaseAnnotation>(`/annotations/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(update),
      }),
  },

  export: {
    jsonlUrl: (params?: { status?: string; min_confidence?: number }) => {
      const qs = new URLSearchParams();
      if (params?.status) qs.set("status", params.status);
      if (params?.min_confidence !== undefined)
        qs.set("min_confidence", String(params.min_confidence));
      return `${BASE}/export/jsonl?${qs}`;
    },
    graph: (id: string) => request(`/export/graph/${id}`),
  },
};
