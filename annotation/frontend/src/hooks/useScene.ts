import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CaseAnnotation } from "../lib/types";

interface UseSceneResult {
  annotation: CaseAnnotation | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
  process: () => Promise<void>;
}

export function useScene(caseId: string): UseSceneResult {
  const [annotation, setAnnotation] = useState<CaseAnnotation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const ann = await api.annotations.get(caseId);
      setAnnotation(ann);
    } catch (e) {
      // 404 = not yet processed
      setAnnotation(null);
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  useEffect(() => {
    load();
  }, [load]);

  const process = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const ann = await api.cases.process(caseId);
      setAnnotation(ann);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Processing failed");
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  return { annotation, loading, error, refresh: load, process };
}
