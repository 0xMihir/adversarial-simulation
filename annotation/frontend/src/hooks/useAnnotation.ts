import { useCallback, useState } from "react";
import { api } from "../lib/api";
import type {
  CaseAnnotation,
  CaseAnnotationUpdate,
  CorrectionRecord,
  ElementStatus,
  LaneAnnotation,
  LaneConnection,
  SelectedElement,
  VehicleAnnotation,
} from "../lib/types";

function toPreviousValue(value: unknown): Record<string, unknown> | null {
  if (value === null || value === undefined) return null;
  return JSON.parse(JSON.stringify(value)) as Record<string, unknown>;
}

interface UseAnnotationResult {
  annotation: CaseAnnotation | null;
  selected: SelectedElement;
  setSelected: (sel: SelectedElement) => void;
  confirmElement: (kind: string, id: string) => Promise<CaseAnnotation | null>;
  rejectElement: (kind: string, id: string) => Promise<void>;
  updateLane: (lane: LaneAnnotation) => Promise<void>;
  updateVehicle: (vehicle: VehicleAnnotation) => Promise<void>;
  addConnection: (conn: LaneConnection) => Promise<void>;
  deleteConnection: (id: string) => Promise<void>;
  hideRoadEdge: (id: string) => Promise<void>;
  undo: () => Promise<void>;
  canUndo: boolean;
  selectNext: (ann?: CaseAnnotation) => void;
  setAnnotation: (ann: CaseAnnotation) => void;
}

export function useAnnotation(
  initial: CaseAnnotation | null
): UseAnnotationResult {
  const [annotation, setAnnotation] = useState<CaseAnnotation | null>(initial);
  const [selected, setSelected] = useState<SelectedElement>(null);
  const [prevState, setPrevState] = useState<CaseAnnotation | null>(null);

  const save = useCallback(
    async (
      update: Omit<CaseAnnotationUpdate, "corrections">,
      correction: CorrectionRecord
    ): Promise<CaseAnnotation | null> => {
      if (!annotation) return null;
      const updated = await api.annotations.update(annotation.case_id, {
        ...update,
        corrections: [correction],
      });
      setAnnotation(updated);
      return updated;
    },
    [annotation]
  );

  const confirmElement = useCallback(
    async (kind: string, id: string): Promise<CaseAnnotation | null> => {
      if (!annotation) return null;
      const now = new Date().toISOString();

      if (kind === "lane") {
        const lanes = annotation.lanes.map((l) =>
          l.id === id ? { ...l, status: "confirmed" as ElementStatus } : l
        );
        return save(
          { lanes },
          { timestamp: now, element_type: "lane", element_id: id, action: "confirmed", previous_value: null }
        );
      } else if (kind === "vehicle") {
        const vehicles = annotation.vehicles.map((v) =>
          v.id === id ? { ...v, status: "confirmed" as ElementStatus } : v
        );
        return save(
          { vehicles },
          { timestamp: now, element_type: "vehicle", element_id: id, action: "confirmed", previous_value: null }
        );
      } else if (kind === "connection") {
        const lane_connections = annotation.lane_connections.map((c) =>
          c.id === id ? { ...c, status: "confirmed" as ElementStatus } : c
        );
        return save(
          { lane_connections },
          { timestamp: now, element_type: "connection", element_id: id, action: "confirmed", previous_value: null }
        );
      }
      return null;
    },
    [annotation, save]
  );

  const rejectElement = useCallback(
    async (kind: string, id: string) => {
      if (!annotation) return;
      const now = new Date().toISOString();

      if (kind === "lane") {
        const prev = annotation.lanes.find((l) => l.id === id);
        const lanes = annotation.lanes.map((l) =>
          l.id === id ? { ...l, status: "rejected" as ElementStatus } : l
        );
        await save(
          { lanes },
          {
            timestamp: now,
            element_type: "lane",
            element_id: id,
            action: "rejected",
            previous_value: toPreviousValue(prev),
          }
        );
      } else if (kind === "vehicle") {
        const prev = annotation.vehicles.find((v) => v.id === id);
        const vehicles = annotation.vehicles.map((v) =>
          v.id === id ? { ...v, status: "rejected" as ElementStatus } : v
        );
        await save(
          { vehicles },
          {
            timestamp: now,
            element_type: "vehicle",
            element_id: id,
            action: "rejected",
            previous_value: toPreviousValue(prev),
          }
        );
      }
    },
    [annotation, save]
  );

  const updateLane = useCallback(
    async (lane: LaneAnnotation) => {
      if (!annotation) return;
      const now = new Date().toISOString();
      const prev = annotation.lanes.find((l) => l.id === lane.id);
      const lanes = annotation.lanes.map((l) =>
        l.id === lane.id ? { ...lane, status: "corrected" as ElementStatus } : l
      );
      await save(
        { lanes },
        {
          timestamp: now,
          element_type: "lane",
          element_id: lane.id,
          action: "corrected",
          previous_value: toPreviousValue(prev),
        }
      );
    },
    [annotation, save]
  );

  const updateVehicle = useCallback(
    async (vehicle: VehicleAnnotation) => {
      if (!annotation) return;
      const now = new Date().toISOString();
      const prev = annotation.vehicles.find((v) => v.id === vehicle.id);
      const vehicles = annotation.vehicles.map((v) =>
        v.id === vehicle.id
          ? { ...vehicle, status: "corrected" as ElementStatus }
          : v
      );
      await save(
        { vehicles },
        {
          timestamp: now,
          element_type: "vehicle",
          element_id: vehicle.id,
          action: "corrected",
          previous_value: toPreviousValue(prev),
        }
      );
    },
    [annotation, save]
  );

  const addConnection = useCallback(
    async (conn: LaneConnection) => {
      if (!annotation) return;
      const now = new Date().toISOString();
      setPrevState(annotation);
      const connections = [...annotation.lane_connections, conn];
      await save(
        { lane_connections: connections },
        { timestamp: now, element_type: "connection", element_id: conn.id, action: "created", previous_value: null }
      );
    },
    [annotation, save]
  );

  const deleteConnection = useCallback(
    async (id: string) => {
      if (!annotation) return;
      const now = new Date().toISOString();
      setPrevState(annotation);
      const lane_connections = annotation.lane_connections.filter((c) => c.id !== id);
      await save(
        { lane_connections },
        { timestamp: now, element_type: "connection", element_id: id, action: "deleted", previous_value: null }
      );
    },
    [annotation, save]
  );

  const hideRoadEdge = useCallback(
    async (id: string) => {
      if (!annotation) return;
      const now = new Date().toISOString();
      const hidden_roadway_ids = [...(annotation.hidden_roadway_ids ?? []), id];
      await save(
        { hidden_roadway_ids },
        { timestamp: now, element_type: "roadway", element_id: id, action: "hidden", previous_value: null }
      );
    },
    [annotation, save]
  );

  const undo = useCallback(async () => {
    if (!annotation || !prevState) return;
    const updated = await api.annotations.update(annotation.case_id, {
      lane_connections: prevState.lane_connections,
    });
    setAnnotation(updated);
    setPrevState(null);
  }, [annotation, prevState]);

  const selectNext = useCallback((ann?: CaseAnnotation) => {
    const a = ann ?? annotation;
    if (!a) return;
    const unreviewedLane = a.lanes.find((l) => l.status === "auto");
    if (unreviewedLane) {
      setSelected({ kind: "lane", id: unreviewedLane.id });
      return;
    }
    const unreviewedVehicle = a.vehicles.find((v) => v.status === "auto");
    if (unreviewedVehicle) {
      setSelected({ kind: "vehicle", id: unreviewedVehicle.id });
    }
  }, [annotation]);

  return {
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
    canUndo: prevState !== null,
    selectNext,
    setAnnotation,
  };
}
