"""
SyntheticSceneGenerator — converts one AV2 or WOMD scenario into a
(ParsedScene, SyntheticGroundTruth) pair that is structurally identical to a
real FARO-parsed scene. Ground-truth labels are derived from the source dataset.

Determinism: seed_used = hash(f"{seed}:{scenario_id}:{stage}:{index}") % 2**31
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np

from schema.scene import AffineMatrix, ParsedScene, Point2D, SceneElement, TextElement
from synthetic.config import CurriculumConfig, CurriculumStage
from synthetic.loaders.base import LaneSegmentData, ScenarioLoader
from synthetic.normalization import DiagramNormalizer, _pts_to_np
from synthetic.randomization import (
    _apply_layout_to_segments,
    _make_element,
    _reset_id_counter,
    add_annotation_clutter,
    add_vehicle_markings,
    apply_faro_styling,
    drop_primitives,
    jitter_vertices,
    resample_vertices,
    subset_network,
    transform_layout,
    vary_crosswalk,
)
from synthetic.schema import (
    BoundaryAssignment,
    ElementClass,
    InverseTransform,
    LaneTopology,
    SyntheticGroundTruth,
    WOMDRoadLineType,
)

if TYPE_CHECKING:
    pass

_IDENTITY_AFFINE = AffineMatrix(values=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# ElementClasses that belong to roadway vs road_marking categories
_ROADWAY_CLASSES = {ElementClass.ROAD_EDGE, ElementClass.SHOULDER_LINE}
_MARKING_CLASSES = {
    ElementClass.LANE_MARKING_SOLID,
    ElementClass.LANE_MARKING_DASHED,
    ElementClass.STOP_LINE,
    ElementClass.CROSSWALK_STRIPE,
}


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _mark_type_to_class(womd_type: WOMDRoadLineType) -> ElementClass:
    """Map a WOMD road line type to an ElementClass for the synthesized boundary element."""
    dashed = {
        WOMDRoadLineType.TYPE_BROKEN_SINGLE_WHITE,
        WOMDRoadLineType.TYPE_BROKEN_SINGLE_YELLOW,
    }
    if womd_type in dashed:
        return ElementClass.LANE_MARKING_DASHED
    if womd_type == WOMDRoadLineType.TYPE_UNKNOWN:
        return ElementClass.ROAD_EDGE  # conservative: unknown → treat as road edge
    return ElementClass.LANE_MARKING_SOLID


def _is_shoulder(seg: LaneSegmentData, all_segments: list[LaneSegmentData]) -> bool:
    """Heuristic: a boundary is a shoulder line if the segment has no left or right neighbor."""
    return seg.left_neighbor_id is None or seg.right_neighbor_id is None


class SyntheticSceneGenerator:
    """
    Converts one scenario (AV2 or WOMD) into a (ParsedScene, SyntheticGroundTruth) pair.

    Args:
        loader: any ScenarioLoader (AV2ScenarioLoader or WOMDScenarioLoader)
        curriculum_cfg: stage A/B/C randomization configs
        apply_pca: if False, skip PCA rotation during normalization (ablation flag)
        seed: base seed; per-sample seed = hash(f"{seed}:{scenario_id}:{stage}:{index}") % 2**31
        source_dataset: label for provenance in SyntheticGroundTruth ("av2" | "womd")
    """

    def __init__(
        self,
        loader: ScenarioLoader,
        curriculum_cfg: CurriculumConfig | None = None,
        apply_pca: bool = True,
        seed: int = 42,
        source_dataset: str = "av2",
    ) -> None:
        self.loader = loader
        self.curriculum_cfg = curriculum_cfg or CurriculumConfig()
        self.apply_pca = apply_pca
        self.seed = seed
        self.source_dataset = source_dataset

    def generate(
        self,
        scenario_id: str,
        stage: CurriculumStage,
        index: int = 0,
    ) -> tuple[ParsedScene, SyntheticGroundTruth]:
        """
        Generate one (ParsedScene, SyntheticGroundTruth) pair.
        Deterministic: same (scenario_id, stage, index) always yields identical output.
        """
        _reset_id_counter()
        seed_used = abs(hash(f"{self.seed}:{scenario_id}:{stage.value}:{index}")) % (2 ** 31)
        rng = np.random.default_rng(seed_used)
        cfg = self.curriculum_cfg.for_stage(stage)

        segments = self.loader.load_scenario(scenario_id)
        if not segments:
            raise ValueError(f"Scenario {scenario_id!r} yielded no surface lane segments")

        # Module 7: pre-normalization layout transform
        M_layout = transform_layout(rng, cfg)
        segments = _apply_layout_to_segments(segments, M_layout)

        # Module 6: crop to a focal region
        all_cl = np.concatenate([s.centerline_xy for s in segments if s.centerline_xy.shape[0] > 0], axis=0)
        focal_point = all_cl.mean(axis=0)
        segments, crop_bbox = subset_network(segments, focal_point, rng, cfg)

        if not segments:
            raise ValueError(f"Scenario {scenario_id!r}: no segments survived subsetting")

        # Convert segments to (SceneElement, ElementClass) tuples — Option A (§1.4)
        (
            lane_markings,
            stop_lines,
            crosswalks,
            shoulders,
            road_edges,
            lane_id_to_elem_ids,
        ) = self._segments_to_elements(segments, rng)

        # Module 1: drop primitives
        lane_markings, stop_lines, crosswalks, shoulders, dropped_ids = drop_primitives(
            lane_markings, stop_lines, crosswalks, shoulders, rng, cfg,
        )

        # Modules 2 & 3: per-element vertex transforms
        def _apply_per_elem(items: list[tuple[SceneElement, ElementClass]]) -> list[tuple[SceneElement, ElementClass]]:
            out = []
            for elem, cls in items:
                elem = resample_vertices(elem, rng, cfg)
                elem = jitter_vertices(elem, rng, cfg)
                out.append((elem, cls))
            return out

        lane_markings = _apply_per_elem(lane_markings)
        stop_lines = _apply_per_elem(stop_lines)
        crosswalks = _apply_per_elem(crosswalks)
        shoulders = _apply_per_elem(shoulders)
        road_edges = _apply_per_elem(road_edges)

        # Module 4: crosswalk variation (replace existing crosswalks)
        if rng.random() < cfg.p_vary_crosswalk and segments:
            seg = segments[rng.integers(0, len(segments))]
            if seg.left_boundary_xy.shape[0] >= 2:
                crosswalks = vary_crosswalk(
                    seg.left_boundary_xy[0], seg.left_boundary_xy[-1],
                    road_width_m=float(rng.uniform(3.0, 12.0)),
                    rng=rng, cfg=cfg,
                )

        # Module 8: vehicle and impact markings
        vehicle_elems: list[tuple[SceneElement, ElementClass]] = []
        if rng.random() < cfg.p_add_vehicle_markings:
            vehicle_elems = add_vehicle_markings(segments, rng, cfg)

        # Gather all elements before clutter (for bbox computation)
        all_elems_pre: list[tuple[SceneElement, ElementClass]] = (
            road_edges + shoulders + lane_markings + stop_lines + crosswalks + vehicle_elems
        )

        # Module 9: FARO styling
        all_elems_pre = [(apply_faro_styling(e, c, rng, cfg), c) for e, c in all_elems_pre]

        # Compute bounding box for clutter placement (in current metric coords)
        all_pts = np.concatenate(
            [_pts_to_np(e.resampled_points) for e, _ in all_elems_pre if e.resampled_points],
            axis=0,
        ) if all_elems_pre else np.zeros((0, 2))
        scene_bbox: tuple[float, float, float, float] = (
            (float(all_pts[:, 0].min()), float(all_pts[:, 1].min()),
             float(all_pts[:, 0].max()), float(all_pts[:, 1].max()))
            if all_pts.shape[0] > 0 else (0.0, 0.0, 1.0, 1.0)
        )

        # Module 5: annotation clutter
        clutter_elems, clutter_texts = add_annotation_clutter(scene_bbox, rng, cfg)

        all_elems: list[tuple[SceneElement, ElementClass]] = all_elems_pre + clutter_elems
        all_texts: list[TextElement] = clutter_texts

        # Assign categorical index lists
        roadway_idx, marking_idx, other_idx = [], [], []
        for i, (_, cls) in enumerate(all_elems):
            if cls in _ROADWAY_CLASSES:
                roadway_idx.append(i)
            elif cls in _MARKING_CLASSES:
                marking_idx.append(i)
            else:
                other_idx.append(i)

        # Assemble ParsedScene (pre-normalization, AV2 metric coords)
        now_iso = "2000-01-01T00:00:00+00:00"  # fixed sentinel — synthetic data has no real timestamp
        scene_id = f"syn_{scenario_id}_{stage.value}_{index}"
        raw_scene = ParsedScene(
            case_id=scene_id,
            far_filename=f"synthetic_{scenario_id}.far",
            parsed_at=now_iso,
            coordinate_unit="m",
            scale_bar=None,
            elements=[e for e, _ in all_elems],
            texts=all_texts,
            images=[],
            vehicles=[],
            roadway_indices=roadway_idx,
            road_marking_indices=marking_idx,
            other_indices=other_idx,
        )

        # Normalize (§0.1)
        norm_scene, M_inv = DiagramNormalizer.normalize(raw_scene, apply_pca=self.apply_pca)

        # Build inverse transform model
        vertices_pre = DiagramNormalizer.collect_all_vertices(raw_scene)
        centroid, pca_angle, scale = DiagramNormalizer.compute_normalization_params(vertices_pre, self.apply_pca)
        inv_transform = InverseTransform(
            values=M_inv.tolist(),
            original_centroid_x=float(centroid[0]),
            original_centroid_y=float(centroid[1]),
            scale_factor=scale,
            pca_rotation_rad=pca_angle,
        )

        # Extract ground truth (apply same forward transform to AV2 centerlines)
        M_fwd = DiagramNormalizer.build_forward_transform(centroid, pca_angle, scale)
        gt = self._extract_ground_truth(
            segments=segments,
            all_elems=all_elems,
            dropped_ids=dropped_ids,
            lane_id_to_elem_ids=lane_id_to_elem_ids,
            M_fwd=M_fwd,
            scene_id=scene_id,
            scenario_id=scenario_id,
            stage=stage,
            inv_transform=inv_transform,
        )

        return norm_scene, gt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segments_to_elements(
        self,
        segments: list[LaneSegmentData],
        rng: np.random.Generator,
    ) -> tuple[
        list[tuple[SceneElement, ElementClass]],  # lane_markings
        list[tuple[SceneElement, ElementClass]],  # stop_lines
        list[tuple[SceneElement, ElementClass]],  # crosswalks (placeholder — vary_crosswalk generates real ones)
        list[tuple[SceneElement, ElementClass]],  # shoulders
        list[tuple[SceneElement, ElementClass]],  # road_edges
        dict[str, list[str]],                     # lane_id → [left_elem_id, right_elem_id]
    ]:
        """
        Convert AV2/WOMD LaneSegmentData to SceneElement lists.
        Option A (§1.4): one element per boundary polyline, same granularity as FARO operators.
        Boundaries are categorised as: ROAD_EDGE, LANE_MARKING_SOLID, LANE_MARKING_DASHED,
        or SHOULDER_LINE based on mark type and neighbor heuristic.
        """
        lane_markings: list[tuple[SceneElement, ElementClass]] = []
        stop_lines: list[tuple[SceneElement, ElementClass]] = []
        crosswalks: list[tuple[SceneElement, ElementClass]] = []
        shoulders: list[tuple[SceneElement, ElementClass]] = []
        road_edges: list[tuple[SceneElement, ElementClass]] = []
        lane_id_to_elem_ids: dict[str, list[str]] = {}

        seen_left: dict[str, str] = {}   # canonical boundary key → elem_id (dedup shared boundaries)
        seen_right: dict[str, str] = {}

        for seg in segments:
            left_id_existing = seen_left.get(f"{seg.lane_id}_L")
            right_id_existing = seen_right.get(f"{seg.lane_id}_R")

            # --- Left boundary ---
            if left_id_existing:
                left_elem_id = left_id_existing
            else:
                left_cls = self._classify_boundary(seg, side="left")
                pts = seg.left_boundary_xy
                if pts.shape[0] >= 2:
                    is_dashed = seg.left_womd_type in {
                        WOMDRoadLineType.TYPE_BROKEN_SINGLE_WHITE,
                        WOMDRoadLineType.TYPE_BROKEN_SINGLE_YELLOW,
                    }
                    color = "yellow" if "YELLOW" in seg.left_womd_type.value else "white"
                    if left_cls == ElementClass.ROAD_EDGE:
                        color = "black"
                    layer = "ROADWAY" if left_cls in _ROADWAY_CLASSES else "LANE_MARKINGS"
                    elem = _make_element(
                        pts, left_cls,
                        is_dashed=is_dashed, layer=layer, color=color,
                        faro_item_name=f"av2_{seg.lane_id}_L",
                    )
                    left_elem_id = elem.id
                    _assign(left_cls, elem, lane_markings, stop_lines, shoulders, road_edges)
                    seen_left[f"{seg.lane_id}_L"] = left_elem_id
                else:
                    left_elem_id = ""

            # --- Right boundary ---
            if right_id_existing:
                right_elem_id = right_id_existing
            else:
                right_cls = self._classify_boundary(seg, side="right")
                pts = seg.right_boundary_xy
                if pts.shape[0] >= 2:
                    is_dashed = seg.right_womd_type in {
                        WOMDRoadLineType.TYPE_BROKEN_SINGLE_WHITE,
                        WOMDRoadLineType.TYPE_BROKEN_SINGLE_YELLOW,
                    }
                    color = "yellow" if "YELLOW" in seg.right_womd_type.value else "white"
                    if right_cls == ElementClass.ROAD_EDGE:
                        color = "black"
                    layer = "ROADWAY" if right_cls in _ROADWAY_CLASSES else "LANE_MARKINGS"
                    elem = _make_element(
                        pts, right_cls,
                        is_dashed=is_dashed, layer=layer, color=color,
                        faro_item_name=f"av2_{seg.lane_id}_R",
                    )
                    right_elem_id = elem.id
                    _assign(right_cls, elem, lane_markings, stop_lines, shoulders, road_edges)
                    seen_right[f"{seg.lane_id}_R"] = right_elem_id
                else:
                    right_elem_id = ""

            lane_id_to_elem_ids[seg.lane_id] = [left_elem_id, right_elem_id]

        return lane_markings, stop_lines, crosswalks, shoulders, road_edges, lane_id_to_elem_ids

    @staticmethod
    def _classify_boundary(seg: LaneSegmentData, side: str) -> ElementClass:
        """Classify one lane boundary as ROAD_EDGE, LANE_MARKING_SOLID, LANE_MARKING_DASHED, or SHOULDER_LINE."""
        womd_type = seg.left_womd_type if side == "left" else seg.right_womd_type

        # Outermost boundary heuristic: if this side has no neighbor → road edge or shoulder
        has_neighbor = seg.left_neighbor_id is not None if side == "left" else seg.right_neighbor_id is not None
        if not has_neighbor:
            if womd_type == WOMDRoadLineType.TYPE_UNKNOWN:
                return ElementClass.ROAD_EDGE
            return ElementClass.SHOULDER_LINE

        return _mark_type_to_class(womd_type)

    def _extract_ground_truth(
        self,
        segments: list[LaneSegmentData],
        all_elems: list[tuple[SceneElement, ElementClass]],
        dropped_ids: set[str],
        lane_id_to_elem_ids: dict[str, list[str]],
        M_fwd: np.ndarray,
        scene_id: str,
        scenario_id: str,
        stage: CurriculumStage,
        inv_transform: InverseTransform,
    ) -> SyntheticGroundTruth:
        from synthetic.normalization import _apply_2d, _pts_to_np

        # element_classes: all elements in the scene
        element_classes: dict[str, str] = {}
        for elem, cls in all_elems:
            if elem.id not in dropped_ids:
                element_classes[elem.id] = cls.value

        # lane_centerlines: apply forward transform to AV2/WOMD centerlines (normalized coords)
        lane_centerlines: dict[str, list[Point2D]] = {}
        for seg in segments:
            if seg.centerline_xy.shape[0] < 2:
                continue
            cl_norm = _apply_2d(M_fwd, seg.centerline_xy)
            lane_centerlines[seg.lane_id] = [
                Point2D(x=float(p[0]), y=float(p[1])) for p in cl_norm
            ]

        # Build set of surviving lane_ids (whose elements weren't all dropped)
        surviving_lane_ids = {
            lane_id for lane_id, (lid, rid) in lane_id_to_elem_ids.items()
            if lid not in dropped_ids or rid not in dropped_ids
        }

        # topology: map source successor/predecessor IDs to surviving lane_ids
        topology: dict[str, LaneTopology] = {}
        for seg in segments:
            if seg.lane_id not in surviving_lane_ids:
                continue
            topology[seg.lane_id] = LaneTopology(
                entry_lane_ids=[p for p in seg.predecessor_ids if p in surviving_lane_ids],
                exit_lane_ids=[s for s in seg.successor_ids if s in surviving_lane_ids],
                left_neighbor_id=seg.left_neighbor_id if seg.left_neighbor_id in surviving_lane_ids else None,
                right_neighbor_id=seg.right_neighbor_id if seg.right_neighbor_id in surviving_lane_ids else None,
            )

        # boundary_assignments: left/right element per lane
        boundary_assignments: dict[str, BoundaryAssignment] = {}
        for seg in segments:
            if seg.lane_id not in surviving_lane_ids:
                continue
            ids = lane_id_to_elem_ids.get(seg.lane_id, ["", ""])
            left_id = ids[0] if ids[0] not in dropped_ids else None
            right_id = ids[1] if len(ids) > 1 and ids[1] not in dropped_ids else None
            boundary_assignments[seg.lane_id] = BoundaryAssignment(
                left_boundary_element_id=left_id or None,
                right_boundary_element_id=right_id or None,
                left_mark_type=seg.left_womd_type,
                right_mark_type=seg.right_womd_type,
            )

        # marking_types: per boundary element
        marking_types: dict[str, str] = {}
        for seg in segments:
            ids = lane_id_to_elem_ids.get(seg.lane_id, ["", ""])
            if ids[0] and ids[0] not in dropped_ids:
                marking_types[ids[0]] = seg.left_womd_type.value
            if len(ids) > 1 and ids[1] and ids[1] not in dropped_ids:
                marking_types[ids[1]] = seg.right_womd_type.value

        # endpoint_adjacency: group elements that share endpoints (within 0.05 normalized units)
        endpoint_adjacency = self._build_endpoint_adjacency(all_elems, dropped_ids, M_fwd)

        return SyntheticGroundTruth(
            scene_id=scene_id,
            source_scenario_id=scenario_id,
            source_dataset=self.source_dataset,
            curriculum_stage=stage.value,
            element_classes=element_classes,
            lane_centerlines=lane_centerlines,
            topology=topology,
            boundary_assignments=boundary_assignments,
            marking_types=marking_types,
            endpoint_adjacency=endpoint_adjacency,
            inverse_transform=inv_transform,
        )

    @staticmethod
    def _build_endpoint_adjacency(
        all_elems: list[tuple[SceneElement, ElementClass]],
        dropped_ids: set[str],
        M_fwd: np.ndarray,
    ) -> dict[str, list[str]]:
        """Group elements by shared endpoints in normalized coords (threshold 0.05 units)."""
        from synthetic.normalization import _apply_2d, _pts_to_np
        # Collect (elem_id, endpoint_xy) pairs
        endpoints: list[tuple[str, np.ndarray]] = []
        for elem, _ in all_elems:
            if elem.id in dropped_ids or not elem.resampled_points:
                continue
            pts = _pts_to_np(elem.resampled_points)
            pts_norm = _apply_2d(M_fwd, pts)
            endpoints.append((elem.id, pts_norm[0]))    # start
            endpoints.append((elem.id, pts_norm[-1]))   # end

        threshold = 0.05
        adjacency: dict[str, set[str]] = {eid: set() for eid, _ in endpoints}

        for i in range(len(endpoints)):
            eid_i, pt_i = endpoints[i]
            for j in range(i + 1, len(endpoints)):
                eid_j, pt_j = endpoints[j]
                if eid_i == eid_j:
                    continue
                if float(np.linalg.norm(pt_i - pt_j)) < threshold:
                    adjacency[eid_i].add(eid_j)
                    adjacency[eid_j].add(eid_i)

        return {k: sorted(v) for k, v in adjacency.items()}


# ---------------------------------------------------------------------------
# Helper: assign element to the correct list by class
# ---------------------------------------------------------------------------

def _assign(
    cls: ElementClass,
    elem: SceneElement,
    lane_markings: list,
    stop_lines: list,
    shoulders: list,
    road_edges: list,
) -> None:
    if cls == ElementClass.ROAD_EDGE:
        road_edges.append((elem, cls))
    elif cls == ElementClass.SHOULDER_LINE:
        shoulders.append((elem, cls))
    elif cls == ElementClass.STOP_LINE:
        stop_lines.append((elem, cls))
    else:
        lane_markings.append((elem, cls))


