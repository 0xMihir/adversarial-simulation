from dataclasses import dataclass, field
from enum import Enum


class CurriculumStage(str, Enum):
    A = "A"  # 0-25%: no randomization — clean source maps with FARO schema only
    B = "B"  # 25-60%: drop_primitives, resample_vertices, apply_faro_styling
    C = "C"  # 60-100%: all 9 modules active


@dataclass
class RandomizationConfig:
    # --- drop_primitives (Module 1) ---
    p_drop_lane_markings: float = 0.0       # per-element Bernoulli drop
    p_drop_stop_lines: float = 0.0
    p_drop_crosswalk_stripes: float = 0.0
    p_drop_shoulder_lines: float = 0.0

    # --- resample_vertices (Module 2) ---
    p_resample_vertices: float = 0.0        # probability of triggering resampling
    vertex_count_min: int = 4
    vertex_count_max: int = 20
    p_convert_to_polycurve: float = 0.0     # if resampling fires, prob of polycurve conversion

    # --- jitter_vertices (Module 3) ---
    p_jitter_vertices: float = 0.0
    jitter_sigma_max: float = 0.3           # metres, before normalization
    p_jitter_perpendicular_only: float = 0.5

    # --- vary_crosswalk (Module 4) ---
    p_vary_crosswalk: float = 0.0
    crosswalk_stripe_count_min: int = 4
    crosswalk_stripe_count_max: int = 20
    crosswalk_spacing_min_m: float = 0.3
    crosswalk_spacing_max_m: float = 1.2
    crosswalk_stripe_length_min_m: float = 2.0
    crosswalk_stripe_length_max_m: float = 6.0
    crosswalk_angle_noise_deg: float = 10.0
    crosswalk_max_occluded_stripes: int = 3

    # --- add_annotation_clutter (Module 5) ---
    p_add_annotation_clutter: float = 0.0
    clutter_max_rp_markers: int = 8
    clutter_max_grade_callouts: int = 4
    p_clutter_north_arrow: float = 0.5
    p_clutter_scale_bar: float = 0.5
    p_clutter_dim_callouts: float = 0.2

    # --- subset_network (Module 6) ---
    crop_size_min_m: float = 40.0
    crop_size_max_m: float = 200.0
    p_show_full_intersection: float = 1.0   # 1.0 = always show full; lower = partial crops

    # --- transform_layout (Module 7) ---
    p_random_rotation: float = 0.0
    p_random_scale: float = 0.0
    layout_scale_min: float = 0.5
    layout_scale_max: float = 2.0
    p_random_aspect_ratio: float = 0.0

    # --- add_vehicle_markings (Module 8) ---
    p_add_vehicle_markings: float = 0.0
    vehicle_count_min: int = 1
    vehicle_count_max: int = 4
    p_add_impact_markers: float = 0.5       # conditional on p_add_vehicle_markings firing

    # --- apply_faro_styling (Module 9) ---
    p_apply_faro_styling: float = 0.0
    faro_line_weight_min: float = 0.1
    faro_line_weight_max: float = 2.0

    def __post_init__(self) -> None:
        for fname, fval in self.__dataclass_fields__.items():
            if fname.startswith("p_"):
                v = getattr(self, fname)
                assert 0.0 <= v <= 1.0, f"{fname}={v} out of [0,1]"


def _stage_b_config() -> "RandomizationConfig":
    return RandomizationConfig(
        p_drop_lane_markings=0.4,
        p_drop_stop_lines=0.5,
        p_drop_shoulder_lines=0.3,
        p_resample_vertices=0.8,
        p_apply_faro_styling=0.7,
    )


def _stage_c_config() -> "RandomizationConfig":
    return RandomizationConfig(
        p_drop_lane_markings=0.7,
        p_drop_stop_lines=0.9,
        p_drop_crosswalk_stripes=0.4,
        p_drop_shoulder_lines=0.5,
        p_resample_vertices=0.9,
        p_convert_to_polycurve=0.5,
        p_jitter_vertices=0.8,
        p_vary_crosswalk=0.6,
        p_add_annotation_clutter=0.5,
        p_show_full_intersection=0.3,
        p_random_rotation=0.9,
        p_random_scale=0.5,
        p_random_aspect_ratio=0.3,
        p_add_vehicle_markings=0.8,
        p_apply_faro_styling=0.9,
    )


@dataclass
class CurriculumConfig:
    stage_a: RandomizationConfig = field(default_factory=RandomizationConfig)
    stage_b: RandomizationConfig = field(default_factory=_stage_b_config)
    stage_c: RandomizationConfig = field(default_factory=_stage_c_config)

    def for_stage(self, stage: CurriculumStage) -> RandomizationConfig:
        return {
            CurriculumStage.A: self.stage_a,
            CurriculumStage.B: self.stage_b,
            CurriculumStage.C: self.stage_c,
        }[stage]
