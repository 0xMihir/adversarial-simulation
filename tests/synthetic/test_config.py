import pytest
from synthetic.config import CurriculumConfig, CurriculumStage, RandomizationConfig


def test_all_probability_fields_in_range():
    for stage in CurriculumStage:
        cfg = CurriculumConfig().for_stage(stage)
        for fname in cfg.__dataclass_fields__:
            if fname.startswith("p_"):
                v = getattr(cfg, fname)
                assert 0.0 <= v <= 1.0, f"{stage}: {fname}={v} out of [0,1]"


def test_for_stage_returns_correct_config():
    cc = CurriculumConfig()
    assert cc.for_stage(CurriculumStage.A) is cc.stage_a
    assert cc.for_stage(CurriculumStage.B) is cc.stage_b
    assert cc.for_stage(CurriculumStage.C) is cc.stage_c


_MODULE_ENABLE_FIELDS = {
    # Fields that should be 0 in Stage A (enable/disable a module entirely)
    "p_drop_lane_markings", "p_drop_stop_lines", "p_drop_crosswalk_stripes", "p_drop_shoulder_lines",
    "p_resample_vertices", "p_convert_to_polycurve", "p_jitter_vertices", "p_vary_crosswalk",
    "p_add_annotation_clutter", "p_random_rotation", "p_random_scale", "p_random_aspect_ratio",
    "p_add_vehicle_markings", "p_apply_faro_styling",
    # Note: p_jitter_perpendicular_only, p_clutter_*, p_add_impact_markers, p_show_full_intersection
    # are conditional/behaviour probabilities, not module-enable flags.
}


def test_stage_a_is_all_zeros():
    cfg = CurriculumConfig().for_stage(CurriculumStage.A)
    for fname in _MODULE_ENABLE_FIELDS:
        assert getattr(cfg, fname) == 0.0, f"Stage A: {fname} should be 0"


def test_stage_c_has_all_modules_active():
    cfg = CurriculumConfig().for_stage(CurriculumStage.C)
    active = [fname for fname in cfg.__dataclass_fields__ if fname.startswith("p_") and getattr(cfg, fname) > 0.0]
    assert len(active) >= 9, "Stage C should activate all 9 module probability knobs"


def test_invalid_probability_raises():
    with pytest.raises(AssertionError):
        RandomizationConfig(p_drop_lane_markings=1.5)
