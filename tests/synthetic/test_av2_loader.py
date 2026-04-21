import pytest
from synthetic.loaders.av2_loader import AV2ScenarioLoader, AV2_TO_WOMD
from synthetic.schema import WOMDRoadLineType


def test_all_six_av2_types_map_correctly():
    mapping = {
        "SOLID_WHITE": WOMDRoadLineType.TYPE_SOLID_SINGLE_WHITE,
        "SOLID_YELLOW": WOMDRoadLineType.TYPE_SOLID_SINGLE_YELLOW,
        "DASHED_WHITE": WOMDRoadLineType.TYPE_BROKEN_SINGLE_WHITE,
        "DASHED_YELLOW": WOMDRoadLineType.TYPE_BROKEN_SINGLE_YELLOW,
        "DOUBLE_SOLID_YELLOW": WOMDRoadLineType.TYPE_SOLID_DOUBLE_YELLOW,
        "DOUBLE_SOLID_WHITE": WOMDRoadLineType.TYPE_SOLID_DOUBLE_WHITE,
    }
    for av2_type, expected in mapping.items():
        result = AV2ScenarioLoader.map_marking_type(av2_type)
        assert result == expected, f"{av2_type}: expected {expected}, got {result}"


def test_unknown_av2_type_maps_to_unknown():
    assert AV2ScenarioLoader.map_marking_type("SOLID_BLUE") == WOMDRoadLineType.TYPE_UNKNOWN
    assert AV2ScenarioLoader.map_marking_type("") == WOMDRoadLineType.TYPE_UNKNOWN
    assert AV2ScenarioLoader.map_marking_type("NONE") == WOMDRoadLineType.TYPE_UNKNOWN


def test_av2_to_womd_dict_has_exactly_six_entries():
    assert len(AV2_TO_WOMD) == 6
