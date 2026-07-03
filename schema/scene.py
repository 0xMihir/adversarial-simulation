import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer, field_validator


class Point2D(BaseModel):
    x: float
    y: float


class AffineMatrix(BaseModel):
    """Row-major 3x3 affine: [[a,b,tx],[c,d,ty],[0,0,1]]"""
    values: list[list[float]]


def _empty_xy() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float64)


class SceneElement(BaseModel):
    """
    Geometry is stored as (N,2) float64 numpy arrays (control_xy/bezier_xy/resampled_xy)
    — the internal hot-path representation. control_points/bezier_handles/resampled_points
    are read-only Point2D views computed on demand, for boundary consumers (FARO parsing,
    annotation backend) that validate/serialize per-vertex. Construction takes ndarrays
    only; convert Point2D lists to arrays explicitly at the call site (see
    synthetic/normalization.py:pts_to_xy).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    faro_item_name: str | None = None
    layer: str | None = None
    element_type: str  # polyline | polycurve | line | arc | label | symbol
    control_xy: np.ndarray  # (N, 2) float64
    bezier_xy: np.ndarray = Field(default_factory=_empty_xy)  # (M, 2) float64
    interpolation_method: str  # passthrough | cubic_bezier | catmull_rom | linear | none
    resampled_xy: np.ndarray  # (K, 2) float64
    transform: AffineMatrix
    is_closed: bool
    is_dashed: bool
    line_width: float | None = None
    color: str | None = None
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy in world coords

    @field_validator("control_xy", "bezier_xy", "resampled_xy")
    @classmethod
    def _validate_xy(cls, v: np.ndarray) -> np.ndarray:
        if v.size == 0:
            return _empty_xy()
        if v.ndim != 2 or v.shape[1] != 2 or v.dtype != np.float64:
            raise ValueError(f"expected (N,2) float64 array, got shape={v.shape} dtype={v.dtype}")
        return v

    @field_serializer("control_xy", "bezier_xy", "resampled_xy")
    def _ser_xy(self, v: np.ndarray, _info) -> list[list[float]]:
        return v.tolist()

    @computed_field
    @property
    def control_points(self) -> list[Point2D]:
        return [Point2D(x=float(r[0]), y=float(r[1])) for r in self.control_xy]

    @computed_field
    @property
    def bezier_handles(self) -> list[Point2D]:
        return [Point2D(x=float(r[0]), y=float(r[1])) for r in self.bezier_xy]

    @computed_field
    @property
    def resampled_points(self) -> list[Point2D]:
        return [Point2D(x=float(r[0]), y=float(r[1])) for r in self.resampled_xy]


class TextElement(BaseModel):
    id: str
    text: str
    position: Point2D
    font_size: float | None = None
    rotation: float = 0.0


class ImageElement(BaseModel):
    id: str
    center: Point2D
    sizx: float
    sizy: float
    oriz: float        # radians, local rotation
    img: str           # raw base64 PNG, no data URI prefix
    layer: str | None = None


class VehicleDetection(BaseModel):
    id: str
    source_element_ids: list[str]
    obb: list[Point2D]  # 4 corners, oriented bounding box
    center: Point2D
    heading: float  # Radians
    classification_score: float
    predicted_class: str | None = None
    label_text: str | None = None


class ScaleBar(BaseModel):
    length_pixels: float
    length_real: float  # In feet (FARO default)
    unit: str = "ft"


class ParsedScene(BaseModel):
    case_id: str
    far_filename: str
    parsed_at: str
    coordinate_unit: str = "ft"
    scale_bar: ScaleBar | None = None
    elements: list[SceneElement]
    texts: list[TextElement]
    images: list[ImageElement] = []
    vehicles: list[VehicleDetection]
    roadway_indices: list[int]
    road_marking_indices: list[int]
    other_indices: list[int]
