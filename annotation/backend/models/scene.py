from pydantic import BaseModel


class Point2D(BaseModel):
    x: float
    y: float


class AffineMatrix(BaseModel):
    """Row-major 3x3 affine: [[a,b,tx],[c,d,ty],[0,0,1]]"""
    values: list[list[float]]


class SceneElement(BaseModel):
    id: str
    faro_item_name: str | None = None
    layer: str | None = None
    element_type: str  # polyline | polycurve | line | arc | label | symbol
    control_points: list[Point2D]
    bezier_handles: list[Point2D] = []
    interpolation_method: str  # passthrough | cubic_bezier | catmull_rom | linear | none
    resampled_points: list[Point2D]
    transform: AffineMatrix
    is_closed: bool
    is_dashed: bool
    line_width: float | None = None
    color: str | None = None
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy in world coords


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
