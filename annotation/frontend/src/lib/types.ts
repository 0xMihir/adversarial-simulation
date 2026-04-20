// Mirror of backend Pydantic models

export interface Point2D {
  x: number;
  y: number;
}

export interface AffineMatrix {
  values: number[][];
}

export interface SceneElement {
  id: string;
  faro_item_name: string | null;
  layer: string | null;
  element_type: string;
  control_points: Point2D[];
  bezier_handles: Point2D[];
  interpolation_method: string;
  resampled_points: Point2D[];
  transform: AffineMatrix;
  is_closed: boolean;
  is_dashed: boolean;
  line_width: number | null;
  color: string | null;
  bbox: [number, number, number, number];
}

export interface TextElement {
  id: string;
  text: string;
  position: Point2D;
  font_size: number | null;
  rotation: number;
}

export interface ImageElement {
  id: string;
  center: Point2D;
  sizx: number;
  sizy: number;
  oriz: number;
  img: string;
  layer: string | null;
}

export interface VehicleDetection {
  id: string;
  source_element_ids: string[];
  obb: Point2D[];
  center: Point2D;
  heading: number;
  classification_score: number;
  predicted_class: string | null;
  label_text: string | null;
}

export interface ScaleBar {
  length_pixels: number;
  length_real: number;
  unit: string;
}

export interface ParsedScene {
  case_id: string;
  far_filename: string;
  parsed_at: string;
  coordinate_unit: string;
  scale_bar: ScaleBar | null;
  elements: SceneElement[];
  texts: TextElement[];
  images: ImageElement[];
  vehicles: VehicleDetection[];
  roadway_indices: number[];
  road_marking_indices: number[];
  other_indices: number[];
}

export type ElementStatus = "auto" | "confirmed" | "corrected" | "rejected";

export interface LaneAnnotation {
  id: string;
  polyline: Point2D[];
  raw_control_points: Point2D[];
  lane_type: string;
  left_boundary_type: string;
  right_boundary_type: string;
  entry_lanes: string[];
  exit_lanes: string[];
  speed_limit_mph: number | null;
  status: ElementStatus;
  source_element_ids: string[];
  notes: string | null;
}

export interface VehicleWaypoint {
  position: Point2D;
  heading: number;
  timestamp_index: number;
  phase: string;
  lane_id: string | null;
  speed_estimate: number | null;
}

export interface VehicleAnnotation {
  id: string;
  vehicle_type: string;
  waypoints: VehicleWaypoint[];
  status: ElementStatus;
}

export interface LaneConnection {
  id: string;
  from_lane_id: string;
  to_lane_id: string;
  from_end: "start" | "end";
  to_end: "start" | "end";
  connection_type: string;
  control_points: Point2D[] | null;
  status: ElementStatus;
}

export interface CorrectionRecord {
  timestamp: string;
  element_type: string;
  element_id: string;
  action: string;
  previous_value: Record<string, unknown> | null;
}

export interface CaseAnnotation {
  case_id: string;
  far_filename: string;
  scene: ParsedScene;
  auto_centerlines: LaneAnnotation[];
  lanes: LaneAnnotation[];
  vehicles: VehicleAnnotation[];
  lane_connections: LaneConnection[];
  corrections: CorrectionRecord[];
  created_at: string;
  updated_at: string;
  annotator: string | null;
  workflow_status: string;
  auto_confidence: number | null;
  hidden_roadway_ids: string[];
}

export interface CaseAnnotationUpdate {
  lanes?: LaneAnnotation[];
  vehicles?: VehicleAnnotation[];
  lane_connections?: LaneConnection[];
  workflow_status?: string;
  annotator?: string;
  corrections?: CorrectionRecord[];
  hidden_roadway_ids?: string[];
}

export interface CaseSummary {
  id: string;
  filename: string;
  annotated: boolean;
  workflow_status: string;
  updated_at: string | null;
  lane_count: number;
  vehicle_count: number;
  auto_confidence: number | null;
}

// UI-only types
export type MouseMode = "select" | "connect" | "edit";

export type LayerKey =
  | "images"
  | "roadway"
  | "road_markings"
  | "centerlines"
  | "vehicles"
  | "trajectories"
  | "connections"
  | "texts";

export interface LayerVisibility {
  images: boolean;
  roadway: boolean;
  road_markings: boolean;
  centerlines: boolean;
  vehicles: boolean;
  trajectories: boolean;
  connections: boolean;
  texts: boolean;
}

export type SelectedElement =
  | { kind: "lane"; id: string }
  | { kind: "vehicle"; id: string }
  | { kind: "connection"; id: string }
  | { kind: "roadway"; id: string }
  | null;
