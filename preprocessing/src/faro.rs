//! FARO XML Scene Graph Parser
//!
//! This module provides a high-performance parser for FARO (.far) files,
//! which are XML-based crash scene diagrams from NHTSA's CISS database.
//!
//! # Example
//!
//! ```no_run
//! use preprocessing::faro::{FaroParser, ParserConfig};
//!
//! let config = ParserConfig::default()
//!     .with_bezier_points_per_segment(10)
//!     .with_closure_threshold(0.25);
//!
//! let mut parser = FaroParser::new(config);
//! let scene = parser.parse("scene.far").unwrap();
//!
//! println!("Found {} vehicles", scene.vehicle_ids.len());
//! ```

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use thiserror::Error;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during FARO file parsing
#[derive(Error, Debug)]
pub enum FaroError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XML parsing error: {0}")]
    Xml(#[from] quick_xml::Error),

    #[error("Invalid vertex format in attribute")]
    InvalidVertexFormat,

    #[error("Missing required attribute: {0}")]
    MissingAttribute(String),

    #[error("Invalid attribute value for {attr}: {value}")]
    InvalidAttributeValue { attr: String, value: String },

    #[error("Unexpected end of file")]
    UnexpectedEof,

    #[error("Malformed scene structure: {0}")]
    MalformedStructure(String),

    #[error("UTF-8 decoding error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
}

// ============================================================================
// Core Geometry Types
// ============================================================================

/// 2D point - Copy for efficient use in algorithms
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)] // C-compatible layout for FFI/SIMD
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    /// Create a new point
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point
    #[inline]
    pub fn distance_to(&self, other: Point2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// Squared distance (avoids sqrt for comparisons)
    #[inline]
    pub fn distance_sq(&self, other: Point2D) -> f64 {
        (self.x - other.x).powi(2) + (self.y - other.y).powi(2)
    }

    /// Convert to array representation
    #[inline]
    pub fn as_array(&self) -> [f64; 2] {
        [self.x, self.y]
    }

    /// Create from tuple
    #[inline]
    pub fn from_tuple(t: (f64, f64)) -> Self {
        Self { x: t.0, y: t.1 }
    }
}

impl From<(f64, f64)> for Point2D {
    fn from(t: (f64, f64)) -> Self {
        Self::from_tuple(t)
    }
}

impl From<[f64; 2]> for Point2D {
    fn from(arr: [f64; 2]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
        }
    }
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BBox {
    pub min: Point2D,
    pub max: Point2D,
}

impl BBox {
    /// Create a new bounding box
    pub const fn new(min: Point2D, max: Point2D) -> Self {
        Self { min, max }
    }

    /// Create an empty/invalid bounding box for initialization
    pub fn empty() -> Self {
        Self {
            min: Point2D::new(f64::INFINITY, f64::INFINITY),
            max: Point2D::new(f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    /// Check if the bounding box is valid (non-empty)
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y
    }

    /// Diagonal length of the bounding box
    pub fn diagonal(&self) -> f64 {
        self.min.distance_to(self.max)
    }

    /// Width of the bounding box
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    /// Height of the bounding box
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }

    /// Center point of the bounding box
    pub fn center(&self) -> Point2D {
        Point2D {
            x: (self.min.x + self.max.x) / 2.0,
            y: (self.min.y + self.max.y) / 2.0,
        }
    }

    /// Compute union of two bounding boxes
    pub fn union(&self, other: &BBox) -> BBox {
        BBox {
            min: Point2D {
                x: self.min.x.min(other.min.x),
                y: self.min.y.min(other.min.y),
            },
            max: Point2D {
                x: self.max.x.max(other.max.x),
                y: self.max.y.max(other.max.y),
            },
        }
    }

    /// Expand bounding box to include a point
    pub fn expand(&mut self, point: Point2D) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
    }

    /// Create bounding box from a slice of points
    pub fn from_points(points: &[Point2D]) -> Self {
        let mut bbox = Self::empty();
        for p in points {
            bbox.expand(*p);
        }
        bbox
    }
}

/// 3x3 affine transform matrix (row-major)
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))] // Aligned for SIMD potential
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Transform2D {
    pub data: [[f64; 3]; 3],
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

impl Transform2D {
    /// Create identity transform
    pub const fn identity() -> Self {
        Self {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Create combined T @ R @ S matrix from transform attributes
    ///
    /// This matches the Python implementation's transformation order:
    /// Translation @ Rotation @ Scale
    ///
    /// # Arguments
    /// * `pos_x` - X translation (posx attribute)
    /// * `pos_y` - Y translation (posy attribute)
    /// * `rotation` - Rotation in radians (oriz attribute)
    /// * `scale_x` - X scale factor (scalex attribute)
    /// * `scale_y` - Y scale factor (scaley attribute)
    pub fn from_attributes(
        pos_x: f64,
        pos_y: f64,
        rotation: f64,
        scale_x: f64,
        scale_y: f64,
    ) -> Self {
        let (sin, cos) = rotation.sin_cos();
        Self {
            data: [
                [cos * scale_x, -sin * scale_y, pos_x],
                [sin * scale_x, cos * scale_y, pos_y],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Create translation-only transform
    pub fn translate(tx: f64, ty: f64) -> Self {
        Self {
            data: [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        }
    }

    /// Create rotation-only transform (radians)
    pub fn rotate(angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            data: [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Create scale-only transform
    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            data: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Matrix multiplication: self @ other
    #[inline]
    pub fn multiply(&self, other: &Self) -> Self {
        let a = &self.data;
        let b = &other.data;
        Self {
            data: [
                [
                    a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
                    a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
                    a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
                ],
                [
                    a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
                    a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
                    a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
                ],
                [0.0, 0.0, 1.0], // Affine: last row always [0, 0, 1]
            ],
        }
    }

    /// Transform a single point
    #[inline]
    pub fn apply(&self, point: Point2D) -> Point2D {
        let m = &self.data;
        Point2D {
            x: m[0][0] * point.x + m[0][1] * point.y + m[0][2],
            y: m[1][0] * point.x + m[1][1] * point.y + m[1][2],
        }
    }

    /// Transform multiple points, returning a new vector
    pub fn apply_batch(&self, points: &[Point2D]) -> Vec<Point2D> {
        points.iter().map(|&p| self.apply(p)).collect()
    }

    /// Transform multiple points in place
    pub fn apply_batch_inplace(&self, points: &mut [Point2D]) {
        for p in points.iter_mut() {
            *p = self.apply(*p);
        }
    }
}

// ============================================================================
// Primitive and Symbol Types
// ============================================================================

/// Type of geometric primitive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PrimitiveType {
    Polyline,
    Polycurve,
    Line,
    Label,
    Ellipse,
    FlexConcreteBarrier,
    Unknown,
}

impl PrimitiveType {
    fn from_str(s: &str) -> Self {
        match s {
            "polyline" => Self::Polyline,
            "polycurve" => Self::Polycurve,
            "line" => Self::Line,
            "label" => Self::Label,
            "ellipse" => Self::Ellipse,
            "flexconcretebarrier" => Self::FlexConcreteBarrier,
            _ => Self::Unknown,
        }
    }
}

/// Classification labels for symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ClassificationLabel {
    Vehicle,
    TrafficLight,
    RoadMarking,
    DirectionArrow,
    TurnDirection,
    Pedestrian,
    Background,
}

/// Line style attributes
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LineStyle {
    pub dashed: bool,
    pub thick: bool,
    pub closed: bool,
}

/// A geometric primitive with all computed properties
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Primitive {
    pub ptype: PrimitiveType,
    pub name: Option<String>,
    pub layer: Option<String>,

    // Geometry (local coordinates)
    pub verts: Vec<Point2D>,
    pub center: Point2D,
    pub bbox: BBox,

    // Transformed geometry (global coordinates)
    pub transformed_verts: Vec<Point2D>,
    pub transformed_center: Point2D,

    // Transform that was applied
    pub transform: Transform2D,

    // Attributes
    pub style: LineStyle,
    pub vehicle2d: bool,
    pub text: Option<String>, // For labels
}

impl Primitive {
    /// Create a new primitive with computed properties
    pub fn new(
        ptype: PrimitiveType,
        verts: Vec<Point2D>,
        transform: Transform2D,
        name: Option<String>,
        layer: Option<String>,
        style: LineStyle,
        vehicle2d: bool,
        text: Option<String>,
    ) -> Self {
        let bbox = BBox::from_points(&verts);
        let center = if verts.is_empty() {
            Point2D::default()
        } else {
            let sum_x: f64 = verts.iter().map(|p| p.x).sum();
            let sum_y: f64 = verts.iter().map(|p| p.y).sum();
            Point2D::new(sum_x / verts.len() as f64, sum_y / verts.len() as f64)
        };

        let transformed_verts = transform.apply_batch(&verts);
        let transformed_center = transform.apply(center);

        Self {
            ptype,
            name,
            layer,
            verts,
            center,
            bbox,
            transformed_verts,
            transformed_center,
            transform,
            style,
            vehicle2d,
            text,
        }
    }
}

/// Index into the primitives vector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PrimitiveId(pub usize);

/// Index into the symbols vector
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SymbolId(pub usize);

/// A symbol containing nested primitives and/or other symbols
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Symbol {
    pub name: Option<String>,
    pub layer: Option<String>,

    // Nested content stored directly (not as indices)
    pub items: Vec<SymbolItem>,

    // Computed properties
    pub bbox: BBox,
    pub center: Point2D,
    pub transformed_center: Point2D,
    pub transform: Transform2D,

    // Classification
    pub vehicle2d: bool,
    pub predicted_class: Option<ClassificationLabel>,
    pub predicted_probability: Option<f64>,
    pub associated_text: Vec<String>,
}

/// Item within a symbol (can be primitive or nested symbol)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SymbolItem {
    Primitive(Primitive),
    Symbol(Box<Symbol>),
}

impl SymbolItem {
    pub fn bbox(&self) -> BBox {
        match self {
            SymbolItem::Primitive(p) => p.bbox,
            SymbolItem::Symbol(s) => s.bbox,
        }
    }
}

/// Scene output with categorized objects
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Scene {
    // All symbols (flat storage)
    pub symbols: Vec<Symbol>,
    // All top-level primitives
    pub primitives: Vec<Primitive>,

    // Categorized indices
    pub vehicle_ids: Vec<usize>,
    pub roadway_ids: Vec<usize>,
    pub road_marking_ids: Vec<usize>,
    pub text_ids: Vec<usize>,
    pub misc_primitive_ids: Vec<usize>,
    pub misc_symbol_ids: Vec<usize>,
}

impl Scene {
    /// Get roadway primitives (for Delaunay input)
    pub fn roadway_primitives(&self) -> impl Iterator<Item = &Primitive> {
        self.roadway_ids.iter().map(|&id| &self.primitives[id])
    }

    /// Get vehicles
    pub fn vehicles(&self) -> impl Iterator<Item = &Symbol> {
        self.vehicle_ids.iter().map(|&id| &self.symbols[id])
    }

    /// Get text labels
    pub fn texts(&self) -> impl Iterator<Item = &Primitive> {
        self.text_ids.iter().map(|&id| &self.primitives[id])
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Parser configuration with sensible defaults
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ParserConfig {
    /// Points per Bezier curve segment (default: 10)
    pub bezier_points_per_segment: usize,

    /// Threshold for auto-closing polycurves (fraction of bbox diagonal, default: 0.25)
    pub closure_threshold: f64,

    /// Max distance for text-to-vehicle association (default: 5.0)
    pub text_association_distance: f64,

    /// Vehicle classification probability threshold (default: 0.7)
    pub vehicle_probability_threshold: f64,

    /// Whether to compute transformed vertices (default: true)
    pub compute_transforms: bool,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            bezier_points_per_segment: 10,
            closure_threshold: 0.25,
            text_association_distance: 5.0,
            vehicle_probability_threshold: 0.7,
            compute_transforms: true,
        }
    }
}

impl ParserConfig {
    /// Set points per Bezier segment
    pub fn with_bezier_points_per_segment(mut self, n: usize) -> Self {
        self.bezier_points_per_segment = n;
        self
    }

    /// Set closure threshold
    pub fn with_closure_threshold(mut self, t: f64) -> Self {
        self.closure_threshold = t;
        self
    }

    /// Set text association distance
    pub fn with_text_association_distance(mut self, d: f64) -> Self {
        self.text_association_distance = d;
        self
    }

    /// Set vehicle probability threshold
    pub fn with_vehicle_probability_threshold(mut self, t: f64) -> Self {
        self.vehicle_probability_threshold = t;
        self
    }

    /// Set whether to compute transforms
    pub fn with_compute_transforms(mut self, compute: bool) -> Self {
        self.compute_transforms = compute;
        self
    }
}

// ============================================================================
// Bezier Interpolation
// ============================================================================

/// Cubic Bezier evaluation: B(t) = (1-t)³P₀ + 3(1-t)²tC₁ + 3(1-t)t²C₂ + t³P₁
#[inline]
pub fn bezier_cubic(p0: Point2D, c1: Point2D, c2: Point2D, p1: Point2D, t: f64) -> Point2D {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;

    Point2D {
        x: mt3 * p0.x + 3.0 * mt2 * t * c1.x + 3.0 * mt * t2 * c2.x + t3 * p1.x,
        y: mt3 * p0.y + 3.0 * mt2 * t * c1.y + 3.0 * mt * t2 * c2.y + t3 * p1.y,
    }
}

/// Interpolate polycurve with configurable points per segment
///
/// # Arguments
/// * `anchors` - Anchor points of the curve
/// * `controls` - Control points (2 per segment, so 2*(n-1) for n anchors)
/// * `points_per_segment` - Number of output points per segment
///
/// # Returns
/// Interpolated points, or the original anchors if control points are invalid
pub fn interpolate_polycurve(
    anchors: &[Point2D],
    controls: &[Point2D],
    points_per_segment: usize,
) -> Vec<Point2D> {
    // Validate control point count: 2*(n-1) for n anchors
    let expected = 2 * anchors.len().saturating_sub(1);
    if controls.len() != expected || anchors.len() < 2 || points_per_segment < 2 {
        return anchors.to_vec(); // Fallback
    }

    let num_segments = anchors.len() - 1;
    let capacity = num_segments * points_per_segment - (num_segments - 1);
    let mut result = Vec::with_capacity(capacity);

    for i in 0..num_segments {
        let p0 = anchors[i];
        let p1 = anchors[i + 1];
        let c1 = controls[2 * i];
        let c2 = controls[2 * i + 1];

        let start_j = if i == 0 { 0 } else { 1 }; // Avoid duplicating shared points
        for j in start_j..points_per_segment {
            let t = j as f64 / (points_per_segment - 1) as f64;
            result.push(bezier_cubic(p0, c1, c2, p1, t));
        }
    }

    result
}

// ============================================================================
// Vertex Parsing
// ============================================================================

/// Parse vertex list format: "x,y,z;x,y,z;..." (z is ignored)
///
/// Optimized to minimize allocations with capacity hints
pub fn parse_vertex_list(input: &str) -> Result<Vec<Point2D>, FaroError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Pre-count semicolons for capacity hint
    let approx_count = input.bytes().filter(|&b| b == b';').count() + 1;
    let mut result = Vec::with_capacity(approx_count);

    for vertex_str in input.split(';') {
        if vertex_str.is_empty() {
            continue;
        }

        let mut parts = vertex_str.split(',');
        let x: f64 = parts
            .next()
            .ok_or(FaroError::InvalidVertexFormat)?
            .parse()
            .map_err(|_| FaroError::InvalidVertexFormat)?;
        let y: f64 = parts
            .next()
            .ok_or(FaroError::InvalidVertexFormat)?
            .parse()
            .map_err(|_| FaroError::InvalidVertexFormat)?;
        // Skip z coordinate if present

        result.push(Point2D::new(x, y));
    }

    Ok(result)
}

// ============================================================================
// XML Attribute Helpers
// ============================================================================

/// Get string attribute value from XML element
fn get_attr<'a>(event: &'a BytesStart<'a>, name: &[u8]) -> Option<String> {
    event
        .attributes()
        .filter_map(|a| a.ok())
        .find(|a| a.key.as_ref() == name)
        .and_then(|a| String::from_utf8(a.value.to_vec()).ok())
}

/// Get float attribute with default
fn get_attr_f64(event: &BytesStart, name: &[u8], default: f64) -> f64 {
    get_attr(event, name)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Get boolean attribute (checks for "T" or "t")
fn get_attr_bool(event: &BytesStart, name: &[u8]) -> bool {
    get_attr(event, name)
        .map(|s| s == "T" || s == "t" || s == "true")
        .unwrap_or(false)
}

/// Extract transform from element attributes
fn get_transform(event: &BytesStart) -> Transform2D {
    let pos_x = get_attr_f64(event, b"posx", 0.0);
    let pos_y = get_attr_f64(event, b"posy", 0.0);
    let scale_x = get_attr_f64(event, b"scalex", 1.0);
    let scale_y = get_attr_f64(event, b"scaley", 1.0);
    // Also check for sclx/scly variants
    let scale_x = if scale_x == 1.0 {
        get_attr_f64(event, b"sclx", 1.0)
    } else {
        scale_x
    };
    let scale_y = if scale_y == 1.0 {
        get_attr_f64(event, b"scly", 1.0)
    } else {
        scale_y
    };
    let rotation = get_attr_f64(event, b"oriz", 0.0);

    Transform2D::from_attributes(pos_x, pos_y, rotation, scale_x, scale_y)
}

// ============================================================================
// Parser Implementation
// ============================================================================

/// FARO file parser
pub struct FaroParser {
    config: ParserConfig,

    // Intermediate results during parsing
    symbols: Vec<Symbol>,
    primitives: Vec<Primitive>,
}

impl FaroParser {
    /// Create a new parser with the given configuration
    pub fn new(config: ParserConfig) -> Self {
        Self {
            config,
            symbols: Vec::new(),
            primitives: Vec::new(),
        }
    }

    /// Create a parser with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ParserConfig::default())
    }

    /// Parse a FARO file and return the scene
    pub fn parse<P: AsRef<Path>>(&mut self, path: P) -> Result<Scene, FaroError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        self.parse_reader(reader)
    }

    /// Parse from a reader
    pub fn parse_reader<R: std::io::BufRead>(&mut self, reader: R) -> Result<Scene, FaroError> {
        // Reset state
        self.symbols.clear();
        self.primitives.clear();

        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let identity = Transform2D::identity();
        let mut current_layer: Option<String> = None;
        let mut in_scene = false;

        loop {
            buf.clear();
            match xml_reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => break,
                Ok(Event::Start(ref e)) => {
                    let name = e.name();
                    match name.as_ref() {
                        b"scene" => {
                            in_scene = true;
                        }
                        b"layer" if in_scene => {
                            current_layer = get_attr(e, b"n");
                            // Continue parsing children
                        }
                        b"item" if in_scene => {
                            let item_type = get_attr(e, b"type").unwrap_or_default();
                            if item_type == "symbol" {
                                // Extract attributes before recursive call
                                let local_transform = get_transform(e);
                                let sym_name = get_attr(e, b"nam");
                                let vehicle2d = get_attr_bool(e, b"vehicle2d");
                                let layer_clone = current_layer.clone();

                                // Parse symbol recursively with fresh buffer
                                if let Some(symbol) = self.parse_symbol_inner(
                                    &mut xml_reader,
                                    local_transform,
                                    sym_name,
                                    vehicle2d,
                                    identity,
                                    layer_clone,
                                )? {
                                    self.symbols.push(symbol);
                                }
                            } else {
                                // Parse primitive
                                if let Some(primitive) =
                                    self.extract_primitive(e, identity, current_layer.clone())?
                                {
                                    self.primitives.push(primitive);
                                }
                                // Skip to end of item with fresh buffer
                                let end_name = e.name();
                                let mut skip_buf = Vec::new();
                                xml_reader.read_to_end_into(end_name, &mut skip_buf)?;
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name().as_ref() == b"layer" {
                        current_layer = None;
                    }
                }
                Ok(Event::Empty(ref e)) if in_scene => {
                    // Handle self-closing <item .../> tags
                    if e.name().as_ref() == b"item" {
                        let item_type = get_attr(e, b"type").unwrap_or_default();
                        if item_type != "symbol" {
                            if let Some(primitive) =
                                self.extract_primitive(e, identity, current_layer.clone())?
                            {
                                self.primitives.push(primitive);
                            }
                        }
                    }
                }
                Err(e) => return Err(FaroError::Xml(e)),
                _ => {}
            }
        }

        // Phase 2: Classify and categorize
        self.classify_and_categorize()
    }

    /// Parse a symbol element recursively (uses its own buffer)
    fn parse_symbol_inner<R: std::io::BufRead>(
        &self,
        reader: &mut Reader<R>,
        local_transform: Transform2D,
        name: Option<String>,
        vehicle2d: bool,
        parent_transform: Transform2D,
        layer: Option<String>,
    ) -> Result<Option<Symbol>, FaroError> {
        // Combine transforms
        let current_transform = parent_transform.multiply(&local_transform);

        let mut items: Vec<SymbolItem> = Vec::new();
        let mut buf = Vec::new();

        // Parse children until </item>
        loop {
            buf.clear();
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    if e.name().as_ref() == b"item" {
                        let item_type = get_attr(e, b"type").unwrap_or_default();
                        if item_type == "symbol" {
                            // Extract nested symbol attributes
                            let nested_transform = get_transform(e);
                            let nested_name = get_attr(e, b"nam");
                            let nested_vehicle2d = get_attr_bool(e, b"vehicle2d");

                            // Nested symbol (recursive)
                            if let Some(nested) = self.parse_symbol_inner(
                                reader,
                                nested_transform,
                                nested_name,
                                nested_vehicle2d,
                                current_transform,
                                layer.clone(),
                            )? {
                                items.push(SymbolItem::Symbol(Box::new(nested)));
                            }
                        } else {
                            // Primitive within symbol
                            if let Some(primitive) =
                                self.extract_primitive(e, current_transform, layer.clone())?
                            {
                                items.push(SymbolItem::Primitive(primitive));
                            }
                            // Skip to end with fresh buffer
                            let end_name = e.name();
                            let mut skip_buf = Vec::new();
                            reader.read_to_end_into(end_name, &mut skip_buf)?;
                        }
                    }
                }
                Ok(Event::Empty(ref e)) => {
                    if e.name().as_ref() == b"item" {
                        let item_type = get_attr(e, b"type").unwrap_or_default();
                        if item_type != "symbol" {
                            if let Some(primitive) =
                                self.extract_primitive(e, current_transform, layer.clone())?
                            {
                                items.push(SymbolItem::Primitive(primitive));
                            }
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name().as_ref() == b"item" {
                        break;
                    }
                }
                Ok(Event::Eof) => return Err(FaroError::UnexpectedEof),
                Err(e) => return Err(FaroError::Xml(e)),
                _ => {}
            }
        }

        // Compute bounding box from items
        let mut bbox = BBox::empty();
        for item in &items {
            let item_bbox = item.bbox();
            if item_bbox.is_valid() {
                bbox = bbox.union(&item_bbox);
            }
        }

        let center = bbox.center();
        let transformed_center = current_transform.apply(center);

        Ok(Some(Symbol {
            name,
            layer,
            items,
            bbox,
            center,
            transformed_center,
            transform: current_transform,
            vehicle2d,
            predicted_class: None,
            predicted_probability: None,
            associated_text: Vec::new(),
        }))
    }

    /// Extract a primitive from an XML element
    fn extract_primitive(
        &self,
        event: &BytesStart,
        parent_transform: Transform2D,
        layer: Option<String>,
    ) -> Result<Option<Primitive>, FaroError> {
        let type_str = get_attr(event, b"type").unwrap_or_default();
        let ptype = PrimitiveType::from_str(&type_str);

        // Get local transform and combine
        let local_transform = get_transform(event);
        let global_transform = parent_transform.multiply(&local_transform);

        let name = get_attr(event, b"nam");
        let vehicle2d = get_attr_bool(event, b"vehicle2d");
        let mut closed = get_attr_bool(event, b"closed");

        let mut verts: Vec<Point2D> = Vec::new();
        let mut text: Option<String> = None;
        let mut dashed = false;
        let mut thick = false;

        match ptype {
            PrimitiveType::Polyline => {
                if let Some(vlist) = get_attr(event, b"vlist") {
                    verts = parse_vertex_list(&vlist)?;
                }
            }
            PrimitiveType::Polycurve => {
                let anchors = get_attr(event, b"pnts")
                    .map(|s| parse_vertex_list(&s))
                    .transpose()?
                    .unwrap_or_default();
                let controls = get_attr(event, b"ctrl")
                    .map(|s| parse_vertex_list(&s))
                    .transpose()?
                    .unwrap_or_default();

                // Check for auto-closure
                if !closed && anchors.len() >= 2 {
                    let first = anchors[0];
                    let last = anchors[anchors.len() - 1];
                    let bbox = BBox::from_points(&anchors);
                    let diag = bbox.diagonal().max(1e-10);
                    if first.distance_to(last) < self.config.closure_threshold * diag {
                        closed = true;
                    }
                }

                verts = interpolate_polycurve(&anchors, &controls, self.config.bezier_points_per_segment);

                // Close the curve if needed
                if closed && !verts.is_empty() && verts[0] != verts[verts.len() - 1] {
                    verts.push(verts[0]);
                }
            }
            PrimitiveType::Line => {
                // Check for arrows - skip if present
                // Note: lndata is a child element, but we check attributes here
                let arrow_e = get_attr_bool(event, b"arrowshowe");
                let arrow_s = get_attr_bool(event, b"arrowshows");
                if arrow_e || arrow_s {
                    return Ok(None);
                }

                let start_x = get_attr_f64(event, b"pntSx", 0.0);
                let start_y = get_attr_f64(event, b"pntSy", 0.0);
                let end_x = get_attr_f64(event, b"pntEx", 0.0);
                let end_y = get_attr_f64(event, b"pntEy", 0.0);
                verts = vec![Point2D::new(start_x, start_y), Point2D::new(end_x, end_y)];
            }
            PrimitiveType::Label => {
                let size_x = get_attr_f64(event, b"sizex", 0.0);
                let size_y = get_attr_f64(event, b"sizey", 0.0);
                // Text anchor at center of label box
                verts = vec![Point2D::new(size_x / 2.0, size_y / 2.0)];
                text = get_attr(event, b"text");
            }
            PrimitiveType::Ellipse => {
                // Extract ellipse center
                let size_x = get_attr_f64(event, b"sizx", 0.0);
                let size_y = get_attr_f64(event, b"sizy", 0.0);
                verts = vec![Point2D::new(size_x / 2.0, size_y / 2.0)];
            }
            PrimitiveType::FlexConcreteBarrier => {
                // Parse like polyline
                if let Some(pnts) = get_attr(event, b"pnts") {
                    verts = parse_vertex_list(&pnts)?;
                }
            }
            PrimitiveType::Unknown => {
                // Skip unknown types
                return Ok(None);
            }
        }

        if verts.is_empty() {
            return Ok(None);
        }

        // Check for line data attributes (dashed, thick)
        // These might be in lndata child element, but we approximate from attributes
        if get_attr(event, b"lt").map(|s| s == "1").unwrap_or(false) {
            dashed = true;
        }
        if get_attr(event, b"thickness")
            .and_then(|s| s.parse::<f64>().ok())
            .map(|t| t > 0.0)
            .unwrap_or(false)
        {
            thick = true;
        }

        let style = LineStyle {
            dashed,
            thick,
            closed,
        };

        Ok(Some(Primitive::new(
            ptype,
            verts,
            global_transform,
            name,
            layer,
            style,
            vehicle2d,
            text,
        )))
    }

    /// Phase 2: Classify and categorize objects
    fn classify_and_categorize(&mut self) -> Result<Scene, FaroError> {
        let mut scene = Scene::default();

        // Move collected data to scene
        scene.symbols = std::mem::take(&mut self.symbols);
        scene.primitives = std::mem::take(&mut self.primitives);

        // Classify symbols
        let mut vehicles: Vec<usize> = Vec::new();
        let mut road_markings: Vec<usize> = Vec::new();
        let mut misc_symbols: Vec<usize> = Vec::new();

        for (idx, symbol) in scene.symbols.iter_mut().enumerate() {
            if self.check_vehicle(symbol) {
                // Collect associated text from nested labels
                for item in &symbol.items {
                    if let SymbolItem::Primitive(p) = item {
                        if p.ptype == PrimitiveType::Label {
                            if let Some(ref txt) = p.text {
                                symbol.associated_text.push(txt.clone());
                            }
                        }
                    }
                }
                vehicles.push(idx);
            } else if symbol.predicted_class == Some(ClassificationLabel::RoadMarking)
                || symbol.predicted_class == Some(ClassificationLabel::TurnDirection)
            {
                road_markings.push(idx);
            } else {
                misc_symbols.push(idx);
            }
        }

        // Categorize primitives
        let mut texts: Vec<usize> = Vec::new();
        let mut roadway: Vec<usize> = Vec::new();
        let mut misc_primitives: Vec<usize> = Vec::new();

        for (idx, primitive) in scene.primitives.iter().enumerate() {
            match primitive.ptype {
                PrimitiveType::Label => texts.push(idx),
                PrimitiveType::Polyline
                | PrimitiveType::Polycurve
                | PrimitiveType::Line
                | PrimitiveType::FlexConcreteBarrier => {
                    // Classify all line-type primitives as roadway
                    // Layer-based classification is commented out in the original notebook
                    roadway.push(idx);
                }
                _ => misc_primitives.push(idx),
            }
        }

        // Associate text to vehicles (spatial join)
        let text_primitives: Vec<_> = texts
            .iter()
            .map(|&idx| &scene.primitives[idx])
            .collect();

        for &veh_idx in &vehicles {
            let vehicle = &mut scene.symbols[veh_idx];
            let veh_center = vehicle.transformed_center;

            for text_prim in &text_primitives {
                let txt_center = text_prim.transformed_center;
                let dist = veh_center.distance_to(txt_center);
                if dist < self.config.text_association_distance {
                    if let Some(ref txt) = text_prim.text {
                        if !vehicle.associated_text.contains(txt) {
                            vehicle.associated_text.push(txt.clone());
                        }
                    }
                }
            }
        }

        scene.vehicle_ids = vehicles;
        scene.roadway_ids = roadway;
        scene.road_marking_ids = road_markings;
        scene.text_ids = texts;
        scene.misc_primitive_ids = misc_primitives;
        scene.misc_symbol_ids = misc_symbols;

        Ok(scene)
    }

    /// Check if a symbol represents a vehicle
    fn check_vehicle(&self, symbol: &mut Symbol) -> bool {
        // Explicit vehicle2d attribute
        if symbol.vehicle2d {
            symbol.predicted_class = Some(ClassificationLabel::Vehicle);
            symbol.predicted_probability = Some(1.0);
            return true;
        }

        // Check name-based heuristics
        if let Some(ref name) = symbol.name {
            let name_lower = name.to_lowercase();
            if name_lower.contains("vehicle")
                || name_lower.contains("car")
                || name_lower.contains("truck")
                || name_lower.contains("suv")
                || name_lower.contains("pickup")
                || name_lower.contains("bus")
                || name_lower.contains("motorcycle")
            {
                symbol.predicted_class = Some(ClassificationLabel::Vehicle);
                symbol.predicted_probability = Some(0.8);
                return true;
            }

            // Check for road markings
            if name_lower.contains("lane")
                || name_lower.contains("marking")
                || name_lower.contains("arrow")
            {
                symbol.predicted_class = Some(ClassificationLabel::RoadMarking);
                symbol.predicted_probability = Some(0.7);
                return false;
            }

            // Check for direction arrows
            if name_lower.contains("north")
                || name_lower.contains("south")
                || name_lower.contains("east")
                || name_lower.contains("west")
                || name_lower.contains("compass")
            {
                symbol.predicted_class = Some(ClassificationLabel::DirectionArrow);
                symbol.predicted_probability = Some(0.7);
                return false;
            }
        }

        // Check nested symbols recursively
        for item in &mut symbol.items {
            if let SymbolItem::Symbol(ref mut nested) = item {
                if self.check_vehicle(nested) {
                    // Propagate classification up
                    if symbol.predicted_class.is_none() {
                        symbol.predicted_class = nested.predicted_class;
                        symbol.predicted_probability = nested.predicted_probability;
                    }
                    return true;
                }
            }
        }

        // Default to background
        symbol.predicted_class = Some(ClassificationLabel::Background);
        symbol.predicted_probability = Some(0.5);
        false
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance_to(p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_identity() {
        let t = Transform2D::identity();
        let p = Point2D::new(5.0, 10.0);
        let result = t.apply(p);
        assert!((result.x - 5.0).abs() < 1e-10);
        assert!((result.y - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_translate() {
        let t = Transform2D::translate(10.0, 20.0);
        let p = Point2D::new(5.0, 5.0);
        let result = t.apply(p);
        assert!((result.x - 15.0).abs() < 1e-10);
        assert!((result.y - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_scale() {
        let t = Transform2D::scale(2.0, 3.0);
        let p = Point2D::new(5.0, 10.0);
        let result = t.apply(p);
        assert!((result.x - 10.0).abs() < 1e-10);
        assert!((result.y - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_composition() {
        // T @ R @ S should match from_attributes
        let manual = Transform2D::from_attributes(10.0, 20.0, 0.5, 2.0, 3.0);

        let s = Transform2D::scale(2.0, 3.0);
        let r = Transform2D::rotate(0.5);
        let t = Transform2D::translate(10.0, 20.0);
        let composed = t.multiply(&r.multiply(&s));

        // Compare matrices element-wise
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (manual.data[i][j] - composed.data[i][j]).abs() < 1e-10,
                    "Mismatch at [{i}][{j}]: {} vs {}",
                    manual.data[i][j],
                    composed.data[i][j]
                );
            }
        }
    }

    #[test]
    fn test_parse_vertex_list() {
        let input = "1.5,2.5,0;3.0,4.0,0;5.5,6.5,0";
        let result = parse_vertex_list(input).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0].x - 1.5).abs() < 1e-10);
        assert!((result[0].y - 2.5).abs() < 1e-10);
        assert!((result[2].x - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_vertex_list_2d() {
        let input = "1.0,2.0;3.0,4.0";
        let result = parse_vertex_list(input).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0].x - 1.0).abs() < 1e-10);
        assert!((result[1].y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_bezier_endpoints() {
        let p0 = Point2D::new(0.0, 0.0);
        let p1 = Point2D::new(10.0, 10.0);
        let c1 = Point2D::new(5.0, 0.0);
        let c2 = Point2D::new(5.0, 10.0);

        let start = bezier_cubic(p0, c1, c2, p1, 0.0);
        let end = bezier_cubic(p0, c1, c2, p1, 1.0);

        assert!((start.x - p0.x).abs() < 1e-10);
        assert!((start.y - p0.y).abs() < 1e-10);
        assert!((end.x - p1.x).abs() < 1e-10);
        assert!((end.y - p1.y).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_polycurve() {
        let anchors = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(20.0, 0.0),
        ];
        let controls = vec![
            Point2D::new(3.0, 5.0),
            Point2D::new(7.0, 5.0),
            Point2D::new(13.0, 5.0),
            Point2D::new(17.0, 5.0),
        ];

        let result = interpolate_polycurve(&anchors, &controls, 5);
        // 2 segments * 5 points - 1 shared = 9 points
        assert_eq!(result.len(), 9);
        assert!((result[0].x - 0.0).abs() < 1e-10);
        assert!((result[8].x - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_from_points() {
        let points = vec![
            Point2D::new(1.0, 2.0),
            Point2D::new(5.0, 8.0),
            Point2D::new(3.0, 4.0),
        ];
        let bbox = BBox::from_points(&points);
        assert!((bbox.min.x - 1.0).abs() < 1e-10);
        assert!((bbox.min.y - 2.0).abs() < 1e-10);
        assert!((bbox.max.x - 5.0).abs() < 1e-10);
        assert!((bbox.max.y - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let config = ParserConfig::default()
            .with_bezier_points_per_segment(20)
            .with_closure_threshold(0.5);

        assert_eq!(config.bezier_points_per_segment, 20);
        assert!((config.closure_threshold - 0.5).abs() < 1e-10);
    }
}
