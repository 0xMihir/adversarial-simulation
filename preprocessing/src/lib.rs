//! FARO Scene Graph Processing Library
//!
//! This library provides tools for parsing FARO (.far) files and extracting
//! road centerlines using Delaunay triangulation.
//!
//! # Features
//!
//! - **FARO Parsing**: Parse NHTSA CISS crash scene diagrams
//! - **Geometry Types**: Efficient 2D point, bounding box, and transform types
//! - **Centerline Detection**: Extract road centerlines from scene geometry
//!
//! # Example
//!
//! ```no_run
//! use preprocessing::faro::{FaroParser, ParserConfig};
//!
//! let config = ParserConfig::default()
//!     .with_bezier_points_per_segment(10);
//!
//! let mut parser = FaroParser::new(config);
//! let scene = parser.parse("scene.far").unwrap();
//!
//! println!("Vehicles: {}", scene.vehicle_ids.len());
//! println!("Roadway segments: {}", scene.roadway_ids.len());
//! ```

pub mod faro;
pub mod geometry;

// Re-export commonly used types at crate root
pub use faro::{
    BBox, ClassificationLabel, FaroError, FaroParser, LineStyle, ParserConfig, Point2D,
    Primitive, PrimitiveType, Scene, Symbol, SymbolItem, Transform2D,
};

pub use geometry::{CenterlineConfig, CenterlineResult};
