//! Delaunay-based Road Centerline Detection
//!
//! This module implements road centerline extraction using Delaunay triangulation,
//! based on the algorithm from the Python data_exploration notebook.
//!
//! # Algorithm Overview
//!
//! 1. Resample roadway polylines to uniform point spacing
//! 2. Compute Delaunay triangulation of all roadway points
//! 3. Filter triangles by:
//!    - Edge length (must be within road width range)
//!    - Vertex clustering (avoid problematic convergence points)
//!    - Parallel edge detection (triangle edges parallel to roadway edges)
//! 4. Extract skeleton by connecting triangle centroids
//! 5. Connect midpoints into continuous centerline polylines using BFS

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::faro::{Point2D, Primitive};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for centerline detection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CenterlineConfig {
    /// Min/max road width for triangle filtering (default: 5.0, 15.0)
    pub road_width_range: (f64, f64),

    /// Resampling step distance (default: 1.0)
    pub resample_step: f64,

    /// Vertex cluster threshold - triangles with vertices in >N triangles are filtered (default: 10)
    pub vertex_cluster_threshold: usize,

    /// Angle epsilon for parallel edge detection in degrees (default: 15.0)
    pub parallel_angle_epsilon: f64,

    /// Distance threshold for connecting centerline points (default: 8.0)
    pub connection_distance: f64,

    /// Angle threshold for BFS connection in degrees (default: 15.0)
    pub connection_angle_threshold: f64,

    /// Minimum polyline length to keep (default: 5.0)
    pub minimum_polyline_length: f64,
}

impl Default for CenterlineConfig {
    fn default() -> Self {
        Self {
            road_width_range: (5.0, 15.0),
            resample_step: 1.0,
            vertex_cluster_threshold: 10,
            parallel_angle_epsilon: 15.0,
            connection_distance: 8.0,
            connection_angle_threshold: 15.0,
            minimum_polyline_length: 5.0,
        }
    }
}

impl CenterlineConfig {
    /// Set road width range
    pub fn with_road_width_range(mut self, min: f64, max: f64) -> Self {
        self.road_width_range = (min, max);
        self
    }

    /// Set resampling step
    pub fn with_resample_step(mut self, step: f64) -> Self {
        self.resample_step = step;
        self
    }

    /// Set vertex cluster threshold
    pub fn with_vertex_cluster_threshold(mut self, threshold: usize) -> Self {
        self.vertex_cluster_threshold = threshold;
        self
    }

    /// Set parallel angle epsilon (degrees)
    pub fn with_parallel_angle_epsilon(mut self, epsilon: f64) -> Self {
        self.parallel_angle_epsilon = epsilon;
        self
    }

    /// Set connection distance
    pub fn with_connection_distance(mut self, distance: f64) -> Self {
        self.connection_distance = distance;
        self
    }

    /// Set connection angle threshold (degrees)
    pub fn with_connection_angle_threshold(mut self, threshold: f64) -> Self {
        self.connection_angle_threshold = threshold;
        self
    }

    /// Set minimum polyline length
    pub fn with_minimum_polyline_length(mut self, length: f64) -> Self {
        self.minimum_polyline_length = length;
        self
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of centerline extraction
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CenterlineResult {
    /// Extracted centerline polylines
    pub centerlines: Vec<Vec<Point2D>>,

    /// Debug: all triangle centroid connections (skeleton segments)
    pub centroid_segments: Vec<(Point2D, Point2D)>,

    /// Debug: midpoints before connection
    pub midpoints: Vec<Point2D>,

    /// Debug: outlier points that couldn't be connected
    pub outliers: Vec<Point2D>,

    /// Statistics
    pub stats: CenterlineStats,
}

/// Statistics about the centerline extraction
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CenterlineStats {
    pub total_input_points: usize,
    pub total_triangles: usize,
    pub valid_triangles: usize,
    pub filtered_by_size: usize,
    pub filtered_by_cluster: usize,
    pub filtered_by_parallel: usize,
    pub skeleton_segments: usize,
    pub centerline_count: usize,
    pub total_centerline_length: f64,
}

// ============================================================================
// Geometry Utilities
// ============================================================================

/// Resample polyline to uniform point spacing
pub fn resample_polyline(points: &[Point2D], step: f64) -> Vec<Point2D> {
    if points.len() < 2 || step <= 0.0 {
        return points.to_vec();
    }

    let mut result = vec![points[0]];

    for i in 1..points.len() {
        let p1 = points[i - 1];
        let p2 = points[i];
        let dist = p1.distance_to(p2);

        if dist > step {
            let n = (dist / step).ceil() as usize;
            for j in 1..=n {
                let t = j as f64 / n as f64;
                result.push(Point2D::new(
                    p1.x + t * (p2.x - p1.x),
                    p1.y + t * (p2.y - p1.y),
                ));
            }
        } else {
            result.push(p2);
        }
    }

    result
}

/// Compute angle of line segment (in radians)
#[inline]
fn compute_edge_angle(start: Point2D, end: Point2D) -> f64 {
    (end.y - start.y).atan2(end.x - start.x)
}

/// Check if two angles are parallel (within epsilon radians)
#[inline]
fn angles_are_parallel(angle1: f64, angle2: f64, epsilon_rad: f64) -> bool {
    let diff = (angle1 - angle2).sin().atan2((angle1 - angle2).cos());
    let abs_diff = diff.abs();
    abs_diff < epsilon_rad || abs_diff > (std::f64::consts::PI - epsilon_rad)
}

/// Point to line segment distance
#[inline]
fn point_to_segment_distance(p: Point2D, a: Point2D, b: Point2D) -> f64 {
    let edge_x = b.x - a.x;
    let edge_y = b.y - a.y;
    let len_sq = edge_x * edge_x + edge_y * edge_y;

    if len_sq < 1e-12 {
        return p.distance_to(a);
    }

    let t = ((p.x - a.x) * edge_x + (p.y - a.y) * edge_y) / len_sq;
    let t = t.clamp(0.0, 1.0);

    let closest = Point2D::new(a.x + t * edge_x, a.y + t * edge_y);
    p.distance_to(closest)
}

/// Counter-clockwise test for three points
#[inline]
fn ccw(a: Point2D, b: Point2D, c: Point2D) -> bool {
    (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)
}

/// Check if two line segments intersect
#[inline]
fn segments_intersect(a1: Point2D, a2: Point2D, b1: Point2D, b2: Point2D) -> bool {
    ccw(a1, b1, b2) != ccw(a2, b1, b2) && ccw(a1, a2, b1) != ccw(a1, a2, b2)
}

// ============================================================================
// Roadway Edge Structure
// ============================================================================

/// Edge of a roadway polyline
#[derive(Debug, Clone, Copy)]
struct RoadwayEdge {
    start: Point2D,
    end: Point2D,
    angle: f64,
    roadway_idx: usize,
}

/// Spatial index entry for roadway edges
struct EdgeIndex {
    edges: Vec<RoadwayEdge>,
    // Simple grid-based spatial index
    grid: HashMap<(i32, i32), Vec<usize>>,
    cell_size: f64,
}

impl EdgeIndex {
    fn new(edges: Vec<RoadwayEdge>, cell_size: f64) -> Self {
        let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();

        for (idx, edge) in edges.iter().enumerate() {
            let min_x = edge.start.x.min(edge.end.x);
            let max_x = edge.start.x.max(edge.end.x);
            let min_y = edge.start.y.min(edge.end.y);
            let max_y = edge.start.y.max(edge.end.y);

            let start_cell_x = (min_x / cell_size).floor() as i32;
            let end_cell_x = (max_x / cell_size).floor() as i32;
            let start_cell_y = (min_y / cell_size).floor() as i32;
            let end_cell_y = (max_y / cell_size).floor() as i32;

            for cx in start_cell_x..=end_cell_x {
                for cy in start_cell_y..=end_cell_y {
                    grid.entry((cx, cy)).or_default().push(idx);
                }
            }
        }

        Self {
            edges,
            grid,
            cell_size,
        }
    }

    /// Find nearest edge to a point within search distance
    fn find_nearest(&self, point: Point2D, roadway_idx: usize, search_dist: f64) -> Option<&RoadwayEdge> {
        let cx = (point.x / self.cell_size).floor() as i32;
        let cy = (point.y / self.cell_size).floor() as i32;
        let cell_range = (search_dist / self.cell_size).ceil() as i32 + 1;

        let mut best_dist = search_dist;
        let mut best_edge = None;

        for dx in -cell_range..=cell_range {
            for dy in -cell_range..=cell_range {
                if let Some(indices) = self.grid.get(&(cx + dx, cy + dy)) {
                    for &idx in indices {
                        let edge = &self.edges[idx];
                        if edge.roadway_idx == roadway_idx {
                            let dist = point_to_segment_distance(point, edge.start, edge.end);
                            if dist < best_dist {
                                best_dist = dist;
                                best_edge = Some(edge);
                            }
                        }
                    }
                }
            }
        }

        best_edge
    }

    /// Check if a segment crosses any edge from the given roadways
    fn crosses_roadway(&self, a: Point2D, b: Point2D, relevant_roadways: &HashSet<usize>) -> bool {
        let min_x = a.x.min(b.x);
        let max_x = a.x.max(b.x);
        let min_y = a.y.min(b.y);
        let max_y = a.y.max(b.y);

        let start_cell_x = (min_x / self.cell_size).floor() as i32;
        let end_cell_x = (max_x / self.cell_size).floor() as i32;
        let start_cell_y = (min_y / self.cell_size).floor() as i32;
        let end_cell_y = (max_y / self.cell_size).floor() as i32;

        for cx in start_cell_x..=end_cell_x {
            for cy in start_cell_y..=end_cell_y {
                if let Some(indices) = self.grid.get(&(cx, cy)) {
                    for &idx in indices {
                        let edge = &self.edges[idx];
                        if relevant_roadways.contains(&edge.roadway_idx) {
                            if segments_intersect(a, b, edge.start, edge.end) {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        false
    }
}

// ============================================================================
// Simple Delaunay Triangulation
// ============================================================================

/// Triangle from Delaunay triangulation
#[derive(Debug, Clone, Copy)]
struct Triangle {
    /// Indices into the points array
    vertices: [usize; 3],
    /// Indices of neighboring triangles (-1 if no neighbor)
    neighbors: [i32; 3],
}

/// Simple Delaunay triangulation using Bowyer-Watson algorithm
fn compute_delaunay(points: &[Point2D]) -> Vec<Triangle> {
    if points.len() < 3 {
        return Vec::new();
    }

    // Use spade crate for Delaunay triangulation
    use spade::DelaunayTriangulation;
    use spade::Triangulation;

    let mut triangulation: DelaunayTriangulation<spade::Point2<f64>> =
        DelaunayTriangulation::new();

    // Insert points - store mapping from spade vertex index to our point index
    let mut spade_to_orig: HashMap<usize, usize> = HashMap::new();
    for (idx, p) in points.iter().enumerate() {
        let handle = triangulation.insert(spade::Point2::new(p.x, p.y));
        if let Ok(h) = handle {
            spade_to_orig.insert(h.index(), idx);
        }
    }

    // Extract triangles
    let mut triangles = Vec::new();
    for face in triangulation.inner_faces() {
        let vertices = face.vertices();
        // vertices() returns [VertexHandle; 3]
        let v0 = spade_to_orig.get(&vertices[0].index()).copied();
        let v1 = spade_to_orig.get(&vertices[1].index()).copied();
        let v2 = spade_to_orig.get(&vertices[2].index()).copied();

        if let (Some(v0), Some(v1), Some(v2)) = (v0, v1, v2) {
            triangles.push(Triangle {
                vertices: [v0, v1, v2],
                neighbors: [-1, -1, -1], // We'll compute neighbors separately if needed
            });
        }
    }

    // Build neighbor relationships using edge-to-triangles map
    // Each edge can be shared by at most 2 triangles
    let mut edge_to_tris: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for (tri_idx, tri) in triangles.iter().enumerate() {
        for i in 0..3 {
            let v0 = tri.vertices[i];
            let v1 = tri.vertices[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            edge_to_tris
                .entry(edge)
                .or_insert_with(Vec::new)
                .push((tri_idx, i));
        }
    }

    // For each edge shared by two triangles, set them as neighbors
    for (_edge, tris) in edge_to_tris.iter() {
        if tris.len() == 2 {
            let (t1, edge1) = tris[0];
            let (t2, edge2) = tris[1];
            triangles[t1].neighbors[edge1] = t2 as i32;
            triangles[t2].neighbors[edge2] = t1 as i32;
        }
    }

    triangles
}

// ============================================================================
// Main Centerline Extraction
// ============================================================================

/// Extract centerlines from roadway primitives
///
/// # Arguments
/// * `roadway` - Slice of roadway primitives
/// * `config` - Centerline detection configuration
///
/// # Returns
/// CenterlineResult containing extracted centerlines and debug information
pub fn extract_centerlines(roadway: &[&Primitive], config: &CenterlineConfig) -> CenterlineResult {
    let mut result = CenterlineResult::default();

    if roadway.is_empty() {
        return result;
    }

    // 1. Collect and resample all roadway points
    let mut all_points: Vec<Point2D> = Vec::new();
    let mut point_to_roadway: Vec<usize> = Vec::new();
    let mut roadway_edges: Vec<RoadwayEdge> = Vec::new();

    for (roadway_idx, prim) in roadway.iter().enumerate() {
        let verts = &prim.verts;
        if verts.len() < 2 {
            continue;
        }

        // Collect edges
        for i in 0..verts.len() - 1 {
            let start = verts[i];
            let end = verts[i + 1];
            roadway_edges.push(RoadwayEdge {
                start,
                end,
                angle: compute_edge_angle(start, end),
                roadway_idx,
            });
        }

        // Resample and collect points
        let resampled = resample_polyline(verts, config.resample_step);
        for pt in resampled {
            all_points.push(pt);
            point_to_roadway.push(roadway_idx);
        }
    }

    result.stats.total_input_points = all_points.len();

    if all_points.len() < 3 {
        return result;
    }

    // 2. Build edge spatial index
    let edge_index = EdgeIndex::new(roadway_edges, config.road_width_range.1 * 2.0);

    // 3. Compute Delaunay triangulation
    let triangles = compute_delaunay(&all_points);
    result.stats.total_triangles = triangles.len();

    if triangles.is_empty() {
        return result;
    }

    // 4. Compute triangle edge lengths and filter
    let (min_width, max_width) = config.road_width_range;
    let epsilon_rad = config.parallel_angle_epsilon.to_radians();

    // Count triangles per vertex for cluster detection
    let mut vertex_tri_count: HashMap<usize, usize> = HashMap::new();
    for tri in &triangles {
        for &v in &tri.vertices {
            *vertex_tri_count.entry(v).or_insert(0) += 1;
        }
    }

    let problematic_vertices: HashSet<usize> = vertex_tri_count
        .iter()
        .filter(|(_, &count)| count >= config.vertex_cluster_threshold)
        .map(|(&v, _)| v)
        .collect();

    // Filter triangles
    let mut valid_triangles: Vec<(usize, Point2D)> = Vec::new(); // (index, centroid)

    for (tri_idx, tri) in triangles.iter().enumerate() {
        let p0 = all_points[tri.vertices[0]];
        let p1 = all_points[tri.vertices[1]];
        let p2 = all_points[tri.vertices[2]];

        // Compute edge lengths
        let d01 = p0.distance_to(p1);
        let d12 = p1.distance_to(p2);
        let d20 = p2.distance_to(p0);
        let max_edge = d01.max(d12).max(d20);

        // Filter by size
        if max_edge < min_width || max_edge > max_width {
            result.stats.filtered_by_size += 1;
            continue;
        }

        // Filter by cluster
        let is_clustered = tri
            .vertices
            .iter()
            .any(|v| problematic_vertices.contains(v));
        if is_clustered {
            result.stats.filtered_by_cluster += 1;
            continue;
        }

        // Check roadway diversity (vertices from at least 2 different roadways)
        let roadway_set: HashSet<_> = tri
            .vertices
            .iter()
            .map(|&v| point_to_roadway[v])
            .collect();
        if roadway_set.len() < 2 {
            result.stats.filtered_by_parallel += 1;
            continue;
        }

        // Check parallel edges
        let mut is_parallel = false;
        let mut vertex_angles: HashMap<usize, Vec<f64>> = HashMap::new();

        for &v in &tri.vertices {
            let point = all_points[v];
            let roadway_idx = point_to_roadway[v];

            if let Some(edge) = edge_index.find_nearest(point, roadway_idx, 2.0) {
                vertex_angles
                    .entry(roadway_idx)
                    .or_default()
                    .push(edge.angle);
            }
        }

        // Check if angles from different roadways are parallel
        if vertex_angles.len() >= 2 {
            let roadway_mean_angles: Vec<f64> = vertex_angles
                .values()
                .map(|angles| {
                    let sum: f64 = angles.iter().sum();
                    sum / angles.len() as f64
                })
                .collect();

            for i in 0..roadway_mean_angles.len() {
                for j in (i + 1)..roadway_mean_angles.len() {
                    if angles_are_parallel(roadway_mean_angles[i], roadway_mean_angles[j], epsilon_rad)
                    {
                        is_parallel = true;
                        break;
                    }
                }
                if is_parallel {
                    break;
                }
            }
        }

        if !is_parallel {
            result.stats.filtered_by_parallel += 1;
            continue;
        }

        // Triangle passed all filters
        let centroid = Point2D::new((p0.x + p1.x + p2.x) / 3.0, (p0.y + p1.y + p2.y) / 3.0);
        valid_triangles.push((tri_idx, centroid));
    }

    result.stats.valid_triangles = valid_triangles.len();

    // 5. Extract skeleton segments (centroid connections)
    let valid_set: HashSet<usize> = valid_triangles.iter().map(|(idx, _)| *idx).collect();
    let centroid_map: HashMap<usize, Point2D> = valid_triangles.iter().copied().collect();

    for (tri_idx, centroid) in &valid_triangles {
        let tri = &triangles[*tri_idx];

        for &neighbor_idx in &tri.neighbors {
            if neighbor_idx < 0 {
                continue;
            }
            let neighbor_idx = neighbor_idx as usize;

            // Only process each pair once (smaller index first)
            if neighbor_idx <= *tri_idx {
                continue;
            }

            if !valid_set.contains(&neighbor_idx) {
                continue;
            }

            let neighbor_centroid = centroid_map[&neighbor_idx];

            // Check distance
            if centroid.distance_to(neighbor_centroid) > max_width / 2.0 {
                continue;
            }

            // Check if segment crosses any roadway edge
            let tri_roadways: HashSet<_> = tri
                .vertices
                .iter()
                .map(|&v| point_to_roadway[v])
                .collect();
            let neighbor_roadways: HashSet<_> = triangles[neighbor_idx]
                .vertices
                .iter()
                .map(|&v| point_to_roadway[v])
                .collect();
            let relevant_roadways: HashSet<_> = tri_roadways.union(&neighbor_roadways).copied().collect();

            if !edge_index.crosses_roadway(*centroid, neighbor_centroid, &relevant_roadways) {
                result.centroid_segments.push((*centroid, neighbor_centroid));
            }
        }
    }

    result.stats.skeleton_segments = result.centroid_segments.len();

    // 6. Compute midpoints
    let midpoints: Vec<Point2D> = result
        .centroid_segments
        .iter()
        .map(|(a, b)| Point2D::new((a.x + b.x) / 2.0, (a.y + b.y) / 2.0))
        .collect();

    result.midpoints = midpoints.clone();

    // 7. Connect midpoints into polylines
    if !midpoints.is_empty() {
        let (centerlines, outliers) = connect_midpoints(
            &midpoints,
            config.connection_distance,
            config.connection_angle_threshold,
            config.minimum_polyline_length,
        );

        result.centerlines = centerlines;
        result.outliers = outliers;
    }

    result.stats.centerline_count = result.centerlines.len();
    result.stats.total_centerline_length = result
        .centerlines
        .iter()
        .map(|line| {
            line.windows(2)
                .map(|w| w[0].distance_to(w[1]))
                .sum::<f64>()
        })
        .sum();

    result
}

// ============================================================================
// Midpoint Connection (BFS)
// ============================================================================

/// Connect midpoints into continuous polylines using BFS
fn connect_midpoints(
    points: &[Point2D],
    distance_threshold: f64,
    angle_threshold_deg: f64,
    min_length: f64,
) -> (Vec<Vec<Point2D>>, Vec<Point2D>) {
    if points.len() < 2 {
        return (Vec::new(), points.to_vec());
    }

    let epsilon_rad = angle_threshold_deg.to_radians();
    let dist_sq_threshold = distance_threshold * distance_threshold;

    // Build adjacency list using simple distance check
    // For larger datasets, this should use a KD-tree
    let mut adjacency: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); points.len()];

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dist_sq = points[i].distance_sq(points[j]);
            if dist_sq < dist_sq_threshold {
                let dist = dist_sq.sqrt();
                let angle = compute_edge_angle(points[i], points[j]);
                adjacency[i].push((j, dist, angle));
                adjacency[j].push((i, dist, angle + std::f64::consts::PI));
            }
        }
    }

    // BFS to build polylines
    let mut visited: HashSet<usize> = HashSet::new();
    let mut polylines: Vec<Vec<Point2D>> = Vec::new();

    for start in 0..points.len() {
        if visited.contains(&start) {
            continue;
        }

        let segment = bfs_connect(start, &adjacency, &mut visited, points, epsilon_rad);

        if segment.len() >= 2 {
            let length: f64 = segment
                .windows(2)
                .map(|w| w[0].distance_to(w[1]))
                .sum();

            if length >= min_length {
                // Filter by average segment length (avoid noisy segments)
                let avg_len = length / (segment.len() - 1) as f64;
                if avg_len <= distance_threshold {
                    polylines.push(segment);
                }
            }
        }
    }

    let outliers: Vec<Point2D> = (0..points.len())
        .filter(|i| !visited.contains(i))
        .map(|i| points[i])
        .collect();

    (polylines, outliers)
}

/// BFS connection with angle-based filtering
fn bfs_connect(
    start: usize,
    adjacency: &[Vec<(usize, f64, f64)>],
    visited: &mut HashSet<usize>,
    points: &[Point2D],
    epsilon_rad: f64,
) -> Vec<Point2D> {
    let mut segment: VecDeque<usize> = VecDeque::new();
    segment.push_back(start);
    visited.insert(start);

    // Forward BFS
    let mut current = start;
    loop {
        let mut candidates: BinaryHeap<std::cmp::Reverse<(i64, usize)>> = adjacency[current]
            .iter()
            .filter(|(nbr, _, _)| !visited.contains(nbr))
            .filter(|(_, _, angle)| {
                if segment.len() <= 1 {
                    return true;
                }
                let prev = segment[segment.len() - 2];
                let prev_angle =
                    compute_edge_angle(points[prev], points[current]);
                angles_are_parallel(prev_angle, *angle, epsilon_rad)
            })
            .map(|(nbr, dist, _)| std::cmp::Reverse(((*dist * 1000.0) as i64, *nbr)))
            .collect();

        if let Some(std::cmp::Reverse((_, next))) = candidates.pop() {
            visited.insert(next);
            segment.push_back(next);
            current = next;
        } else {
            break;
        }
    }

    // Backward BFS (prepend to deque)
    if segment.len() >= 2 {
        let forward_angle =
            compute_edge_angle(points[segment[0]], points[segment[1]]);
        current = start;

        loop {
            let mut candidates: BinaryHeap<std::cmp::Reverse<(i64, usize)>> = adjacency[current]
                .iter()
                .filter(|(nbr, _, _)| !visited.contains(nbr))
                .filter(|(_, _, angle)| {
                    // Want opposite direction for prepending
                    let diff = (forward_angle - *angle).sin().atan2((forward_angle - *angle).cos());
                    diff.abs() > (std::f64::consts::PI - epsilon_rad)
                })
                .map(|(nbr, dist, _)| std::cmp::Reverse(((*dist * 1000.0) as i64, *nbr)))
                .collect();

            if let Some(std::cmp::Reverse((_, next))) = candidates.pop() {
                visited.insert(next);
                segment.push_front(next);
                current = next;
            } else {
                break;
            }
        }
    }

    segment.iter().map(|&i| points[i]).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_polyline() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
        ];
        let resampled = resample_polyline(&points, 2.0);
        assert!(resampled.len() >= 5);
        assert!((resampled[0].x - 0.0).abs() < 1e-10);
        assert!((resampled.last().unwrap().x - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_angles_are_parallel() {
        let epsilon = 15.0_f64.to_radians();

        // Same angle
        assert!(angles_are_parallel(0.0, 0.0, epsilon));

        // Opposite directions (parallel)
        assert!(angles_are_parallel(0.0, std::f64::consts::PI, epsilon));

        // Perpendicular (not parallel)
        assert!(!angles_are_parallel(0.0, std::f64::consts::FRAC_PI_2, epsilon));
    }

    #[test]
    fn test_point_to_segment_distance() {
        let p = Point2D::new(5.0, 5.0);
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(10.0, 0.0);

        let dist = point_to_segment_distance(p, a, b);
        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_segments_intersect() {
        let a1 = Point2D::new(0.0, 0.0);
        let a2 = Point2D::new(10.0, 10.0);
        let b1 = Point2D::new(0.0, 10.0);
        let b2 = Point2D::new(10.0, 0.0);

        assert!(segments_intersect(a1, a2, b1, b2));

        // Non-intersecting
        let c1 = Point2D::new(0.0, 0.0);
        let c2 = Point2D::new(5.0, 0.0);
        let d1 = Point2D::new(0.0, 5.0);
        let d2 = Point2D::new(5.0, 5.0);

        assert!(!segments_intersect(c1, c2, d1, d2));
    }

    #[test]
    fn test_config_builder() {
        let config = CenterlineConfig::default()
            .with_road_width_range(8.0, 100.0)
            .with_vertex_cluster_threshold(15);

        assert!((config.road_width_range.0 - 8.0).abs() < 1e-10);
        assert!((config.road_width_range.1 - 100.0).abs() < 1e-10);
        assert_eq!(config.vertex_cluster_threshold, 15);
    }
}
