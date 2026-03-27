use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

mod faro;
mod geometry;
mod classification;

use faro::{FaroParser, ParserConfig};
use geometry::{extract_centerlines, CenterlineConfig};

fn main() {
    // Initialize logging
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_file> [--json] [--centerlines]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --json         Output scene as JSON");
        eprintln!("  --centerlines  Extract and output road centerlines");
        eprintln!("  --bezier N     Set Bezier interpolation points per segment (default: 10)");
        eprintln!("  --road-width MIN MAX  Set road width range for centerlines (default: 5.0 15.0)");
        std::process::exit(1);
    }

    let input_file = &args[1];

    // Check if file exists
    if !Path::new(input_file).exists() {
        eprintln!("Error: File '{}' does not exist", input_file);
        std::process::exit(1);
    }

    // Parse command line options
    let output_json = args.contains(&"--json".to_string());
    let extract_centerlines_flag = args.contains(&"--centerlines".to_string());

    // Parse optional parameters
    let mut bezier_points = 10usize;
    let mut road_width_min = 5.0f64;
    let mut road_width_max = 15.0f64;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--bezier" if i + 1 < args.len() => {
                bezier_points = args[i + 1].parse().unwrap_or(10);
                i += 2;
            }
            "--road-width" if i + 2 < args.len() => {
                road_width_min = args[i + 1].parse().unwrap_or(5.0);
                road_width_max = args[i + 2].parse().unwrap_or(15.0);
                i += 3;
            }
            _ => i += 1,
        }
    }

    println!("Processing file: {}", input_file);
    println!("Bezier points per segment: {}", bezier_points);

    // Configure parser
    let config = ParserConfig::default()
        .with_bezier_points_per_segment(bezier_points);

    // Parse the file
    let start = Instant::now();
    let mut parser = FaroParser::new(config);

    match parser.parse(input_file) {
        Ok(scene) => {
            let parse_duration = start.elapsed();

            println!("\nParsing completed in {:.2?}", parse_duration);
            println!("\n=== Scene Summary ===");
            println!("Symbols: {}", scene.symbols.len());
            println!("Primitives: {}", scene.primitives.len());
            println!("  - Vehicles: {}", scene.vehicle_ids.len());
            println!("  - Roadway segments: {}", scene.roadway_ids.len());
            println!("  - Road markings: {}", scene.road_marking_ids.len());
            println!("  - Text labels: {}", scene.text_ids.len());
            println!("  - Misc primitives: {}", scene.misc_primitive_ids.len());
            println!("  - Misc symbols: {}", scene.misc_symbol_ids.len());

            // Print vehicle details
            if !scene.vehicle_ids.is_empty() {
                println!("\n=== Vehicles ===");
                for (i, veh) in scene.vehicles().enumerate() {
                    let name = veh.name.as_deref().unwrap_or("<unnamed>");
                    let text = if veh.associated_text.is_empty() {
                        String::new()
                    } else {
                        format!(" (text: {})", veh.associated_text.join(", "))
                    };
                    println!(
                        "  {}. {} at ({:.2}, {:.2}){}",
                        i + 1,
                        name,
                        veh.transformed_center.x,
                        veh.transformed_center.y,
                        text
                    );
                }
            }

            // Extract centerlines if requested
            if extract_centerlines_flag {
                println!("\n=== Centerline Extraction ===");
                println!("Road width range: {:.1} - {:.1}", road_width_min, road_width_max);

                let centerline_config = CenterlineConfig::default()
                    .with_road_width_range(road_width_min, road_width_max);

                let roadway_refs: Vec<_> = scene.roadway_primitives().collect();

                let cl_start = Instant::now();
                let result = extract_centerlines(&roadway_refs, &centerline_config);
                let cl_duration = cl_start.elapsed();

                println!("Centerline extraction completed in {:.2?}", cl_duration);
                println!("\nStatistics:");
                println!("  Input points: {}", result.stats.total_input_points);
                println!("  Total triangles: {}", result.stats.total_triangles);
                println!("  Valid triangles: {}", result.stats.valid_triangles);
                println!("  Filtered by size: {}", result.stats.filtered_by_size);
                println!("  Filtered by cluster: {}", result.stats.filtered_by_cluster);
                println!("  Filtered by parallel: {}", result.stats.filtered_by_parallel);
                println!("  Skeleton segments: {}", result.stats.skeleton_segments);
                println!("  Centerlines found: {}", result.stats.centerline_count);
                println!("  Total centerline length: {:.2}", result.stats.total_centerline_length);

                #[cfg(feature = "serde")]
                if output_json {
                    if let Ok(json) = serde_json::to_string_pretty(&result) {
                        let output_path = format!("{}.centerlines.json", input_file);
                        if let Err(e) = fs::write(&output_path, json) {
                            eprintln!("Error writing centerlines JSON: {}", e);
                        } else {
                            println!("\nCenterlines written to: {}", output_path);
                        }
                    }
                }
            }

            // Output JSON if requested
            #[cfg(feature = "serde")]
            if output_json {
                if let Ok(json) = serde_json::to_string_pretty(&scene) {
                    let output_path = format!("{}.json", input_file);
                    if let Err(e) = fs::write(&output_path, json) {
                        eprintln!("Error writing JSON: {}", e);
                    } else {
                        println!("\nScene JSON written to: {}", output_path);
                    }
                }
            }

            #[cfg(not(feature = "serde"))]
            if output_json {
                eprintln!("Warning: JSON output requires the 'serde' feature");
            }
        }
        Err(e) => {
            eprintln!("Error parsing file: {}", e);
            std::process::exit(1);
        }
    }
}
