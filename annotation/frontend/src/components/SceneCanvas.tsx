import * as d3 from "d3";
import { useEffect, useRef } from "react";
import { colors, statusColors } from "../lib/theme";
import type {
  CaseAnnotation,
  LayerVisibility,
  MouseMode,
  SelectedElement,
} from "../lib/types";

// Tab10 color palette for vehicles
const TAB10 = d3.schemeTableau10;

interface SceneCanvasProps {
  annotation: CaseAnnotation;
  layers: LayerVisibility;
  mode: MouseMode;
  selected: SelectedElement;
  onSelect: (sel: SelectedElement) => void;
  onAddConnection?: (fromId: string, toId: string, fromEnd: "start" | "end", toEnd: "start" | "end") => void;
  onHideRoadEdge?: (id: string) => void;
  width?: number;
  height?: number;
}

export default function SceneCanvas({
  annotation,
  layers,
  mode,
  selected,
  onSelect,
  onAddConnection,
  onHideRoadEdge,
}: SceneCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const connectRef = useRef<string | null>(null); // first clicked endpoint in connect mode
  const transformRef = useRef<d3.ZoomTransform | null>(null); // persisted zoom/pan state
  const prevCaseIdRef = useRef<string | null>(null); // detect scene change

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const isFirefox = /firefox/i.test(navigator.userAgent);

    // Clear previous render
    svg.selectAll("*").remove();

    svg
      .attr("shape-rendering", "geometricPrecision")
      .attr("text-rendering", "geometricPrecision")
      .style("image-rendering", isFirefox ? "-moz-crisp-edges" : "auto");

    const width = svgRef.current.clientWidth || 1200;
    const height = svgRef.current.clientHeight || 800;

    // Compute bounding box of all roadway elements to set initial viewbox
    const scene = annotation.scene;
    const allPts = scene.elements.flatMap((e) =>
      e.resampled_points.map((p) => [p.x, p.y] as [number, number])
    );

    const xExtent = d3.extent(allPts, (d) => d[0]) as [number, number];
    const yExtent = d3.extent(allPts, (d) => d[1]) as [number, number];
    const xRange = (xExtent[1] - xExtent[0]) || 100;
    const yRange = (yExtent[1] - yExtent[0]) || 100;
    const padding = Math.max(xRange, yRange) * 0.1;

    // SVG coordinate system: flip Y (FARO uses math-style Y-up, SVG is Y-down)
    // Use a uniform scale so the data aspect ratio is preserved across screen sizes.
    const dataXRange = xRange + 2 * padding;
    const dataYRange = yRange + 2 * padding;
    const scalePerUnit = Math.min(width / dataXRange, height / dataYRange);
    const scaledW = dataXRange * scalePerUnit;
    const scaledH = dataYRange * scalePerUnit;
    const offsetX = (width - scaledW) / 2;
    const offsetY = (height - scaledH) / 2;

    const xScale = d3
      .scaleLinear()
      .domain([xExtent[0] - padding, xExtent[1] + padding])
      .range([offsetX, offsetX + scaledW]);
    const yScale = d3
      .scaleLinear()
      .domain([yExtent[1] + padding, yExtent[0] - padding]) // flip Y
      .range([offsetY, offsetY + scaledH]);

    const toSvg = (pt: { x: number; y: number }) => ({
      x: xScale(pt.x),
      y: yScale(pt.y),
    });

    // Root group for zoom/pan
    const root = svg.append("g").attr("class", "root");

    // --- Grid background ---
    const gridStep = Math.max(xRange, yRange) / 20;
    const gridG = root.append("g").attr("class", "grid");
    for (let gx = Math.floor(xExtent[0] / gridStep) * gridStep; gx <= xExtent[1] + padding; gx += gridStep) {
      gridG.append("line")
        .attr("x1", xScale(gx)).attr("x2", xScale(gx))
        .attr("y1", 0).attr("y2", height)
        .attr("stroke", colors.canvas.grid).attr("stroke-width", 0.5);
    }
    for (let gy = Math.floor(yExtent[0] / gridStep) * gridStep; gy <= yExtent[1] + padding; gy += gridStep) {
      gridG.append("line")
        .attr("x1", 0).attr("x2", width)
        .attr("y1", yScale(gy)).attr("y2", yScale(gy))
        .attr("stroke", colors.canvas.grid).attr("stroke-width", 0.5);
    }

    const lineGen = (pts: { x: number; y: number }[]) =>
      d3.line<{ x: number; y: number }>()
        .x((d) => toSvg(d).x)
        .y((d) => toSvg(d).y)(pts) ?? "";

    // --- Layer 1: Road signs (embedded images) ---
    if (layers.images) {
      const imgG = root.append("g").attr("class", "road-signs");
      for (const img of scene.images ?? []) {
        if (!img.img) continue;
        const cx = xScale(img.center.x);
        const cy = yScale(img.center.y);
        const svgW = Math.abs(xScale(img.center.x + img.sizx / 2) - xScale(img.center.x - img.sizx / 2));
        const svgH = Math.abs(yScale(img.center.y - img.sizy / 2) - yScale(img.center.y + img.sizy / 2));
        if (svgW === 0 || svgH === 0) continue;
        const rotDeg = -(img.oriz * 180 / Math.PI); // negate for Y-axis flip
        imgG.append("image")
          .attr("href", `data:image/png;base64,${img.img}`)
          .attr("x", cx - svgW / 2)
          .attr("y", cy - svgH / 2)
          .attr("width", svgW)
          .attr("height", svgH)
          .attr("transform", `rotate(${rotDeg}, ${cx}, ${cy})`)
          .attr("preserveAspectRatio", "none")
          .style("image-rendering", isFirefox ? "-moz-crisp-edges" : "auto");
      }
    }

    // --- Layer 2: Roadway polylines ---
    if (layers.roadway) {
      const roadG = root.append("g").attr("class", "roadway");
      const hiddenIds = new Set(annotation.hidden_roadway_ids ?? []);
      for (const idx of scene.roadway_indices) {
        const elem = scene.elements[idx];
        if (hiddenIds.has(elem.id)) continue;
        const pts = elem.resampled_points;
        if (pts.length < 2) continue;
        const isSelected = selected?.kind === "roadway" && selected.id === elem.id;
        roadG.append("path")
          .attr("d", lineGen(pts))
          .attr("fill", "none")
          .attr("stroke", isSelected ? colors.accent.selected : colors.canvas.roadway)
          .attr("stroke-width", isSelected ? 2.5 : 1.5)
          .attr("stroke-dasharray", elem.is_dashed ? "5, 10" : "none");

        if (mode === "edit") {
          // Invisible wide hit area for easier clicking
          roadG.append("path")
            .attr("d", lineGen(pts))
            .attr("fill", "none")
            .attr("stroke", colors.util.transparent)
            .attr("stroke-width", 12)
            .attr("cursor", "pointer")
            .on("click", (event) => {
              event.stopPropagation();
              onSelect({ kind: "roadway", id: elem.id });
            });
        }
      }
    }

    // --- Layer 3: Road markings ---
    if (layers.road_markings) {
      const markG = root.append("g").attr("class", "road-markings");
      for (const idx of scene.road_marking_indices) {
        const elem = scene.elements[idx];
        const pts = elem.resampled_points;
        if (pts.length < 2) continue;
        markG.append("path")
          .attr("d", lineGen(pts))
          .attr("fill", "none")
          .attr("stroke", colors.canvas.roadMarking)
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", elem.is_dashed ? "5, 10" : "none")
          .attr("opacity", 0.7);
      }
    }

    // --- Layer 4: Centerlines / Lanes ---
    if (layers.centerlines) {
      const laneG = root.append("g").attr("class", "lanes");
      for (const lane of annotation.lanes) {
        const isSelected = selected?.kind === "lane" && selected.id === lane.id;
        const isRejected = lane.status === "rejected";
        if (isRejected && !isSelected) continue;

        const strokeColor = isRejected
          ? colors.accent.danger
          : isSelected
            ? colors.accent.selected
            : statusColors.element[lane.status];
        const strokeOpacity = isRejected ? 0.35 : 0.9;

        const path = laneG.append("path")
          .attr("d", lineGen(lane.polyline))
          .attr("fill", "none")
          .attr("stroke", strokeColor)
          .attr("stroke-width", isSelected ? 3 : 1.5)
          .attr("opacity", strokeOpacity)
          .attr("cursor", "pointer")
          .attr("data-id", lane.id)
          .attr("data-kind", "lane");

        path.on("click", (event) => {
          event.stopPropagation();
          if (mode === "select") {
            onSelect({ kind: "lane", id: lane.id });
          }
        });

        // Invisible wide hit area for easier clicking
        laneG.append("path")
          .attr("d", lineGen(lane.polyline))
          .attr("fill", "none")
          .attr("stroke", colors.util.transparent)
          .attr("stroke-width", 12)
          .attr("cursor", "pointer")
          .on("click", (event) => {
            event.stopPropagation();
            if (mode === "select") {
              onSelect({ kind: "lane", id: lane.id });
            }
          });

      }
    }

    // Build stable label → color map based on label_text from scene detections
    const uniqueLabels = [...new Set(
      scene.vehicles
        .map(v => v.label_text)
        .filter((l): l is string => l !== null)
    )].sort();
    const labelColorMap = new Map(
      uniqueLabels.map((label, i) => [label, TAB10[i % TAB10.length]])
    );

    // --- Layer 5: Vehicle OBBs ---
    if (layers.vehicles) {
      const vehG = root.append("g").attr("class", "vehicles");
      for (let vi = 0; vi < annotation.vehicles.length; vi++) {
        const veh = annotation.vehicles[vi];
        if (veh.status === "rejected") continue;
        const isSelected = selected?.kind === "vehicle" && selected.id === veh.id;

        // Find OBB from scene
        const det = scene.vehicles.find((v) => v.id === veh.id);
        const label = det?.label_text ?? veh.id;
        const color = labelColorMap.get(label) ?? TAB10[vi % TAB10.length];
        if (det?.obb && det.obb.length === 4) {
          const corners = det.obb.map((p) => toSvg(p));
          const points = corners.map((c) => `${c.x},${c.y}`).join(" ");
          vehG.append("polygon")
            .attr("points", points)
            .attr("fill", color)
            .attr("fill-opacity", 0.2)
            .attr("stroke", isSelected ? colors.text.white : color)
            .attr("stroke-width", isSelected ? 2.5 : 1.5)
            .attr("cursor", "pointer")
            .on("click", (ev) => {
              ev.stopPropagation();
              if (mode === "select") onSelect({ kind: "vehicle", id: veh.id });
            });

          // Vehicle label
          const tc = toSvg(det.center);
          vehG.append("text")
            .attr("x", tc.x)
            .attr("y", tc.y)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("fill", color)
            .attr("font-size", 11)
            .attr("font-family", "JetBrains Mono, monospace")
            .attr("font-weight", "bold")
            .attr("pointer-events", "none")
            .text(label);
        }
      }
    }

    // --- Layer 6: Trajectories ---
    if (layers.trajectories) {
      const trajG = root.append("g").attr("class", "trajectories");
      for (let vi = 0; vi < annotation.vehicles.length; vi++) {
        const veh = annotation.vehicles[vi];
        if (veh.status === "rejected" || veh.waypoints.length < 2) continue;
        const det = scene.vehicles.find((v) => v.id === veh.id);
        const label = det?.label_text ?? veh.id;
        const color = labelColorMap.get(label) ?? TAB10[vi % TAB10.length];

        // Group waypoints by phase
        const preWpts = veh.waypoints.filter((w) =>
          ["pre_crash", "collision"].includes(w.phase)
        );
        const postWpts = veh.waypoints.filter((w) => w.phase === "post_crash");

        if (preWpts.length >= 2) {
          trajG.append("path")
            .attr("d", lineGen(preWpts.map((w) => w.position)))
            .attr("fill", "none")
            .attr("stroke", color)
            .attr("stroke-width", 1.5)
            .attr("stroke-dasharray", "8,3")
            .attr("opacity", 0.7);
        }
        if (postWpts.length >= 2) {
          trajG.append("path")
            .attr("d", lineGen(postWpts.map((w) => w.position)))
            .attr("fill", "none")
            .attr("stroke", colors.canvas.trajectoryCollision)
            .attr("stroke-width", 1.5)
            .attr("stroke-dasharray", "3,3")
            .attr("opacity", 0.7);
        }

        // Waypoint index circles
        for (const [i, wp] of veh.waypoints.entries()) {
          const pt = toSvg(wp.position);
          const isCollision = wp.phase === "collision";
          if (isCollision) {
            // X marker
            trajG.append("text")
              .attr("x", pt.x).attr("y", pt.y)
              .attr("text-anchor", "middle").attr("dominant-baseline", "middle")
              .attr("fill", colors.text.black).attr("font-size", 14).attr("font-weight", "bold")
              .attr("pointer-events", "none").text("✕");
          } else {
            trajG.append("circle")
              .attr("cx", pt.x).attr("cy", pt.y).attr("r", 5)
              .attr("fill", color).attr("stroke", colors.canvas.labelBackground).attr("stroke-width", 1);
            trajG.append("text")
              .attr("x", pt.x).attr("y", pt.y)
              .attr("text-anchor", "middle").attr("dominant-baseline", "middle")
              .attr("fill", colors.canvas.labelBackground).attr("font-size", 7).attr("font-weight", "bold")
              .attr("pointer-events", "none").text(i + 1);
          }
        }
      }
    }

    // --- Layer 7: Text labels ---
    if (layers.texts) {
      const textG = root.append("g").attr("class", "texts");
      for (const txt of scene.texts) {
        const pt = toSvg(txt.position);
        textG.append("text")
          .attr("x", pt.x).attr("y", pt.y)
          .attr("fill", colors.text.muted)
          .attr("font-size", 9)
          .attr("font-family", "JetBrains Mono, monospace")
          .attr("pointer-events", "none")
          .attr("transform", `rotate(${-txt.rotation * 180 / Math.PI}, ${pt.x}, ${pt.y})`) // negate for Y-axis flip
          .text(txt.text);
      }
    }

    // --- Layer 8: Lane connections (top-most scene layer so arrows render above everything) ---
    if (layers.connections) {
      const connG = root.append("g").attr("class", "connections");

      // Returns the unit tangent pointing outward from `end` of `polyline`, in SVG coords.
      // For "end": direction of last segment (forward). For "start": reversed (backward).
      const outwardTangentSvg = (
        polyline: { x: number; y: number }[],
        end: "start" | "end"
      ): { x: number; y: number } => {
        if (polyline.length < 2) return { x: 1, y: 0 };
        const [p1w, p2w] =
          end === "end"
            ? [polyline[polyline.length - 2], polyline[polyline.length - 1]]
            : [polyline[1], polyline[0]]; // reversed so tangent exits the lane at start
        const p1 = toSvg(p1w);
        const p2 = toSvg(p2w);
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        return len > 0 ? { x: dx / len, y: dy / len } : { x: 1, y: 0 };
      };

      for (const conn of annotation.lane_connections) {
        if (conn.status === "rejected") continue;
        const fromLane = annotation.lanes.find((l) => l.id === conn.from_lane_id);
        const toLane = annotation.lanes.find((l) => l.id === conn.to_lane_id);
        if (!fromLane || !toLane || !fromLane.polyline.length || !toLane.polyline.length) continue;
        const isSelected = selected?.kind === "connection" && selected.id === conn.id;

        const fromEnd = (conn.from_end ?? "end") as "start" | "end";
        const toEnd = (conn.to_end ?? "start") as "start" | "end";

        const fromPt = fromEnd === "start" ? fromLane.polyline[0] : fromLane.polyline[fromLane.polyline.length - 1];
        const toPt = toEnd === "start" ? toLane.polyline[0] : toLane.polyline[toLane.polyline.length - 1];
        const f = toSvg(fromPt);
        const t = toSvg(toPt);

        // Cubic Bezier approximation of a clothoid transition:
        // CP1 is placed along the departure tangent of fromLane,
        // CP2 is placed along the outward tangent of toLane (so the curve arrives flowing into it).
        // Scale control arms to 1/3 of the chord length — this matches Euler spiral curvature onset.
        const chord = Math.sqrt((t.x - f.x) ** 2 + (t.y - f.y) ** 2);
        const arm = chord / 3;
        const fromDir = outwardTangentSvg(fromLane.polyline, fromEnd);
        const toDir = outwardTangentSvg(toLane.polyline, toEnd);
        const cp1 = { x: f.x + fromDir.x * arm, y: f.y + fromDir.y * arm };
        const cp2 = { x: t.x + toDir.x * arm, y: t.y + toDir.y * arm };
        const curvePath = `M ${f.x},${f.y} C ${cp1.x},${cp1.y} ${cp2.x},${cp2.y} ${t.x},${t.y}`;

        connG.append("path")
          .attr("d", curvePath)
          .attr("fill", "none")
          .attr("stroke", isSelected ? colors.accent.selected : colors.accent.connection)
          .attr("stroke-width", isSelected ? 2 : 1)
          .attr("stroke-dasharray", "5,3")
          .attr("marker-end", "url(#arrowhead)")
          .attr("cursor", "pointer")
          .on("click", (ev) => {
            ev.stopPropagation();
            onSelect({ kind: "connection", id: conn.id });
          });

        // Invisible wide hit area for easier clicking
        connG.append("path")
          .attr("d", curvePath)
          .attr("fill", "none")
          .attr("stroke", colors.util.transparent)
          .attr("stroke-width", 12)
          .attr("cursor", "pointer")
          .on("click", (ev) => {
            ev.stopPropagation();
            onSelect({ kind: "connection", id: conn.id });
          });
      }

      // Arrowhead marker
      svg.append("defs").append("marker")
        .attr("id", "arrowhead")
        .attr("markerWidth", 8).attr("markerHeight", 8)
        .attr("refX", 6).attr("refY", 3)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,0 L0,6 L9,3 z")
        .attr("fill", colors.accent.connection);
    }

    // --- Layer 9: Connect-mode endpoint dots (drawn last so they sit on top of connections) ---
    if (mode === "connect" && layers.connections && layers.centerlines) {
      const endptG = root.append("g").attr("class", "endpoints");
      const drawEndpoint = (pt: { x: number; y: number }, endId: string) => {
        endptG.append("circle")
          .attr("cx", toSvg(pt).x)
          .attr("cy", toSvg(pt).y)
          .attr("r", 6)
          .attr("fill", colors.canvas.endpoint)
          .attr("stroke", colors.text.white)
          .attr("stroke-width", 1.5)
          .attr("cursor", "pointer")
          .attr("data-endpoint", endId)
          .on("click", (ev) => {
            ev.stopPropagation();
            const first = connectRef.current;
            if (!first) {
              connectRef.current = endId;
              d3.select(ev.target as SVGCircleElement).attr("fill", colors.canvas.endpointHover);
            } else if (first !== endId) {
              const [fromLane, fromEnd] = first.split("__") as [string, "start" | "end"];
              const [toLane, toEnd] = endId.split("__") as [string, "start" | "end"];
              if (fromLane !== toLane && onAddConnection) {
                onAddConnection(fromLane, toLane, fromEnd, toEnd);
              }
              connectRef.current = null;
              endptG.selectAll("[data-endpoint]").attr("fill", colors.canvas.endpoint);
            }
          });
      };
      for (const lane of annotation.lanes) {
        if (lane.status === "rejected" || lane.polyline.length === 0) continue;
        drawEndpoint(lane.polyline[0], `${lane.id}__start`);
        drawEndpoint(lane.polyline[lane.polyline.length - 1], `${lane.id}__end`);
      }
    }

    // --- Click-to-deselect / cancel connect on background ---
    const cancelConnect = () => {
      connectRef.current = null;
      root.selectAll("[data-endpoint]").attr("fill", colors.canvas.endpoint);
    };

    svg.on("click", () => {
      if (mode === "select" || mode === "edit") onSelect(null);
      else if (mode === "connect") cancelConnect();
    });

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && mode === "connect") cancelConnect();
    };
    window.addEventListener("keydown", onKeyDown);

    // --- Zoom/pan ---
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 50])
      .on("zoom", (event) => {
        root.attr("transform", event.transform.toString());
        transformRef.current = event.transform;
        // Keep endpoint dots constant screen size
        const k_clamped = Math.max(1, Math.min(5, event.transform.k));

        root.selectAll<SVGCircleElement, unknown>(".endpoints circle")
          .attr("r", 6 / k_clamped)
          .attr("stroke-width", 1.5 / k_clamped);
      });
    svg.call(zoom);

    const annotationChanged = prevCaseIdRef.current !== annotation.scene.case_id;
    prevCaseIdRef.current = annotation.scene.case_id;

    if (transformRef.current && !annotationChanged) {
      // Restore previous zoom/pan without triggering the zoom event
      svg.call(zoom.transform, transformRef.current!);
    } else {
      // Initial fit-to-view (first render or new annotation)
      // Scales already center and fit the data, so just apply a 5% margin via a scale-around-center.
      const k = 0.95;
      svg.call(zoom.transform, d3.zoomIdentity.translate(width * (1 - k) / 2, height * (1 - k) / 2).scale(k));
    }

    return () => window.removeEventListener("keydown", onKeyDown);
  }, [annotation, layers, mode, selected, onSelect, onAddConnection, onHideRoadEdge]);

  return (
    <svg
      ref={svgRef}
      style={{
        width: "100%",
        height: "100%",
        background: colors.surface.page,
        display: "block",
      }}
    />
  );
}
