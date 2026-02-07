from lxml import etree
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


class FaroSceneGraphReader:
    # MNLI Classification Labels
    LABEL_VEHICLE = "vehicle (car, suv, truck, pickup, bus)"
    LABEL_TRAFFIC_LIGHT = "traffic light / signal"
    LABEL_ROAD_MARKING = "road marking / lane line"
    LABEL_DIRECTION_ARROW = "direction arrow (north, south, east, west)"  # capture compass direction in diagram
    LABEL_TURN_DIRECTION = "turn direction"
    LABEL_PEDESTRIAN = "pedestrian"  # TODO: cyclist, deer etc
    LABEL_BACKGROUND = "background / decoration"

    CLASSIFICATION_LABELS = [
        LABEL_VEHICLE,
        LABEL_TRAFFIC_LIGHT,
        LABEL_ROAD_MARKING,
        LABEL_DIRECTION_ARROW,
        LABEL_TURN_DIRECTION,
        LABEL_PEDESTRIAN,
        LABEL_BACKGROUND,
    ]

    def __init__(self, file_path, clf_pipeline, cls_cache=None):
        self.tree = etree.parse(file_path, etree.XMLParser(huge_tree=True))
        self.root = self.tree.getroot()
        self.clf = clf_pipeline

        self.symbols = []
        self.primitives = []
        # Pass 2 Output: Structured objects
        self.scene_objects = {
            "vehicles": [],
            "roadway": [],
            "road_markings": [],
            "misc": [],
            "texts": [],
            "scalebar": None,
        }
        self.cls_cache = cls_cache if cls_cache is not None else {}

    def _get_transform_matrix(self, item):
        """
        Creates a 3x3 affine transform matrix from item attributes.
        Order: Scale -> Rotate -> Translate
        """
        tx = float(item.get("posx", "0"))
        ty = float(item.get("posy", "0"))
        sx = float(item.get("scalex", "1"))
        sy = float(item.get("scaley", "1"))
        rot = float(item.get("oriz", "0"))  # Assuming radians. If degrees, convert!

        # 1. Scale Matrix
        S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

        # 2. Rotation Matrix
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # 3. Translation Matrix
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

        # M = T * R * S
        return T @ R @ S

    def _apply_transform(self, points, matrix):
        """
        Applies a 3x3 matrix to a list of 2D points [(x,y), ...].
        """
        if not points:
            return []

        # Convert to homogeneous coordinates (x, y, 1)
        pts = np.array(points)
        ones = np.ones((pts.shape[0], 1))
        pts_homo = np.hstack([pts, ones])

        # Apply matrix (Transposed because points are rows)
        # Result = (M @ P.T).T
        transformed = (matrix @ pts_homo.T).T

        # Return as list of (x, y) tuples
        return [tuple(row[:2]) for row in transformed]

    def parse(self):
        """
        Main entry point.
        """
        logger.info("Phase 1: Flattening Scene Graph...")
        scene = self.root.find("scene")
        # Identity matrix as root transform
        identity = np.eye(3)
        self._traverse_recursive(scene, identity)

        print(
            f"Extracted {len(self.primitives)} primitives and {len(self.symbols)} symbols."
        )

        logger.info("Phase 2: Semantic Association...")
        self._cluster_and_classify()
        return self.scene_objects

    def _traverse_recursive(self, element, parent_matrix, parent_props=[], current_layer=None):
        current_matrix = parent_matrix
        current_props = parent_props.copy()
        current_props.append(element.attrib)

        el_tag = element.tag
        el_type = element.get("type")
        layer = current_layer

        if el_tag == "item":
            # Compute local transform
            if el_type == "symbol":
                self.symbols.append(
                    self._traverse_symbol(element, current_matrix, current_props, layer)
                )
            else:
                current_matrix = parent_matrix @ self._get_transform_matrix(element)
                primitive = self._extract_primitive(
                    element, current_matrix, current_props, layer
                )
                if primitive:
                    self.primitives.append(primitive)
        else:
            if el_tag == "layer":
                layer = element.get("n", None)
            for child in element:
                self._traverse_recursive(child, current_matrix, current_props, layer)

    def _traverse_symbol(self, element, parent_matrix, parent_props, layer):
        current_matrix = parent_matrix
        current_props = parent_props.copy()
        current_props.append(element.attrib)

        # Symbol may have its own transform
        local_matrix = self._get_transform_matrix(element)
        current_matrix = parent_matrix @ local_matrix

        symbol_items = []

        for child in element:
            if child.tag == "item":
                if child.get("type") == "symbol":
                    symbol_result = self._traverse_symbol(
                        child, current_matrix, current_props, layer
                    )
                    if symbol_result:
                        symbol_items.append(symbol_result)
                else:
                    primitive = self._extract_primitive(
                        child, current_matrix, current_props, layer
                    )
                    if primitive:
                        symbol_items.append(primitive)

        for prop in reversed(current_props):
            if "nam" in prop and prop["nam"]:
                name = prop["nam"]
                break
        else:
            name = None

        bbox = (float("inf"), float("inf"), float("-inf"), float("-inf"))
        for item in symbol_items:
            ibox = item["bbox"]
            bbox = (
                min(bbox[0], ibox[0]),
                min(bbox[1], ibox[1]),
                max(bbox[2], ibox[2]),
                max(bbox[3], ibox[3]),
            )

        center = (np.mean([bbox[0], bbox[2]]), np.mean([bbox[1], bbox[3]]))

        # Propagate dashed property from child items (for trajectory vs final position detection)
        is_dashed = any(
            item.get("dashed", False) for item in symbol_items if item["type"] != "symbol"
        ) or any(
            item.get("dashed", False) for item in symbol_items if item["type"] == "symbol"
        )

        return {
            "type": "symbol",
            "name": name,
            "items": symbol_items,
            "bbox": bbox,
            "center": center,
            "transformed_center": self._apply_transform([center], current_matrix)[0],
            "vehicle2d": element.get("vehicle2d", "F") == "T",
            "transform": current_matrix,
            "layer": layer,
            "predicted_class": None,
            "predicted_probability": None,
            "dashed": is_dashed,
        }

    def _extract_primitive(self, element, global_matrix, inherited_props, layer):
        """
        Extracts geometry, applies global transform, and saves to flat list.
        """
        el_type = element.get("type")
        current_props = inherited_props.copy()
        current_props.append(element.attrib)

        # Helper to parse vertex strings "x,y;x,y"
        def parse_verts(attr):
            raw = element.get(attr)
            if not raw:
                return []
            return [tuple(map(float, v.split(",")))[:2] for v in raw.split(";") if v]

        # Cubic Bezier evaluation
        def bezier_cubic(P0, C1, C2, P1, t):
            # t is (m,) -> returns (m,3)
            t = t[:, None]
            return (
                ((1 - t) ** 3) * P0
                + 3 * ((1 - t) ** 2) * t * C1
                + 3 * (1 - t) * (t**2) * C2
                + (t**3) * P1
            )

        verts = []
        text_content = None
        dashed = False
        thick = False
        closed = element.get("closed", "F") == "T"
        if el_type == "polyline":
            verts = parse_verts("vlist")

        elif el_type == "polycurve":
            raw_verts = parse_verts("pnts")
            ctrl_pts = parse_verts("ctrl")
            if raw_verts and not closed:
                first_pt = raw_verts[0]
                last_pt = raw_verts[-1]
                bbox = (min(x for x, y in raw_verts), min(y for x, y in raw_verts),
                        max(x for x, y in raw_verts), max(y for x, y in raw_verts))
                if np.hypot(first_pt[0]-last_pt[0], first_pt[1]-last_pt[1]) < 0.25 * max(bbox[2]-bbox[0], bbox[3]-bbox[1]):
                    closed = True
                    
            if raw_verts and ctrl_pts and 2 * len(raw_verts) - 2 == len(ctrl_pts):
                # Build cubic Bezier segments
                curve_verts = []
                num_segments = len(raw_verts) - 1
                for i in range(num_segments):
                    P0 = np.array(raw_verts[i])
                    P1 = np.array(raw_verts[i + 1])
                    C1 = np.array(ctrl_pts[2 * i])
                    C2 = np.array(ctrl_pts[2 * i + 1])
                    t_vals = np.linspace(0, 1, num=10)  # 10 points per segment
                    segment_pts = bezier_cubic(P0, C1, C2, P1, t_vals)
                    if i > 0:
                        segment_pts = segment_pts[1:]  # Avoid duplicating points
                    curve_verts.extend([tuple(pt) for pt in segment_pts])
                verts = curve_verts
            else:
                verts = raw_verts

            if closed and verts:
                if verts[0] != verts[-1]:
                    verts.append(verts[0])

        elif el_type == "line":
            line_data = element.find("lndata")
            if (
                line_data.get("arrowshowe", "F") == "T"
                or line_data.get("arrowshows", "F") == "T"
            ):
                # Ignore arrows
                return None
            startX = float(element.get("pntSx", "0"))
            startY = float(element.get("pntSy", "0"))
            endX = float(element.get("pntEx", "0"))
            endY = float(element.get("pntEy", "0"))
            verts = [(startX, startY), (endX, endY)]
        elif el_type == "label":
            # For text, we might care about the insertion point
            sizeX = float(element.get("sizex", "0"))
            sizeY = float(element.get("sizey", "0"))
            offPosx = sizeX / 2.0
            offPosy = sizeY / 2.0
            text_point = (offPosx, offPosy)  # Text anchor point
            verts = [text_point]
            # verts = [(0, 0)]
            text_content = element.get("text", "")
            if not text_content:
                print("Warning: Label element without text content.")
        elif el_type == "scalebar":
            if self.scene_objects["scalebar"] is not None:
                print("Warning: Multiple scalebars found; using the first one.")
                return None
            x = float(element.get("px", "0"))
            y = float(element.get("py", "0"))
            sizeX = float(element.get("szx", "0"))
            sizeY = float(element.get("szy", "0"))
            self.scene_objects["scalebar"] = {
                "type": el_type,
                "position": (x, y),
                "size": (sizeX, sizeY),
            }
        elif el_type == "flexconcretebarrier":
            print("Flex concrete barrier encountered; skipping for now.")
            pass

        lndata = element.find("lndata")  # could check for arrows here
        if lndata is not None:
            if lndata.get("lt") == "1":
                dashed = True
            if lndata.get("thickness") > "0":
                thick = True

        if verts:
            xs, ys = zip(*verts)
            center = (np.mean(xs), np.mean(ys))
            bbox = (min(xs), min(ys), max(xs), max(ys))  # (xmin, ymin, xmax, ymax)

            for prop in reversed(current_props):
                if "nam" in prop and prop["nam"]:
                    name = prop["nam"]
                    break
            else:
                name = None

            return {
                "type": el_type,
                "name": name,
                "verts": verts,
                "transformed_verts": self._apply_transform(verts, global_matrix),
                "center": center,
                "transformed_center": self._apply_transform([center], global_matrix)[0],
                "bbox": bbox,
                "vehicle2d": element.get("vehicle2d", "F") == "T",
                "transform": global_matrix,
                "text": text_content,
                "dashed": dashed,
                "thick": thick,
                "closed": closed,
                "layer": layer
            }
        else:
            pass
            # logger.warning(f"Warning: No verts found for element of type {el_type}")

    def _check_name_vehicle(self, name):
        """
        Use MNLI to classify a name and return (is_vehicle, predicted_class, probability).
        """
        out = self.clf(
            name.lower().replace("_", " "),
            candidate_labels=self.CLASSIFICATION_LABELS,
            multi_label=True,
            hypothesis_template="This item is a {}.",
        )
        # print(f"Classifying name '{name}': {list(zip(out['labels'], out['scores']))}")
        scores = out["scores"]
        max_idx = np.argmax(scores)
        predicted_class = out["labels"][max_idx]
        predicted_prob = scores[max_idx]

        is_vehicle = predicted_class == self.LABEL_VEHICLE and predicted_prob > 0.7

        # Cache the result
        self.cls_cache[name] = {
            "is_vehicle": is_vehicle,
            "predicted_class": predicted_class,
            "predicted_probability": predicted_prob,
        }

        return is_vehicle, predicted_class, predicted_prob

    def _check_vehicle(self, symbol):
        """
        Simple heuristic to check if a symbol is a vehicle based on attributes.
        Propagates predicted_class and predicted_probability to the symbol dict.
        """

        if symbol["vehicle2d"]:
            symbol["predicted_class"] = self.LABEL_VEHICLE
            symbol["predicted_probability"] = 1.0
            return True

        if symbol["name"]:
            if symbol["name"] in self.cls_cache:
                cache_entry = self.cls_cache[symbol["name"]]
                symbol["predicted_class"] = cache_entry["predicted_class"]
                symbol["predicted_probability"] = cache_entry["predicted_probability"]
                return cache_entry["is_vehicle"]
            else:
                is_vehicle, predicted_class, predicted_prob = self._check_name_vehicle(
                    symbol["name"]
                )
                symbol["predicted_class"] = predicted_class
                symbol["predicted_probability"] = predicted_prob
                return is_vehicle

        # Check nested symbols
        for item in symbol["items"]:
            if item["type"] == "symbol":
                if self._check_vehicle(item):
                    # Propagate nested classification if no classification yet
                    if not symbol["predicted_class"]:
                        symbol["predicted_class"] = item.get("predicted_class")
                        symbol["predicted_probability"] = item.get(
                            "predicted_probability"
                        )
                    return True

        # if no indicators found
        # TODO: Add aspect ratio or size heuristics
        # TODO: rasterize + vlm?
        return False

    def _cluster_and_classify(self):
        """
        Phase 2: Reconstruct objects based on attributes and SPATIAL PROXIMITY.
        """
        # 1. Identify Vehicle Candidates
        # Heuristics: Explicit attribute OR Name match OR Aspect Ratio check
        vehicle_candidates = []
        texts = []
        others = []
        roadway = []
        road_markings = []

        print("Classifying symbols for vehicle candidates...")
        for p in self.symbols:
            if self._check_vehicle(p):
                # print(f"Identified vehicle candidate: {p['name']}")
                p["associated_text"] = []
                for it in p["items"]:
                    if it["type"] == "label":
                        p["associated_text"].append(it["text"])
                vehicle_candidates.append(p)
            elif (
                p["predicted_probability"]
                and p["predicted_probability"] > 0.5
                and p["predicted_class"]
                in [self.LABEL_ROAD_MARKING, self.LABEL_TURN_DIRECTION]
            ):
                road_markings.append(p)

            else:
                others.append(p)

        for p in self.primitives:
            if p["type"] == "label":
                texts.append(p)
            elif p["type"] in ["polycurve", "polyline", "line", "flexconcretebarrier"]: #and p["layer"] == "Line Work":
                roadway.append(p)
            else:
                others.append(p)

        # 2. Associate Text to Vehicles (Spatial Join)
        # Simple N*M check (fast enough for <1000 items)
        for txt in texts:
            txt_pos = txt["transformed_center"]
            best_dist = float("inf")
            best_vehicle = None

            for veh in vehicle_candidates:
                # Check if text is inside or near vehicle bbox
                # Using simple center-to-center distance for demo
                vx, vy = veh["transformed_center"]

                dist = np.hypot(vx - txt_pos[0], vy - txt_pos[1])
                # Heuristic: Text must be within X meters (units depend on file)
                if dist < best_dist:
                    best_dist = dist
                    best_vehicle = veh

            # Threshold distance (adjust based on your coordinate system units)
            if best_vehicle and best_dist < 5.0:
                best_vehicle["associated_text"].append(txt["text"])
                # Or append the raw XML text content

        self.scene_objects["vehicles"] = vehicle_candidates
        self.scene_objects["misc"] = others
        self.scene_objects["roadway"] = roadway
        self.scene_objects["road_markings"] = road_markings
        self.scene_objects["texts"] = texts