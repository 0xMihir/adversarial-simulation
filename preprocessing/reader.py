from lxml import etree
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

MAX_ROADWAY_CURVATURE_DEG = 100.0  # deg per ft


def _avg_curvature_deg(verts: np.ndarray) -> float:
    """Average curvature in degrees per unit length."""
    if len(verts) < 3:
        return 0.0
    diffs = np.diff(verts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    dtheta = np.abs(np.diff(headings))
    dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
    avg_lens = (seg_lens[:-1] + seg_lens[1:]) / 2.0
    curv = np.where(avg_lens > 0, dtheta / avg_lens, 0.0)
    return float(np.degrees(np.mean(curv)))


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

    def __init__(self, file_path, clf_pipeline, cls_cache=None, texture_cache=None):
        self.file_path = file_path
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
            "images": [],
            "scalebar": None,
        }
        self.cls_cache = cls_cache if cls_cache is not None else {}
        self.texture_cache = texture_cache  # flat {normalized_key: base64_data} from corpus
        self.textures = {}

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
        Applies a 3x3 matrix to an (N,2) array (or list) of 2D points.
        Returns an (N,2) numpy array.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if len(pts) == 0:
            return np.empty((0, 2))

        ones = np.ones((pts.shape[0], 1))
        pts_homo = np.hstack([pts, ones])
        transformed = (matrix @ pts_homo.T).T
        return transformed[:, :2]

    def _extract_textures(self, scene_element):
        """Build {normalized_key: base64_data} from <textures> block."""
        textures = {}
        tex_block = scene_element.find("textures")
        if tex_block is None:
            return textures
        for tex in tex_block.findall("tex"):
            key = tex.get("key", "").lower().replace("\\", "/")
            file_elem = tex.find("file")
            if file_elem is not None and file_elem.get("data"):
                textures[key] = file_elem.get("data")
        return textures

    def _lookup_sign_texture(self, fname):
        if not fname:
            return None
        norm = fname.lower().replace("\\", "/")
        # Absolute Windows path — direct key lookup
        if norm.startswith("c:/"):
            return self.textures.get(norm)
        # [Signs] relative prefix — match against key suffix
        if norm.startswith("[signs]"):
            suffix = norm[len("[signs]"):]  # e.g. "preset/us/.../stop.png"
            for key, data in self.textures.items():
                if key.endswith("signs/" + suffix):
                    return data
        return None

    def parse(self):
        """
        Main entry point.
        """
        logger.info("Phase 1: Flattening Scene Graph...")
        scene = self.root.find("scene")
        # Seed from global corpus cache so files without a <textures> block can resolve signs
        self.textures = dict(self.texture_cache or {})
        # Overlay local <textures> block — same file's data always wins
        self.textures.update(self._extract_textures(scene))
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
                return np.empty((0, 2))
            pts = [tuple(map(float, v.split(",")))[:2] for v in raw.split(";") if v]
            return np.array(pts) if pts else np.empty((0, 2))

        # Cubic Bezier evaluation
        def bezier_cubic(P0, C1, C2, P1, t):
            # t is (m,) -> returns (m,2)
            t = t[:, None]
            return (
                ((1 - t) ** 3) * P0
                + 3 * ((1 - t) ** 2) * t * C1
                + 3 * (1 - t) * (t**2) * C2
                + (t**3) * P1
            )

        # Quadratic Bezier evaluation
        def bezier_quadratic(P0, C, P1, t):
            # t is (m,) -> returns (m,2)
            t = t[:, None]
            return (
                ((1 - t) ** 2) * P0
                + 2 * (1 - t) * t * C
                + (t**2) * P1
            )

        verts = None
        text_content = None
        dashed = False
        thick = False
        oriz = float(element.get("oriz", "0"))
        closed = element.get("closed", "F") == "T"
        control_points = np.empty((0, 2))   # raw knot points (pre-transform)
        bezier_handles = np.empty((0, 2))   # raw bezier handle points (pre-transform)
        interpolation_method = "passthrough"

        if el_type == "polyline":
            verts = parse_verts("vlist")
            control_points = verts
            interpolation_method = "passthrough"

        elif el_type == "polycurve":
            raw_verts = parse_verts("pnts")
            ctrl_pts = parse_verts("ctrl")
            control_points = raw_verts
            bezier_handles = ctrl_pts
            if len(raw_verts) > 0 and not closed:
                first_pt = raw_verts[0]
                last_pt = raw_verts[-1]
                bbox = (raw_verts[:, 0].min(), raw_verts[:, 1].min(),
                        raw_verts[:, 0].max(), raw_verts[:, 1].max())
                if np.hypot(first_pt[0]-last_pt[0], first_pt[1]-last_pt[1]) < 0.25 * max(bbox[2]-bbox[0], bbox[3]-bbox[1]):
                    closed = True

            if len(raw_verts) > 0 and len(ctrl_pts) > 0:
                num_segments = len(raw_verts) - 1
                if 2 * num_segments == len(ctrl_pts):
                    # Build cubic Bezier segments (2 control points per segment)
                    interpolation_method = "cubic_bezier"
                    curve_segments = []
                    for i in range(num_segments):
                        P0 = raw_verts[i]
                        P1 = raw_verts[i + 1]
                        C1 = ctrl_pts[2 * i]
                        C2 = ctrl_pts[2 * i + 1]
                        t_vals = np.linspace(0, 1, num=10)
                        segment_pts = bezier_cubic(P0, C1, C2, P1, t_vals)
                        if i > 0:
                            segment_pts = segment_pts[1:]
                        curve_segments.append(segment_pts)
                    verts = np.vstack(curve_segments)
                elif num_segments == len(ctrl_pts):
                    # Build quadratic Bezier segments (1 control point per segment)
                    interpolation_method = "cubic_bezier"
                    curve_segments = []
                    for i in range(num_segments):
                        P0 = raw_verts[i]
                        P1 = raw_verts[i + 1]
                        C = ctrl_pts[i]
                        t_vals = np.linspace(0, 1, num=10)
                        segment_pts = bezier_quadratic(P0, C, P1, t_vals)
                        if i > 0:
                            segment_pts = segment_pts[1:]
                        curve_segments.append(segment_pts)
                    verts = np.vstack(curve_segments)

            else:
                interpolation_method = "catmull_rom"
                # verts = self._interp_catmull_rom(raw_verts)
                verts = self._interp_bezier_composite(raw_verts, tension=0.33)

            if closed and len(verts) > 0:
                if not np.array_equal(verts[0], verts[-1]):
                    verts = np.vstack([verts, verts[0:1]])

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
            verts = np.array([[startX, startY], [endX, endY]])
            control_points = verts
            interpolation_method = "linear"
        elif el_type == "label":
            # For text, we might care about the insertion point
            sizeX = float(element.get("sizex", "0"))
            sizeY = float(element.get("sizey", "0"))
            offPosx = sizeX / 2.0
            offPosy = sizeY / 2.0
            verts = np.array([[offPosx, offPosy]])

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

        elif el_type == "image":
            world_cx = float(global_matrix[0, 2])
            world_cy = float(global_matrix[1, 2])
            sizx = float(element.get("sizx", "0"))
            sizy = float(element.get("sizy", "0"))
            oriz = float(element.get("oriz", "0"))
            img_data = element.get("img", "")
            if not img_data:
                return None
            for prop in reversed(current_props):
                if "nam" in prop and prop["nam"]:
                    name = prop["nam"]
                    break
            else:
                name = None
            return {
                "type": "image",
                "name": name,
                "posx": world_cx,
                "posy": world_cy,
                "sizx": sizx,
                "sizy": sizy,
                "oriz": oriz,
                "img": img_data,
                "layer": layer,
                "verts": None,
                "transformed_verts": np.empty((0, 2)),
                "control_points": np.empty((0, 2)),
                "bezier_handles": np.empty((0, 2)),
                "transformed_control_points": np.empty((0, 2)),
                "transformed_bezier_handles": np.empty((0, 2)),
                "interpolation_method": "none",
                "center": np.array([world_cx, world_cy]),
                "transformed_center": np.array([world_cx, world_cy]),
                "bbox": (world_cx - sizx / 2, world_cy - sizy / 2, world_cx + sizx / 2, world_cy + sizy / 2),
                "vehicle2d": False,
                "transform": global_matrix,
                "text": None,
                "dashed": False,
                "thick": False,
                "closed": False,
                "lndata": {},
            }

        elif el_type == "sign":
            world_cx = float(global_matrix[0, 2])
            world_cy = float(global_matrix[1, 2])
            fsx = float(element.get("fsx", "1"))
            fsy = float(element.get("fsy", "1"))
            oriz = float(element.get("oriz", "0"))
            fname = element.get("fname", "")
            img_data = self._lookup_sign_texture(fname)
            if not img_data:
                return None
            for prop in reversed(current_props):
                if "nam" in prop and prop["nam"]:
                    name = prop["nam"]
                    break
            else:
                name = None
            return {
                "type": "image",
                "name": name,
                "posx": world_cx,
                "posy": world_cy,
                "sizx": fsx,
                "sizy": fsy,
                "oriz": oriz,
                "img": img_data,
                "layer": layer,
                "verts": None,
                "transformed_verts": np.empty((0, 2)),
                "control_points": np.empty((0, 2)),
                "bezier_handles": np.empty((0, 2)),
                "transformed_control_points": np.empty((0, 2)),
                "transformed_bezier_handles": np.empty((0, 2)),
                "interpolation_method": "none",
                "center": np.array([world_cx, world_cy]),
                "transformed_center": np.array([world_cx, world_cy]),
                "bbox": (world_cx - fsx / 2, world_cy - fsy / 2, world_cx + fsx / 2, world_cy + fsy / 2),
                "vehicle2d": False,
                "transform": global_matrix,
                "text": None,
                "dashed": False,
                "thick": False,
                "closed": False,
                "lndata": {},
            }

        lndata_dict = {}
        lndata = element.find("lndata")
        if lndata is not None:
            for key in ['lt', 'thickness', 'dshlen', 'dshspc', 'lnspc', 'extrude',
                        'dotradius', 'dotspacing', 'arrowsize']:
                val = lndata.get(key)
                if val is not None:
                    lndata_dict[key] = float(val)
            for key in ['dotfilled', 'arrowshows', 'arrowshowe']:
                lndata_dict[key] = lndata.get(key, 'F') == 'T'

            # Derive legacy flags for backwards compat
            dashed = lndata_dict.get('lt', 0) == 1
            thick = lndata_dict.get('thickness', 0) > 0

        if verts is not None and len(verts) > 0:
            center = verts.mean(axis=0)
            bbox = (verts[:, 0].min(), verts[:, 1].min(),
                    verts[:, 0].max(), verts[:, 1].max())

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
                "control_points": control_points,
                "bezier_handles": bezier_handles,
                "transformed_control_points": self._apply_transform(control_points, global_matrix),
                "transformed_bezier_handles": self._apply_transform(bezier_handles, global_matrix),
                "interpolation_method": interpolation_method,
                "center": center,
                "transformed_center": self._apply_transform(center[np.newaxis], global_matrix)[0],
                "bbox": bbox,
                "vehicle2d": element.get("vehicle2d", "F") == "T",
                "transform": global_matrix,
                "oriz": oriz,
                "text": text_content,
                "dashed": dashed,
                "thick": thick,
                "closed": closed,
                "layer": layer,
                "lndata": lndata_dict,
            }
        else:
            pass
            # logger.warning(f"Warning: No verts found for element of type {el_type}")

    def _interp_bezier_composite(self, pts_2d, tension):
        """Composite cubic Bézier with Hobby-like tangent estimation."""
        N_SAMPLES = 10
        n = len(pts_2d)

        # Estimate tangents at each point
        tangents = np.zeros_like(pts_2d)
        for i in range(n):
            if i == 0:
                tangents[i] = pts_2d[1] - pts_2d[0]
            elif i == n - 1:
                tangents[i] = pts_2d[-1] - pts_2d[-2]
            else:
                tangents[i] = pts_2d[i+1] - pts_2d[i-1]

        all_pts = []
        for i in range(n - 1):
            p0 = pts_2d[i]
            p3 = pts_2d[i + 1]
            seg_len = np.linalg.norm(p3 - p0)
            p1 = p0 + tension * tangents[i] * seg_len / np.linalg.norm(tangents[i] + 1e-12)
            p2 = p3 - tension * tangents[i+1] * seg_len / np.linalg.norm(tangents[i+1] + 1e-12)

            t_seg = np.linspace(0, 1, max(N_SAMPLES // (n-1), 20))
            curve = (
                np.outer((1-t_seg)**3, p0) +
                np.outer(3*(1-t_seg)**2*t_seg, p1) +
                np.outer(3*(1-t_seg)*t_seg**2, p2) +
                np.outer(t_seg**3, p3)
            )
            all_pts.append(curve if i == 0 else curve[1:])  # avoid duplicating junction points

        return np.vstack(all_pts)
        

    def _interp_catmull_rom(self, pts_2d):
        N_SAMPLES = 10
        alpha = 0.5

        # Extend endpoints
        p_start = 2 * pts_2d[0] - pts_2d[1]
        p_end = 2 * pts_2d[-1] - pts_2d[-2]

        pts_ext = np.vstack([p_start, pts_2d, p_end])

        def tj(ti, pi, pj):
            return ti + np.sqrt((pj[0]-pi[0])**2 + (pj[1]-pi[1])**2)**alpha

        segments = []
        seg_lengths = []
        for i in range(len(pts_ext) - 3):
            p0, p1, p2, p3 = pts_ext[i], pts_ext[i+1], pts_ext[i+2], pts_ext[i+3]
            t0 = 0
            t1 = tj(t0, p0, p1)
            t2 = tj(t1, p1, p2)
            t3 = tj(t2, p2, p3)
            seg_lengths.append(t2 - t1)
            segments.append((p0, p1, p2, p3, t0, t1, t2, t3))

        # Distribute t_fine samples proportional to segment parameter length
        total_len = sum(seg_lengths)
        all_pts = []
        for seg_idx, (p0, p1, p2, p3, t0, t1, t2, t3) in enumerate(segments):
            n_seg = max(int(N_SAMPLES * seg_lengths[seg_idx] / total_len), 10)
            t_local = np.linspace(t1, t2, n_seg, endpoint=(seg_idx == len(segments)-1))

            A1 = np.outer((t1-t_local)/(t1-t0), p0) + np.outer((t_local-t0)/(t1-t0), p1)
            A2 = np.outer((t2-t_local)/(t2-t1), p1) + np.outer((t_local-t1)/(t2-t1), p2)
            A3 = np.outer((t3-t_local)/(t3-t2), p2) + np.outer((t_local-t2)/(t3-t2), p3)
            B1 = np.outer((t2-t_local)/(t2-t0), A1[:, 0]) + np.outer((t_local-t0)/(t2-t0), A2[:, 0])
            B1 = np.column_stack([
                (t2-t_local)/(t2-t0) * A1[:, 0] + (t_local-t0)/(t2-t0) * A2[:, 0],
                (t2-t_local)/(t2-t0) * A1[:, 1] + (t_local-t0)/(t2-t0) * A2[:, 1],
            ])
            B2 = np.column_stack([
                (t3-t_local)/(t3-t1) * A2[:, 0] + (t_local-t1)/(t3-t1) * A3[:, 0],
                (t3-t_local)/(t3-t1) * A2[:, 1] + (t_local-t1)/(t3-t1) * A3[:, 1],
            ])
            C = np.column_stack([
                (t2-t_local)/(t2-t1) * B1[:, 0] + (t_local-t1)/(t2-t1) * B2[:, 0],
                (t2-t_local)/(t2-t1) * B1[:, 1] + (t_local-t1)/(t2-t1) * B2[:, 1],
            ])
            all_pts.append(C)

        return np.vstack(all_pts)

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

    def _batch_classify_names(self, names):
        """
        Run a single batched MNLI call for all unique uncached names and populate cls_cache.
        """
        uncached = [n for n in names if n not in self.cls_cache]
        if not uncached:
            return

        texts = [n.lower().replace("_", " ") for n in uncached]
        results = self.clf(
            texts,
            candidate_labels=self.CLASSIFICATION_LABELS,
            multi_label=True,
            hypothesis_template="This item is a {}.",
        )

        # Pipeline returns a list when given a list
        for name, out in zip(uncached, results):
            scores = out["scores"]
            max_idx = int(np.argmax(scores))
            predicted_class = out["labels"][max_idx]
            predicted_prob = scores[max_idx]
            is_vehicle = predicted_class == self.LABEL_VEHICLE and predicted_prob > 0.7
            self.cls_cache[name] = {
                "is_vehicle": is_vehicle,
                "predicted_class": predicted_class,
                "predicted_probability": predicted_prob,
            }

    def _cluster_and_classify(self):
        """
        Phase 2: Reconstruct objects based on attributes and SPATIAL PROXIMITY.
        """
        # Pre-classify all unique symbol names in one batched request
        names_to_classify = [
            s["name"]
            for s in self.symbols
            if s["name"] and not s["vehicle2d"]
        ]
        self._batch_classify_names(names_to_classify)

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
            elif p["type"] in ["polycurve", "polyline", "line", "flexconcretebarrier"]:
                # if _avg_curvature_deg(p["verts"]) <= MAX_ROADWAY_CURVATURE_DEG:
                    roadway.append(p)
                # else:
                    # others.append(p)
            elif p["type"] == "image":
                self.scene_objects["images"].append(p)
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