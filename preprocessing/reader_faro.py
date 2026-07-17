"""
FARO Zone (.far) scene-graph reader.

Phase 1 flattens the XML scene graph into primitive/symbol dicts (geometry in
local frame + composed world transforms); Phase 2 delegates semantic
classification to preprocessing.classify.SceneClassifier, shared with the
.blz reader.
"""
import logging
import sys

import numpy as np
from lxml import etree

from . import base
from .classify import CLASSIFICATION_LABELS, LABEL_VEHICLE, SceneClassifier

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

MAX_ROADWAY_CURVATURE_DEG = 100.0  # deg per ft


class FaroSceneGraphReader:
    # Kept as class attributes for backward compatibility with older callers
    LABEL_VEHICLE = LABEL_VEHICLE
    CLASSIFICATION_LABELS = CLASSIFICATION_LABELS

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
        """3x3 affine from item attributes, composed T @ R @ S."""
        tx = float(item.get("posx", "0"))
        ty = float(item.get("posy", "0"))
        sx = float(item.get("scalex", "1"))
        sy = float(item.get("scaley", "1"))
        rot = float(item.get("oriz", "0"))  # Assuming radians. If degrees, convert!
        return base.make_affine(tx, ty, sx, sy, rot)

    def _apply_transform(self, points, matrix):
        return base.apply_transform(points, matrix)

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
        classifier = SceneClassifier(self.clf, self.cls_cache)
        classifier.classify(self.symbols, self.primitives, self.scene_objects)
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
        current_props = parent_props.copy()
        current_props.append(element.attrib)

        # Symbol may have its own transform
        current_matrix = parent_matrix @ self._get_transform_matrix(element)

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

        return base.make_symbol_dict(
            name=_nearest_name(current_props),
            items=symbol_items,
            transform=current_matrix,
            layer=layer,
            vehicle2d=element.get("vehicle2d", "F") == "T",
        )

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
                        segment_pts = base.bezier_cubic(P0, C1, C2, P1, t_vals)
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
                        segment_pts = base.bezier_quadratic(P0, C, P1, t_vals)
                        if i > 0:
                            segment_pts = segment_pts[1:]
                        curve_segments.append(segment_pts)
                    verts = np.vstack(curve_segments)

            else:
                interpolation_method = "catmull_rom"
                verts = base.interp_bezier_composite(raw_verts, tension=0.33)

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
            img_data = element.get("img", "")
            if not img_data:
                return None
            return base.make_image_dict(
                name=_nearest_name(current_props),
                cx=world_cx,
                cy=world_cy,
                sizx=float(element.get("sizx", "0")),
                sizy=float(element.get("sizy", "0")),
                oriz=float(element.get("oriz", "0")),
                img_data=img_data,
                layer=layer,
                transform=global_matrix,
            )

        elif el_type == "sign":
            world_cx = float(global_matrix[0, 2])
            world_cy = float(global_matrix[1, 2])
            img_data = self._lookup_sign_texture(element.get("fname", ""))
            if not img_data:
                return None
            return base.make_image_dict(
                name=_nearest_name(current_props),
                cx=world_cx,
                cy=world_cy,
                sizx=float(element.get("fsx", "1")),
                sizy=float(element.get("fsy", "1")),
                oriz=float(element.get("oriz", "0")),
                img_data=img_data,
                layer=layer,
                transform=global_matrix,
            )

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
            return base.make_primitive_dict(
                el_type,
                name=_nearest_name(current_props),
                verts=verts,
                transform=global_matrix,
                control_points=control_points,
                bezier_handles=bezier_handles,
                interpolation_method=interpolation_method,
                layer=layer,
                oriz=oriz,
                text=text_content,
                dashed=dashed,
                thick=thick,
                closed=closed,
                vehicle2d=element.get("vehicle2d", "F") == "T",
                lndata=lndata_dict,
            )
        return None


def _nearest_name(props) -> str | None:
    """Nearest ancestor 'nam' attribute, walking outward from the element."""
    for prop in reversed(props):
        if "nam" in prop and prop["nam"]:
            return prop["nam"]
    return None
