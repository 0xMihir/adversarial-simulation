"""
Phase-2 semantic association for parsed scene graphs: vehicle identification,
primitive routing, and text-to-vehicle joining. Dialect-neutral — both the
.far and .blz readers hand their flattened symbols/primitives to
SceneClassifier so classification behaves identically for both formats.
"""
import numpy as np

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

VEHICLE_PROB_THRESHOLD = 0.7
TEXT_JOIN_MAX_DIST = 5.0  # ft, center-to-center


class SceneClassifier:
    """
    Classifies flattened symbols/primitives into scene_objects buckets.
    Mutates cls_cache in place (shared with the caller so results persist
    across scenes).
    """

    def __init__(self, clf_pipeline, cls_cache=None, vehicle_bank=None):
        self.clf = clf_pipeline
        self.cls_cache = cls_cache if cls_cache is not None else {}
        self.vehicle_bank = vehicle_bank  # reserved for flat-vehicle detection

    def check_name_vehicle(self, name):
        """
        Use MNLI to classify a name and return (is_vehicle, predicted_class, probability).
        """
        out = self.clf(
            name.lower().replace("_", " "),
            candidate_labels=CLASSIFICATION_LABELS,
            multi_label=True,
            hypothesis_template="This item is a {}.",
        )
        scores = out["scores"]
        max_idx = np.argmax(scores)
        predicted_class = out["labels"][max_idx]
        predicted_prob = scores[max_idx]

        is_vehicle = (
            predicted_class == LABEL_VEHICLE and predicted_prob > VEHICLE_PROB_THRESHOLD
        )

        self.cls_cache[name] = {
            "is_vehicle": is_vehicle,
            "predicted_class": predicted_class,
            "predicted_probability": predicted_prob,
        }

        return is_vehicle, predicted_class, predicted_prob

    def check_vehicle(self, symbol):
        """
        Heuristic cascade: explicit vehicle2d attribute, then name
        classification (cached), then nested symbols. Propagates
        predicted_class/predicted_probability onto the symbol dict.
        """
        if symbol["vehicle2d"]:
            symbol["predicted_class"] = LABEL_VEHICLE
            symbol["predicted_probability"] = 1.0
            return True

        if symbol["name"]:
            if symbol["name"] in self.cls_cache:
                cache_entry = self.cls_cache[symbol["name"]]
                symbol["predicted_class"] = cache_entry["predicted_class"]
                symbol["predicted_probability"] = cache_entry["predicted_probability"]
                return cache_entry["is_vehicle"]
            else:
                is_vehicle, predicted_class, predicted_prob = self.check_name_vehicle(
                    symbol["name"]
                )
                symbol["predicted_class"] = predicted_class
                symbol["predicted_probability"] = predicted_prob
                return is_vehicle

        # Check nested symbols
        for item in symbol["items"]:
            if item["type"] == "symbol":
                if self.check_vehicle(item):
                    # Propagate nested classification if no classification yet
                    if not symbol["predicted_class"]:
                        symbol["predicted_class"] = item.get("predicted_class")
                        symbol["predicted_probability"] = item.get(
                            "predicted_probability"
                        )
                    return True

        return False

    def batch_classify_names(self, names):
        """
        Run a single batched MNLI call for all unique uncached names and populate cls_cache.
        """
        uncached = [n for n in names if n not in self.cls_cache]
        if not uncached:
            return

        texts = [n.lower().replace("_", " ") for n in uncached]
        results = self.clf(
            texts,
            candidate_labels=CLASSIFICATION_LABELS,
            multi_label=True,
            hypothesis_template="This item is a {}.",
        )

        # Pipeline returns a list when given a list
        for name, out in zip(uncached, results):
            scores = out["scores"]
            max_idx = int(np.argmax(scores))
            predicted_class = out["labels"][max_idx]
            predicted_prob = scores[max_idx]
            is_vehicle = (
                predicted_class == LABEL_VEHICLE
                and predicted_prob > VEHICLE_PROB_THRESHOLD
            )
            self.cls_cache[name] = {
                "is_vehicle": is_vehicle,
                "predicted_class": predicted_class,
                "predicted_probability": predicted_prob,
            }

    def classify(self, symbols, primitives, scene_objects):
        """
        Route symbols/primitives into scene_objects buckets and associate
        nearby text labels with vehicles. Mutates and returns scene_objects.
        """
        # Pre-classify all unique symbol names in one batched request
        names_to_classify = [
            s["name"] for s in symbols if s["name"] and not s["vehicle2d"]
        ]
        self.batch_classify_names(names_to_classify)

        vehicle_candidates = []
        texts = []
        others = []
        roadway = []
        road_markings = []

        print("Classifying symbols for vehicle candidates...")
        for p in symbols:
            if self.check_vehicle(p):
                p["associated_text"] = []
                for it in p["items"]:
                    if it["type"] == "label":
                        p["associated_text"].append(it["text"])
                vehicle_candidates.append(p)
            elif (
                p["predicted_probability"]
                and p["predicted_probability"] > 0.5
                and p["predicted_class"] in [LABEL_ROAD_MARKING, LABEL_TURN_DIRECTION]
            ):
                road_markings.append(p)
            else:
                others.append(p)

        for p in primitives:
            if p["type"] == "label":
                texts.append(p)
            elif p["type"] in ["polycurve", "polyline", "line", "flexconcretebarrier"]:
                roadway.append(p)
            elif p["type"] == "image":
                scene_objects["images"].append(p)
            else:
                others.append(p)

        # Associate text to vehicles (spatial nearest-neighbor join)
        for txt in texts:
            txt_pos = txt["transformed_center"]
            best_dist = float("inf")
            best_vehicle = None

            for veh in vehicle_candidates:
                vx, vy = veh["transformed_center"]
                dist = np.hypot(vx - txt_pos[0], vy - txt_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_vehicle = veh

            if best_vehicle and best_dist < TEXT_JOIN_MAX_DIST:
                best_vehicle["associated_text"].append(txt["text"])

        scene_objects["vehicles"] = vehicle_candidates
        scene_objects["misc"] = others
        scene_objects["roadway"] = roadway
        scene_objects["road_markings"] = road_markings
        scene_objects["texts"] = texts
        return scene_objects
