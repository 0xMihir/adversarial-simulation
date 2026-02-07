from transformers import pipeline

clf = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli",
    device=-1,  # CPU; set to 0 for GPU
)

text = "suv dashed midsize V1"

labels = [
    "vehicle (car, suv, truck, bus)",
    "traffic light / signal",
    "road marking / lane line",
    "direction arrow",
    "pedestrian",
    "background / decoration",
]

out = clf(
    text,
    candidate_labels=labels,
    multi_label=True,          # <-- important for multi-label
    hypothesis_template="This item is a {}."
)

print(out["labels"])
print(out["scores"])
