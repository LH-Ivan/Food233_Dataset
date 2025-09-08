import json

# File paths
# json1_path = "900image.json"   # Main dataset
# json2_path = "100image.json"   # Dataset to merge
json1_path = r"C:\Users\ivanc\FYP_Project\train\labels_my-project-name_2025-08-07-03-13-38.json"
json2_path = r"C:\Users\ivanc\FYP_Project\train\corrupted.json"
output_path = "merged_1000image.json"

# Load JSON files
with open(json1_path, 'r') as f:
    coco1 = json.load(f)
with open(json2_path, 'r') as f:
    coco2 = json.load(f)

# Determine ID offsets to avoid collisions
max_image_id = max(img["id"] for img in coco1["images"])
max_ann_id = max(ann["id"] for ann in coco1["annotations"])

# Adjust IDs in second dataset
for img in coco2["images"]:
    img["id"] += max_image_id  # Shift image IDs

for ann in coco2["annotations"]:
    ann["id"] += max_ann_id  # Shift annotation IDs
    ann["image_id"] += max_image_id  # Shift image_id references

# Merge images and annotations
coco1["images"].extend(coco2["images"])
coco1["annotations"].extend(coco2["annotations"])

# (Optional) Ensure categories are consistent
# Here we assume both have the same categories, so no change is needed

# Save merged file
with open(output_path, 'w') as f:
    json.dump(coco1, f, indent=2)

print(f"Merged dataset saved to {output_path}")
