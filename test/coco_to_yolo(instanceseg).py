import json
import os

# ==== CONFIG ====
json_path = r"C:\Users\ivanc\OneDrive\Desktop\010825_Dataset\Annotate_Images\split_images\val\labels_my-project-name_2025-08-06-01-10-29.json"  # your COCO-format JSON
images_dir = r"C:\Users\ivanc\OneDrive\Desktop\010825_Dataset\Annotate_Images\split_images\val\images"           # folder with your images
labels_dir = r"C:\Users\ivanc\OneDrive\Desktop\010825_Dataset\Annotate_Images\split_images\val\labels"           # output YOLOv11 label files
os.makedirs(labels_dir, exist_ok=True)

# Load COCO JSON
with open(json_path, "r") as f:
    coco_data = json.load(f)

# Map image_id to file_name, width, height
image_info = {img["id"]: img for img in coco_data["images"]}

# Map category_id to YOLO class index (0-based)
cat_id_to_class = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

# Create label files
for ann in coco_data["annotations"]:
    img_id = ann["image_id"]
    category_id = ann["category_id"]
    segmentation = ann["segmentation"][0]  # single polygon

    img_data = image_info[img_id]
    img_w, img_h = img_data["width"], img_data["height"]

    # Normalize polygon points
    norm_points = []
    for i in range(0, len(segmentation), 2):
        x = segmentation[i] / img_w
        y = segmentation[i + 1] / img_h
        norm_points.extend([x, y])

    # Class ID (0-based)
    class_id = cat_id_to_class[category_id]

    # Label file path
    label_file = os.path.splitext(img_data["file_name"])[0] + ".txt"
    label_path = os.path.join(labels_dir, label_file)

    # Append polygon to file
    with open(label_path, "a") as lf:
        lf.write(f"{class_id} " + " ".join(map(str, norm_points)) + "\n")

print(f"✅ Conversion completed! Labels saved in '{labels_dir}'")
