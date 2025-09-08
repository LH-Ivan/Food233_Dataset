import os
import shutil

# Paths
source_folder = "labelme_annotations"  # Replace with your source folder
destination_folder = "labelme_annotations2"  # Replace with your destination folder

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through files in the source folder
for filename in os.listdir(source_folder):
    # Check if file is an original (does not contain '_aug_')
    if "_aug_" not in filename and filename.lower().endswith((".json")):
        source_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(destination_folder, filename)
        shutil.copy2(source_path, dest_path)  # Copy with metadata
        print(f"Copied: {filename}")

print("Done copying original images.")
