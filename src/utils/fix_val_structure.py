import os
import shutil

val_dir = "tiny-imagenet-200/val"
images_dir = os.path.join(val_dir, "images")
annotations_file = os.path.join(val_dir, "val_annotations.txt")

# Read annotations
with open(annotations_file, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split("\t")
    img_name = parts[0]
    class_name = parts[1]

    class_dir = os.path.join(val_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    src = os.path.join(images_dir, img_name)
    dst = os.path.join(class_dir, img_name)

    if os.path.exists(src):
        shutil.move(src, dst)

# Remove empty images folder
os.rmdir(images_dir)

print("Validation folder reorganized!")