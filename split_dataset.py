import os, random, shutil

# Paths
#input path
img_labels_dirs = ["../TACO/data/batch_1", "../TACO/data/batch_2"]
# label_dir = "datasets/taco/labels_all"
out_dir = "./datasets/taco"

# Output structure
os.makedirs(f"{out_dir}/images/train", exist_ok=True)
os.makedirs(f"{out_dir}/images/val", exist_ok=True)
os.makedirs(f"{out_dir}/labels/train", exist_ok=True)
os.makedirs(f"{out_dir}/labels/val", exist_ok=True)

# Split ratio
val_ratio = 0.2

for img_labels_dir in img_labels_dirs:
    # List all images
    images = [f for f in os.listdir(img_labels_dir) if f.endswith(".jpg")]
    random.shuffle(images)
    val_count = int(len(images) * val_ratio)

    for i, img_name in enumerate(images):
        base = os.path.splitext(img_name)[0]
        label_name = base + ".txt"

        if i < val_count:
            subset = "val"
        else:
            subset = "train"

        shutil.copy(f"{img_labels_dir}/{img_name}", f"{out_dir}/images/{subset}/{img_name}")
        # if os.path.exists(f"{label_dir}/{label_name}"):
        shutil.copy(f"{img_labels_dir}/{label_name}", f"{out_dir}/labels/{subset}/{label_name}")



