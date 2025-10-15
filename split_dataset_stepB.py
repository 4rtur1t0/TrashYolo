import os, random, shutil

# Paths
#input path
# add directories
img_labels_dirs = ["../TACO/data/batch_1",
                   "../TACO/data/batch_2",
                   "../TACO/data/batch_8",
                   "../TACO/data/batch_9",
                   "../TACO/data/batch_10",
                   "../TACO/data/batch_11",
                   "../TACO/data/batch_12",
                   "../TACO/data/batch_13",
                   "../TACO/data/batch_14",
                   "../TACO/data/batch_15"]
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
    # get the batch number. Som
    batch_name = img_labels_dir.split('_')[1]
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
        # finally, copy to directory with the batch name
        shutil.copy(f"{img_labels_dir}/{img_name}", f"{out_dir}/images/{subset}/{batch_name+'_'+img_name}")
        # if os.path.exists(f"{label_dir}/{label_name}"):
        shutil.copy(f"{img_labels_dir}/{label_name}", f"{out_dir}/labels/{subset}/{batch_name+'_'+label_name}")



