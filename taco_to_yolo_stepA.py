import json, os
from tqdm import tqdm

input_json = '../TACO/data/annotations.json'
output_dir = '../TACO/data/'
os.makedirs(output_dir, exist_ok=True)

with open(input_json) as f:
    data = json.load(f)

for img in tqdm(data['images']):
    img_id = img['id']
    file_name = img['file_name']
    width, height = img['width'], img['height']

    anns = [a for a in data['annotations'] if a['image_id'] == img_id]

    lines = []
    for a in anns:
        cat_id = a['category_id'] - 1  # YOLO expects 0-based
        bbox = a['bbox']  # [x, y, width, height]
        x_c = (bbox[0] + bbox[2] / 2) / width
        y_c = (bbox[1] + bbox[3] / 2) / height
        w = bbox[2] / width
        h = bbox[3] / height
        lines.append(f"{cat_id} {x_c} {y_c} {w} {h}\n")
    output_filepath = os.path.join(output_dir, file_name.replace('.jpg', '.txt'))
    with open(os.path.join(output_dir, file_name.replace('.jpg', '.txt')), 'w') as f:
        f.writelines(lines)

