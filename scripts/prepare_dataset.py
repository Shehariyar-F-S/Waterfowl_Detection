import os
import pandas as pd
from PIL import Image
import random
import shutil

# ==== CONFIGURATION ====
base_dir = "/Users/shehariyarfs/Desktop/MAI./CV/46k66mz9sz-2/00_UAV-derived Thermal Waterfowl Dataset/00_UAV-derived Waterfowl Thermal Imagery Dataset/01_Thermal Images and Ground Truth (used for detector training and testing)"

pos_img_dir = os.path.join(base_dir, "01_Posistive Image")
neg_img_dir = os.path.join(base_dir, "03_Negative Images")
csv_path = os.path.join(base_dir, "02_Groundtruth Label for Positive Images", "Bounding Box Label.csv")

output_image_dir = "data/images/all"
output_label_dir = "data/labels/all"
split_root = "data/images"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

print("🐦 Starting dataset preparation...")

# ==== STEP 1: Convert .tif → .jpg (Positive + Negative) ====
def convert_tif_to_jpg(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for f in os.listdir(src_dir):
        if f.lower().endswith(".tif"):
            img_path = os.path.join(src_dir, f)
            out_path = os.path.join(dst_dir, f.replace(".tif", ".jpg"))
            try:
                Image.open(img_path).convert("RGB").save(out_path, "JPEG")
                count += 1
            except Exception as e:
                print(f"⚠️ Error converting {f}: {e}")
    print(f"✅ Converted {count} .tif images from {src_dir}")

print("🔄 Converting .tif to .jpg ...")
convert_tif_to_jpg(pos_img_dir, output_image_dir)
convert_tif_to_jpg(neg_img_dir, output_image_dir)

# ==== STEP 2: Generate YOLO Labels ====
def generate_yolo_labels():
    print("\n🧾 Generating YOLO labels from CSV ...")

    # Load CSV correctly (comma-separated)
    df = pd.read_csv(csv_path, header=0, sep=',')
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        'imageFilename': 'filename',
        'x(column)': 'x_min',
        'y(row)': 'y_min',
        'width': 'width',
        'height': 'height'
    }
    df = df.rename(columns=rename_map)

    # Clean & validate
    df = df.dropna(subset=['filename', 'x_min', 'y_min', 'width', 'height'])
    for c in ['x_min', 'y_min', 'width', 'height']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['x_min', 'y_min', 'width', 'height'])

    print("First 5 parsed rows:")
    print(df.head())

    os.makedirs(output_label_dir, exist_ok=True)
    label_count = 0

    # Group and write YOLO labels
    for filename, group in df.groupby('filename'):
        jpg_name = filename.strip().replace('.TIF', '.jpg').replace('.tif', '.jpg')
        img_path = os.path.join(output_image_dir, jpg_name)
        if not os.path.exists(img_path):
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        label_path = os.path.join(output_label_dir, jpg_name.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                x_center = (row['x_min'] + row['width'] / 2) / img_w
                y_center = (row['y_min'] + row['height'] / 2) / img_h
                w = row['width'] / img_w
                h = row['height'] / img_h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                label_count += 1

    print(f"✅ YOLO label generation complete ({label_count} boxes written).\n")

# === Check if label directory exists or is empty ===
if not os.path.exists(output_label_dir) or len(os.listdir(output_label_dir)) == 0:
    generate_yolo_labels()
else:
    empty_labels = [f for f in os.listdir(output_label_dir)
                    if os.path.getsize(os.path.join(output_label_dir, f)) == 0]
    if len(empty_labels) > 0:
        print(f"⚠️ Found {len(empty_labels)} empty label files, regenerating ...")
        generate_yolo_labels()
    else:
        print("✅ Label files already exist and are non-empty — skipping regeneration.\n")

# ==== STEP 3: Split train/val/test ====
def split_dataset():
    print("✂️ Splitting dataset into train/val/test ...")

    import random
    from glob import glob

    os.makedirs("data/images/train", exist_ok=True)
    os.makedirs("data/images/val", exist_ok=True)
    os.makedirs("data/images/test", exist_ok=True)
    os.makedirs("data/labels/train", exist_ok=True)
    os.makedirs("data/labels/val", exist_ok=True)
    os.makedirs("data/labels/test", exist_ok=True)

    all_images = glob(os.path.join(output_image_dir, "*.jpg"))
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:]
    }

    for split, files in splits.items():
        for f in files:
            shutil.copy(f, f"data/images/{split}/")
            label_file = os.path.basename(f).replace(".jpg", ".txt")
            src_label = os.path.join(output_label_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy(src_label, f"data/labels/{split}/{label_file}")

    print(f"✅ Split complete:\n  Train: {n_train}\n  Val: {n_val}\n  Test: {n_test}\n")

split_dataset()

# ==== SUMMARY ====
def count_files(path, ext):
    if not os.path.exists(path): return 0
    return len([f for f in os.listdir(path) if f.endswith(ext)])

train_count = count_files("data/images/train", ".jpg")
val_count = count_files("data/images/val", ".jpg")
test_count = count_files("data/images/test", ".jpg")
total_labels = count_files(output_label_dir, ".txt")

print(f"📊 DATASET SUMMARY:")
print(f"  • Train: {train_count} images")
print(f"  • Val:   {val_count} images")
print(f"  • Test:  {test_count} images")
print(f"  • Total label files: {total_labels}")
print(f"✅ Dataset preparation complete!")