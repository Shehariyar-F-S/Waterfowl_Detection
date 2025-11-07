import os
import pandas as pd
from PIL import Image
import random
import shutil

# ==== CONFIGURATION ====
base_dir = "/Users/shehariyarfs/Desktop/MAI./CV/46k66mz9sz-2/00_UAV-derived Thermal Waterfowl Dataset/00_UAV-derived Waterfowl Thermal Imagery Dataset/01_Thermal Images and Ground Truth (used for detector training and testing)"

pos_dir = os.path.join(base_dir, "01_Posistive Image")
neg_dir = os.path.join(base_dir, "03_Negative Images")
csv_path = os.path.join(base_dir, "02_Groundtruth Label for Positive Images", "Bounding Box Label.csv")

output_image_dir = "data/images/all"
output_label_dir = "data/labels/all"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# ==== STEP 1: Convert positive .tif → .jpg ====
print("🔄 Converting positive .tif images to .jpg ...")
for file in os.listdir(pos_dir):
    if file.endswith(".tif"):
        jpg_path = os.path.join(output_image_dir, file.replace(".tif", ".jpg"))
        if not os.path.exists(jpg_path):  # skip if already converted
            img = Image.open(os.path.join(pos_dir, file))
            rgb_img = img.convert("RGB")
            rgb_img.save(jpg_path)
print("✅ Positive image conversion complete.\n")

# ==== STEP 2: Generate YOLO labels from CSV ====
print("🧾 Generating YOLO labels from CSV ...")

# Read as comma-separated with header
df = pd.read_csv(csv_path, header=0, sep=',')

# Normalize/rename columns
df.columns = [c.strip() for c in df.columns]  # strip spaces
rename_map = {
    'imageFilename': 'filename',
    'x(column)': 'x_min',
    'y(row)': 'y_min',
    'width': 'width',
    'height': 'height'
}
df = df.rename(columns=rename_map)

# Basic sanity: drop rows with missing values and coerce numeric types
df = df.dropna(subset=['filename', 'x_min', 'y_min', 'width', 'height'])
for c in ['x_min', 'y_min', 'width', 'height']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['x_min', 'y_min', 'width', 'height'])

print("First 5 parsed rows:")
print(df.head())

print("✅ YOLO label generation complete.\n")

# ==== STEP 3: Handle negative images ====
print("🚫 Processing negative images (no waterfowl) ...")
for file in os.listdir(neg_dir):
    if file.endswith(".tif"):
        jpg_path = os.path.join(output_image_dir, file.replace(".tif", ".jpg"))
        txt_path = os.path.join(output_label_dir, file.replace(".tif", ".txt"))
        if not os.path.exists(jpg_path):
            img = Image.open(os.path.join(neg_dir, file))
            rgb_img = img.convert("RGB")
            rgb_img.save(jpg_path)
        if not os.path.exists(txt_path):
            open(txt_path, 'w').close()
print("✅ Negative images processed.\n")

# ==== STEP 4: Split dataset ====
random.seed(42)  # ensures reproducible splits

if os.path.exists("data/images/train") and os.listdir("data/images/train"):
    print("⚠️  Dataset split already exists — skipping splitting step.\n")
else:
    print("✂️ Splitting dataset into train/val/test ...")
    images = [f for f in os.listdir(output_image_dir) if f.endswith(".jpg")]
    random.shuffle(images)

    n = len(images)
    train_split = int(0.8 * n)
    val_split = int(0.9 * n)

    splits = {
        'train': images[:train_split],
        'val': images[train_split:val_split],
        'test': images[val_split:]
    }

    for split, files in splits.items():
        os.makedirs(f"data/images/{split}", exist_ok=True)
        os.makedirs(f"data/labels/{split}", exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(output_image_dir, f), f"data/images/{split}/{f}")
            label_src = os.path.join(output_label_dir, f.replace(".jpg", ".txt"))
            label_dst = f"data/labels/{split}/{f.replace('.jpg', '.txt')}"

            # if missing, create empty file
            if not os.path.exists(label_src):
                print(f"⚠️ Warning: Missing label for {f} — creating empty file.")
                open(label_dst, 'w').close()
            else:
                shutil.copy(label_src, label_dst)

    print("✅ Dataset splitting complete.\n")

# ==== STEP 5: Dataset Summary ====
def count_files(path, ext):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return 0
    return len([f for f in os.listdir(path) if f.endswith(ext)])

train_count = count_files("data/images/train", ".jpg")
val_count = count_files("data/images/val", ".jpg")
test_count = count_files("data/images/test", ".jpg")
total_labels = count_files(output_label_dir, ".txt")

print("\n📊 DATASET SUMMARY:")
print(f"  • Train: {train_count} images")
print(f"  • Val:   {val_count} images")
print(f"  • Test:  {test_count} images")
print(f"  • Total label files (all): {total_labels}")
print(f"  • Total images overall: {train_count + val_count + test_count}")

print("\n✅ Dataset preparation completed successfully!")

