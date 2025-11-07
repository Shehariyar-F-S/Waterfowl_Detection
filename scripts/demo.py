import os, pandas as pd

csv_path = "/Users/shehariyarfs/Desktop/MAI./CV/46k66mz9sz-2/00_UAV-derived Thermal Waterfowl Dataset/00_UAV-derived Waterfowl Thermal Imagery Dataset/01_Thermal Images and Ground Truth (used for detector training and testing)/02_Groundtruth Label for Positive Images/Bounding Box Label.csv"
pos_dir = "/Users/shehariyarfs/Desktop/MAI./CV/46k66mz9sz-2/00_UAV-derived Thermal Waterfowl Dataset/00_UAV-derived Waterfowl Thermal Imagery Dataset/01_Thermal Images and Ground Truth (used for detector training and testing)/01_Posistive Image"


df = pd.read_csv(csv_path)
df['filename'] = df['filename'].str.strip()  # remove extra spaces

csv_files = set(df['filename'].str.replace('.tif', '.jpg'))
jpg_files = set(os.listdir(pos_dir))

missing = csv_files - jpg_files
print(f"CSV labels without matching images: {len(missing)}")
print(list(missing)[:10])
