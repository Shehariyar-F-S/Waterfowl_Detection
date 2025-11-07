import pandas as pd

csv_path = "/Users/shehariyarfs/Desktop/MAI./CV/46k66mz9sz-2/00_UAV-derived Thermal Waterfowl Dataset/00_UAV-derived Waterfowl Thermal Imagery Dataset/01_Thermal Images and Ground Truth (used for detector training and testing)/02_Groundtruth Label for Positive Images/Bounding Box Label.csv"

# Try reading a few lines raw
with open(csv_path, "r") as f:
    for _ in range(5):
        print(repr(f.readline()))
