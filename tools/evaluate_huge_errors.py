import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .utils import *

LAYOUT_START = 20  # Layouts to evaluate 20-25
LAYOUT_END = 25
GROUP_SCALE = 4

POS_ERROR_THRESHOLD = 1.2  # Threshold for errors
ROTATION_ERROR_THRESHOLD = 60.0  # Threshold for rotation errors

group_id = 2000

folder = "/workspace/shiyuan_ws/CoViS-Net"

csv_gt = folder + "/eval_plots/gt.csv"
csv_pred = folder + "/eval_plots/eval_results.csv"

pred_data = load_csv_with_error(csv_pred)

layout = LAYOUT_START - 1
for i, pred_entry in enumerate(pred_data):
    
    # Increment layout at the start of each group (every 4 entries)
    if  pred_entry["image_id"] == 0:  # Assuming 4 groups per layout
        layout += 1 
        
    # Only process group leaders (first entry of each group)
    if not (i + 1) % 4 == 1:
        continue

    # Get current layout paths
    current_folder = f"/workspace/shiyuan_ws/CoViS-Net/datasets/simple_room_0_square_formation/{layout}/rgb"
    current_eval_error_csv = f"/workspace/shiyuan_ws/CoViS-Net/datasets/simple_room_0_square_formation/{layout}/eval_error.csv"
    current_eval_error_image = f"/workspace/shiyuan_ws/CoViS-Net/datasets/simple_room_0_square_formation/{layout}/eval_error_rgb"
    Path(current_eval_error_image).mkdir(parents=True, exist_ok=True)

    # Check if any entry in the current group has high error
    group_has_huge_error = False
    group_id = -1  # Reset group_id for each new group

    # Check all 4 entries in the current group
    for j in range(i, min(i + GROUP_SCALE, len(pred_data))):
        current_entry = pred_data[j]
        if current_entry["pos_error"] > POS_ERROR_THRESHOLD or current_entry["pitch_error"] > ROTATION_ERROR_THRESHOLD:
            print(f"******Layout {layout}, Index {current_entry["index"]}, Image {j%200:06d} - Position Error: {current_entry['pos_error']:.2f}, Rotation Error: {current_entry['pitch_error']:.2f}******")
            group_has_huge_error = True
            group_id = current_entry["group_id"]
    
    # If any entry in the group has error, save the entire group
    if group_has_huge_error:
        for k in range(i - GROUP_SCALE, min(i + GROUP_SCALE, len(pred_data))):
            eval_error_dict = pred_data[k].copy()
            if eval_error_dict["group_id"] != group_id:
                continue
            
            n = eval_error_dict["image_id"] 
            if k % 200 != n:
                print(f"Warning: Image ID mismatch, expected {k % 200}, got {n}")
                
            # Append the error dict to a CSV file for further analysis
            with open(current_eval_error_csv, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=eval_error_dict.keys())
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow(eval_error_dict)
                print(f"Saved error entry, image_id: {n}")

            src_img = Path(current_folder) / f"{n:06d}.png"
            if src_img.exists():
                print(f"Processing Image {n:06d}")
                # Copy the error image to the error image folder
                dst_img = Path(current_eval_error_image) / f"{n:06d}.png"
                dst_img.write_bytes(src_img.read_bytes())
            else:
                print(f"Warning: Image not found: {src_img}")



