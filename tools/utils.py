import csv

def load_csv_simple(csv_path):
    """Load ground truth CSV into a list using standard csv module"""
    gt_data = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_entry = {
                    "index": int(row["index"]),
                    "group_id": int(row["group_id"]),
                    "image_id": int(row["image_id"]),
                    "topdown_id": int(row["topdown_id"]),
                    "pos_x": float(row["pos_x"]),
                    "pos_y": float(row["pos_y"]),
                    "pos_z": float(row["pos_z"]),
                    "quat_w": float(row["quat_w"]),
                    "quat_x": float(row["quat_x"]),
                    "quat_y": float(row["quat_y"]),
                    "quat_z": float(row["quat_z"]),
                    "cam_fov": float(row["cam_fov"]),
                    "pixel_pos_x": float(row["pixel_pos_x"]),
                    "pixel_pos_y": float(row["pixel_pos_y"]),
                }
                gt_data.append(gt_entry)

        print(f"Loaded {len(gt_data)} ground truth entries")
        return gt_data

    except Exception as e:
        print(f"Error loading ground truth CSV: {e}")
        return []
    
def load_csv_with_error(csv_path):
    """Load ground truth CSV into a list using standard csv module"""
    gt_data = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt_entry = {
                    "index": int(row["index"]),
                    "group_id": int(row["group_id"]),
                    "image_id": int(row["image_id"]),
                    "topdown_id": int(row["topdown_id"]),
                    "pos_x": float(row["pos_x"]),
                    "pos_y": float(row["pos_y"]),
                    "pos_z": float(row["pos_z"]),
                    "quat_w": float(row["quat_w"]),
                    "quat_x": float(row["quat_x"]),
                    "quat_y": float(row["quat_y"]),
                    "quat_z": float(row["quat_z"]),
                    "cam_fov": float(row["cam_fov"]),
                    "pixel_pos_x": float(row["pixel_pos_x"]),
                    "pixel_pos_y": float(row["pixel_pos_y"]),
                    "pos_error": float(row["pos_error"]),
                    "pitch_error": float(row["pitch_error"]),
                }
                gt_data.append(gt_entry)

        print(f"Loaded {len(gt_data)} ground truth entries")
        return gt_data

    except Exception as e:
        print(f"Error loading ground truth CSV: {e}")
        return []
    