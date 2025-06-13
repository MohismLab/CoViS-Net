'''This script evaluates a group of agents with a leader-follower formation,
writes the positions and orientations of the agents to a CSV file'''

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from train.rendering import render_single
from pathlib import Path
import csv
from dataclasses import dataclass

CUDA = False
METERS_PER_PIXEL = 0.1
TOTAL_IMAGES = 200
CAM_FOV = 120
LAYOUT_RANGE = range(20, 25)  # Layouts to evaluate

@dataclass
class PosRange:
    """Position range definition for 3D space boundaries."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def get_width(self) -> float:
        """Get width of the range (x-axis)."""
        return self.x_max - self.x_min

    def get_length(self) -> float:
        """Get length of the range (y-axis)."""
        return self.y_max - self.y_min

    def get_center(self) -> tuple[float, float, float]:
        """Get center point of the range."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        )


map_range = PosRange(
    x_min=-5.0,
    x_max=5.0,
    y_min=-3.6,
    y_max=5.1,
    z_min=-0.64,
    z_max=2.1,
)


def load_img(path):
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            # RandomCenterCrop(0.5),
            transforms.Resize(
                224,
                antialias=True,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float),
        ]
    )
    # img = Image.open(path)
    img = Image.open(path).convert("RGB")  # Ensure the image has 3 channels (RGB)
    img = transform(img)
    # print(f"Loaded image shape: {img.shape}")
    return img


folder = [
    f"/workspace/shiyuan_ws/CoViS-Net/datasets/simple_room_0_square_formation/{layout}/rgb"
    for layout in LAYOUT_RANGE
]
gt_csv = [
    f"/workspace/shiyuan_ws/CoViS-Net/datasets/simple_room_0_square_formation/{layout}/data.csv"
    for layout in LAYOUT_RANGE]

def load_gt_csv_simple(csv_path):
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


def find_gt_entry(gt_data, image_id):
    """Find ground truth entry by image_id"""
    for gt_entry in gt_data:
        # print(f"Checking gt_entry: {gt_entry['image_id']} against {image_id}")
        if gt_entry["image_id"] == image_id:
            return gt_entry
    return None


def quaternion_rotate_point(point, quat):
    """
    Rotate a 3D point by a quaternion
    point: [x, y, z]
    quat: [w, x, y, z]
    """
    # Convert point to quaternion: [0, x, y, z]
    point_quat = [0.0, point[0], point[1], point[2]]

    # Quaternion conjugate of rotation quat
    quat_conj = [quat[0], -quat[1], -quat[2], -quat[3]]

    # Rotate: result = quat * point_quat * quat_conjugate
    temp = quaternion_multiply(quat, point_quat)
    result = quaternion_multiply(temp, quat_conj)

    # Return the vector part [x, y, z]
    return [result[1], result[2], result[3]]


def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 * q2, both in [w,x,y,z] format"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
    ]


def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion [w, x, y, z]"""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    quat_w = (cr * cp * cy) + (sr * sp * sy)
    quat_x = (sr * cp * cy) - (cr * sp * sy)
    quat_y = (cr * sp * cy) + (sr * cp * sy)
    quat_z = (cr * cp * sy) - (sr * sp * cy)

    return [quat_w.item(), quat_x.item(), quat_y.item(), quat_z.item()]


def run(model_base):
    if not CUDA:
        enc = torch.jit.load(f"models/{model_base}_float32_jit_cpu_enc.ts")
        msg = torch.jit.load(f"models/{model_base}_float32_jit_cpu_msg.ts")
        post = torch.jit.load(f"models/{model_base}_float32_jit_cpu_post.ts")
        bev = torch.jit.load(f"models/{model_base}_float32_jit_cpu_bev.ts")
        bev_dec = torch.jit.load(f"models/{model_base}_float32_jit_cpu_bevdec.ts")
    else:
        enc = torch.jit.load(f"models/{model_base}_float32_jit_cuda_enc.ts")
        msg = torch.jit.load(f"models/{model_base}_float32_jit_cuda_msg.ts")
        post = torch.jit.load(f"models/{model_base}_float32_jit_cuda_post.ts")
        bev = torch.jit.load(f"models/{model_base}_float32_jit_cuda_bev.ts")
        bev_dec = torch.jit.load(f"models/{model_base}_float32_jit_cuda_bevdec.ts")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("cuda available:", torch.cuda.is_available())

    # Collect all evaluation data and ground truth data from all layouts
    all_data_eval = []
    all_data_gt = []
    global_index = 0
    
    # Iterate through each layout
    for layout_idx, layout in enumerate(LAYOUT_RANGE):
        print(f"\n{'='*60}")
        print(f"Processing Layout {layout} ({layout_idx + 1}/{len(LAYOUT_RANGE)})")
        print(f"{'='*60}")
        
        # Get current layout paths
        current_folder = f"/workspace/shiyuan_ws/CoViS-Net/datasets/simple_room_0_square_formation/{layout}/rgb"
        current_gt_csv = f"/workspace/shiyuan_ws/CoViS-Net/datasets/simple_room_0_square_formation/{layout}/data.csv"
        
        # Count available images for current layout
        available_images = 0
        while Path(current_folder + f"/{available_images:06d}.png").exists():
            available_images += 1

        print(f"Found {available_images} images in layout {layout}")
        
        if available_images == 0:
            print(f"No images found in layout {layout}, skipping...")
            continue

        # Load ground truth for current layout
        data_gt = load_gt_csv_simple(current_gt_csv)
        if not data_gt:
            print(f"Failed to load ground truth for layout {layout}, skipping...")
            continue
            
        layout_data_eval = []
        layout_data_gt = []
        
        # Process images in groups of 4
        max_images = min(available_images, TOTAL_IMAGES)
        for l in range(0, max_images - 3, 4):
            scene_paths = [
                current_folder + f"/{l:06d}.png",
                current_folder + f"/{l+1:06d}.png",
                current_folder + f"/{l+2:06d}.png",
                current_folder + f"/{l+3:06d}.png",
            ]
            
            # Check if all images in the group exist
            if not all(Path(path).exists() for path in scene_paths):
                print(f"Some images in group {l//4} don't exist, skipping group...")
                continue
                
            is_group_leader = True
            leader_pos_gt = {}

            for k, path in enumerate(scene_paths):
                index = image_id = l + k
                group_id = (l + k) // 4

                print(f"scene_path: {path}")
                print(f"group_id: {group_id}, image_id: {image_id}")
                
                # Add ground truth entry with updated group_id
                gt_entry = find_gt_entry(data_gt, image_id)
                if gt_entry is not None:
                    unique_group_id = group_id + layout * 1000  # Same as eval results
                    layout_data_gt.append([
                        global_index,                    # Global unique index (same as eval)
                        unique_group_id,                 # Unique group ID across layouts
                        gt_entry["image_id"],            # Original image ID
                        gt_entry["topdown_id"],          # topdown_id
                        gt_entry["pos_x"],               # world x position
                        gt_entry["pos_y"],               # world y position  
                        gt_entry["pos_z"],               # world z position
                        gt_entry["quat_w"],              # world quaternion w
                        gt_entry["quat_x"],              # world quaternion x
                        gt_entry["quat_y"],              # world quaternion y
                        gt_entry["quat_z"],              # world quaternion z
                        gt_entry["cam_fov"],             # camera FOV
                        gt_entry["pixel_pos_x"],         # pixel x position
                        gt_entry["pixel_pos_y"],         # pixel y position
                    ])
                
                data = {
                    "img": torch.stack(
                        [load_img(scene_paths[0]), load_img(scene_paths[k])],
                        dim=0,
                    ),
                    "encs": None,
                    "edge_index": [[], []],
                    "edge_preds": {
                        "msg": [],
                        "pos": [],
                        "rot": [],
                        "pos_var": [],
                        "rot_var": [],
                    },
                    "node_preds": [],
                }

                with torch.no_grad():
                    if not CUDA:
                        data["encs"] = [enc(img.unsqueeze(0)) for img in data["img"]]
                    else:
                        data["encs"] = [
                            enc(img.unsqueeze(0).to(device)) for img in data["img"]
                        ]

                    for i, enc_i in enumerate(data["encs"]):
                        for j, enc_j in enumerate(data["encs"]):
                            if i == j:
                                continue
                            
                            if i == 1 and j == 0:
                                m = msg(enc_i, enc_j)
                                data["edge_preds"]["msg"].append(m)

                                pos, pos_var, heading, heading_var = post(m)
                                # Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)
                                q = heading.squeeze()
                                x, y, z, w = q[0], q[1], q[2], q[3]

                                # Compute Euler angles
                                t0 = 2.0 * (w * x + y * z)
                                t1 = 1.0 - 2.0 * (x * x + y * y)
                                roll = torch.atan2(t0, t1)

                                t2 = 2.0 * (w * y - z * x)
                                t2 = torch.clamp(t2, -1.0, 1.0)
                                pitch = torch.asin(t2)

                                t3 = 2.0 * (w * z + x * y)
                                t4 = 1.0 - 2.0 * (y * y + z * z)
                                yaw = torch.atan2(t3, t4)

                                if is_group_leader:
                                    is_group_leader = False
                                    
                                    # Leader uses ground truth position and orientation
                                    if gt_entry is None:
                                        print(f"No ground truth found for image_id {image_id}, skipping...")
                                        continue
                                        
                                    leader_pos_gt = {
                                        "pos_x": gt_entry["pos_x"],
                                        "pos_y": gt_entry["pos_y"],
                                        "pos_z": gt_entry["pos_z"],
                                        "quat_w": gt_entry["quat_w"],
                                        "quat_x": gt_entry["quat_x"],
                                        "quat_y": gt_entry["quat_y"],
                                        "quat_z": gt_entry["quat_z"],
                                    }
                                    
                                    # For leader, use ground truth pose directly
                                    world_pos_x = leader_pos_gt["pos_x"]
                                    world_pos_y = leader_pos_gt["pos_y"]
                                    world_pos_z = leader_pos_gt["pos_z"]
                                    world_quat_w = leader_pos_gt["quat_w"]
                                    world_quat_x = leader_pos_gt["quat_x"]
                                    world_quat_y = leader_pos_gt["quat_y"]
                                    world_quat_z = leader_pos_gt["quat_z"]
                                    
                                else:
                                    # For followers, transform from leader's local coordinate system to world
                                    
                                    # 1. Local position in leader's coordinate system
                                    pos_x = -pos[0, 0].item()
                                    pos_y = pos[0, 1].item()
                                    pos_z = -pos[0, 2].item()
                                    local_pos = [pos_x, pos_y, pos_z]
                                    
                                    # 2. Leader's orientation quaternion
                                    leader_quat = [
                                        leader_pos_gt["quat_w"],
                                        leader_pos_gt["quat_x"],
                                        leader_pos_gt["quat_y"],
                                        leader_pos_gt["quat_z"]
                                    ]
                                    
                                    # 3. Rotate local position by leader's orientation
                                    rotated_pos = quaternion_rotate_point(local_pos, leader_quat)
                                    
                                    # 4. Translate to world coordinates
                                    world_pos_x = leader_pos_gt["pos_x"] + rotated_pos[0]
                                    world_pos_y = leader_pos_gt["pos_y"] + rotated_pos[1]
                                    world_pos_z = leader_pos_gt["pos_z"] + rotated_pos[2]
                                    
                                    # 5. Transform orientation
                                    # Convert local Euler angles to quaternion
                                    local_quat = quaternion_from_euler(roll, -pitch, yaw)
                                    
                                    # Compose with leader's orientation: world_quat = leader_quat * local_quat
                                    world_quat = quaternion_multiply(leader_quat, local_quat)
                                    world_quat_w = world_quat[0]
                                    world_quat_x = world_quat[1]
                                    world_quat_y = world_quat[2]
                                    world_quat_z = world_quat[3]

                                print(f"World pos: ({world_pos_x:.3f}, {world_pos_y:.3f}, {world_pos_z:.3f})")

                                # Calculate pixel positions based on world coordinates
                                pixel_pos_x = float(
                                    "{:.3f}".format(
                                        (world_pos_x - map_range.x_min) / METERS_PER_PIXEL
                                    )
                                )
                                pixel_pos_y = float(
                                    "{:.3f}".format(
                                        (world_pos_z - map_range.y_min) / METERS_PER_PIXEL
                                    )
                                )

                                # Use global_index for unique indexing across all layouts
                                # Keep original group_id and image_id, but make group_id unique across layouts
                                unique_group_id = group_id + layout * 1000  # Make group IDs unique across layouts
                                
                                layout_data_eval.append([
                                    global_index,        # Global unique index
                                    unique_group_id,     # Unique group ID across layouts
                                    image_id,            # Original image ID within layout
                                    0,                   # topdown_id
                                    world_pos_x,         # world x position
                                    world_pos_y,         # world y position
                                    world_pos_z,         # world z position
                                    world_quat_w,        # world quaternion w
                                    world_quat_x,        # world quaternion x
                                    world_quat_y,        # world quaternion y
                                    world_quat_z,        # world quaternion z
                                    CAM_FOV,
                                    pixel_pos_x,
                                    pixel_pos_y,
                                ])
                                global_index += 1

        print(f"Layout {layout} processed: {len(layout_data_eval)} eval entries, {len(layout_data_gt)} GT entries")
        all_data_eval.extend(layout_data_eval)
        all_data_gt.extend(layout_data_gt)

    # Write evaluation results to CSV file
    print(f"\n{'='*60}")
    print(f"Writing evaluation results to eval_results.csv")
    print(f"Total eval entries: {len(all_data_eval)}")
    print(f"{'='*60}")
    
    with open("eval_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(
            [
                "index",
                "group_id",
                "image_id",
                "topdown_id",
                "pos_x",
                "pos_y",
                "pos_z",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "cam_fov",
                "pixel_pos_x",
                "pixel_pos_y",
            ]
        )
        # Write all rows from all layouts
        writer.writerows(all_data_eval)
    
    print("Evaluation results saved to eval_results.csv")

    # Write ground truth data to CSV file
    print(f"\n{'='*60}")
    print(f"Writing ground truth data to gt.csv")
    print(f"Total GT entries: {len(all_data_gt)}")
    print(f"{'='*60}")
    
    with open("gt.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Write header (same as eval_results.csv)
        writer.writerow(
            [
                "index",
                "group_id",
                "image_id",
                "topdown_id",
                "pos_x",
                "pos_y",
                "pos_z",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "cam_fov",
                "pixel_pos_x",
                "pixel_pos_y",
            ]
        )
        # Write all ground truth rows from all layouts
        writer.writerows(all_data_gt)
    
    print("Ground truth data saved to gt.csv")
    
    # Print summary statistics
    layouts_processed = []
    for layout in LAYOUT_RANGE:
        eval_count = sum(1 for row in all_data_eval if row[1] // 1000 == layout)
        gt_count = sum(1 for row in all_data_gt if row[1] // 1000 == layout)
        if eval_count > 0 or gt_count > 0:
            layouts_processed.append(layout)
            print(f"  Layout {layout}: {eval_count} eval entries, {gt_count} GT entries")
    
    print(f"Successfully processed {len(layouts_processed)} layouts: {layouts_processed}")
    print(f"\nOutput files generated:")
    print(f"  - eval_results.csv: {len(all_data_eval)} prediction entries")
    print(f"  - gt.csv: {len(all_data_gt)} ground truth entries")


if __name__ == "__main__":
    # run("0kc5po4ee18")
    # my model
    run("7v8j82qce29")
    # run("mr5eierxe09")
