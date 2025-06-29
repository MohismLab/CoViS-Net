import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

folder = "/workspace/shiyuan_ws/CoViS-Net"

csv_gt = folder + "/eval_plots/gt.csv"
csv_pred = folder + "/eval_plots/eval_results.csv"


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


def quaternion_to_euler_degrees(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) in degrees
    quat: quaternion as [w, x, y, z]
    Returns: (roll, pitch, yaw) in degrees
    """
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)

    return roll_deg, pitch_deg, yaw_deg


def pitch_error_degrees(q1, q2):
    """
    Calculate pitch error between two quaternions in degrees
    Properly handles pitch periodicity and edge cases
    q1, q2: quaternions as [w, x, y, z]
    Returns: minimum pitch error in degrees [0, 180]
    """
    # Convert both quaternions to Euler angles
    roll1, pitch1, yaw1 = quaternion_to_euler_degrees(q1)
    roll2, pitch2, yaw2 = quaternion_to_euler_degrees(q2)

    # Function to calculate minimum angular distance
    def angular_distance_degrees(a1, a2):
        """Calculate minimum angular distance between two angles in degrees"""
        diff = abs(a1 - a2)
        return min(diff, 360 - diff)

    return angular_distance_degrees(pitch1, pitch2)


def normalize_angle_degrees(angle):
    """
    Normalize angle to [-180, 180] degrees range
    """
    angle = angle % 360  # First bring to [0, 360)
    if angle > 180:
        angle -= 360  # Convert to [-180, 180]
    return angle


def is_leader(entry):
    """Check if an entry is a leader (first camera in each group)"""
    return entry["index"] % 4 == 0


def filter_followers_only(data):
    """Filter data to keep only follower entries (exclude leaders)"""
    return [entry for entry in data if not is_leader(entry)]


def calculate_detailed_statistics(gt_data, pred_data):
    """Calculate detailed statistics including min, max, std for both position and pitch errors"""

    pos_errors = []
    pitch_errors = []

    for gt_entry, pred_entry in zip(gt_data, pred_data):
        if gt_entry["index"] != pred_entry["index"]:
            continue

        # Position error (2D: x, y only)
        pos_error = (
            (gt_entry["pos_x"] - pred_entry["pos_x"]) ** 2
            + (gt_entry["pos_y"] - pred_entry["pos_y"]) ** 2
        ) ** 0.5

        # Pitch error
        gt_quat = [
            gt_entry["quat_w"],
            gt_entry["quat_x"],
            gt_entry["quat_y"],
            gt_entry["quat_z"],
        ]
        pred_quat = [
            pred_entry["quat_w"],
            pred_entry["quat_x"],
            pred_entry["quat_y"],
            pred_entry["quat_z"],
        ]
        pitch_err = pitch_error_degrees(gt_quat, pred_quat)

        pos_errors.append(pos_error)
        pitch_errors.append(pitch_err)

    if not pos_errors:
        return None

    # Calculate statistics
    pos_errors.sort()
    pitch_errors.sort()
    n = len(pos_errors)

    stats = {
        "count": n,
        "position": {
            "mean": sum(pos_errors) / n,
            "median": pos_errors[n // 2],
            "min": min(pos_errors),
            "max": max(pos_errors),
            "std": (sum((x - sum(pos_errors) / n) ** 2 for x in pos_errors) / n) ** 0.5,
            "percentile_75": pos_errors[int(0.75 * n)],
            "percentile_95": pos_errors[int(0.95 * n)],
        },
        "orientation": {
            "mean": sum(pitch_errors) / n,
            "median": pitch_errors[n // 2],
            "min": min(pitch_errors),
            "max": max(pitch_errors),
            "std": (sum((x - sum(pitch_errors) / n) ** 2 for x in pitch_errors) / n)
            ** 0.5,
            "percentile_75": pitch_errors[int(0.75 * n)],
            "percentile_95": pitch_errors[int(0.95 * n)],
        },
    }

    return stats


def draw_error_histograms(
    gt_data, pred_data, save_path=None, filter_outliers=False, outlier_range=[5, 95]
):
    """
    Draw histograms for position and pitch errors with optional outlier filtering
    """
    pos_errors = []
    pitch_errors = []

    for gt_entry, pred_entry in zip(gt_data, pred_data):
        if gt_entry["index"] != pred_entry["index"]:
            continue

        # Position error (2D: x, y only)
        pos_error = (
            (gt_entry["pos_x"] - pred_entry["pos_x"]) ** 2
            + (gt_entry["pos_y"] - pred_entry["pos_y"]) ** 2
        ) ** 0.5

        # Pitch error
        gt_quat = [
            gt_entry["quat_w"],
            gt_entry["quat_x"],
            gt_entry["quat_y"],
            gt_entry["quat_z"],
        ]
        pred_quat = [
            pred_entry["quat_w"],
            pred_entry["quat_x"],
            pred_entry["quat_y"],
            pred_entry["quat_z"],
        ]
        pitch_err = pitch_error_degrees(gt_quat, pred_quat)

        pos_errors.append(pos_error)
        pitch_errors.append(pitch_err)

    if not pos_errors:
        print("No data to plot histograms")
        return

    # Filter outliers if requested
    if filter_outliers:
        pos_min_bound = np.percentile(pos_errors, outlier_range[0])
        pos_max_bound = np.percentile(pos_errors, outlier_range[1])
        pitch_min_bound = np.percentile(pitch_errors, outlier_range[0])
        pitch_max_bound = np.percentile(pitch_errors, outlier_range[1])

        pos_errors_plot = [e for e in pos_errors if pos_min_bound <= e <= pos_max_bound]
        pitch_errors_plot = [
            e for e in pitch_errors if pitch_min_bound <= e <= pitch_max_bound
        ]

        title_suffix = f" (filtered {outlier_range[0]}-{outlier_range[1]}%)"
        print(f"Basic histogram outlier filtering:")
        print(f"  Position: {len(pos_errors)} -> {len(pos_errors_plot)} samples")
        print(f"  Pitch: {len(pitch_errors)} -> {len(pitch_errors_plot)} samples")
    else:
        pos_errors_plot = pos_errors
        pitch_errors_plot = pitch_errors
        title_suffix = ""

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Position error histogram
    ax1.hist(pos_errors_plot, bins=20, alpha=0.7, color="blue", edgecolor="black")
    ax1.set_xlabel("Position Error (meters)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Position Error Distribution (2D){title_suffix}")
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    pos_mean = np.mean(pos_errors_plot)
    pos_median = np.median(pos_errors_plot)
    ax1.axvline(
        pos_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {pos_mean:.4f}m",
    )
    ax1.axvline(
        pos_median,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {pos_median:.4f}m",
    )
    ax1.legend()

    # Pitch error histogram
    ax2.hist(pitch_errors_plot, bins=20, alpha=0.7, color="orange", edgecolor="black")
    ax2.set_xlabel("Pitch Error (degrees)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Pitch Error Distribution{title_suffix}")
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    pitch_mean = np.mean(pitch_errors_plot)
    pitch_median = np.median(pitch_errors_plot)
    ax2.axvline(
        pitch_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {pitch_mean:.2f}°",
    )
    ax2.axvline(
        pitch_median,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {pitch_median:.2f}°",
    )
    ax2.legend()

    # Overall title
    n_samples = len(pos_errors_plot)
    total_samples = len(pos_errors)
    if filter_outliers:
        title = f"Error Distribution Analysis (n={n_samples}/{total_samples} samples){title_suffix}"
    else:
        title = f"Error Distribution Analysis (n={n_samples} samples)"

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Histogram saved to: {save_path}")
    else:
        plt.savefig("error_histograms.png", dpi=300, bbox_inches="tight")
        print("Histogram saved to: error_histograms.png")

    plt.show()


def draw_detailed_histograms(
    gt_data,
    pred_data,
    save_path=None,
    percentiles=[75, 95],
    filter_outliers=False,
    outlier_range=[5, 95],
):
    """
    Draw more detailed histograms with additional statistics

    Args:
        gt_data: Ground truth data
        pred_data: Predicted data
        save_path: Path to save the plot
        percentiles: List of percentiles to show as vertical lines
        filter_outliers: If True, only plot errors within outlier_range percentiles
        outlier_range: [min_percentile, max_percentile] to include in plots
    """
    pos_errors = []
    pitch_errors = []

    for gt_entry, pred_entry in zip(gt_data, pred_data):
        if gt_entry["index"] != pred_entry["index"]:
            continue

        # Position error (2D: x, y only)
        pos_error = (
            (gt_entry["pos_x"] - pred_entry["pos_x"]) ** 2
            + (gt_entry["pos_y"] - pred_entry["pos_y"]) ** 2
        ) ** 0.5

        # Pitch error
        gt_quat = [
            gt_entry["quat_w"],
            gt_entry["quat_x"],
            gt_entry["quat_y"],
            gt_entry["quat_z"],
        ]
        pred_quat = [
            pred_entry["quat_w"],
            pred_entry["quat_x"],
            pred_entry["quat_y"],
            pred_entry["quat_z"],
        ]
        pitch_err = pitch_error_degrees(gt_quat, pred_quat)

        pos_errors.append(pos_error)
        pitch_errors.append(pitch_err)

    if not pos_errors:
        print("No data to plot histograms")
        return

    # Filter outliers if requested
    if filter_outliers:
        # Calculate percentile bounds
        pos_min_bound = np.percentile(pos_errors, outlier_range[0])
        pos_max_bound = np.percentile(pos_errors, outlier_range[1])
        pitch_min_bound = np.percentile(pitch_errors, outlier_range[0])
        pitch_max_bound = np.percentile(pitch_errors, outlier_range[1])

        # Filter errors within bounds
        pos_errors_filtered = [
            e for e in pos_errors if pos_min_bound <= e <= pos_max_bound
        ]
        pitch_errors_filtered = [
            e for e in pitch_errors if pitch_min_bound <= e <= pitch_max_bound
        ]

        print(f"Outlier filtering applied:")
        print(
            f"  Position errors: {len(pos_errors)} -> {len(pos_errors_filtered)} "
            + f"(removed {len(pos_errors) - len(pos_errors_filtered)} outliers)"
        )
        print(
            f"  Pitch errors: {len(pitch_errors)} -> {len(pitch_errors_filtered)} "
            + f"(removed {len(pitch_errors) - len(pitch_errors_filtered)} outliers)"
        )
        print(f"  Position range: [{pos_min_bound:.4f}, {pos_max_bound:.4f}] meters")
        print(f"  Pitch range: [{pitch_min_bound:.2f}, {pitch_max_bound:.2f}] degrees")

        # Use filtered data for plotting
        pos_errors_plot = pos_errors_filtered
        pitch_errors_plot = pitch_errors_filtered
        title_suffix = f" (filtered {outlier_range[0]}-{outlier_range[1]}%)"
    else:
        pos_errors_plot = pos_errors
        pitch_errors_plot = pitch_errors
        title_suffix = ""

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Position error histogram
    n_pos, bins_pos, patches_pos = ax1.hist(
        pos_errors_plot, bins=30, alpha=0.7, color="blue", edgecolor="black"
    )
    ax1.set_xlabel("Position Error (meters)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Position Error Distribution (2D){title_suffix}")
    ax1.grid(True, alpha=0.3)

    # Add statistics lines for position
    pos_mean = np.mean(pos_errors_plot)
    pos_median = np.median(pos_errors_plot)

    # Calculate and plot custom percentiles
    pos_percentile_values = [np.percentile(pos_errors_plot, p) for p in percentiles]
    colors = ["purple", "orange", "brown", "pink", "gray"]
    linestyles = [":", "-.", "--", ":", "-."]

    ax1.axvline(
        pos_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {pos_mean:.4f}m",
    )
    ax1.axvline(
        pos_median,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {pos_median:.4f}m",
    )

    for i, (percentile, value) in enumerate(zip(percentiles, pos_percentile_values)):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        ax1.axvline(
            value,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{percentile}th %: {value:.4f}m",
        )

    ax1.legend()

    # Position error cumulative distribution
    ax2.hist(
        pos_errors_plot,
        bins=30,
        alpha=0.7,
        color="blue",
        edgecolor="black",
        cumulative=True,
        density=True,
    )
    ax2.set_xlabel("Position Error (meters)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title(f"Position Error Cumulative Distribution{title_suffix}")
    ax2.grid(True, alpha=0.3)

    # Add horizontal lines for percentiles
    ax2.axhline(0.5, color="green", linestyle="--", label="50th percentile")
    for i, percentile in enumerate(percentiles):
        color = colors[i % len(colors)]
        ax2.axhline(
            percentile / 100,
            color=color,
            linestyle=":",
            label=f"{percentile}th percentile",
        )
    ax2.legend()

    # Pitch error histogram
    n_pitch, bins_pitch, patches_pitch = ax3.hist(
        pitch_errors_plot, bins=30, alpha=0.7, color="orange", edgecolor="black"
    )
    ax3.set_xlabel("Pitch Error (degrees)")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Pitch Error Distribution{title_suffix}")
    ax3.grid(True, alpha=0.3)

    # Add statistics lines for pitch
    pitch_mean = np.mean(pitch_errors_plot)
    pitch_median = np.median(pitch_errors_plot)

    # Calculate and plot custom percentiles
    pitch_percentile_values = [np.percentile(pitch_errors_plot, p) for p in percentiles]

    ax3.axvline(
        pitch_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {pitch_mean:.2f}°",
    )
    ax3.axvline(
        pitch_median,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {pitch_median:.2f}°",
    )

    for i, (percentile, value) in enumerate(zip(percentiles, pitch_percentile_values)):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        ax3.axvline(
            value,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{percentile}th %: {value:.2f}°",
        )

    ax3.legend()

    # Pitch error cumulative distribution
    ax4.hist(
        pitch_errors_plot,
        bins=30,
        alpha=0.7,
        color="orange",
        edgecolor="black",
        cumulative=True,
        density=True,
    )
    ax4.set_xlabel("Pitch Error (degrees)")
    ax4.set_ylabel("Cumulative Probability")
    ax4.set_title(f"Pitch Error Cumulative Distribution{title_suffix}")
    ax4.grid(True, alpha=0.3)

    # Add horizontal lines for percentiles
    ax4.axhline(0.5, color="green", linestyle="--", label="50th percentile")
    for i, percentile in enumerate(percentiles):
        color = colors[i % len(colors)]
        ax4.axhline(
            percentile / 100,
            color=color,
            linestyle=":",
            label=f"{percentile}th percentile",
        )
    ax4.legend()

    # Overall title
    n_samples = len(pos_errors_plot)
    total_samples = len(pos_errors)
    if filter_outliers:
        title = f"Detailed Error Distribution Analysis (n={n_samples}/{total_samples} samples){title_suffix}"
    else:
        title = f"Detailed Error Distribution Analysis (n={n_samples} samples)"

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Detailed histogram saved to: {save_path}")
    else:
        plt.savefig("detailed_error_histograms.png", dpi=300, bbox_inches="tight")
        print("Detailed histogram saved to: detailed_error_histograms.png")

    plt.show()


def calculate_filtered_statistics(gt_data, pred_data, outlier_range=[5, 95]):
    """
    Calculate detailed statistics for filtered data within percentile range
    
    Args:
        gt_data: Ground truth data
        pred_data: Predicted data  
        outlier_range: [min_percentile, max_percentile] to include in analysis
    
    Returns:
        stats: Dictionary with filtered statistics
    """
    pos_errors = []
    pitch_errors = []

    for gt_entry, pred_entry in zip(gt_data, pred_data):
        if gt_entry["index"] != pred_entry["index"]:
            continue

        # Position error (2D: x, y only)
        pos_error = (
            (gt_entry["pos_x"] - pred_entry["pos_x"]) ** 2
            + (gt_entry["pos_y"] - pred_entry["pos_y"]) ** 2
        ) ** 0.5

        # Pitch error
        gt_quat = [
            gt_entry["quat_w"],
            gt_entry["quat_x"],
            gt_entry["quat_y"],
            gt_entry["quat_z"],
        ]
        pred_quat = [
            pred_entry["quat_w"],
            pred_entry["quat_x"],
            pred_entry["quat_y"],
            pred_entry["quat_z"],
        ]
        pitch_err = pitch_error_degrees(gt_quat, pred_quat)

        pos_errors.append(pos_error)
        pitch_errors.append(pitch_err)

    if not pos_errors:
        return None

    # Calculate percentile bounds
    pos_min_bound = np.percentile(pos_errors, outlier_range[0])
    pos_max_bound = np.percentile(pos_errors, outlier_range[1])
    pitch_min_bound = np.percentile(pitch_errors, outlier_range[0])
    pitch_max_bound = np.percentile(pitch_errors, outlier_range[1])

    # Filter errors within bounds
    pos_errors_filtered = [e for e in pos_errors if pos_min_bound <= e <= pos_max_bound]
    pitch_errors_filtered = [e for e in pitch_errors if pitch_min_bound <= e <= pitch_max_bound]

    if not pos_errors_filtered:
        return None

    # Calculate statistics for filtered data
    pos_errors_filtered.sort()
    pitch_errors_filtered.sort()
    n_pos = len(pos_errors_filtered)
    n_pitch = len(pitch_errors_filtered)

    stats = {
        "outlier_range": outlier_range,
        "original_count": len(pos_errors),
        "filtered_count_pos": n_pos,
        "filtered_count_pitch": n_pitch,
        "removed_count_pos": len(pos_errors) - n_pos,
        "removed_count_pitch": len(pitch_errors) - n_pitch,
        "bounds": {
            "pos_min": pos_min_bound,
            "pos_max": pos_max_bound,
            "pitch_min": pitch_min_bound,
            "pitch_max": pitch_max_bound,
        },
        "position": {
            "mean": sum(pos_errors_filtered) / n_pos,
            "median": pos_errors_filtered[n_pos // 2],
            "min": min(pos_errors_filtered),
            "max": max(pos_errors_filtered),
            "std": (sum((x - sum(pos_errors_filtered) / n_pos) ** 2 for x in pos_errors_filtered) / n_pos) ** 0.5,
            "percentile_75": pos_errors_filtered[int(0.75 * n_pos)],
            "percentile_95": pos_errors_filtered[int(0.95 * n_pos)],
        },
        "orientation": {
            "mean": sum(pitch_errors_filtered) / n_pitch,
            "median": pitch_errors_filtered[n_pitch // 2],
            "min": min(pitch_errors_filtered),
            "max": max(pitch_errors_filtered),
            "std": (sum((x - sum(pitch_errors_filtered) / n_pitch) ** 2 for x in pitch_errors_filtered) / n_pitch) ** 0.5,
            "percentile_75": pitch_errors_filtered[int(0.75 * n_pitch)],
            "percentile_95": pitch_errors_filtered[int(0.95 * n_pitch)],
        },
    }

    return stats

def print_statistics(stats):
    """Print formatted statistics"""
    if not stats:
        print("No statistics available")
        return
    
    print(f"\nDetailed Statistics:")
    print(f"  Total samples: {stats['count']}")
    
    print(f"\n  Position Error Statistics (meters, 2D):")
    print(f"    Min:     {stats['position']['min']:.4f}")
    print(f"    Max:     {stats['position']['max']:.4f}")
    print(f"    Mean:    {stats['position']['mean']:.4f}")
    print(f"    Median:  {stats['position']['median']:.4f}")
    print(f"    Std:     {stats['position']['std']:.4f}")
    print(f"    75th %:  {stats['position']['percentile_75']:.4f}")
    print(f"    95th %:  {stats['position']['percentile_95']:.4f}")

    print(f"\n  Orientation Error Statistics (degrees):")
    print(f"    Min:     {stats['orientation']['min']:.2f}")
    print(f"    Max:     {stats['orientation']['max']:.2f}")
    print(f"    Mean:    {stats['orientation']['mean']:.2f}")
    print(f"    Median:  {stats['orientation']['median']:.2f}")
    print(f"    Std:     {stats['orientation']['std']:.2f}")
    print(f"    75th %:  {stats['orientation']['percentile_75']:.2f}")
    print(f"    95th %:  {stats['orientation']['percentile_95']:.2f}")


def print_filtered_statistics(stats):
    """Print formatted filtered statistics"""
    if not stats:
        print("No filtered statistics available")
        return
    
    range_str = f"{stats['outlier_range'][0]}-{stats['outlier_range'][1]}%"
    
    print(f"\nFiltered Statistics ({range_str} percentile range):")
    print(f"  Original samples: {stats['original_count']}")
    print(f"  Filtered samples: {stats['filtered_count_pos']} position, {stats['filtered_count_pitch']} pitch")
    print(f"  Removed outliers: {stats['removed_count_pos']} position, {stats['removed_count_pitch']} pitch")
    print(f"  Position bounds: [{stats['bounds']['pos_min']:.4f}, {stats['bounds']['pos_max']:.4f}] meters")
    print(f"  Pitch bounds: [{stats['bounds']['pitch_min']:.2f}, {stats['bounds']['pitch_max']:.2f}] degrees")
    
    print(f"\n  Filtered Position Error Statistics (meters, 2D):")
    print(f"    Min:     {stats['position']['min']:.4f}")
    print(f"    Max:     {stats['position']['max']:.4f}")
    print(f"    Mean:    {stats['position']['mean']:.4f}")
    print(f"    Median:  {stats['position']['median']:.4f}")
    print(f"    Std:     {stats['position']['std']:.4f}")
    print(f"    75th %:  {stats['position']['percentile_75']:.4f}")
    print(f"    95th %:  {stats['position']['percentile_95']:.4f}")

    print(f"\n  Filtered Orientation Error Statistics (degrees):")
    print(f"    Min:     {stats['orientation']['min']:.2f}")
    print(f"    Max:     {stats['orientation']['max']:.2f}")
    print(f"    Mean:    {stats['orientation']['mean']:.2f}")
    print(f"    Median:  {stats['orientation']['median']:.2f}")
    print(f"    Std:     {stats['orientation']['std']:.2f}")
    print(f"    75th %:  {stats['orientation']['percentile_75']:.2f}")
    print(f"    95th %:  {stats['orientation']['percentile_95']:.2f}")


def add_errors_to_csv(gt_data, pred_data, csv_path):
    """
    Add position and pitch error columns to the existing CSV file
    
    Args:
        gt_data: Ground truth data
        pred_data: Predicted data
        csv_path: Path to the CSV file to update
    """
    # Calculate errors for each entry
    updated_data = []
    
    for gt_entry, pred_entry in zip(gt_data, pred_data):
        if gt_entry["index"] != pred_entry["index"]:
            print(f"Index mismatch: {gt_entry['index']} vs {pred_entry['index']}")
            continue
        
        # Calculate position error (2D: x, y only)
        pos_error = (
            (gt_entry["pos_x"] - pred_entry["pos_x"]) ** 2
            + (gt_entry["pos_y"] - pred_entry["pos_y"]) ** 2
        ) ** 0.5
        
        # Calculate pitch error in degrees
        gt_quat = [
            gt_entry["quat_w"],
            gt_entry["quat_x"],
            gt_entry["quat_y"],
            gt_entry["quat_z"],
        ]
        pred_quat = [
            pred_entry["quat_w"],
            pred_entry["quat_x"],
            pred_entry["quat_y"],
            pred_entry["quat_z"],
        ]
        pitch_err = pitch_error_degrees(gt_quat, pred_quat)
        
        # Create updated entry with error columns
        updated_entry = pred_entry.copy()
        updated_entry["pos_error"] = pos_error
        updated_entry["pitch_error"] = pitch_err
        
        updated_data.append(updated_entry)
    
    # Write updated data back to CSV
    if updated_data:
        fieldnames = list(updated_data[0].keys())
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(updated_data)
            
            print(f"Successfully added error columns to {len(updated_data)} entries in: {csv_path}")
            print(f"Added columns: 'pos_error' (meters), 'pitch_error' (degrees)")
            return True
        
        except Exception as e:
            print(f"Error updating CSV file: {e}")
            return False
    else:
        print("No data to update")
        return False


def calculate_error():
    """Main function to calculate and display all error metrics"""
    gt_data = load_csv_simple(csv_gt)
    pred_data = load_csv_simple(csv_pred)

    if not gt_data or not pred_data:
        print("No data to compare")
        return

    # Add error columns to the CSV file first
    print("Adding error columns to CSV file...")
    success = add_errors_to_csv(gt_data, pred_data, csv_pred)
    
    if success:
        print(f"Error columns added to: {csv_pred}")
    else:
        print("Failed to add error columns to CSV")
    
    # # Continue with existing analysis...
    # # Filter out leaders for analysis
    # gt_followers = filter_followers_only(gt_data)
    # pred_followers = filter_followers_only(pred_data)

    # print(f"Total GT entries: {len(gt_data)}, Follower entries: {len(gt_followers)}")
    # print(
    #     f"Total Pred entries: {len(pred_data)}, Follower entries: {len(pred_followers)}"
    # )

    # # Count leaders that were skipped
    # leaders_count = len(gt_data) - len(gt_followers)
    # print(f"Skipped {leaders_count} leader entries from analysis")

    # # Print results
    # print("\n" + "=" * 60)
    # print("POSITION AND ORIENTATION ERROR ANALYSIS (FOLLOWERS ONLY)")
    # print("=" * 60)

    # # Calculate detailed statistics
    # detailed_stats = calculate_detailed_statistics(gt_followers, pred_followers)
    # print_statistics(detailed_stats)

    # # Calculate and print filtered statistics
    # print("\n" + "=" * 60)
    # print("FILTERED ERROR ANALYSIS")
    # print("=" * 60)
    
    # # Calculate filtered statistics for 5-95% range
    # filtered_stats_5_95 = calculate_filtered_statistics(gt_followers, pred_followers, [0, 95])
    # print_filtered_statistics(filtered_stats_5_95)
    
    # # Calculate filtered statistics for 10-90% range  
    # filtered_stats_10_90 = calculate_filtered_statistics(gt_followers, pred_followers, [0, 90])
    # print_filtered_statistics(filtered_stats_10_90)
    
    # print("=" * 60)

    # # Draw histograms - both with and without outlier filtering
    # print("\nGenerating error histograms...")

    # # Standard histograms (with all data)
    # draw_error_histograms(gt_followers, pred_followers, "error_histograms.png")
    # draw_detailed_histograms(
    #     gt_followers, pred_followers, "detailed_error_histograms.png"
    # )

    # # Filtered histograms (without extreme outliers)
    # print("\nGenerating filtered histograms (5th-95th percentile)...")
    # draw_error_histograms(
    #     gt_followers,
    #     pred_followers,
    #     "error_histograms_filtered.png",
    #     filter_outliers=True,
    #     outlier_range=[0, 95],
    # )
    # draw_detailed_histograms(
    #     gt_followers,
    #     pred_followers,
    #     "detailed_error_histograms_filtered.png",
    #     percentiles=[75, 90, 95],
    #     filter_outliers=True,
    #     outlier_range=[0, 95],
    # )

    # # Very conservative filtering (10th-90th percentile)
    # print("\nGenerating conservative filtered histograms (10th-90th percentile)...")
    # draw_detailed_histograms(
    #     gt_followers,
    #     pred_followers,
    #     "detailed_error_histograms_conservative.png",
    #     percentiles=[75, 85, 90],
    #     filter_outliers=True,
    #     outlier_range=[0, 90],
    # )


if __name__ == "__main__":
    calculate_error()
