import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Arc
from PIL import Image
import argparse
import os
import roma

W = 100
H = 87

def polar(angle, dist):
    """Convert polar coordinates to Cartesian coordinates"""
    return np.array([np.sin(angle), np.cos(angle)]) * dist


def quaternion_to_euler(quat_w, quat_x, quat_y, quat_z):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    # Roll (rotation around x-axis)
    sinr_cosp = 2 * (quat_w * quat_x + quat_y * quat_z)
    cosr_cosp = 1 - 2 * (quat_x * quat_x + quat_y * quat_y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (rotation around y-axis)
    sinp = 2 * (quat_w * quat_y - quat_z * quat_x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    
    # Yaw (rotation around z-axis)
    siny_cosp = 2 * (quat_w * quat_z + quat_x * quat_y)
    cosy_cosp = 1 - 2 * (quat_y * quat_y + quat_z * quat_z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quaternion_to_yaw(quat_w, quat_x, quat_y, quat_z):
    """Convert quaternion to yaw angle (rotation around z-axis)"""
    norm = np.sqrt(quat_w**2 + quat_x**2 + quat_y**2 + quat_z**2)
    if norm < 1e-6:
        print("Warning: Quaternion close to zero, returning 0 angle")
        return 0.0
    
    # Normalize quaternion
    quat_w = quat_w / norm
    quat_x = quat_x / norm
    quat_y = quat_y / norm
    quat_z = quat_z / norm
    
    # Roma library uses XYZW order: [x, y, z, w]
    q = torch.as_tensor([quat_x, quat_y, quat_z, quat_w], dtype=torch.float32)
    
    rotvec = roma.unitquat_to_rotvec(q, shortest_arc=True)
    yaw = float(rotvec[1])
    
    print(f"quat: [{quat_w:.3f}, {quat_x:.3f}, {quat_y:.3f}, {quat_z:.3f}] -> yaw: {np.rad2deg(yaw):.2f}°")
    return yaw


def plot_camera_marker(ax, p, heading, fov=120, dist=10, color='blue', agent_id=None, arc_scale=0.1, forward=True):
    """Plot camera marker with field of view arc"""
    fov_half = fov / 2
    fov_half_rad = np.deg2rad(fov_half)

    ax.scatter(p[0], p[1], c=color, s=100, marker='o', zorder=5)
    
    if agent_id is not None:
        ax.annotate(f'Cam{agent_id}', (p[0], p[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, color=color)
    
    arc_radius = dist * arc_scale
    p_fov_l = p + polar(heading + fov_half_rad, arc_radius)
    ax.plot([p[0], p_fov_l[0]], [p[1], p_fov_l[1]], c=color, linewidth=1.5, alpha=0.7)
    
    p_fov_r = p + polar(heading - fov_half_rad, arc_radius)
    ax.plot([p[0], p_fov_r[0]], [p[1], p_fov_r[1]], c=color, linewidth=1.5, alpha=0.7)

    arc = Arc(
        p,
        arc_radius * 2,
        arc_radius * 2,
        angle=np.rad2deg(-heading + np.pi / 2),
        theta1=-fov_half,
        theta2=fov_half,
        color=color,
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(arc)
    
    theta = np.linspace(heading - fov_half_rad, heading + fov_half_rad, 50)
    fan_x = [p[0]] + [p[0] + arc_radius * np.sin(t) for t in theta] + [p[0]]
    fan_y = [p[1]] + [p[1] + arc_radius * np.cos(t) for t in theta] + [p[1]]
    ax.fill(fan_x, fan_y, color=color, alpha=0.1)


def load_camera_data(csv_path, camera_range=None, forward=True):
    """Load camera data from CSV file"""
    df = pd.read_csv(csv_path)
    cameras = []
    
    if camera_range is not None:
        start, end = camera_range
        print(f"Reading camera data from index {start} to {end}")
        df = df.iloc[start:end]
    else:
        print(f"Reading all {len(df)} camera data entries")

    # CSV format：index,walk_id,image_id,topdown_id,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,cam_fov,pixel_pos_x,pixel_pos_y
    for _, row in df.iterrows():
        heading = quaternion_to_yaw(
            row['quat_w'], 
            row['quat_x'], 
            row['quat_y'], 
            row['quat_z']
        )
        
        camera = {
            'id': row['image_id'],
            'x': row['pixel_pos_x'],
            'y': row['pixel_pos_y'],
            'world_x': row['pos_x'],
            'world_y': row['pos_y'],
            'world_z': row['pos_z'],
            'heading': heading,
            'fov': row['cam_fov'],
            'walk_id': row['group_id'],
            'topdown_id': row['topdown_id']
        }
        camera['heading'] = heading + np.pi 
        cameras.append(camera)
    
    return cameras


def visualize_bev_with_cameras(csv_path, topdown_path, output_path=None, view_distance=50, camera_range=None, forward=True, full_bev=True, fixed_width=None, fixed_height=None, min_width=None, min_height=None):
    """Visualize BEV map with camera field of view"""
    cameras = load_camera_data(csv_path, camera_range, forward)
    print(f"Successfully loaded {len(cameras)} camera data entries")
    
    # 打印相机信息
    for camera in cameras:
        print(f"Camera {camera['id']}: pixel position({camera['x']:.0f}, {camera['y']:.0f}), "
              f"world position({camera['world_x']:.2f}, {camera['world_y']:.2f}), "
              f"heading {np.rad2deg(camera['heading']):.1f}°, FOV{camera['fov']:.0f}°")
    
    # 加载BEV地图
    try:
        bev_image = Image.open(topdown_path)
        bev_array = np.array(bev_image)
        print(f"Successfully loaded BEV image, size: {bev_array.shape}")
    except Exception as e:
        print(f"Failed to load BEV image: {e}")
        return
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 获取BEV图像尺寸
    height, width = bev_array.shape[:2]
    
    # 根据相机像素位置计算显示范围
    all_x = [cam['x'] for cam in cameras]
    all_y = [cam['y'] for cam in cameras]
    
    if full_bev:
        # 显示完整BEV图像
        x_min, x_max = 0, width
        y_min, y_max = 0, height
        print(f"Using full BEV: {width}x{height} pixels")
    elif fixed_width and fixed_height:
        # 固定尺寸，以相机中心为中心
        center_x = sum(cam['x'] for cam in cameras) / len(cameras)
        center_y = sum(cam['y'] for cam in cameras) / len(cameras)
        
        x_min = max(0, center_x - fixed_width / 2)
        x_max = min(width, center_x + fixed_width / 2)
        y_min = max(0, center_y - fixed_height / 2)
        y_max = min(height, center_y + fixed_height / 2)
        
        print(f"Using fixed size: {fixed_width}x{fixed_height} pixels centered at ({center_x:.0f}, {center_y:.0f})")
    else:
        # 原有的动态计算 + 最小尺寸约束
        margin = max(view_distance, 50)  # 至少50像素边距
        x_min = max(0, min(all_x) - margin)
        x_max = min(width, max(all_x) + margin)
        y_min = max(0, min(all_y) - margin)
        y_max = min(height, max(all_y) + margin)
        
        # 应用最小宽度和高度约束
        if min_width and min_height:
            min_x_max = min_x + min_width
            max_x_min = max_x - min_width
            min_y_max = min_y + min_height
            max_y_min = max_y - min_height
            
            x_min = max(x_min, min_x_max)
            x_max = min(x_max, max_x_min)
            y_min = max(y_min, min_y_max)
            y_max = min(y_max, max_y_min)
        
        print(f"Using dynamic range: x({x_min:.0f}, {x_max:.0f}), y({y_min:.0f}, {y_max:.0f}) with margin {margin}px")
    
    # 设置像素坐标范围
    extent = [x_min, x_max, y_max, y_min]  # 注意y轴方向（图像坐标系）
    
    # 显示BEV地图
    ax.imshow(bev_array, extent=extent, alpha=0.7, cmap='gray', origin='upper')
    
    # 获取颜色循环
    colors = plt.cm.tab10(np.linspace(0, 1, len(cameras)))
    
    # 绘制每个相机的视野
    for i, camera in enumerate(cameras):
        pos = np.array([camera['x'], camera['y']])  # 使用像素坐标
        heading = camera['heading']
        fov = camera['fov']
        color = colors[i]
        
        plot_camera_marker(
            ax, pos, heading, 
            fov=fov,  # 使用CSV中的FOV值
            dist=view_distance,  # 视野距离（像素单位）
            color=color, 
            agent_id=camera['id'],
            forward=forward
        )
    
    # 设置图形属性
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [pixels]')
    ax.set_ylabel('Y [pixels]')
    range_str = f"cameras {camera_range[0]}-{camera_range[1]}" if camera_range else "all cameras"
    ax.set_title(f'BEV Field of View Visualization (distance: {view_distance}px, {range_str})')
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], markersize=8,
                                 label=f'Camera {cameras[i]["id"]} (FOV:{cameras[i]["fov"]:.0f}°)')
                      for i in range(len(cameras))]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # 自动生成输出文件名（如果未指定）
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        range_str = f"{camera_range[0]}_{camera_range[1]}" if camera_range else "all"
        output_path = f'bev_pixel_visualization_{range_str}_{timestamp}.png'
    
    # 保存图像为PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    print(f"Image saved to: {output_path}")
    
    # 显示图像
    plt.show()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='BEV camera field of view visualization tool (pixel coordinate based)')
    parser.add_argument('--csv', required=True, help='Camera data CSV file path')
    parser.add_argument('--bev', required=True, help='BEV map image path (topdown_0.png)')
    parser.add_argument('--output', help='Output image path (optional, auto-generated by default)')
    parser.add_argument('--distance', type=float, default=50.0, help='View display distance (pixels), default 50')
    parser.add_argument('--camera_range', type=int, nargs=2, metavar=('START', 'END'), 
                       help='Camera range to process (start end), e.g., --camera_range 0 20')
    parser.add_argument('--full_bev', action='store_true', help='Display full BEV image')
    parser.add_argument('--fixed_width', type=int, help='Fixed width for BEV view (pixels)')
    parser.add_argument('--fixed_height', type=int, help='Fixed height for BEV view (pixels)')
    parser.add_argument('--min_width', type=int, help='Minimum width for dynamic range (pixels)')
    parser.add_argument('--min_height', type=int, help='Minimum height for dynamic range (pixels)')
    parser.add_argument('--max_cameras', type=int, help='Deprecated: use --camera_range instead')
    parser.add_argument('--forward', action='store_true', default=True, help='Deprecated: use --camera_range instead')
    parser.add_argument('--backward', action='store_true', help='Deprecated: use --camera_range instead')
    
    args = parser.parse_args()
    
    camera_range = None
    if args.camera_range:
        start, end = args.camera_range
        if start < 0 or end <= start:
            print(f"Error: Invalid camera range. Start must be >= 0 and end must be > start")
            return
        camera_range = (start, end)
    elif args.max_cameras:
        print("Warning: --max_cameras is deprecated, use --camera_range instead")
        if args.backward:
            import pandas as pd
            df = pd.read_csv(args.csv)
            total = len(df)
            camera_range = (max(0, total - args.max_cameras), total)
        else:
            camera_range = (0, args.max_cameras)
    
    forward = not args.backward if not args.camera_range else True
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file does not exist: {args.csv}")
        return
    
    if not os.path.exists(args.bev):
        print(f"Error: BEV image file does not exist: {args.bev}")
        return
    
    output_file = visualize_bev_with_cameras(
        csv_path=args.csv,
        topdown_path=args.bev,
        output_path=args.output,
        view_distance=args.distance,
        camera_range=camera_range,
        forward=forward,
        full_bev=args.full_bev,
        fixed_width=args.fixed_width,
        fixed_height=args.fixed_height,
        min_width=args.min_width,
        min_height=args.min_height
    )
    
    print(f"Visualization completed, result saved to: {output_file}")


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python test_bev.py --csv camera_data.csv --bev topdown_0.png --output result.png --camera_range 0 20 --distance 50")
        print("python test_bev.py --csv camera_data.csv --bev topdown_0.png --camera_range 10 30")
        print("python test_bev.py --csv camera_data.csv --bev topdown_0.png --camera_range 5 15 --distance 100")
        print("python test_bev.py --csv camera_data.csv --bev topdown_0.png --full_bev")
        print("python test_bev.py --csv camera_data.csv --bev topdown_0.png --fixed_width 400 --fixed_height 300")
        print("\n(Deprecated usage with backward compatibility):")
        print("python test_bev.py --csv camera_data.csv --bev topdown_0.png --max_cameras 20")
        print("python test_bev.py --csv camera_data.csv --bev topdown_0.png --backward --max_cameras 10")
        print("\nCSV file format:")
        print("index,walk_id,image_id,topdown_id,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,cam_fov,pixel_pos_x,pixel_pos_y")
        print("0,0,0,0,0.6763698056154412,-0.5905067571288155,3.330509232464266,0.761834442615509,0.0,0.6477717757225037,0.0,120.0,56,69")
        
        sample_cameras = [
            {
                'id': 0, 
                'x': 56,
                'y': 69,
                'heading': quaternion_to_yaw(0.7618, 0.0, 0.6478, 0.0),
                'fov': 120.0
            },
            {
                'id': 1, 
                'x': 150,
                'y': 200,
                'heading': quaternion_to_yaw(0.5, 0.0, 0.866, 0.0),
                'fov': 120.0
            }
        ]
        
        sample_bev = np.ones((300, 300, 3)) * 0.8
        sample_bev[120:180, 120:180] = [0.2, 0.2, 0.2]
        sample_bev[200:250, 80:130] = [0.2, 0.2, 0.2]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        extent = [0, 300, 300, 0]
        ax.imshow(sample_bev, extent=extent, alpha=0.7, origin='upper')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_cameras)))
        for i, camera in enumerate(sample_cameras):
            pos = np.array([camera['x'], camera['y']])
            plot_camera_marker(ax, pos, camera['heading'], 
                             fov=camera['fov'], dist=50, color=colors[i], agent_id=camera['id'])
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')
        ax.set_title('Example: BEV Camera Field of View Visualization (Pixel Coordinate Based)')
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        example_output = f'example_pixel_bev_{timestamp}.png'
        plt.savefig(example_output, dpi=300, bbox_inches='tight', format='png')
        print(f"Example image saved to: {example_output}")
        plt.show()
    else:
        main()