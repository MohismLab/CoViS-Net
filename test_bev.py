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


def polar(angle, dist):
    """
    将极坐标转换为直角坐标
    参数:
        angle: 角度 (弧度)
        dist: 距离
    返回:
        二维坐标点 [x, y]
    """
    return np.array([np.sin(angle), np.cos(angle)]) * dist


def quaternion_to_euler(quat_w, quat_x, quat_y, quat_z):
    """
    将四元数转换为欧拉角（roll, pitch, yaw）
    参数:
        quat_w, quat_x, quat_y, quat_z: 四元数分量
    返回:
        (roll, pitch, yaw) 欧拉角元组（弧度）
    """
    # Roll (绕x轴旋转)
    sinr_cosp = 2 * (quat_w * quat_x + quat_y * quat_z)
    cosr_cosp = 1 - 2 * (quat_x * quat_x + quat_y * quat_y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (绕y轴旋转)
    sinp = 2 * (quat_w * quat_y - quat_z * quat_x)
    sinp = np.clip(sinp, -1.0, 1.0)  # 防止数值误差
    pitch = np.arcsin(sinp)
    
    # Yaw (绕z轴旋转)
    siny_cosp = 2 * (quat_w * quat_z + quat_x * quat_y)
    cosy_cosp = 1 - 2 * (quat_y * quat_y + quat_z * quat_z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


# def quaternion_to_yaw(quat_w, quat_x, quat_y, quat_z):
#     """
#     将四元数转换为yaw角度（绕y轴旋转）
#     参数:
#         quat_w, quat_x, quat_y, quat_z: 四元数分量
#     返回:
#         yaw角度（弧度）
#     """
#     # 获取绕y轴的旋转角度（pitch）
#     _, pitch, _ = quaternion_to_euler(quat_w, quat_x, quat_y, quat_z)
#     print(f"roll: {np.rad2deg(0):.2f}°, pitch: {np.rad2deg(pitch):.2f}°, yaw: {np.rad2deg(0):.2f}°")
#     return pitch

def quaternion_to_yaw(quat_w, quat_x, quat_y, quat_z):
    # 检查四元数是否有效
    norm = np.sqrt(quat_w**2 + quat_x**2 + quat_y**2 + quat_z**2)
    if norm < 1e-6:
        print("警告: 四元数接近零，返回0角度")
        return 0.0
    
    # 归一化四元数
    quat_w = quat_w / norm
    quat_x = quat_x / norm
    quat_y = quat_y / norm
    quat_z = quat_z / norm
    
    # roma库使用XYZW顺序：[x, y, z, w]
    q = torch.as_tensor([quat_x, quat_y, quat_z, quat_w], dtype=torch.float32)
    
    # 使用roma库转换为旋转向量，设置shortest_arc=True避免不确定性
    rotvec = roma.unitquat_to_rotvec(q, shortest_arc=True)
    yaw = float(rotvec[1])  # 提取z轴旋转分量作为yaw角度
    
    # TODO
    # yaw = -yaw + np.pi  # 调整yaw角度，使其与BEV坐标系一致 
    print(f"quat: [{quat_w:.3f}, {quat_x:.3f}, {quat_y:.3f}, {quat_z:.3f}] -> yaw: {np.rad2deg(yaw):.2f}°")


    return yaw



def plot_camera_marker(ax, p, heading, fov=120, dist=10, color='blue', agent_id=None, arc_scale=1):
    """
    在图上绘制相机标记，包括视野弧形范围
    参数:
        ax: matplotlib轴对象
        p: 相机位置坐标 [x, y]
        heading: 相机朝向角度 (弧度)
        fov: 视野角度(度)，默认120度
        dist: 视野显示距离，默认10米
        color: 标记颜色
        agent_id: 智能体ID，用于标注
        arc_scale: 弧形缩放比例，默认0.6（弧形半径为视野距离的60%）
    """
    fov_half = fov / 2  # 视野半角
    fov_half_rad = np.deg2rad(fov_half)  # 视野半角转弧度
    
    # 绘制相机位置点
    ax.scatter(p[0], p[1], c=color, s=100, marker='o', zorder=5)
    
    # 如果有agent_id，标注相机编号
    if agent_id is not None:
        ax.annotate(f'Cam{agent_id}', (p[0], p[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, color=color)
    
    # 绘制更小的视野弧形范围
    arc_radius = dist * arc_scale  # 弧形半径可调节
    # 绘制视野左边界线（使用弧形半径）
    p_fov_l = p + polar(heading + fov_half_rad, arc_radius)
    ax.plot([p[0], p_fov_l[0]], [p[1], p_fov_l[1]], c=color, linewidth=1.5, alpha=0.7)
    
    # 绘制视野右边界线（使用弧形半径）
    p_fov_r = p + polar(heading - fov_half_rad, arc_radius)
    ax.plot([p[0], p_fov_r[0]], [p[1], p_fov_r[1]], c=color, linewidth=1.5, alpha=0.7)

    # 绘制更小的视野弧形范围
    arc_radius = dist * arc_scale  # 弧形半径可调节
    arc = Arc(
        p,  # 弧心位置
        arc_radius * 2,  # 弧的宽度
        arc_radius * 2,  # 弧的高度
        angle=np.rad2deg(-heading + np.pi / 2),  # 弧的旋转角度
        theta1=-fov_half,  # 弧的起始角度
        theta2=fov_half,   # 弧的结束角度
        color=color,
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(arc)
    
    # 添加半透明扇形区域显示视野范围
    theta = np.linspace(heading - fov_half_rad, heading + fov_half_rad, 50)
    fan_x = [p[0]] + [p[0] + arc_radius * np.sin(t) for t in theta] + [p[0]]
    fan_y = [p[1]] + [p[1] + arc_radius * np.cos(t) for t in theta] + [p[1]]
    ax.fill(fan_x, fan_y, color=color, alpha=0.1)


def load_camera_data(csv_path, max_cameras=20):
    """
    从CSV文件加载相机数据
    参数:
        csv_path: CSV文件路径
        max_cameras: 最大读取相机数量，默认10
    返回:
        相机数据字典列表
    """
    df = pd.read_csv(csv_path)
    cameras = []
    
    # 只读取前max_cameras条数据
    # df = df.head(max_cameras)
    df= df.tail(max_cameras)  # 确保只读取最新的max_cameras条数据

    # CSV格式：index,walk_id,image_id,topdown_id,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,cam_fov,pixel_pos_x,pixel_pos_y
    for _, row in df.iterrows():
        # 从四元数计算朝向角度
        heading = quaternion_to_yaw(
            row['quat_w'], 
            row['quat_x'], 
            row['quat_y'], 
            row['quat_z']
        )
        
        camera = {
            'id': row['image_id'],
            'x': row['pixel_pos_x'],  # 使用像素坐标x
            'y': row['pixel_pos_y'],  # 使用像素坐标y
            'world_x': row['pos_x'],  # 保留世界坐标作为参考
            'world_y': row['pos_y'],
            'world_z': row['pos_z'],
            'heading': heading,  # 已经是弧度
            'fov': row['cam_fov'],  # 相机视野角度
            'walk_id': row['walk_id'],
            'topdown_id': row['topdown_id']
        }
        w = 100
        camera['x'] = w - camera['x']
        # pixel_pos_x = w - pixel_pos_x

        cameras.append(camera)
    
    return cameras


def visualize_bev_with_cameras(csv_path, topdown_path, output_path=None, view_distance=50, max_cameras=20):
    """
    可视化BEV地图和相机视野
    参数:
        csv_path: 相机数据CSV文件路径
        topdown_path: BEV地图图像路径
        output_path: 输出图像路径（可选）
        view_distance: 视野显示距离（像素），默认50像素
        max_cameras: 最大读取相机数量，默认10
    """
    # 加载相机数据（只读取前max_cameras条）
    # try:
    cameras = load_camera_data(csv_path, max_cameras)
    print(f"成功加载 {len(cameras)} 个相机数据（前{max_cameras}条）")
    
    # 打印相机信息
    for camera in cameras:
        print(f"相机 {camera['id']}: 像素位置({camera['x']:.0f}, {camera['y']:.0f}), "
                f"世界位置({camera['world_x']:.2f}, {camera['world_y']:.2f}), "
                f"朝向{np.rad2deg(camera['heading']):.1f}°, FOV{camera['fov']:.0f}°")
            
    # except Exception as e:
    #     print(f"加载CSV文件失败: {e}")
    #     return
    
    # 加载BEV地图
    try:
        bev_image = Image.open(topdown_path)
        bev_array = np.array(bev_image)
        print(f"成功加载BEV图像，尺寸: {bev_array.shape}")
    except Exception as e:
        print(f"加载BEV图像失败: {e}")
        return
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 获取BEV图像尺寸
    height, width = bev_array.shape[:2]
    
    # 根据相机像素位置计算显示范围
    all_x = [cam['x'] for cam in cameras]
    all_y = [cam['y'] for cam in cameras]
    
    # 计算边界，添加一些边距（像素单位）
    margin = max(view_distance, 50)  # 至少50像素边距
    x_min = max(0, min(all_x) - margin)
    x_max = min(width, max(all_x) + margin)
    y_min = max(0, min(all_y) - margin)
    y_max = min(height, max(all_y) + margin)
    
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
            agent_id=camera['id']
        )
    
    # 设置图形属性
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [像素]')
    ax.set_ylabel('Y [像素]')
    ax.set_title(f'鸟瞰图视野可视化 (距离: {view_distance}像素, 前{len(cameras)}个相机)')
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], markersize=8,
                                 label=f'相机 {cameras[i]["id"]} (FOV:{cameras[i]["fov"]:.0f}°)')
                      for i in range(len(cameras))]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # 自动生成输出文件名（如果未指定）
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'bev_pixel_visualization_{timestamp}.png'
    
    # 保存图像为PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    print(f"图像已保存到: {output_path}")
    
    # 显示图像
    plt.show()
    
    return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BEV相机视野可视化工具（基于像素坐标）')
    parser.add_argument('--csv', required=True, help='相机数据CSV文件路径')
    parser.add_argument('--bev', required=True, help='BEV地图图像路径 (topdown_0.png)')
    parser.add_argument('--output', help='输出图像路径（可选，默认自动生成）')
    parser.add_argument('--distance', type=float, default=50.0, help='视野显示距离（像素），默认50')
    parser.add_argument('--max_cameras', type=int, default=20, help='最大读取相机数量，默认10')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.csv):
        print(f"错误: CSV文件不存在: {args.csv}")
        return
    
    if not os.path.exists(args.bev):
        print(f"错误: BEV图像文件不存在: {args.bev}")
        return
    
    # 运行可视化
    output_file = visualize_bev_with_cameras(
        csv_path=args.csv,
        topdown_path=args.bev,
        output_path=args.output,
        view_distance=args.distance,
        max_cameras=args.max_cameras
    )
    
    print(f"可视化完成，结果保存在: {output_file}")


if __name__ == "__main__":
    # 示例用法
    if len(os.sys.argv) == 1:
        # 如果没有命令行参数，使用示例数据
        print("示例用法:")
        print("python test_bev.py --csv camera_data.csv --bev topdown_0.png --output result.png --max_cameras 10 --distance 50")
        print("\nCSV文件格式:")
        print("index,walk_id,image_id,topdown_id,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,cam_fov,pixel_pos_x,pixel_pos_y")
        print("0,0,0,0,0.6763698056154412,-0.5905067571288155,3.330509232464266,0.761834442615509,0.0,0.6477717757225037,0.0,120.0,56,69")
        
        # 创建基于像素坐标的示例
        sample_cameras = [
            {
                'id': 0, 
                'x': 56,    # 使用像素坐标
                'y': 69,    # 使用像素坐标
                'heading': quaternion_to_yaw(0.7618, 0.0, 0.6478, 0.0),
                'fov': 120.0
            },
            {
                'id': 1, 
                'x': 150,   # 示例像素坐标
                'y': 200,   # 示例像素坐标
                'heading': quaternion_to_yaw(0.5, 0.0, 0.866, 0.0),
                'fov': 120.0
            }
        ]
        
        # 创建示例BEV图像
        sample_bev = np.ones((300, 300, 3)) * 0.8  # 灰色背景
        # 添加一些障碍物
        sample_bev[120:180, 120:180] = [0.2, 0.2, 0.2]  # 黑色方块
        sample_bev[200:250, 80:130] = [0.2, 0.2, 0.2]   # 另一个障碍物
        
        # 可视化示例
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        extent = [0, 300, 300, 0]  # 像素坐标范围
        ax.imshow(sample_bev, extent=extent, alpha=0.7, origin='upper')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_cameras)))
        for i, camera in enumerate(sample_cameras):
            pos = np.array([camera['x'], camera['y']])
            plot_camera_marker(ax, pos, camera['heading'], 
                             fov=camera['fov'], dist=50, color=colors[i], agent_id=camera['id'])
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X [像素]')
        ax.set_ylabel('Y [像素]')
        ax.set_title('示例：基于像素坐标的BEV相机视野可视化')
        
        # 保存示例图像
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        example_output = f'example_pixel_bev_{timestamp}.png'
        plt.savefig(example_output, dpi=300, bbox_inches='tight', format='png')
        print(f"示例图像已保存到: {example_output}")
        plt.show()
    else:
        main()