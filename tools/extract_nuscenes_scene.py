#!/usr/bin/env python3
"""
Extract nuScenes scene data: images (symlinks), trajectory visualization, and camera intrinsics
Author: mengmengliu1998
Date: 2025-10-30
"""
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def transform_matrix(translation, rotation):
    """
    Construct 4x4 transformation matrix from translation and quaternion
    
    Args:
        translation: [x, y, z]
        rotation: [w, x, y, z] quaternion
    """
    T = np.eye(4)
    quat = Quaternion(rotation)
    T[:3, :3] = quat.rotation_matrix
    T[:3, 3] = translation
    return T


def get_all_sample_data_tokens(nusc, scene, camera_name):
    """
    Get all sample_data tokens for a camera in chronological order
    """
    first_sample = nusc.get('sample', scene['first_sample_token'])
    first_sd_token = first_sample['data'][camera_name]
    
    sd_tokens = []
    current_token = first_sd_token
    
    while current_token != '':
        sd_tokens.append(current_token)
        sd = nusc.get('sample_data', current_token)
        current_token = sd['next']
    
    return sd_tokens


def extract_images(nusc, scene_name, camera_name, output_dir):
    """
    Extract images using symbolic links
    
    Returns:
        list of (timestamp, symlink_path) tuples
    """
    # Find scene
    scene = None
    for s in nusc.scene:
        if s['name'] == scene_name:
            scene = s
            break
    
    if scene is None:
        raise ValueError(f"Scene {scene_name} not found")
    
    # Get all sample_data tokens
    sd_tokens = get_all_sample_data_tokens(nusc, scene, camera_name)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {len(sd_tokens)} images from {scene_name} ({camera_name})...")
    
    image_info = []
    
    for idx, sd_token in enumerate(sd_tokens):
        sd = nusc.get('sample_data', sd_token)
        
        # Source image path
        src_path = Path(nusc.dataroot) / sd['filename']
        
        # Timestamp in seconds
        timestamp_sec = sd['timestamp'] / 1e6
        
        # Create symbolic link with timestamp as filename
        # Format: {timestamp}_{frame_idx}.jpg
        dst_filename = f"{timestamp_sec:.6f}_{idx:06d}.jpg"
        dst_path = output_dir / dst_filename
        
        # Create symbolic link
        if dst_path.exists():
            dst_path.unlink()  # Remove existing link
        
        try:
            # Create relative symlink if possible
            if src_path.exists():
                os.symlink(src_path, dst_path)
                image_info.append((timestamp_sec, dst_path.name, idx))
            else:
                print(f"Warning: Source image not found: {src_path}")
        except Exception as e:
            print(f"Warning: Failed to create symlink for {dst_filename}: {e}")
    
    print(f"✓ Created {len(image_info)} symbolic links in {output_dir}")
    
    # Save image list to text file
    list_file = output_dir / "image_list.txt"
    with open(list_file, 'w') as f:
        f.write("# timestamp filename frame_index\n")
        for ts, fname, idx in image_info:
            f.write(f"{ts:.9f} {fname} {idx}\n")
    
    print(f"✓ Saved image list to {list_file}")
    
    return image_info


def extract_trajectory(nusc, scene_name, camera_name):
    """
    Extract camera trajectory (camera-to-world transforms)
    
    Returns:
        timestamps: numpy array of timestamps in seconds
        positions: numpy array of [x, y, z] positions
        poses_se3: list of 4x4 SE3 matrices
    """
    # Find scene
    scene = None
    for s in nusc.scene:
        if s['name'] == scene_name:
            scene = s
            break
    
    if scene is None:
        raise ValueError(f"Scene {scene_name} not found")
    
    # Get all sample_data tokens
    sd_tokens = get_all_sample_data_tokens(nusc, scene, camera_name)
    
    timestamps = []
    positions = []
    poses_se3 = []
    
    print(f"Extracting trajectory for {scene_name} ({camera_name})...")
    
    for sd_token in sd_tokens:
        sd = nusc.get('sample_data', sd_token)
        
        # Timestamp
        timestamp = sd['timestamp'] / 1e6
        timestamps.append(timestamp)
        
        # Get camera pose: camera -> ego -> global
        calib_sensor = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
        
        # Camera to ego
        T_cam_to_ego = transform_matrix(
            calib_sensor['translation'],
            calib_sensor['rotation']
        )
        
        # Ego to global
        T_ego_to_global = transform_matrix(
            ego_pose['translation'],
            ego_pose['rotation']
        )
        
        # Camera to global
        T_cam_to_global = T_ego_to_global @ T_cam_to_ego
        
        # Extract position
        position = T_cam_to_global[:3, 3]
        positions.append(position)
        poses_se3.append(T_cam_to_global)
    
    timestamps = np.array(timestamps)
    positions = np.array(positions)
    
    print(f"✓ Extracted {len(positions)} trajectory poses")
    
    return timestamps, positions, poses_se3


def visualize_trajectory(timestamps, positions, output_dir, scene_name, camera_name):
    """
    Visualize and save trajectory plots
    """
    output_dir = Path(output_dir)
    
    print(f"Visualizing trajectory...")
    
    # Calculate trajectory statistics
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    duration = timestamps[-1] - timestamps[0]
    
    # 3D trajectory plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, alpha=0.8, label='Camera Trajectory')
    
    # Mark start and end
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              c='green', s=200, marker='o', edgecolors='black', linewidths=2, 
              label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              c='red', s=200, marker='s', edgecolors='black', linewidths=2, 
              label='End', zorder=5)
    
    # Mark intermediate points
    step = max(1, len(positions) // 20)
    ax.scatter(positions[::step, 0], positions[::step, 1], positions[::step, 2], 
              c='orange', s=50, alpha=0.6, zorder=3)
    
    ax.set_xlabel('X [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z [m]', fontsize=12, fontweight='bold')
    ax.set_title(f'Camera Trajectory - {scene_name} ({camera_name})\n'
                f'Distance: {total_distance:.2f}m | Duration: {duration:.2f}s | Frames: {len(positions)}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set aspect ratio
    ax.set_box_aspect([1, 1, 0.5])
    
    plt.savefig(output_dir / 'trajectory_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved 3D trajectory plot to {output_dir / 'trajectory_3d.png'}")
    
    # 2D top-down view
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot trajectory with color gradient (time progression)
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))
    for i in range(len(positions) - 1):
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], 
               color=colors[i], linewidth=2, alpha=0.8)
    
    # Start and end markers
    ax.scatter(positions[0, 0], positions[0, 1], 
              c='green', s=300, marker='o', edgecolors='black', linewidths=3, 
              label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], 
              c='red', s=300, marker='s', edgecolors='black', linewidths=3, 
              label='End', zorder=5)
    
    # Intermediate points
    ax.scatter(positions[::step, 0], positions[::step, 1], 
              c='orange', s=80, alpha=0.6, edgecolors='white', linewidths=1, zorder=3)
    
    ax.set_xlabel('X [m]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=13, fontweight='bold')
    ax.set_title(f'Top-Down View - {scene_name} ({camera_name})\n'
                f'Distance: {total_distance:.2f}m | Duration: {duration:.2f}s',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                               norm=plt.Normalize(vmin=0, vmax=duration))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Time [s]', fontsize=11, fontweight='bold')
    
    plt.savefig(output_dir / 'trajectory_2d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved 2D trajectory plot to {output_dir / 'trajectory_2d.png'}")
    
    # Height profile
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Z coordinate over time
    ax1.plot(timestamps - timestamps[0], positions[:, 2], 'b-', linewidth=2)
    ax1.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Height (Z) [m]', fontsize=12, fontweight='bold')
    ax1.set_title('Camera Height Over Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # XY distance from start over time
    distances = np.linalg.norm(positions[:, :2] - positions[0, :2], axis=1)
    ax2.plot(timestamps - timestamps[0], distances, 'r-', linewidth=2)
    ax2.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance from Start [m]', fontsize=12, fontweight='bold')
    ax2.set_title('Traveled Distance Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved trajectory profiles to {output_dir / 'trajectory_profiles.png'}")
    
    # Save trajectory statistics
    stats_file = output_dir / 'trajectory_stats.txt'
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Trajectory Statistics - {scene_name} ({camera_name})\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of frames:        {len(positions)}\n")
        f.write(f"Duration:                {duration:.3f} seconds\n")
        f.write(f"Average FPS:             {len(positions) / duration:.2f} Hz\n")
        f.write(f"Total distance:          {total_distance:.3f} meters\n")
        f.write(f"Average speed:           {total_distance / duration:.3f} m/s\n")
        f.write(f"Start position (x,y,z):  ({positions[0, 0]:.3f}, {positions[0, 1]:.3f}, {positions[0, 2]:.3f}) m\n")
        f.write(f"End position (x,y,z):    ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}) m\n")
        f.write(f"Height range:            [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m\n")
        f.write(f"Bounding box (X):        [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m\n")
        f.write(f"Bounding box (Y):        [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m\n")
    
    print(f"✓ Saved trajectory statistics to {stats_file}")


def save_camera_intrinsics(nusc, scene_name, camera_name, output_dir):
    """
    Save camera intrinsics to text file
    """
    output_dir = Path(output_dir)
    
    # Find scene
    scene = None
    for s in nusc.scene:
        if s['name'] == scene_name:
            scene = s
            break
    
    if scene is None:
        raise ValueError(f"Scene {scene_name} not found")
    
    # Get camera calibration from first sample
    first_sample = nusc.get('sample', scene['first_sample_token'])
    cam_token = first_sample['data'][camera_name]
    cam_data = nusc.get('sample_data', cam_token)
    calib_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    
    # Extract intrinsics
    K_matrix = np.array(calib_sensor['camera_intrinsic'])
    fx, fy = K_matrix[0, 0], K_matrix[1, 1]
    cx, cy = K_matrix[0, 2], K_matrix[1, 2]
    
    # Image resolution
    width = cam_data.get('width', 1600)  # Default nuScenes resolution
    height = cam_data.get('height', 900)
    
    print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"Image resolution: {width} x {height}")
    
    # Save intrinsics in multiple formats
    
    # Format 1: Simple (fx fy cx cy) - for DPVO
    intrinsics_file = output_dir / 'camera_intrinsics.txt'
    with open(intrinsics_file, 'w') as f:
        f.write(f"{fx} {fy} {cx} {cy}\n")
    print(f"✓ Saved camera intrinsics (simple format) to {intrinsics_file}")
    
    # Format 2: Detailed
    intrinsics_detailed = output_dir / 'camera_intrinsics_detailed.txt'
    with open(intrinsics_detailed, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Camera Intrinsics - {scene_name} ({camera_name})\n")
        f.write("="*60 + "\n\n")
        f.write("Focal Length:\n")
        f.write(f"  fx = {fx:.6f} pixels\n")
        f.write(f"  fy = {fy:.6f} pixels\n\n")
        f.write("Principal Point:\n")
        f.write(f"  cx = {cx:.6f} pixels\n")
        f.write(f"  cy = {cy:.6f} pixels\n\n")
        f.write("Image Resolution:\n")
        f.write(f"  width  = {width} pixels\n")
        f.write(f"  height = {height} pixels\n\n")
        f.write("Intrinsic Matrix (K):\n")
        f.write(f"  [{fx:.6f},  0.000000,  {cx:.6f}]\n")
        f.write(f"  [0.000000,  {fy:.6f},  {cy:.6f}]\n")
        f.write(f"  [0.000000,  0.000000,  1.000000]\n\n")
        f.write("Distortion Coefficients:\n")
        f.write("  k1 = 0.0 (not provided by nuScenes)\n")
        f.write("  k2 = 0.0\n")
        f.write("  p1 = 0.0\n")
        f.write("  p2 = 0.0\n")
        f.write("  k3 = 0.0\n\n")
        f.write("Note: nuScenes typically provides undistorted images.\n")
    
    print(f"✓ Saved detailed camera intrinsics to {intrinsics_detailed}")
    
    # Format 3: JSON
    import json
    intrinsics_json = output_dir / 'camera_intrinsics.json'
    intrinsics_dict = {
        "scene": scene_name,
        "camera": camera_name,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": int(width),
        "height": int(height),
        "K_matrix": K_matrix.tolist(),
        "distortion": {
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
            "note": "nuScenes typically provides undistorted images"
        }
    }
    
    with open(intrinsics_json, 'w') as f:
        json.dump(intrinsics_dict, f, indent=2)
    
    print(f"✓ Saved camera intrinsics (JSON format) to {intrinsics_json}")
    
    return intrinsics_dict


def save_trajectory_to_file(timestamps, poses_se3, output_dir):
    """
    Save trajectory in TUM format (timestamp tx ty tz qx qy qz qw)
    """
    output_dir = Path(output_dir)
    traj_file = output_dir / 'trajectory.txt'
    
    from scipy.spatial.transform import Rotation
    
    with open(traj_file, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write("# Camera-to-World transformation (nuScenes format)\n")
        
        for ts, T in zip(timestamps, poses_se3):
            # Extract translation
            tx, ty, tz = T[:3, 3]
            
            # Extract rotation as quaternion
            rot = Rotation.from_matrix(T[:3, :3])
            quat = rot.as_quat()  # [x, y, z, w]
            qx, qy, qz, qw = quat
            
            f.write(f"{ts:.9f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    
    print(f"✓ Saved trajectory (TUM format) to {traj_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract nuScenes scene data: images, trajectory, and camera intrinsics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract scene-0061 from v1.0-mini
  python extract_nuscenes_scene.py \\
      --dataroot /data/sets/nuscenes \\
      --version v1.0-mini \\
      --scene scene-0061 \\
      --output_dir output/scene-0061

  # Extract with specific camera
  python extract_nuscenes_scene.py \\
      --dataroot /data/sets/nuscenes \\
      --version v1.0-trainval \\
      --scene scene-0103 \\
      --camera CAM_FRONT \\
      --output_dir output/scene-0103_front
        """
    )
    
    parser.add_argument('--dataroot', type=str, default="/mnt/cfs-baidu/public/mengmeng.liu/code/HRegNet2/datasets/nuscenes",
                       help='Path to nuScenes dataset root directory')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                       choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                       help='nuScenes dataset version')
    parser.add_argument('--scene', type=str, default="scene-0103",
                       help='Scene name (e.g., scene-0061)')
    parser.add_argument('--camera', type=str, default='CAM_FRONT',
                       choices=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                       help='Camera channel to extract')
    parser.add_argument('--output_dir', type=str, default='output/scene-0103',
                       help='Output directory for extracted data')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" nuScenes Scene Data Extraction Tool")
    print("="*70)
    print(f"Dataset root:    {args.dataroot}")
    print(f"Version:         {args.version}")
    print(f"Scene:           {args.scene}")
    print(f"Camera:          {args.camera}")
    print(f"Output dir:      {args.output_dir}")
    print("="*70 + "\n")
    
    # Load nuScenes
    print("Loading nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print("✓ Dataset loaded\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Step 1: Extract images (symbolic links)
    print("\n" + "-"*70)
    print("Step 1: Extracting images...")
    print("-"*70)
    image_info = extract_images(nusc, args.scene, args.camera, images_dir)
    
    # Step 2: Extract and visualize trajectory
    print("\n" + "-"*70)
    print("Step 2: Extracting trajectory...")
    print("-"*70)
    timestamps, positions, poses_se3 = extract_trajectory(nusc, args.scene, args.camera)
    
    print("\n" + "-"*70)
    print("Step 3: Visualizing trajectory...")
    print("-"*70)
    visualize_trajectory(timestamps, positions, output_dir, args.scene, args.camera)
    
    # Save trajectory to file
    save_trajectory_to_file(timestamps, poses_se3, output_dir)
    
    # Step 3: Save camera intrinsics
    print("\n" + "-"*70)
    print("Step 4: Saving camera intrinsics...")
    print("-"*70)
    intrinsics = save_camera_intrinsics(nusc, args.scene, args.camera, output_dir)
    
    # Generate summary
    print("\n" + "="*70)
    print(" Extraction Summary")
    print("="*70)
    print(f"✓ Extracted {len(image_info)} images to: {images_dir}")
    print(f"✓ Generated trajectory visualizations in: {output_dir}")
    print(f"✓ Saved camera intrinsics to: {output_dir}")
    print("\nOutput files:")
    print(f"  - images/                      : Symbolic links to images")
    print(f"  - image_list.txt               : List of all images with timestamps")
    print(f"  - trajectory_3d.png            : 3D trajectory visualization")
    print(f"  - trajectory_2d.png            : Top-down trajectory view")
    print(f"  - trajectory_profiles.png      : Height and distance profiles")
    print(f"  - trajectory_stats.txt         : Trajectory statistics")
    print(f"  - trajectory.txt               : Trajectory in TUM format")
    print(f"  - camera_intrinsics.txt        : Camera parameters (simple)")
    print(f"  - camera_intrinsics_detailed.txt : Detailed camera parameters")
    print(f"  - camera_intrinsics.json       : Camera parameters (JSON)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()


"""

python tools/extract_nuscenes_scene.py \
    --dataroot /mnt/cfs-baidu/public/mengmeng.liu/code/HRegNet2/datasets/nuscenes \
    --version v1.0-mini \
    --scene scene-0103 \
    --output_dir output/scene-0103
"""