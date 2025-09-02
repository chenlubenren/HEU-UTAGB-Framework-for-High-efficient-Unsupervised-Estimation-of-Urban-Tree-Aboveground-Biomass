#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-1  点云预处理 + 地面/建筑提取
改进：坡度自适应切片 + 局部 RANSAC
"""


import pathlib, sys, os
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils.io import read_las_xyz, save_pcd_as_las
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from tqdm import tqdm

# ---- 可选 GPU 依赖 ----
try:
    import cupy as cp
    import cuml.neighbors as cu_neighbors
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("[WARN] cuML/cupy not found, falling back to CPU-KDTree")

# -------------------------------------------------
# 工具函数
# -------------------------------------------------
def anisotropic_downsample(xyz, vx, vy, vz):
    """非均匀体素降采样，保留每个体素内 Z 最高的点索引"""
    vox = np.floor(np.column_stack([
        xyz[:, 0] / vx,
        xyz[:, 1] / vy,
        xyz[:, 2] / vz
    ])).astype(np.int64)
    vox_id = (vox[:, 0].astype(np.int64) << 40) | \
             (vox[:, 1].astype(np.int64) << 20) | \
             vox[:, 2].astype(np.int64)
    order = np.lexsort((-xyz[:, 2], vox_id))
    vox_id = vox_id[order]
    mask = np.concatenate(([True], vox_id[1:] != vox_id[:-1]))
    return order[mask]

def compute_avg_density(pcd):
    """计算点云平均点间距（CPU）"""
    xyz = np.asarray(pcd.points)
    tree = KDTree(xyz)
    nn_dists = []
    for i in tqdm(range(len(xyz)), desc="Compute density"):
        dist, _ = tree.query([xyz[i]], k=2)
        nn_dists.append(dist[0, 1])
    return np.mean(nn_dists)

# -------------------------------------------------
# 核心：切片 + 局部 RANSAC
# -------------------------------------------------
def extract_ground_tiled_ransac(xyz, tile_size, overlap, ransac_th, ransac_iter):
    """
    将点云切成若干 tile，每个 tile 内部跑 RANSAC 平面提取地面。
    返回地面点布尔掩码（True=ground）。
    """
    minx, maxx = xyz[:, 0].min(), xyz[:, 0].max()
    miny, maxy = xyz[:, 1].min(), xyz[:, 1].max()

    ground_mask = np.zeros(len(xyz), dtype=bool)

    # 切片网格
    x_edges = np.arange(minx - overlap, maxx + tile_size + overlap, tile_size)
    y_edges = np.arange(miny - overlap, maxy + tile_size + overlap, tile_size)

    for y0 in tqdm(y_edges[:-1], desc='Tiling+RANSAC'):
        for x0 in x_edges[:-1]:
            # 当前 tile 范围（含重叠）
            mask_tile = ((xyz[:, 0] >= x0 - overlap) &
                         (xyz[:, 0] < x0 + tile_size + overlap) &
                         (xyz[:, 1] >= y0 - overlap) &
                         (xyz[:, 1] < y0 + tile_size + overlap))
            idx_tile = np.where(mask_tile)[0]
            if len(idx_tile) < 100:
                continue

            sub_xyz = xyz[idx_tile]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sub_xyz)

            plane, inliers = pcd.segment_plane(
                distance_threshold=ransac_th,
                ransac_n=3,
                num_iterations=ransac_iter)
            ground_mask[idx_tile[inliers]] = True

    return ground_mask

# -------------------------------------------------
# 主函数
# -------------------------------------------------
def run_step1(las_path, out_dir, cfg):
    print("==========  Step-1 Tiling + Local-RANSAC ground/building extraction started  ==========")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取 LAS
    print("Reading LAS file ...")
    xyz, las = read_las_xyz(las_path)
    print(f"Original point count: {len(xyz)}")

    # 2. 非均匀体素降采样（转 float32 省内存）
    xyz = xyz.astype(np.float32)
    idx = anisotropic_downsample(xyz, cfg['VOXEL_XY'], cfg['VOXEL_XY'], cfg['VOXEL_Z'])
    xyz = xyz[idx]
    print(f"After downsampling: {len(xyz)}")

    # 3) 统计离群值滤波
    print("Noise removal ...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _, ind = pcd.remove_radius_outlier(
        nb_points=cfg['RADIUS_OUTLIER_NB'],
        radius=cfg['RADIUS_OUTLIER_R'])
    xyz = xyz[ind]
    print(f"After noise removal: {len(xyz)}")

    # 4) 计算 ransac_th
    low_mask = xyz[:, 2] <= np.percentile(xyz[:, 2], 90)
    low_pcd = o3d.geometry.PointCloud()
    low_pcd.points = o3d.utility.Vector3dVector(xyz[low_mask])

    if cfg.get('GROUND_FILTER_AUTO', True):
        avg_density = compute_avg_density(low_pcd)
        print(f"avg_density = {avg_density:.4f} m")
        ransac_th = avg_density * cfg['GROUND_FILTER_LEVEL']
    else:
        ransac_th = cfg['GROUND_FILTER_DISTANCE_MANUAL']
    print(f"ransac_th   = {ransac_th:.4f} m")

    # 5) 切片 + 局部 RANSAC 提取地面
    ground_mask = extract_ground_tiled_ransac(
        xyz,
        tile_size=cfg.get('TILE_SIZE', 100.0),
        overlap=cfg.get('TILE_OVERLAP', 0),
        ransac_th=ransac_th,
        ransac_iter=cfg.get('RANSAC_MAX_ITER', 100))
    ground_xyz = xyz[ground_mask]
    non_ground_xyz = xyz[~ground_mask]
    print(f"Ground points: {len(ground_xyz)}")
    print(f"Non-ground points: {len(non_ground_xyz)}")

    # 6) 建筑提取（与原来相同）
    mask_build = np.zeros(len(non_ground_xyz), dtype=bool)
    batch_size = 500000
    radius = cfg['KD_TREE_RADIUS']

    for start in tqdm(range(0, len(non_ground_xyz), batch_size), desc='Building'):
        end = min(start + batch_size, len(non_ground_xyz))
        sub_xyz = non_ground_xyz[start:end]

        if GPU_AVAILABLE:
            import cupy as cp
            sub_gpu = cp.asarray(sub_xyz)
            nn = cu_neighbors.NearestNeighbors(
                n_neighbors=min(200, len(non_ground_xyz)),
                metric='euclidean')
            nn.fit(sub_gpu)
            dists_gpu, idxs_gpu = nn.kneighbors(sub_gpu)
            dists = dists_gpu.get()
            idxs = idxs_gpu.get() + start
            for i, (d_vec, id_vec) in enumerate(zip(dists, idxs)):
                mask_local = d_vec <= radius
                neigh = non_ground_xyz[id_vec[mask_local]]
                if len(neigh) < 6:
                    continue
                cov = np.cov(neigh.T)
                eigvals = np.sort(np.linalg.eigvals(cov))
                lam1, lam3 = eigvals[-1], eigvals[0]
                if lam3 > 0 and (lam1 - lam3) / lam1 > cfg['EIG_RATIO_THRES']:
                    mask_build [id_vec[mask_local]] = True
        else:
            tree = KDTree(sub_xyz)
            idx_list = tree.query_radius(sub_xyz, r=radius)
            for i, idx in enumerate(idx_list):
                if len(idx) < 6:
                    continue
                neigh = non_ground_xyz[idx + start]
                cov = np.cov(neigh.T)
                eigvals = np.sort(np.linalg.eigvals(cov))
                lam1, lam3 = eigvals[-1], eigvals[0]
                if lam3 > 0 and (lam1 - lam3) / lam1 > cfg['EIG_RATIO_THRES']:
                    mask_build[idx + start] = True

    # 7) 保存结果
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground_xyz)

    building_pcd = o3d.geometry.PointCloud()
    building_pcd.points = o3d.utility.Vector3dVector(non_ground_xyz[mask_build])

    other_pcd = o3d.geometry.PointCloud()
    other_pcd.points = o3d.utility.Vector3dVector(non_ground_xyz[~mask_build])

    print(f"Building points: {len(building_pcd.points)}")
    print(f"Other points: {len(other_pcd.points)}")

    save_pcd_as_las(ground_pcd,   out_dir / 'ground.las',   las)
    save_pcd_as_las(building_pcd, out_dir / 'building.las', las)
    save_pcd_as_las(other_pcd,    out_dir / 'other.las',    las)

    # 8) 可选可视化
    if cfg.get('VISUALIZE_STEP1', False):
        ground_pcd.paint_uniform_color([0.4, 0.2, 0.1])
        building_pcd.paint_uniform_color([0.9, 0.2, 0.2])
        other_pcd.paint_uniform_color([0.2, 0.6, 0.2])
        o3d.visualization.draw_geometries(
            [ground_pcd, building_pcd, other_pcd],
            window_name="Step1 Tiling+RANSAC Result",
            width=1280, height=720)

    # 清理内存
    del xyz, pcd, ground_pcd, building_pcd, other_pcd
    if GPU_AVAILABLE:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    import gc
    gc.collect()