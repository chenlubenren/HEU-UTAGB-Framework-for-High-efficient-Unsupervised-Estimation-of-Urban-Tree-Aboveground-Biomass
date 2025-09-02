#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-2  生成 DEM / CHM / SUB_CNT
行列号方向完全对齐
"""
import pathlib, sys, numpy as np, rasterio, laspy, yaml, argparse
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# -------------------------------------------------
# GPU 支持：优先使用 cupy，失败回退 numpy
# -------------------------------------------------
try:
    import cupy as cp
    def _as_numpy(x):
        return cp.asnumpy(x) if hasattr(x, 'get') else x
    xp = cp
except ImportError:
    cp = None
    def _as_numpy(x):
        return x
    xp = np

# -------------------------------------------------
# 读取 LAS（x,y,z）
# -------------------------------------------------
def read_las_xyz(path):
    las = laspy.read(path)
    xyz = np.vstack((las.x, las.y, las.z)).T
    return xp.asarray(xyz)  # 直接搬上 GPU（如果可用）

# -------------------------------------------------
# 粗栅格 DEM：4×CHM 分辨率 → 最近邻上采样
# -------------------------------------------------
def build_dem(xyz, res):
    coarse_res = res * 4.0
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = float(y.min()), float(y.max())

    # 用 floor 向上取整再 +1，避免 ceil 导致的越界
    coarse_cols = int(np.floor((max_x - min_x) / coarse_res)) + 1
    coarse_rows = int(np.floor((max_y - min_y) / coarse_res)) + 1
    coarse_transform = from_origin(min_x, max_y, coarse_res, coarse_res)

    coarse_dem = np.full((coarse_rows, coarse_cols), np.nan, dtype=np.float32)

    # 计算索引并裁剪到合法范围
    coarse_col = ((x - min_x) / coarse_res).astype(int)
    coarse_row = ((max_y - y) / coarse_res).astype(int)

    mask = (coarse_row >= 0) & (coarse_row < coarse_rows) & \
           (coarse_col >= 0) & (coarse_col < coarse_cols)
    coarse_row = _as_numpy(coarse_row[mask])
    coarse_col = _as_numpy(coarse_col[mask])
    z_np = _as_numpy(z[mask])

    coarse_dem[coarse_row, coarse_col] = z_np

    # 以下与原代码一致 …
    Yc, Xc = np.meshgrid(np.arange(coarse_dem.shape[0]),
                         np.arange(coarse_dem.shape[1]), indexing='ij')
    valid = ~np.isnan(coarse_dem)
    coarse_dem = griddata(np.column_stack((Xc[valid], Yc[valid])),
                          coarse_dem[valid],
                          (Xc, Yc), method='nearest')
    coarse_dem = gaussian_filter(coarse_dem.astype(np.float32), sigma=1.0)

    # 上采样到目标分辨率
    target_cols = int(np.floor((max_x - min_x) / res)) + 1
    target_rows = int(np.floor((max_y - min_y) / res)) + 1
    scale_x = coarse_dem.shape[1] / target_cols
    scale_y = coarse_dem.shape[0] / target_rows

    src_rows = (np.arange(target_rows) * scale_y).astype(int)
    src_cols = (np.arange(target_cols) * scale_x).astype(int)
    src_rows = np.clip(src_rows, 0, coarse_dem.shape[0] - 1)
    src_cols = np.clip(src_cols, 0, coarse_dem.shape[1] - 1)
    dem = coarse_dem[np.ix_(src_rows, src_cols)]

    transform = from_origin(min_x, max_y, res, res)
    return dem.astype(np.float32), transform
# -------------------------------------------------
# run_step2：完整流程（含越界植被点掩码）
# -------------------------------------------------
def run_step2(out_dir, cfg, las_path):
    print("==========  Step-2 DEM / CHM / SUB_CNT generation started  ==========")
    out_dir = pathlib.Path(out_dir)

    # 1) 读地面 & 植被
    ground_xyz = read_las_xyz(out_dir / 'ground.las')
    other_xyz = read_las_xyz(out_dir / 'other.las')

    # 2) 生成 DEM
    dem, transform = build_dem(ground_xyz, cfg['RESOLUTION'])
    dem = _as_numpy(dem)        # 回到 CPU
    rows, cols = dem.shape

    # 3) 植被点行列号（GPU）
    x, y, z = other_xyz[:, 0], other_xyz[:, 1], other_xyz[:, 2]
    col = ((x - transform.c) / transform.a).astype(int)
    row = xp.floor((transform.f - y) / (-transform.e)).astype(int)

    # 4) 有效范围掩码
    valid = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    x, y, z, row, col = x[valid], y[valid], z[valid], row[valid], col[valid]

    # 5) 相对高度 & 过滤
    row_np = _as_numpy(row)
    col_np = _as_numpy(col)
    z_np = _as_numpy(z)
    ground_z = dem[row_np, col_np]
    h_rel = z_np - ground_z
    h_rel = np.where((h_rel >= cfg['LOWEST_THRESHOLD']) &
                     (h_rel <= cfg['HIGHEST_THRESHOLD']), h_rel, 0)
    veg_xyz = np.column_stack((_as_numpy(x), _as_numpy(y), h_rel))

    # 6) 栅格化 CHM：取每个栅格第一个点
    chm = np.zeros(dem.shape, dtype=np.float32)
    # 先按行列排序，保证“第一个”含义一致
    order = np.lexsort((col_np, row_np))
    row_first = row_np[order]
    col_first = col_np[order]
    h_first = h_rel[order]

    # 使用 unique 取每个 (row,col) 第一次出现
    _, idx_first = np.unique(np.stack([row_first, col_first], axis=1),
                             axis=0, return_index=True)
    row_unique = row_first[idx_first]
    col_unique = col_first[idx_first]
    h_unique = h_first[idx_first]
    chm[row_unique, col_unique] = h_unique

    profile = dict(driver='GTiff', dtype='float32', count=1,
                   crs=f"EPSG:{cfg['EPSG_CODE']}",
                   transform=transform, height=rows,
                   width=cols, nodata=0, compress='lzw')
    with rasterio.open(out_dir / 'DEM.tif', 'w', **profile) as dst:
        dst.write(dem, 1)
    with rasterio.open(out_dir / 'CHM.tif', 'w', **profile) as dst:
        dst.write(chm, 1)

    # 8) 生成 SUB_CNT.tif：用原始 LAS
    full_xyz = read_las_xyz(las_path)
    x_all, y_all, z_all = full_xyz[:, 0], full_xyz[:, 1], full_xyz[:, 2]

    sub_res = cfg['SUB_RESOLUTION']
    min_x, max_x = float(x_all.min()), float(x_all.max())
    min_y, max_y = float(y_all.min()), float(y_all.max())
    sub_cols = int(np.ceil((max_x - min_x) / sub_res))
    sub_rows = int(np.ceil((max_y - min_y) / sub_res))
    sub_trans = from_origin(min_x, max_y, sub_res, sub_res)

    # 粗栅格索引（GPU）
    sub_cols_idx = ((x_all - sub_trans.c) / sub_trans.a).astype(int)
    sub_rows_idx = xp.floor((sub_trans.f - y_all) / (-sub_trans.e)).astype(int)

    # 细栅格索引（GPU）
    fine_cols_idx = ((x_all - transform.c) / transform.a).astype(int)
    fine_rows_idx = xp.floor((transform.f - y_all) / (-transform.e)).astype(int)

    # 掩码（GPU）
    valid_all = (fine_rows_idx >= 0) & (fine_rows_idx < rows) & \
                (fine_cols_idx >= 0) & (fine_cols_idx < cols) & \
                (sub_rows_idx >= 0) & (sub_rows_idx < sub_rows) & \
                (sub_cols_idx >= 0) & (sub_cols_idx < sub_cols)

    sub_rows_idx = sub_rows_idx[valid_all]
    sub_cols_idx = sub_cols_idx[valid_all]
    fine_rows_idx = fine_rows_idx[valid_all]
    fine_cols_idx = fine_cols_idx[valid_all]
    z_all = z_all[valid_all]

    # 重新回到 CPU 进行后续 numpy 计算（rasterio 写入需要）
    sub_rows_idx_np = _as_numpy(sub_rows_idx)
    sub_cols_idx_np = _as_numpy(sub_cols_idx)
    fine_rows_idx_np = _as_numpy(fine_rows_idx)
    fine_cols_idx_np = _as_numpy(fine_cols_idx)
    z_all_np = _as_numpy(z_all)

    ground_z_full = dem[fine_rows_idx_np, fine_cols_idx_np]
    canopy_top = chm[fine_rows_idx_np, fine_cols_idx_np]
    abs_heights = z_all_np
    voxel_top = ground_z_full + canopy_top - cfg['SUB_CHM_OFFSET']
    voxel_bottom = voxel_top - sub_res

    mask = (abs_heights >= voxel_bottom) & (abs_heights <= voxel_top) & (canopy_top > 0)
    sub_rows_idx_np = sub_rows_idx_np[mask]
    sub_cols_idx_np = sub_cols_idx_np[mask]

    # 使用 GPU unique+count（cupy 支持）
    idx = sub_rows_idx_np * sub_cols + sub_cols_idx_np
    unique, counts = np.unique(idx, return_counts=True)
    unr_rows, unr_cols = np.divmod(unique, sub_cols)
    sub_cnt = np.zeros((sub_rows, sub_cols), dtype=np.uint32)
    sub_cnt[unr_rows, unr_cols] = counts.astype(np.uint32)

    sub_profile = dict(
        driver='GTiff',
        dtype='uint32',
        count=1,
        width=sub_cols,
        height=sub_rows,
        crs=f"EPSG:{cfg['EPSG_CODE']}",
        transform=sub_trans,
        nodata=0,
        compress='lzw'
    )
    with rasterio.open(out_dir / 'SUB_CNT.tif', 'w', **sub_profile) as dst:
        dst.write(sub_cnt, 1)
    print("==========  Step-2 DEM / CHM / SUB_CNT generation finished  ==========")

