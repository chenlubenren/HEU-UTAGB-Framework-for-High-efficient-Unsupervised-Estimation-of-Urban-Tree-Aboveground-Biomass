#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版Step-5：通过临时TIFF拼接实现分水岭分割
每个label窗口保存为带坐标的int8临时文件，最终合并为uint16全局结果
"""

import pathlib
import numpy as np
import rasterio
import glob
import tempfile
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.merge import merge
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import measure
from tqdm import tqdm
from rasterio.windows import Window
from rasterio.transform import Affine


# ---------- 主流程 ----------
def run_step5(out_dir, cfg):
    print("==========  Step-5 Watershed segmentation (Temp TIFF merge)  ==========")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    chm_path = out_dir / 'CHM.tif'
    mask_path = out_dir / 'FILTERED_MASK.tif'

    if not chm_path.exists():
        raise FileNotFoundError(f"CHM 不存在: {chm_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"FILTERED_MASK 不存在: {mask_path}")

    # -------------------------------------------------
    # 0. 参数与临时目录设置
    dst_res = cfg.get('DOWNSAMPLE_RES', 0.5)

    temp_dir = tempfile.TemporaryDirectory(prefix='watershed_temp_', dir=out_dir)
    temp_path = pathlib.Path(temp_dir.name)

    # -------------------------------------------------
    # 1. 以原始 FILTERED_MASK 为基准，计算对齐的低分辨率 transform
    with rasterio.open(mask_path) as mask_src:
        orig_transform = mask_src.transform
        orig_crs       = mask_src.crs
        orig_bounds    = mask_src.bounds
        orig_shape     = (mask_src.height, mask_src.width)

    # 左上角保持不变，仅改分辨率
    aligned_transform = rasterio.Affine(dst_res, 0, orig_bounds.left,
                                        0, -dst_res, orig_bounds.top)
    dst_width  = int(np.round((orig_bounds.right  - orig_bounds.left) / dst_res))
    dst_height = int(np.round((orig_bounds.top    - orig_bounds.bottom) / dst_res))


    # -------------------------------------------------
    # 2. 重采样CHM和MASK到低分辨率
    def _resample_once(src_path, out_name, dtype, resampling, nodata=None):
        with rasterio.open(src_path) as src:
            profile = src.profile.copy()
            profile.update({
                'height': dst_height,
                'width': dst_width,
                'transform': aligned_transform,
                'dtype': dtype,
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
            })
            if nodata is not None:
                profile['nodata'] = nodata

            out_path = out_dir / out_name
            with rasterio.open(out_path, 'w', **profile) as dst:
                data = src.read(1, masked=True).filled(src.nodata or 0)
                dst_arr = np.empty((dst_height, dst_width), dtype=dtype)
                reproject(
                    data, dst_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=aligned_transform,
                    dst_crs=src.crs,
                    resampling=resampling,
                    src_nodata=src.nodata,
                    dst_nodata=profile.get('nodata'),
                )
                dst.write(dst_arr, 1)
            return out_path

    # 重采样CHM（低分辨率）
    chm_low_path = _resample_once(
        chm_path, f'CHM_{dst_res}m.tif', 'float32', Resampling.average, nodata=0
    )
    # 重采样MASK（低分辨率）
    mask_low_path = _resample_once(
        mask_path, f'MASK_{dst_res}m.tif', 'uint16', Resampling.nearest, nodata=0
    )
    print("重采样完成")


    # -------------------------------------------------
    # 3. 读取低分辨率数据
    with rasterio.open(chm_low_path) as chm_src, \
         rasterio.open(mask_low_path) as mask_src:
        chm_low = chm_src.read(1)
        mask_low = mask_src.read(1)
    res_low = dst_res  # 低分辨率像素大小（米）

    # -------------------------------------------------
    # 4. 分水岭分割（按label处理并保存临时TIFF）
    label_ids = np.unique(mask_low)
    label_ids = label_ids[label_ids != 0]  # 排除背景
    temp_files = []  # 存储临时文件路径

    # 后处理工具函数
    def _area_circularity(lab_win):
        """计算区域面积和近圆度"""
        props = measure.regionprops(lab_win.astype(int))
        if not props:
            return 0, 0.0
        p = props[0]
        return p.area, 4 * np.pi * p.area / max(p.perimeter ** 2, 1e-6)

    def _resegment_underseg(chm_win, mask_win, cfg, res_low):
        """欠分割区域重新分水岭"""
        chm_smooth = gaussian_filter(chm_win, sigma=cfg['GAUSSIAN_SIGMA'])
        min_dist_px = max(2, int((cfg['MIN_DISTANCE'] / 2) / res_low))
        coords = peak_local_max(
            chm_smooth,
            min_distance=min_dist_px,
            threshold_abs=cfg['THRESHOLD_ABS'],
            labels=mask_win,
        )
        markers = np.zeros_like(chm_win, dtype=int)
        if coords.size:
            markers[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1)
        return watershed(-chm_smooth, markers=markers, mask=mask_win).astype(np.uint16)

    global_id = 0
    # 逐个label处理
    for lid in tqdm(label_ids, desc=f"处理label并生成临时文件 @ {dst_res}m"):
        mask_bool = mask_low == lid
        rows, cols = np.where(mask_bool)
        if len(rows) == 0:
            continue  # 空区域跳过

        # 计算窗口范围（全局坐标）
        min_r, max_r = rows.min(), rows.max() + 1
        min_c, max_c = cols.min(), cols.max() + 1
        win_h, win_w = max_r - min_r, max_c - min_c
        if win_h == 0 or win_w == 0:
            continue  # 无效窗口

        # 提取窗口内数据
        win = np.s_[min_r:max_r, min_c:max_c]
        chm_win = chm_low[win].copy()
        mask_win = mask_bool[win]
        chm_win[~mask_win] = 0

        # 初始分水岭
        chm_smooth = gaussian_filter(chm_win, sigma=cfg['GAUSSIAN_SIGMA'])
        coords = peak_local_max(
            chm_smooth,
            min_distance=max(1, int(cfg['MIN_DISTANCE'] / res_low)),
            threshold_abs=cfg['THRESHOLD_ABS'],
            labels=mask_win,
        )
        labels_win = np.zeros_like(chm_win, dtype=np.uint16)
        if coords.size == 0:
            labels_win[mask_win] = 1  # 无峰值时整体标为1
        else:
            markers = np.zeros_like(chm_win, dtype=int)
            markers[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1)
            labels_win = watershed(-chm_smooth, markers=markers, mask=mask_win).astype(np.uint16)

        # 后处理迭代
        max_iter = int(cfg.get('MAX_ITER', 0))
        for _ in range(max_iter):
            region_ids = np.unique(labels_win)
            region_ids = region_ids[region_ids != 0]
            new_labels = labels_win.copy()
            merged_or_resplit = False

            for rid in region_ids:
                mask_r = labels_win == rid
                area_px, circ = _area_circularity(mask_r.astype(int))
                area_m2 = area_px * (res_low ** 2)  # 转换为平方米

                # 判断过分割/欠分割
                is_over = (area_m2 < cfg['ST_MIN_AREA_M2']) or \
                          (area_m2 < cfg['ST_MAX_AREA_M2'] and circ < cfg['MIN_CIRCULARITY'])
                is_under = (area_m2 > cfg['MAX_AREA_M2'] and circ < cfg['MIN_CIRCULARITY'])

                if is_over:
                    new_labels[mask_r] = 0  # 标记为待合并
                    merged_or_resplit = True
                elif is_under:
                    # 重新分割当前区域
                    chm_sub = chm_win.copy()
                    mask_sub = mask_r
                    chm_sub[~mask_sub] = 0
                    sub_lab = _resegment_underseg(chm_sub, mask_sub, cfg, res_low)
                    # 更新标签（局部编号，1开始）
                    new_labels[mask_sub] = 0
                    max_sub = int(sub_lab.max())
                    if max_sub > 0:
                        new_labels[mask_sub] = sub_lab[mask_sub]
                    merged_or_resplit = True

            # 合并过分割区域（填充0值区域为最近邻）
            if merged_or_resplit:
                del_mask = new_labels == 0
                if np.any(del_mask):
                    from scipy.ndimage import distance_transform_edt
                    _, idx = distance_transform_edt(del_mask, return_indices=True)
                    nearest = labels_win[idx[0], idx[1]]
                    new_labels[del_mask] = nearest[del_mask]

            labels_win = new_labels
            if not merged_or_resplit:
                break  # 无变化则退出迭代

        # 保存当前窗口为临时TIFF（带地理坐标）
        # 计算窗口的地理变换（基于低分辨率aligned_transform）
        # 低分辨率左上角像素坐标对应的地理坐标
        x_min = aligned_transform.c + min_c * aligned_transform.a
        y_max = aligned_transform.f + min_r * aligned_transform.e

        # 构造窗口的 transform
        win_transform = Affine(
            aligned_transform.a, 0, x_min,
            0, aligned_transform.e, y_max
        )

        # 临时文件命名（包含label和坐标范围）
        temp_file = temp_path / f"temp_lid{lid}_r{min_r}-{max_r}_c{min_c}-{max_c}.tif"
        # 写入TIFF（uint16，0为nodata）
        profile = {
            'driver': 'GTiff',
            'height': win_h,
            'width': win_w,
            'count': 1,
            'dtype': 'uint16',
            'crs': orig_crs,
            'transform': win_transform,
            'nodata': 0,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }
        # 直接平移成全局 ID 后再写入
        with rasterio.open(temp_file, 'w', **profile) as dst:
            dst.write(labels_win, 1)

        # ---------- 立即平移为全局唯一 ID ----------
        with rasterio.open(temp_file, 'r+') as dst:
            arr = dst.read(1)
            mask = arr > 0
            if np.any(mask):
                max_local = int(arr.max())
                arr[mask] += global_id
                dst.write(arr, 1)
                global_id += max_local        # 更新全局计数器

        temp_files.append(temp_file)

    # -------------------------------------------------
    # 5. 合并所有临时TIFF（低分辨率）
    print("合并临时TIFF文件...")
    srcs = [rasterio.open(p) for p in temp_files]
    merged_low, merged_transform = merge(srcs)
    merged_low = merged_low[0]  # 取第一个波段
    for src in srcs:
        src.close()

    # -------------------------------------------------
    # 6. 映射回原始分辨率
    print("重采样至原始分辨率...")
    labels_high = np.empty(orig_shape, dtype=np.uint16)
    reproject(
        merged_low.astype(np.uint16), labels_high,
        src_transform=merged_transform,
        src_crs=orig_crs,
        dst_transform=orig_transform,
        dst_crs=orig_crs,
        resampling=Resampling.nearest,
    )

    # 读取原始掩膜并修正背景
    with rasterio.open(mask_path) as mask_src:
        mask_orig = mask_src.read(1)
    labels_high[mask_orig == 0] = 0

    # -------------------------------------------------
    # 7. 标签重排序（连续编号）并保存最终结果
    print("重排序标签并保存结果...")
    # 提取非零唯一标签
    unique_ids = np.unique(labels_high)
    unique_ids = unique_ids[unique_ids != 0]
    # 映射表：旧ID → 新ID（从1开始连续）
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids, 1)}
    # 应用映射
    labels_remapped = np.zeros_like(labels_high, dtype=np.uint16)
    for old_id, new_id in tqdm(id_map.items(), desc="重映射标签"):
        labels_remapped[labels_high == old_id] = new_id

    # 保存最终结果（uint16）
    seg_final_path = out_dir / 'INDIVIDUAL_TREE.tif'
    final_profile = {
        'driver': 'GTiff',
        'height': orig_shape[0],
        'width': orig_shape[1],
        'count': 1,
        'dtype': 'uint16',
        'crs': orig_crs,
        'transform': orig_transform,
        'nodata': 0,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 1024,
        'blockysize': 1024,
    }
    with rasterio.open(seg_final_path, 'w',** final_profile) as dst:
        dst.write(labels_high, 1)

    print("==========  完成  ==========")
    print(f"最终结果：{seg_final_path}")
    print(f"中间文件：{chm_low_path}")
    print(f"中间文件：{mask_low_path}")
