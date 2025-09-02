# -*- coding: utf-8 -*-
"""
step4_filter_feature_v2.py
1. 先用面积、mean_chm 做“硬过滤”：
   area < AREA_SMALL_TH   OR
   mean_chm <= MEAN_CHM_MIN OR mean_chm >= MEAN_CHM_MAX
   直接丢弃，不再参与 regionprops。
2. 对剩余对象建立完整指标，并按新规则分层保留：
   a) aspect_ratio >= ASPECT_RATIO_MAX          → 剔除
   b) sub_cnt_ratio >= SUB_CNT_RATIO
      且 glcm_contrast >= GLCM_CON              → 直接保留
   c) 其余（小目标）再按
      near_circular > ST_NEAR_CIRC_MIN
      且 ST_CHM_MIN < mean_chm < ST_CHM_MAX
      且 ST_AREA_MIN <= area <= ST_AREA_MAX     → 保留
3. 输出：
   FILTERED_MASK.tif      uint16  保留对象的 label
   indicator/*.tif        float32 各指标栅格
"""
import pathlib
import numpy as np
import rasterio
import pandas as pd
import cv2
from skimage.measure import regionprops, label as sk_label
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import remove_small_objects, remove_small_holes
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

# -------------------------------------------------
# 工具：内存重采样
# -------------------------------------------------
def resample_sub_in_mem(mask_path, sub_path):
    with rasterio.open(mask_path) as msk:
        dst_shape = msk.shape
        dst_trans = msk.transform
        dst_crs = msk.crs
    dst = np.empty(dst_shape, dtype=np.int16)

    with rasterio.open(sub_path) as sub:
        reproject(
            source=rasterio.band(sub, 1),
            destination=dst,
            src_transform=sub.transform,
            src_crs=sub.crs,
            dst_transform=dst_trans,
            dst_crs=dst_crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return dst

# -------------------------------------------------
# GLCM 对比度
# -------------------------------------------------
def _calc_glcm_contrast(patch_1d, levels=32, dist=1):
    if patch_1d.size == 0 or patch_1d.min() == patch_1d.max():
        return 0.0
    q = ((patch_1d - patch_1d.min()) /
         (patch_1d.max() - patch_1d.min()) * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(
        q.reshape(-1, 1),
        distances=[dist],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=levels,
        symmetric=True,
        normed=True
    )
    return float(graycoprops(glcm, 'contrast')[0, 0])

# -------------------------------------------------
# 主流程
# -------------------------------------------------
def run_step4(out_dir, cfg):
    print("==========  Step-4 Feature-based patch filtering v2 ==========")
    out_dir = pathlib.Path(out_dir)
    mask_path = out_dir / "CHM_SEGMENTS.tif"
    chm_path  = out_dir / "CHM.tif"
    sub_path  = out_dir / "SUB_CNT.tif"

    # 读数据
    with rasterio.open(mask_path) as msk:
        profile = msk.profile
        res = abs(msk.transform[0])
        mask_arr = msk.read(1).astype(np.uint16)   # === CHANGED ===
    chm_arr = rasterio.open(chm_path).read(1)
    sub_arr = resample_sub_in_mem(mask_path, sub_path)

    # 1. 先 label（此时 label 已用 uint16）
    labeled = sk_label(mask_arr, connectivity=2)

    # 2. 先做一次“硬过滤”：面积 & mean_chm
    area_px_th = int(cfg["AREA_SMALL_TH"] / (res ** 2))
    labeled = remove_small_objects(labeled, min_size=area_px_th)

    props = regionprops(labeled, intensity_image=chm_arr)
    rows = []
    for p in tqdm(props):
        valid_chm = chm_arr[p.slice][p.image]
        valid_chm = valid_chm[valid_chm > 0]
        mean_chm = float(valid_chm.mean()) if valid_chm.size else 0.0

        # 高度硬过滤
        if mean_chm <= cfg["MEAN_CHM_MIN"] or mean_chm >= cfg["MEAN_CHM_MAX"]:
            labeled[labeled == p.label] = 0
            continue

        area = float(p.area * res ** 2)
        peri = p.perimeter
        near_circ = 4 * np.pi * p.area / (peri ** 2 + 1e-8)

        # aspect ratio
        mask_bin = (labeled == p.label).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            aspect = 1.0
        else:
            (_, _), (w, h), _ = cv2.minAreaRect(contours[0])
            aspect = max(w, h) / (min(w, h) + 1e-8)

        # sub_cnt_ratio
        sub_mask = sub_arr[p.slice][p.image] != 0
        sub_cnt_ratio = float(sub_mask.sum()) / (p.area + 1e-8)

        # glcm
        glcm_con = _calc_glcm_contrast(valid_chm)

        rows.append({
            "label": int(p.label),
            "area": area,
            "near_circular": float(near_circ),
            "aspect_ratio": float(aspect),
            "mean_chm": mean_chm,
            "sub_cnt_ratio": sub_cnt_ratio,
            "glcm_contrast": glcm_con,
        })

    df = pd.DataFrame(rows)

    # 3. 规则过滤
    def keep_rule(r):
        # a) 长宽比过高 且 近圆度过低 → 剔除
        if (r["aspect_ratio"] >= cfg["ASPECT_RATIO_MAX"] and
            r["near_circular"] <= cfg["NEAR_CIRC_MIN"]):
            return False

        # b) sub+glcm 直接晋升
        if (r["sub_cnt_ratio"] >= cfg["SUB_CNT_RATIO"] and
            r["glcm_contrast"] >= cfg["GLCM_CON"]):
            return True

        # c) 小目标额外规则
        if (r["near_circular"] > cfg["ST_NEAR_CIRC_MIN"] and
            cfg["ST_CHM_MIN"] < r["mean_chm"] < cfg["ST_CHM_MAX"] and
            cfg["ST_AREA_MIN"] <= r["area"] <= cfg["ST_AREA_MAX"]):
            return True
        return False

    df["keep"] = df.apply(keep_rule, axis=1)
    keep_labels = set(df[df["keep"]]["label"].values)

    # 4. 生成结果
    filtered = np.where(np.isin(labeled, list(keep_labels)), labeled, 0)
    filtered = sk_label(remove_small_holes(filtered > 0,
                                           area_threshold=int(500 / res ** 2)))

    profile.update(dtype="uint16", nodata=0, compress="lzw")
    with rasterio.open(out_dir / "FILTERED_MASK.tif", "w", **profile) as dst:
        dst.write(filtered.astype("uint16"), 1)

    # 5. indicator 栅格
    indicator_dir = out_dir / "indicator"
    indicator_dir.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", nodata=np.nan, compress="lzw")

    maps = {
        "aspect_ratio": dict(zip(df["label"], df["aspect_ratio"])),
        "area": dict(zip(df["label"], df["area"])),
        "near_circular": dict(zip(df["label"], df["near_circular"])),
        "mean_chm": dict(zip(df["label"], df["mean_chm"])),
        "sub_cnt_ratio": dict(zip(df["label"], df["sub_cnt_ratio"])),
        "glcm_contrast": dict(zip(df["label"], df["glcm_contrast"])),
    }
    for key, mapper in maps.items():
        arr = np.vectorize(mapper.get)(labeled.astype("int32")).astype("float32")
        with rasterio.open(indicator_dir / f"{key}.tif", "w", **profile) as dst:
            dst.write(arr, 1)

    print("==========  Step-4 FINISHED ==========")