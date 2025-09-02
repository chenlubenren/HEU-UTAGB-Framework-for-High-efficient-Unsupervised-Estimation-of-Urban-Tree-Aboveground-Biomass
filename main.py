#!/usr/bin/env python3
import argparse
import yaml
import pathlib
import os
import numpy as np
from utils.batch import process_one_las
from utils.sweep import dict_product
from utils.log import init_logger
from utils.merge_rasters import merge_rasters
import rasterio
from skimage.measure import label as sk_label

DEFAULT_CONFIG = "CONFIG.YAML"

# -------------------------------------------------
# 拍平列表参数
# -------------------------------------------------
def _unwrap_lists(cfg_node):
    if isinstance(cfg_node, dict):
        return {k: _unwrap_lists(v) for k, v in cfg_node.items()}
    elif isinstance(cfg_node, list):
        return cfg_node[0]
    else:
        return cfg_node

# -------------------------------------------------
# 主函数
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=5)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['OUTPUT_ROOT'], exist_ok=True)

    # 1. 初始化日志 + 自动捕获所有 print
    logger = init_logger(cfg['OUTPUT_ROOT'])
    logger.info("=== 日志初始化成功，所有 print 已被重定向 ===")

    # 2. 收集 LAS/LAZ
    las_dir = pathlib.Path(cfg['INPUT_LAS_DIR'])
    las_files = sorted(las_dir.glob('*.las')) + sorted(las_dir.glob('*.laz'))
    if not las_files:
        logger.error('未找到 LAS/LAZ 文件')
        return

    # 3. 参数组合
    if args.sweep:
        sweep_cfgs = list(dict_product(cfg))
        logger.info(f'共 {len(sweep_cfgs)} 组参数组合')
    else:
        cfg = _unwrap_lists(cfg)
        sweep_cfgs = [cfg]

    # 4. 主循环
    for idx, one_cfg in enumerate(sweep_cfgs, 1):
        cfg_out = pathlib.Path(cfg['OUTPUT_ROOT']) / f"cfg_{idx:03d}"
        cfg_out.mkdir(parents=True, exist_ok=True)
        with open(cfg_out / 'params.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(one_cfg, f, allow_unicode=True, default_flow_style=False)

        logger.info(f'===== 开始处理第 {idx} 套参数，输出目录：{cfg_out} =====')

        # Step1-2：逐 LAS 处理
        for las_path in las_files:
            las_stem = las_path.stem
            out_sub = cfg_out / las_stem
            out_sub.mkdir(parents=True, exist_ok=True)

            logger.info(f'  处理 {las_path.name} → {out_sub}')
            try:
                process_one_las(
                    las_path, out_sub, one_cfg,
                    max(args.start, 1), min(args.end, 4)  # 最多到 step4
                )
            except Exception as e:
                logger.exception(f'失败: {las_path.name} -> {e}')
                continue

        if args.end == 2:
            # 收集每个 LAS 的 CHM
            chm_list = []
            for las_path in las_files:
                las_stem = las_path.stem
                out_sub = cfg_out / las_stem
                if (out_sub / 'CHM.tif').exists():
                    chm_list.append(out_sub / 'CHM.tif')
            if chm_list:
                logger.info('开始合并CHM …')
                merge_rasters(chm_list, cfg_out / 'CHM.tif')
                logger.info('CHM合并完成')

        # Step4 完成后合并 FILTERED_MASK
        if args.end == 4:
            mask_paths = [cfg_out / p.stem / 'FILTERED_MASK.tif' for p in las_files]
            mask_paths = [p for p in mask_paths if p.exists()]
            if mask_paths:
                # 1) 先合并
                merged_mask_path = cfg_out / 'FILTERED_MASK.tif'
                merge_rasters(mask_paths, merged_mask_path)

                logger.info('FILTERED_MASK 合并完成')

                # 2) 重新编号（全局连通域）
                with rasterio.open(merged_mask_path, 'r+') as dst:
                    mask = dst.read(1)
                    # 0 为背景，>0 为前景
                    new_labels = sk_label(mask > 0, connectivity=2)  # 8-连通
                    dst.write(new_labels.astype('uint16'), 1)

                logger.info('FILTERED_MASK 已全局重编号')

        # Step5：使用合并后的 FILTERED_MASK 和 CHM
        if args.end >= 5:
            from pipeline.step5_watershed import run_step5
            logger.info('开始在合并栅格上进行单木分割...')
            run_step5(cfg_out, one_cfg['STEP5'])
            logger.info('全部处理完成')


if __name__ == '__main__':
    main()