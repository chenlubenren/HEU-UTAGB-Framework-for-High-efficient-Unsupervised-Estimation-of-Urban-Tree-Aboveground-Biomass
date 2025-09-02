#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
import numpy as np
import rasterio
from skimage.segmentation import felzenszwalb
import subprocess
from utils.Multi_resolution_Seg.run import multi_resolution_seg

def run_step3(out_dir, cfg):
    print("==========  Step-3 Felzenszwalb crown segmentation started  ==========")
    chm_path = pathlib.Path(out_dir) / 'CHM.tif'
    assert chm_path.exists(), f'CHM not found: {chm_path}'


    process = multi_resolution_seg( str(chm_path), str(pathlib.Path(out_dir) /'CHM_SEGMENTS.tif'),
                                    cfg['SCALE'], cfg['COLOR_WEIGHT'], cfg['SHAPE_WEIGHT'])


    print("==========  Step-3 Felzenszwalb crown segmentation finished  ==========")