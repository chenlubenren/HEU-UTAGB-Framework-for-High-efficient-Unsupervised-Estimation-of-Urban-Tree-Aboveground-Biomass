# HEU-UTAGB
Framework for High-efficient Unsupervised Estimation of Urban Tree Aboveground Biomass at the Individual Tree Scale

## Cross-Platform
- Runs on Windows and Linux without modification.  
- GPU acceleration (cupy + cuml) is available only on Linux with CUDA 11.8 or newer.

## Quick Start
```bash
# Linux (GPU)
conda create -n heu python=3.10
conda activate heu
pip install -r requirements.txt
pip install cupy-cuda12x cuml-cuda12x   # match your CUDA version
```


```cmd
# Windows (CPU)
conda create -n heu python=3.10
conda activate heu
pip install -r requirements.txt
```


Place .las or .laz files in las_files/, then:

```bash
python main.py           # single parameter run (the first parameter of each list in CONFIG.yaml)
python main.py --sweep   # parameter sweep 
```

## Outputs
```
output/cfg_001/
├── INDIVIDUAL_TREE.tif   # individual tree IDs
├── CHM.tif               # canopy height model (corase)
├── DEM.tif               # digital elevation model (corase)
├── SUB_CNT.tif           # sub-canopy point density (corase)
└── indicator/            # intermediate feature layers for feature-based filtering
```



## Configuration
All parameters are in CONFIG.yaml; list entries are automatically combined during --sweep