import numpy as np
import os

path = "raw/DEC_16/NOx.npy"
if os.path.exists(path):
    arr = np.load(path)
    print(f"{path}: shape={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}")
else:
    print(f"File not found: {path}")
