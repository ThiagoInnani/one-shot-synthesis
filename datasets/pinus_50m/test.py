import numpy as np
from PIL import Image

img = Image.open("/mnt/f/Machine Learning/one-shot-synthesis/datasets/pinus_50m/mask_class/1.png")
mask = np.array(img)
print("Formato:", mask.shape)
print("Valores únicos:", np.unique(mask))
