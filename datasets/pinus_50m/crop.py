import os
import numpy as np
from PIL import Image
import cv2

def load_image(path):
    return Image.open(path).convert("RGB")

def load_mask(path):
    return Image.open(path).convert("L")

def crop_masked_patches(image_path, mask_path, crop_size=1024, stride=512, output_dir="output"):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)  # grayscale

    h, w = mask.shape

    count = 0
    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            mask_crop = mask[y:y+crop_size, x:x+crop_size]
            if np.any(mask_crop > 0):
                img_crop = img[y:y+crop_size, x:x+crop_size]

                # Salvar os crops
                cv2.imwrite(os.path.join(output_dir, "images", f"{count}.png"), img_crop)
                cv2.imwrite(os.path.join(output_dir, "masks", f"{count}.png"), mask_crop)
                count += 1

    print(f"{count} recortes com máscara salvos em '{output_dir}'.")


# ===== USO COM UMA IMAGEM =====
img_path = "pinus_50m/image/1.jpg"
mask_path = "pinus_50m/mask/1.png"  # mesma imagem se for máscara também

output_image_dir = "pinus_50m/image(1024)"
output_mask_dir = "pinus_50m/mask(1024)"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Carrega e prepara
img = load_image(img_path)
mask = load_mask(mask_path)
img_np = np.array(img)
mask_np = np.array(mask)

crop_masked_patches(img_path, mask_path)
