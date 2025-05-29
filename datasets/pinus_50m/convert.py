import os
import cv2
import numpy as np
from tqdm import tqdm

# Diretórios
input_masks_dir = 'F:\Machine Learning\one-shot-synthesis\datasets\pinus_50m\mask (PNG)'  # onde estão suas máscaras atuais (com 0,1,2)
output_masks_dir = 'F:\Machine Learning\one-shot-synthesis\datasets\pinus_50m\mask'  # onde vai salvar as novas máscaras (com 0,1)

# Cria o diretório de saída se não existir
os.makedirs(output_masks_dir, exist_ok=True)

# Lista todos os arquivos da pasta
mask_files = [f for f in os.listdir(input_masks_dir) if f.endswith('.png') or f.endswith('.jpg')]

# Para cada máscara
for mask_file in tqdm(mask_files, desc="Convertendo máscaras"):
    # Carrega a máscara como grayscale
    mask_path = os.path.join(input_masks_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Converte a classe 2 (Pinus derrubado) para 1 (Pinus)
    mask[mask == 2] = 1

    # Salva a nova máscara
    output_path = os.path.join(output_masks_dir, mask_file)
    cv2.imwrite(output_path, mask)

print(f"\nConversão finalizada! Máscaras salvas em {output_masks_dir}")
