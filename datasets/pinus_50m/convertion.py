import os
import numpy as np
from PIL import Image

# Caminho das máscaras coloridas originais
input_dir = '/mnt/f/Machine Learning/one-shot-synthesis/datasets/pinus_50m/mask'  # <- ajuste aqui
# Caminho onde vamos salvar as máscaras de classes
output_dir = '/mnt/f/Machine Learning/one-shot-synthesis/datasets/pinus_50m/mask_class'  # <- ajuste aqui

# Criar pasta de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# Definir o mapeamento de cor -> índice
color2index = {
    (0, 0, 0): 0,          # Fundo
    (255, 0, 124): 1,      # Roxo (classe 1)
    (50, 183, 250): 2      # Azul (classe 2)
}

# Percorrer todas as imagens da pasta
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Abrir a imagem
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')
        np_img = np.array(img)

        # Criar máscara de classes
        mask = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=np.uint8)

        for color, index in color2index.items():
            mask[np.all(np_img == color, axis=-1)] = index

        # Salvar a máscara
        output_path = os.path.join(output_dir, filename)
        Image.fromarray(mask).save(output_path)

        print(f"Processado: {filename}")

print("✅ Todas as máscaras foram convertidas!")
