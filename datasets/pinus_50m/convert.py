import os
from PIL import Image
import numpy as np
import cv2 # OpenCV é usado para encontrar contornos

def process_image_to_binary_mask(image_path, output_path, select_largest_component=True):
    """
    Converte uma imagem para uma máscara binária (preto e branco),
    onde objetos não-pretos se tornam brancos e o fundo preto.
    Opcionalmente, seleciona apenas o maior componente conectado.

    Args:
        image_path (str): Caminho para a imagem de entrada.
        output_path (str): Caminho para salvar a máscara processada.
        select_largest_component (bool): Se True, mantém apenas o maior
                                         componente/contorno na máscara.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)

        # Identifica pixels pretos ([0,0,0])
        # is_black será True para pixels pretos, False para outros
        is_black = np.all(img_np == [0, 0, 0], axis=-1)

        # Cria a máscara binária:
        # Pixels não-pretos (objeto) se tornam 1, pixels pretos (fundo) se tornam 0.
        binary_mask_np = (~is_black).astype(np.uint8)

        if select_largest_component and np.any(binary_mask_np):
            # Encontra contornos na máscara binária (objetos de valor 1)
            # cv2.findContours espera uma imagem de 8 bits (0-255), então multiplicamos por 255 temporariamente
            # ou podemos trabalhar com 0 e 1 e depois converter para 0 e 255.
            # Aqui, binary_mask_np já é 0 ou 1.
            contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Encontra o maior contorno pela área
                largest_contour = max(contours, key=cv2.contourArea)

                # Cria uma nova máscara preta e desenha apenas o maior contorno nela
                output_contour_mask_np = np.zeros_like(binary_mask_np)
                cv2.drawContours(output_contour_mask_np, [largest_contour], -1, 1, cv2.FILLED) # Desenha com valor 1
                binary_mask_np = output_contour_mask_np
            else:
                # Se não houver contornos (ex: imagem toda preta), a máscara continua sendo de zeros
                binary_mask_np = np.zeros_like(binary_mask_np)


        # Converte a máscara final para valores 0 (fundo) e 255 (objeto)
        final_mask_np = (binary_mask_np * 255).astype(np.uint8)

        # Salva a imagem como escala de cinza ('L')
        output_img_pil = Image.fromarray(final_mask_np, mode='L')
        output_img_pil.save(output_path)
        # print(f"Processada: {image_path} -> {output_path}")

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {image_path}")
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")


def process_directory(input_dir, output_dir, select_largest_component=True):
    """
    Processa todas as imagens em um diretório de entrada e salva
    as máscaras resultantes em um diretório de saída.

    Args:
        input_dir (str): Diretório contendo as imagens originais.
        output_dir (str): Diretório onde as máscaras processadas serão salvas.
        select_largest_component (bool): Passado para process_image_to_binary_mask.
    """
    if not os.path.exists(input_dir):
        print(f"Erro: Diretório de entrada não encontrado: {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório de saída criado: {output_dir}")

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    processed_count = 0
    error_count = 0

    print(f"Iniciando processamento do diretório: {input_dir}")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            image_path = os.path.join(input_dir, filename)
            
            # Mantém o nome base do arquivo, mas garante que a saída seja .png
            base_name = os.path.splitext(filename)[0]
            output_filename = base_name + ".png" # Salva como PNG
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                process_image_to_binary_mask(image_path, output_path, select_largest_component)
                processed_count += 1
            except Exception: # Captura exceções de process_image_to_binary_mask para contagem
                error_count +=1
                
    print(f"\nProcessamento concluído.")
    print(f"Total de imagens processadas com sucesso: {processed_count}")
    if error_count > 0:
        print(f"Total de imagens com erro no processamento: {error_count}")
    print(f"Máscaras salvas em: {output_dir}")


if __name__ == '__main__':
    # --- Configuração ---
    # Substitua pelos seus caminhos de diretório
    diretorio_imagens_originais = "datasets\pinus_50m\mask (PNG)" # Ex: pasta com 1.png
    diretorio_mascaras_saida = "datasets\pinus_50m\mask" # Ex: pasta que será o 'mask' para a GAN

    # Se True, tentará isolar o maior objeto na imagem, tornando-a mais parecida
    # com uma silhueta única (como a do ônibus).
    # Se False, todas as partes não-pretas da imagem original se tornarão brancas.
    manter_apenas_maior_componente = False
    # --- Fim da Configuração ---

    # Verifica se os caminhos de exemplo precisam ser alterados
    if "caminho/para/suas" in diretorio_imagens_originais or \
       "caminho/para/suas" in diretorio_mascaras_saida:
        print("⚠️ Atenção! Por favor, edite o script e substitua os valores de")
        print("'diretorio_imagens_originais' e 'diretorio_mascaras_saida'")
        print("com os caminhos corretos para os seus diretórios antes de executar.")
    else:
        process_directory(diretorio_imagens_originais,
                          diretorio_mascaras_saida,
                          select_largest_component=manter_apenas_maior_componente)