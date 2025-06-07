from .AugmentPipe_kornia import AugmentPipe_kornia
from .AugmentPipe_stylegan2 import AugmentPipe
import torch

class augment_pipe():
    def __init__(self, opt):
        if opt.use_kornia_augm:
            self.augment_func = AugmentPipe_kornia(opt.prob_augm, opt.no_masks).to(opt.device)
            self.kornia_mode = True
        else:
            self.augment_func = AugmentPipe(
                xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
                brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
                imgfilter=1, noise=1, cutout=1,
            ).to(opt.device)
            self.augment_func.p = torch.tensor(opt.prob_augm, device=opt.device)
            self.kornia_mode = False

    def __call__(self, batch):
        if self.kornia_mode:
            return self.augment_func(batch)
        else:
            # images é uma lista. Precisamos pegar o tensor de imagens principal.
            original_images_list = batch["images"]

            # --- Tentativa 1: Pegar o primeiro tensor da lista ---
            # main_images_tensor = original_images_list[0]

            # --- Tentativa 2 (Mais provável para saída de GANs): Pegar o último tensor da lista (maior resolução) ---
            main_images_tensor = original_images_list[-1]




            augmented_main_images = self.augment_func(main_images_tensor)

            # Agora, como a AugmentPipe_stylegan2 só aumentou um tensor,
            # precisamos decidir como reinserir isso na lista original.
            # A forma mais simples é substituir o tensor que foi aumentado.
            # Se você pegou o último, substitua o último.
            original_images_list[-1] = augmented_main_images
            batch["images"] = original_images_list # Retorne a lista modificada

            # IMPORTANTE: Se o seu pipeline espera que *todos* os tensores na lista sejam aumentados
            # com a mesma transformação, a solução seria mais complexa, iterando e aplicando
            # a transformação de forma manual para cada tensor, ou adaptando a AugmentPipe_stylegan2.
            # Por enquanto, estamos focando no caso mais simples de aumentar apenas o tensor final.
            return batch