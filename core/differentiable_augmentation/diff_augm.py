from .AugmentPipe_kornia import AugmentPipe_kornia
from .AugmentPipe_stylegan2 import AugmentPipe # Adicione esta linha


class augment_pipe():
    def __init__(self, opt):
        if opt.use_kornia_augm:
            self.augment_func = AugmentPipe_kornia(opt.prob_augm, opt.no_masks).to(opt.device)
        else:
            # Altere esta seção para usar a AugmentPipe do StyleGAN2
            self.augment_func = AugmentPipe().to(opt.device) # Instancie a AugmentPipe sem argumentos, pois ela gerencia os parâmetros internamente.
            # Você pode precisar passar o 'p' (probabilidade geral) para ela se seu 'opt.prob_augm' for o 'p' do StyleGAN2
            # Se for o caso, mude para: self.augment_func = AugmentPipe(p=opt.prob_augm).to(opt.device)
            # Analise como 'p' é usado no StyleGAN2 e como 'opt.prob_augm' é usado no seu código para decidir.
            # Por padrão, a AugmentPipe do StyleGAN2 tem todos os multiplicadores de probabilidade desabilitados (0).
            # Então, você pode precisar ajustar as configurações dos multiplicadores (xflip, rotate90, etc.) dentro do construtor se quiser usar as aumentações específicas.
            # Exemplo: self.augment_func = AugmentPipe(xflip=1, rotate=1, p=opt.prob_augm).to(opt.device)
    def __call__(self, batch, real=True):
        return self.augment_func(batch)


