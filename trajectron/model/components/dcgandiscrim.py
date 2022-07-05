# modified from pytorch/examples/dcgan
import numpy as np

from torch import nn


class DCGANDiscrim(nn.Module):
    # dcgan discriminator
    def __init__(self, nc, ndf, input_hw, use_spectral_norm=True,  n_classify_head=25):
        # nc is the channel of input, ndf is the channel of hidden layer
        super(DCGANDiscrim, self).__init__()
        h, w = input_hw
        assert h == w  # only consider square input currently

        n_layer = np.floor(np.log2(h)).astype(int) - 1
        m_list = []
        size = h

        self.use_sn = use_spectral_norm

        for i in range(n_layer - 1):
            if i == 0:
                # input is (nc) x h x w
                size = size // 2
                m_list.append(self.sn(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)))
                m_list.append(nn.LeakyReLU(0.2, inplace=True))
            elif i < n_layer - 1:
                # state size. (ndf*i) x h/2^i x w/2^i
                size = size // 2
                m_list.append(self.sn(nn.Conv2d(ndf * i, ndf * (i + 1), 4, 2, 1, bias=False)))
                m_list.append(nn.LeakyReLU(0.2, inplace=True))

        self.main = nn.Sequential(*m_list)

        self.gan_head = self.sn(nn.Conv2d(ndf * (n_layer - 1), 1, size, 1, 0, bias=False))
        self.classify_head = nn.Linear(ndf * (n_layer - 1) * size * size, n_classify_head)

    def forward(self, input):
        hidden = self.main(input)
        gan_score = self.gan_head(hidden)
        class_score = self.classify_head(hidden.reshape(hidden.shape[0], -1))
        return gan_score, class_score

    def sn(self, model):
        return nn.utils.spectral_norm(model) if self.use_sn else model
