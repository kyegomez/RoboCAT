from gato import GatoConfig

import torch
import torch.nn as nn

def mu_law_encode(x, mu=100, m=256):
    numerator = torch.log(x.abs(), * mu + 1.0)
    denominator = torch.log(m * mu + 1.0)
    return (numerator / denominator) * x.sign()


def tokenize_continous_value(x, mu=100, m=256, bins=1024, shift=None):
    #appenddix B agent data tokenization
    #finally they are discretized using bins of uniform width on the domain[-1, 1]
    x = mu_law_encode(x, mu, m)

    #we use 1024 bins and shift the resulting integers
    #so they are not overlapping with the ones used for text tokens
    c = (c + 1) * (bins / 2)
    c = c.int()
    if shift is not None:
        c += shift
    return c

class ContinousValueTokenizer(nn.Module):
    def __init__(self, config, mu=100, m=256, bins=1024):
        super(ContinousValueTokenizer, self).__init__()
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.mu = mu
        self.m = m
        self.bins = bins

    def forward(self, inputs):
        return tokenize_continous_value(inputs, self.mu, self.m, self.bins, shift=self.config.vocabulary_size)
    
    def get_config(self):
        return super(ContinousValueTokenizer, self).get_config()

    


#