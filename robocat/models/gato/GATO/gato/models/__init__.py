

from gato.models.transformer import TransformerBlock
from gato.models.embedding import PatchPositionEncoding, ResidualEmbedding, LocalPositionEncoding, DiscreteEmbedding
from gato.models.tokenizers import ContinousValueTokenizer

from gato import GatoConfig
from typing import Dict, Any, Union


import torch
import torch.nn as nn
import torch.nn.functional as F





class Gato(nn.Module):
    def __init__(self, config):
        super(Gato, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.image_embedding = PatchEmbedding(config)
        self.discrete_embedding = DiscreteEmbedding(config)
        self.continuous_encoding = ContinousValueTokenizer(config)
        self.transformer = Transformer(config)
        self.local_pos_encoding = LocalPositionEncoding(config)

    def forward(self, inputs):
        input_ids, (encoding, row_pos, col_pos), (obs_pos, obs_mask) = inputs
        encoding = F.one_hot(encoding, num_classes=3).float()

        ones = torch.ones((input_ids.size(0), 1, self.config.layer_width))
        image_embed = self.image_embedding((input_ids, (row_pos, col_pos)))
        image_embed *= encoding[..., 0].unsqueeze(-1).matmul(ones)

        continuous_embed = self.continuous_encoding(input_ids[..., 0])
        continuous_embed = self.discrete_embedding(continuous_embed)
        continuous_embed *= encoding[..., 1].unsqueeze(-1).matmul(ones)

        discrete_embed = self.discrete_embedding(input_ids[..., 0])
        discrete_embed *= encoding[..., 2].unsqueeze(-1).matmul(ones)

        embed = image_embed + continuous_embed + discrete_embed
        embed += self.local_pos_encoding((obs_pos, obs_mask))

        hidden_states = self.transformer(embed)
        return hidden_states
    
    def get_config(self):
        return super(Gato, self).get_config()


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.encoders = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_transformer_blocks)])

    def forward(self, inputs):
        x = inputs
        for encoder in self.encoders:
            x = encoder(x)
        
        return x


    def get_config(self):
        return super(Transformer, self).get_config()


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super(PatchEmbedding, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.residual_embedding = ResidualEmbedding(config)
        self.pos_encoding = PatchPositionEncoding(config)
    
    def forward(self, inputs):
        input_ids, (row_pos, col_pos) = inputs
        patch_size = self.config.img_patch_size
        depth = self.config.input_dim // (patch_size * patch_size)

        x = input_ids.view(-1, input_ids.size(1), patch_size, patch_size, depth)
        x = self.residual_embedding(x)
        x = self.pos_encoding((x, (row_pos, col_pos)))
        return x

    def get_config(self):
        return super(PatchEmbedding, self).get_config()
