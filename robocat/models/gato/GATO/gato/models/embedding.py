from typing import Dict, Any, Union
from gato import GatoConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from gato import GatoConfig



def _randomized_positions(from_v, to_v):
    pos = torch.rand_like(from_v) * (to_v - from_v)
    return pos.int()


def _rounded_mean_positions(from_v, to_v):
    pos = (from_v + to_v).float() / 2
    return pos.round()



class PatchPositionEncoding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.layer_width
        self.discretize_depth = config.discretize_depth
        self.patch_size = config.img_patch_size

        self.row_embedding = nn.Embedding(self.discretize_depth, self.embedding_dim)
        self.col_embedding = nn.Embedding(self.discretize_depth, self.embedding_dim)

    def _discretize(self, pos):
        return (pos * self.discretize_depth).round()

    def _discretize_interval(self, interval):
        pos_from, pos_to = interval
        return self._discretize(pos_from), self._discretize(pos_to)

    def forward(self, input_ids, pos):
        row_pos, col_pos = pos

        row_pos_from, row_pos_to = self._discretize_interval(row_pos)
        col_pos_from, col_pos_to = self._discretize_interval(col_pos)

        if self.training:
            row_pos = row_pos_from + _randomized_positions(row_pos_from, row_pos_to)
            col_pos = col_pos_from + _randomized_positions(col_pos_from, col_pos_to)
        else:
            row_pos = _rounded_mean_positions(row_pos_from, row_pos_to)
            col_pos = _rounded_mean_positions(col_pos_from, col_pos_to)

        return input_ids + self.row_embedding(row_pos.long()) + self.col_embedding(col_pos.long())

    
    def get_config(self):
        config = super(PatchPositionEncoding, self).get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config


class ResidualUnit(nn.Module):
    
    def __init__(self, num_groups: int, filters: int):
        super().__init__()
        self.num_groups = num_groups
        self.filters = filters
        self.conv1 = nn.Conv2d(in_channels=filters, out_channels=filters//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters//2, out_channels=filters, kernel_size=3, stride=2, padding=1)
        
        self.conv_proj = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=2, padding=0)
        self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=filters)
        self.gn2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=filters//2)
        self.gn_proj = nn.GroupNorm(num_groups=self.num_groups, num_channels=filters)

    def forward(self, x):
        residual = self.conv_prok(self.gn_proj(x))

        x = F.gelu(self.gn1(x))
        x = self.conv1(x)

        x = F.gelu(self.gn2(x))
        x = self.conv2(x)

        return x + residual



class ResidualEmbedding(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.root_conv = nn.Sequential(
            nn.Conv2d(in_channels=config.input_dim, out_channels=96, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(num_channels=96, num_groups=config.num_group_norm_groups),
            nn.GELU()
        )

        self.residual_units = nn.ModuleList([ResidualUnit(num_groups=config.num_group_norm_groups,
                                                          filters=96 * 2 ** (i + 1))
                                                          for i in range(3)])
        
        if config.input_dim != config.layer_width:
            self.conv_proj = nn.Conv2d(in_channels=96 * 2 ** 3, out_channels=config.layer_width, kernel_size=1, stride=1, padding=0)
    

    def forward(self, images):
        x = self.root_conv(images)

        for unit in self.residual_units:
            x = unit(x)

        if self.config.input_dim != self.config.layer_width:
            x = self.conv_proj(x)

        return x
        

    def get_config(self):
        config = super(ResidualEmbedding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config









class LocalPositionEncoding(nn.Module):

    def __init__(self, config: Union[GatoConfig, Dict[str, Any]], trainable=True, name=None, *args, **kwargs):
        """
        Appendix C.3. Position Encodings > Local Observation Position Encodings
        """
        super(LocalPositionEncoding, self).__init__(trainable=trainable, name=name, *args, **kwargs)

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.embedding = nn.Embedding(self.config.token_sequence_length, self.config.layer_width)

    def forward(self, inputs):
        obs_pos, obs_mask = inputs
        embed = self.embedding(obs_pos)

        ones = torch.ones((embed.shape[0], 1, self.config.layer_width)).to(embed.device)
        obs_mask = obs_mask.float().transpose(-1, -2).matmul(ones)
        return embed * obs_mask

    def get_config(self):
        config = super(LocalPositionEncoding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config


class DiscreteEmbedding(nn.Module):
    def __init__(self, config):
        super(DiscreteEmbedding, self).__init__()
        
        if isinstance(config, dict):
            config = GatoConfig(**config)

        self.config = config
        self.embedding = nn.Embedding(self.config_embedding_input_size, self.config.layer_width)

    def forward(self, inputs):
        return self.embedding(inputs)

    def get_config(self):
        config = super(DiscreteEmbedding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config
