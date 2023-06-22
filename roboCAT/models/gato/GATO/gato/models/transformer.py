from typing import Dict, Any, Union
from gato import GatoConfig

import torch 
import torch.nn as nn
import torch.nn.functional as F

#implement alibi, flash sparse multihead attention + other juicy plug methods
from flash_attn.flash_blocksparse_attention import FlashBlocksparseMHA

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.attention = FlashBlocksparseMHA(embed_dim=config.layer_width,
                                             num_heads=config.num_attention_heads,
                                             dropout=config.dropout_rate,
                                             max_seq_length=config.max_seq_len)
        
        #may be unnecessary
        self.dropout = nn.Dropout(config.dropout_rate)

        self.feed_forward(nn.Sequential(
            nn.Linear(in_features=config.layer_width, out_features=config.feedforward_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(in_features=config.feedforward_hidden_size, out_features=config.layer_width),
            nn.Dropout(config.dropout_rate)
        ))

        self.layer_norm1 = nn.LayerNorm(normalized_shape=config.layer_width, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=config.layer_width, eps=1e-6)


    def forward(self, inputs):
        x_norm1 = self.layer_norm1(inputs)
        x_attention, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x_dropout = self.dropout(x_attention)
        x_residual = x_dropout + inputs

        x_norm2 = self.layer_norm2(x_residual)
        x_ff = self.feed_forward(x_norm2)
        x_residual2 = x_ff + x_residual
        return x_residual2


    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config