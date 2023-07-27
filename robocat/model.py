from robocat.models.MechaZilla.models.rt1.robotic_transformer import MaxViT, RT1

class RoboCAT:
    def __init__(self, num_classes, dim_conv_stem, dim, dim_head, depth, window_size, mbconv_expansion_rate,
                 mbconv_shrinkage_rate, dropout, num_actions, rt_depth, heads, rt_dim_head, cond_drop_prob):
        self.vit = MaxViT(
            num_classes=num_classes,
            dim_conv_stem=dim_conv_stem,
            dim=dim,

            dim_head=dim_head,
            depth = depth,
            window_size = window_size,

            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate = mbconv_shrinkage_rate,
            dropout=dropout
        )

        self.model = RT1(
            vit = self.vit,
            num_actions = num_actions,
            depth = rt_depth,
            heads = heads,
            dim_head = rt_dim_head,
            cond_drop_prob = cond_drop_prob,
        )

    def forward(self, video, instructions, cond_scale=None):
        return self.model(video, instructions, cond_scale)
    
