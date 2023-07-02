import torch
from models.MechaZilla.models.rt1.robotic_transformer import MaxViT, RT1

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
    

#example usage
video = torch.randn(2, 3, 6, 224, 224)
instructions = [
    'Bring me tthat apple on the table',
    'Please bring me the butter'
]

robo_cat = RoboCAT(
    num_classes=1000,
    dim_conv_stem=64,
    dim=96,

    dim_head=32,
    depth=(2, 2, 5, 2),
    window_size = 7,

    mbconv_expansion_rate=4,
    mbconv_shrinkage_rate = 0.25,
    dropout = 0.1,

    num_actions = 11,
    rt_depth = 6,
    heads = 8,

    rt_dim_head = 64,
    cond_drop_prob=0.2
)

train_logits = robo_cat.forward(video, instructions)
robo_cat.model.eval()
eval_logits = robo_cat.forward(video, instructions, cond_scale=3.0)