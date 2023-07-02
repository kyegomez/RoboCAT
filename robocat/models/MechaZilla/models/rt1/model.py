import torch
from robotic_transformer import MaxViT, RT1


vit = MaxViT(
    num_classes = 1000,
    dim_conv_stem = 64,
    dim = 96,
    dim_head = 36,
    depth = (2, 2, 5, 2),
    window_size = 7,
    mb_conv_expansion_rate = 4,
    mbconv_shrinkage_rate = 0.25,
    dropout = 0.1
)

model = RT1(
    vit = vit,
    num_actions = 11,
    depth = 6,
    heads = 8,
    dim_head = 64,
    cond_drop_prob = 0.2
)

video = torch.randn(2, 3, 6, 224, 224)

instructions = [
    'bring me that pizza on that table'
]

train_logits = model(video, instructions) # (2, 6, 11, 256) # (batch, frames, actions, bins)


#after training
model.eval()
eval_logits = model(video, instructions, cond_scale = 3.) # classifier free guidance with conditional scale of 3