import torch
from robocat import PALME, Robocat

model = Robocat(
    palme=PALME(),
    num_actions=11,
    action_bins=256,
    depth=6,
    heads=8,
    dim_head=64,
    token_learner_ff_mult=2,
    token_learner_num_layers=2,
    token_learner_num_output_tokens=8,
    cond_drop_prob=0.2,
    use_attn_conditioner=False,
    conditioner_kwargs=dict()
)

video = torch.rand((1, 3, 224, 224))
texts = ["this is a text"]
output = model(video, texts)
print(output.shape)

torch.save(model.state_dict(), 'rt3_model.pth')

