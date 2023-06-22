import torch
from gato import Gato, GatoConfig

# Create model instance
config = GatoConfig.small()
gato = Gato(config)

# Fake inputs for Gato
input_dim = config.input_dim
input_ids = torch.cat([
    torch.rand((1, 1, input_dim)) for _ in range(20)] +  # 20 image patches
    [torch.full((1, 1, input_dim), 0.25),  # continuous value
    torch.full((1, 1, input_dim), 624.0)] +  # discrete (actions, texts)
    [torch.rand((1, 1, input_dim)) for _ in range(20)] +  # 20 image patches
    [torch.full((1, 1, input_dim), 0.12),  # continuous value
    torch.full((1, 1, input_dim), 295.0)],  # discrete (actions, texts)
    dim=1)
encoding = torch.tensor([
    [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 2]
])
row_pos = (
    torch.tensor([[0.00, 0.25, 0.50, 0.75, 0, 0, 0.00, 0.25, 0.50, 0.75, 0, 0]]),  # pos_from
    torch.tensor([[0.25, 0.50, 0.75, 1.00, 0, 0, 0.25, 0.50, 0.75, 1.00, 0, 0]])  # pos_to
)
col_pos = (
    torch.tensor([[0.00, 0.00, 0.00, 0.80, 0, 0, 0.00, 0.00, 0.00, 0.80, 0, 0]]),  # pos_from
    torch.tensor([[0.20, 0.20, 0.20, 1.00, 0, 0, 0.20, 0.20, 0.20, 1.00, 0, 0]])  # pos_to
)
obs = (
    torch.tensor([[ 0,  1,  2, 19, 20, 21,  0,  1,  2, 19, 20, 21]]),  # obs token
    torch.tensor([[ 1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0]])  # obs token masking (for action tokens)
)
hidden_states = gato((input_ids, (encoding, row_pos, col_pos), obs))
