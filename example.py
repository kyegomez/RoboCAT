import torch 
from robocat.model import RoboCat

model = RoboCat()

video = torch.randn(2, 3, 6, 224, 224)
instructions = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]

result = model.forward(video, instructions)
print(result)

