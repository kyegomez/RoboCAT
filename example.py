import torch

from robocat.model import VQGanVAE, RoboCat



vqgan = VQGanVAE()

# Instantiate the RoboCat model
robo_cat = RoboCat(vqgan=vqgan)

# Generate random inputs
BATCH_SIZE = 4
FRAMES = 5
CHANNELS = 3
WIDTH, HEIGHT = 256, 256  # These values should match the expected input size for VQGanVAE
video_input = torch.rand(BATCH_SIZE, CHANNELS, FRAMES, WIDTH, HEIGHT)
texts = ["Hello world!", "Test input", "Another input", "Final test"]  # Random texts for each item in the batch

# Forward pass
output = robo_cat(video=video_input, texts=texts)

print(output.shape)  # This should give the shape of the output logits

