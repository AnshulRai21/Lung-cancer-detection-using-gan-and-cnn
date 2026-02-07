import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


gan_generator = None

def load_gan_model(model_path):
    global gan_generator
    gan_generator = Generator()
    state_dict = torch.load(model_path, map_location="cpu")
    gan_generator.load_state_dict(state_dict)
    gan_generator.eval()
    print("✅ GAN model loaded (state_dict)")


def enhance_image(preprocessed_image):
    tensor = torch.from_numpy(preprocessed_image).permute(0, 3, 1, 2)
    with torch.no_grad():
        output = gan_generator(tensor)
    output = output.squeeze().cpu().numpy()
    output = np.clip(output, 0, 1)
    return (output * 255).astype(np.uint8)
