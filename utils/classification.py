import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


cnn_model = None

def load_cnn_model(model_path):
    global cnn_model
    cnn_model = CNN()
    state_dict = torch.load(model_path, map_location="cpu")
    cnn_model.load_state_dict(state_dict)
    cnn_model.eval()
    print("✅ CNN model loaded (state_dict)")


def predict_cancer(image):
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(tensor)

    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    label = "Cancer" if pred.item() == 1 else "No Cancer"
    confidence = round(confidence.item() * 100, 2)

    return label, confidence
