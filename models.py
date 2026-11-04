"""Model definitions shared by train.py and the notebook: a small CNN and a small MLP for MNIST."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Two convolution layers and two linear layers; about 1.2 million parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)  # 64 channels x 12 x 12 after the pooling step
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(torch.flatten(x, 1))))


class MLP(nn.Module):
    """Three linear layers on flattened images; about 235 thousand parameters."""

    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        return self.fc3(F.relu(self.fc2(x)))


# Names accepted by build_model and by the --model option of train.py.
MODEL_NAMES = ("cnn", "mlp")


def build_model(name):
    """Returns a new model for the given name ("cnn" or "mlp"); raises ValueError for any other name."""
    if name == "cnn":
        return Net()
    if name == "mlp":
        return MLP()
    raise ValueError(f"unknown model {name!r}; choose one of {MODEL_NAMES}")
