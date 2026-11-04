"""Image transforms for MNIST: normalisation, with an optional light random shift and rotation for training."""
from torchvision import transforms

# MNIST channel statistics, the standard normalisation for this dataset.
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def build_transform(augment_train=False):
    """Returns the transform pipeline; with augment_train=True it adds a random rotation up to 10 degrees and a shift up to 10 percent."""
    steps = []
    if augment_train:
        steps.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
    steps += [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    return transforms.Compose(steps)
