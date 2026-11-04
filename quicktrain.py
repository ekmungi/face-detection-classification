"""Small training and evaluation helpers used by train.py and the notebook, so cells stay short."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def subset_loader(dataset, size, batch_size, seed, shuffle=True):
    """Returns a DataLoader over a reproducible random subset of `size` items of dataset."""
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:size].tolist()
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, order), batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_steps(model, loader, steps, device, lr=1e-3, optimizer=None):
    """Trains model for at most `steps` batches taken from loader and returns the list of losses.

    The loader is cycled when it has fewer batches than `steps`. Pass an `optimizer` to keep its state
    across calls (fit does this); otherwise a fresh Adam optimizer with learning rate `lr` is used.
    """
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    while len(losses) < steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if len(losses) >= steps:
                break
    return losses


def evaluate(model, loader, device):
    """Returns (mean loss, accuracy) of model over every batch in loader; leaves model in train mode."""
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    model.train()
    return loss_sum / total, correct / total


def accuracy(model, loader, device):
    """Returns the fraction of items in loader that model classifies correctly; leaves model in train mode."""
    return evaluate(model, loader, device)[1]
