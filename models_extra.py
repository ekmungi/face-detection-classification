"""A third model kept in its own script, to test notebooks that import code from several files."""
from models import MLP


class WideMLP(MLP):
    """The MLP with a much wider first layer (1024 units); about 0.94 million parameters."""

    def __init__(self):
        super().__init__(hidden=1024)
