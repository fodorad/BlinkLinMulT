import toml
from os import PathLike
from pathlib import Path
from importlib.metadata import version

try:
    __version__ = version("blinklinmult")
except Exception:
    __version__ = "unknown"

# Module level constants
PROJECT_ROOT = Path(__file__).parents[1]
WEIGHTS_DIR = Path().home() / '.cache' / 'torch' / 'hub' / 'checkpoints' / 'blink'

# Type aliases
PathType = str | PathLike