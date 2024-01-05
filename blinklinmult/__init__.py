import toml
from os import PathLike
from pathlib import Path

# Module level constants
PROJECT_ROOT = Path(__file__).parents[1]
WEIGHTS_DIR = Path().home() / '.cache' / 'torch' / 'hub' / 'checkpoints' / 'blink'

__version__ = toml.load(PROJECT_ROOT / 'pyproject.toml')['project']['version']

# Type aliases
PathType = str | PathLike