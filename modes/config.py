""" package configuration
"""

__all__ = ["CONFIG"]

import pathlib

import matplotlib.pylab as plt

plt.rc("image", cmap="coolwarm")


home = pathlib.Path.home()
cwd = pathlib.Path.cwd()
cwd_config = cwd / "config.yml"
home_config = home / ".config" / "modes.yml"
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent
cache = home / ".local" / "cache" / "modes"
cache.mkdir(exist_ok=True, parents=True)


class Config:
    module = module_path
    repo = repo_path
    cache = home / ".local" / "cache" / "modes"


CONFIG = Config()

if __name__ == "__main__":
    print(CONFIG.repo)
