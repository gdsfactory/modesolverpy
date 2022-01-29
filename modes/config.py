"""package config."""

__all__ = ["PATH"]
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


class Path:
    module = module_path
    repo = repo_path
    cache = home / ".local" / "cache" / "modes"


PATH = Path()


if __name__ == "__main__":
    print(PATH.repo)
