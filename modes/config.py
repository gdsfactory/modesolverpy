"""package config."""

__all__ = ["PATH"]
import hashlib
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


def clean_value(value):
    if isinstance(value, float) and int(value) == value:
        value = int(value)
        value = str(value)
    return value


def get_kwargs_hash(**kwargs) -> str:
    """Returns kwargs parameters hash."""
    kwargs_list = [f"{key}={clean_value(kwargs[key])}" for key in sorted(kwargs.keys())]
    kwargs_string = "_".join(kwargs_list)
    kwargs_hash = hashlib.md5(kwargs_string.encode()).hexdigest()[:8]
    return kwargs_hash


if __name__ == "__main__":
    print(PATH.repo)
