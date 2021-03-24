import functools
import hashlib
from inspect import signature
from typing import Any, Dict

import numpy as np

MAX_NAME_LENGTH = 36


def get_component_name(component_type: str, **kwargs) -> str:
    name = component_type

    if kwargs:
        name += "_" + dict2name(**kwargs)

    # If the name is too long, fall back on hashing the longuest arguments
    if len(name) > MAX_NAME_LENGTH:
        name = "{}_{}".format(component_type, hashlib.md5(name.encode()).hexdigest())

    return name


def clean_name(name: str) -> str:
    """Ensures that names are composed of [a-zA-Z0-9]

    FIXME: only a few characters are currently replaced.
        This function has been updated only on case-by-case basis
    """
    replace_map = {
        "=": "",
        ",": "_",
        ")": "",
        "(": "",
        "-": "m",
        ".": "p",
        ":": "_",
        "[": "",
        "]": "",
        " ": "_",
    }
    for k, v in list(replace_map.items()):
        name = name.replace(k, v)
    return name


def clean_value(value: Any) -> str:
    """returns more readable value (integer)
    if number is < 1:
        returns number units in nm (integer)
    """

    def f():
        return

    try:
        if isinstance(value, int):  # integer
            value = str(value)
        elif type(value) in [float, np.float64]:  # float
            value = "{:.4f}".format(value).replace(".", "p").rstrip("0").rstrip("p")
        elif isinstance(value, list):
            value = "_".join(clean_value(v) for v in value)
        elif isinstance(value, tuple):
            value = "_".join(clean_value(v) for v in value)
        elif isinstance(value, dict):
            value = dict2name(**value)
        elif callable(value):
            value = value.__name__
        else:
            value = clean_name(str(value))
        if len(value) > MAX_NAME_LENGTH:
            value = hashlib.md5(value.encode()).hexdigest()
        return value
    except TypeError:  # use the __str__ method
        return clean_name(str(value))


def join_first_letters(name: str) -> str:
    """ join the first letter of a name separated with underscores (taper_length -> TL) """
    return "".join([x[0] for x in name.split("_") if x])


def dict2name(prefix: str = "", **kwargs) -> str:
    """ returns name from a dict """
    if prefix:
        label = [prefix]
    else:
        label = []
    for key in sorted(kwargs):
        value = kwargs[key]
        key = join_first_letters(key)
        value = clean_value(value)
        label += [f"{key.upper()}{value}"]
    label = "_".join(label)
    return clean_name(label)


def autoname(component_function):
    """decorator for auto-naming modesolver functions
    if no Keyword argument `name`  is passed it creates a name by concenating all Keyword arguments

    .. plot::
      :include-source:

      import pp

      @pp.autoname
      def mode_solver(width=0.5):
        ...

      ms = mode_solver(width=1)
      print(ms)
      >> mode_solver_WW1

    """

    @functools.wraps(component_function)
    def wrapper(*args, **kwargs):
        if args:
            raise ValueError("autoname supports only Keyword args")
        kwargs.pop("plot", "")
        kwargs.pop("plot_profile", "")
        name = kwargs.pop(
            "name", get_component_name(component_function.__name__, **kwargs)
        )
        sig = signature(component_function)
        if (
            "args" not in sig.parameters
            and "kwargs" not in sig.parameters
            and "wg_kwargs" not in sig.parameters
        ):
            for key in kwargs.keys():
                assert (
                    key in sig.parameters.keys()
                ), f"{key} key not in {list(sig.parameters.keys())}"

        component = component_function(**kwargs)
        component.name = name
        component.name_function = component_function.__name__
        component.settings.update(
            **{p.name: p.default for p in sig.parameters.values()}
        )
        component.settings.update(**kwargs)
        component.function_name = component_function.__name__
        return component

    return wrapper


class _Dummy:
    name: str = "dummy"
    settings: Dict[str, float] = dict(a=3)


@autoname
def _dummy(plot: bool = True, length: int = 3, width: float = 0.5) -> _Dummy:
    c = _Dummy()
    c.name = ""
    c.settings_exta = dict(plot=plot, length=length, width=width)
    return c


def test_autoname() -> None:
    name_base = _dummy().name
    assert name_base == "_dummy"
    name_plot = _dummy(plot=True).name
    assert name_base == name_plot, "plot argument should be ingored in names"
    name_int = _dummy(length=3).name
    assert name_int == "_dummy_L3"
    name_float = _dummy(width=0.5).name
    assert name_float == "_dummy_W0p5"


def test_clean_value() -> None:
    assert clean_value(0.5) == "0p5"
    assert clean_value(5) == "5"


def test_clean_name() -> None:
    assert clean_name("mode_solver(:_=_2852") == "mode_solver___2852"


if __name__ == "__main__":
    print(clean_name("mode_solver(:_=_2852"))
    # print(clean_value(0.5))
    test_autoname()
    # test_clean_value()
