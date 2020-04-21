import functools
import hashlib
from inspect import signature

import numpy as np

MAX_NAME_LENGTH = 127


def get_component_name(component_type, **kwargs):
    name = component_type

    if kwargs:
        name += "_" + dict2name(**kwargs)

    # If the name is too long, fall back on hashing the longuest arguments
    if len(name) > MAX_NAME_LENGTH:
        name = "{}_{}".format(component_type, hashlib.md5(name.encode()).hexdigest())

    return name


def clean_name(name):
    """ Ensures that names are composed of [a-zA-Z0-9]

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


def clean_value(value):
    """ returns more readable value (integer)
    if number is < 1:
        returns number units in nm (integer)
    """

    def f():
        return

    try:
        if isinstance(value, int):  # integer
            return str(value)
        elif type(value) in [float, np.float64]:  # float
            return "{:.4f}".format(value).replace(".", "p").rstrip("0").rstrip("p")
        elif isinstance(value, list):
            return "_".join(clean_value(v) for v in value)
        elif isinstance(value, tuple):
            return "_".join(clean_value(v) for v in value)
        elif isinstance(value, dict):
            return dict2name(**value)
        elif callable(value):
            return value.__name__
        else:
            return clean_name(str(value))
    except TypeError:  # use the __str__ method
        return clean_name(str(value))


def join_first_letters(name):
    """ join the first letter of a name separated with underscores (taper_length -> TL) """
    return "".join([x[0] for x in name.split("_") if x])


def dict2name(prefix=None, **kwargs):
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
    """ decorator for auto-naming modesolver functions
    if no Keyword argument `name`  is passed it creates a name by concenating all Keyword arguments

    .. plot::
      :include-source:

      import pp

      @pp.autoname
      def mode_solver(wg_width=0.5):
        ...

      ms = mode_solver(wg_width=1)
      print(ms)
      >> mode_solver_WW1

    """

    @functools.wraps(component_function)
    def wrapper(*args, **kwargs):
        if args:
            raise ValueError("autoname supports only Keyword args")
        if "name" in kwargs:
            name = kwargs.pop("name")
        else:
            name = get_component_name(component_function.__name__, **kwargs)

        component = component_function(**kwargs)
        component.name = name
        component.name_function = component_function.__name__
        sig = signature(component_function)
        component.settings.update(
            **{p.name: p.default for p in sig.parameters.values()}
        )
        component.settings.update(**kwargs)
        component.function_name = component_function.__name__
        return component

    return wrapper


if __name__ == "__main__":
    print(clean_name("mode_solver(:_=_2852"))

    print(clean_value(0.5))
