"""Types for modes """
from pathlib import PosixPath
from typing import Literal, Union

PathType = Union[str, PosixPath]

Field = Literal[
    "Ex",
    "Ey",
    "Ez",
    "Hx",
    "Hy",
    "Hz",
]

SemiVectorialMethod = Literal[
    "Ex",
    "Ey",
]
