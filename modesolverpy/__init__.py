import modesolverpy.materials as materials
from modesolverpy.config import CONFIG
from modesolverpy.mode_solver import get_modes_jsonpath
from modesolverpy.mode_solver_full_vectorial import mode_solver_full
from modesolverpy.mode_solver_semi_vectorial import mode_solver_semi
from modesolverpy.waveguide import si, sio2, waveguide

__all__ = [
    "CONFIG",
    "get_modes_jsonpath",
    "mode_solver_semi",
    "mode_solver_full",
    "waveguide",
    "materials",
]

if __name__ == "__main__":
    print(__all__)
