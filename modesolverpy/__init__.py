from modesolverpy import materials
from modesolverpy.config import CONFIG
from modesolverpy.group_index import group_index
from modesolverpy.mode_solver_full import mode_solver_full
from modesolverpy.mode_solver_semi import mode_solver_semi
from modesolverpy.sweep_waveguide import sweep_waveguide
from modesolverpy.sweep_wavelength import sweep_wavelength
from modesolverpy.waveguide import waveguide, waveguide_array, write_material_index

__all__ = [
    "CONFIG",
    "materials",
    "mode_solver_full",
    "mode_solver_semi",
    "sweep_waveguide",
    "sweep_wavelength",
    "group_index",
    "waveguide",
    "waveguide_array",
    "write_material_index",
]

if __name__ == "__main__":
    print(__all__)
