from modes import _analyse as fit
from modes import materials
from modes.config import CONFIG
from modes.group_index import group_index
from modes.mode_solver_full import mode_solver_full
from modes.mode_solver_semi import mode_solver_semi
from modes.sweep_waveguide import sweep_waveguide
from modes.sweep_wavelength import sweep_wavelength
from modes.waveguide import waveguide
from modes.waveguide import waveguide_array
from modes.waveguide import write_material_index

__all__ = [
    "fit",
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


__version__ = "1.0.2"

if __name__ == "__main__":
    print(__all__)
