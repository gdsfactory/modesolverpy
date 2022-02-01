from modes import _analyse as fit
from modes import design, materials
from modes.config import PATH
from modes.coupling_efficiency import coupling_efficiency
from modes.group_index import group_index
from modes.mode_solver_full import mode_solver_full
from modes.mode_solver_semi import mode_solver_semi
from modes.neff import neff
from modes.sweep_waveguide import sweep_waveguide
from modes.sweep_wavelength import sweep_wavelength
from modes.waveguide import waveguide, waveguide_array, write_material_index

__all__ = [
    "fit",
    "PATH",
    "coupling_efficiency",
    "design",
    "materials",
    "mode_solver_full",
    "mode_solver_semi",
    "neff",
    "sweep_waveguide",
    "sweep_wavelength",
    "group_index",
    "waveguide",
    "waveguide_array",
    "write_material_index",
]


__version__ = "1.0.6"

if __name__ == "__main__":
    print(__all__)
