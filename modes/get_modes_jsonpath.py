from pathlib import PosixPath
from typing import Union

from modes._mode_solver_full_vectorial import ModeSolverFullyVectorial
from modes._mode_solver_semi_vectorial import ModeSolverSemiVectorial
from modes.config import CONFIG


def get_modes_jsonpath(
    mode_solver: Union[ModeSolverFullyVectorial, ModeSolverSemiVectorial]
) -> PosixPath:
    return CONFIG.cache / f"{mode_solver.name}.json"
