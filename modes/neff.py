import numpy as np
import pytest

from modes.mode_solver_full import mode_solver_full


def neff(mode: int = 0, overwrite: bool = False, **wg_kwargs) -> float:
    """Returns effective index for a mode

    Args:
        mode: 0 is fundamental
        x_step: 0.02 (um)
        y_step: 0.02 (um)
        thickness: 0.22 (um)
        width: 0.5 (um)
        slab_thickness: 0 (um)
        sub_thickness: 0.5 (um)
        sub_width: 2.0 (um)
        clad_thickness: 0.5 (um)
        n_sub: sio2
        n_wg: si
        n_clad: sio2
        wavelength: 1.55 (um)
        angle: 90.0
    """
    ms = mode_solver_full(overwrite=overwrite, n_modes=mode + 1, **wg_kwargs)
    return ms.get_neff(mode=mode)


@pytest.mark.parametrize("mode, neff_expected", [(0, 2.47), (1, 1.81)])
def test_neff(mode: int, neff_expected: float) -> None:
    assert np.isclose(neff(mode), neff_expected, atol=0.1)


if __name__ == "__main__":
    print(neff(mode=0))
    # test_neff()
