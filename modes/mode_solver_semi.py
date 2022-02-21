import pathlib
from typing import Optional

import h5py
import numpy as np
import pytest

from modes._mode_solver_semi_vectorial import ModeSolverSemiVectorial
from modes.autoname import autoname
from modes.config import PATH, get_kwargs_hash
from modes.types import PathType
from modes.waveguide import waveguide


@pytest.mark.parametrize("overwrite", [True, False])
def test_mode_solver_semi_vectorial_te(overwrite: bool) -> None:
    mode_solver = mode_solver_semi(overwrite=overwrite)
    neff0 = mode_solver.results["n_effs"][0].real
    assert np.isclose(neff0, 2.507954410087166), neff0


@pytest.mark.parametrize("overwrite", [True, False])
def test_mode_solver_semi_vectorial_tm(overwrite: bool) -> None:
    mode_solver = mode_solver_semi(semi_vectorial_method="Ey", overwrite=overwrite)
    neff0 = mode_solver.results["n_effs"][0].real
    assert np.isclose(neff0, 1.859555511265503), neff0


@autoname
def _semi(
    n_modes: int = 2,
    semi_vectorial_method: str = "Ex",
    plot_index: bool = False,
    **wg_kwargs,
) -> ModeSolverSemiVectorial:
    """
    returns mode solver with mode_solver.wg
    writes waveguide material index
    use mode_solver_semi instead

    Args:
        n_modes: 2
        semi_vectorial_method: 'Ey' for TM, 'Ex' for TE
        plot_index: plot index profile
        wg_kwargs: for waveguide
    """

    wg = waveguide(**wg_kwargs)

    if plot_index:
        wg.plot()

    mode_solver = ModeSolverSemiVectorial(
        n_modes, semi_vectorial_method=semi_vectorial_method
    )
    mode_solver.wg = wg
    # modes = mode_solver.solve(wg)
    # mode_solver.write_modes_to_file("example_modes_1.dat")
    return mode_solver


def mode_solver_semi(
    n_modes: int = 2,
    semi_vectorial_method: str = "Ex",
    overwrite: bool = False,
    plot: bool = False,
    plot_index: bool = False,
    logscale: bool = False,
    cache: Optional[PathType] = PATH.cache,
    **wg_kwargs,
) -> ModeSolverSemiVectorial:
    """
    returns semi vectorial mode solver with the computed modes

    Args:
        n_modes: 2
        overwrite: whether to run again even if it finds the modes in PATH.cache
        semi_vectorial_method: 'Ey' for TM, 'Ex' for TE
        overwrite: runs even
        plot: plot modes
        plot_index: plot index profile
        logscale: plots mode in logscale

    Keyword Args:
        x_step: 0.02
        y_step: 0.02
        thickness: 0.22
        width: 0.5
        slab_thickness: 0
        sub_thickness: 0.5
        sub_width: 2.0
        clad_thickness: 0.5
        n_sub: sio2
        n_wg: si
        n_clads: [sio2]
        wavelength: 1.55
        angle: 90.0

    .. plot::
      :include-source:

      import modes as ms

      m = ms.mode_solver_semi(plot_index=True, plot=True)
      print(m.results.keys())

    """
    mode_solver = _semi(
        n_modes=n_modes,
        semi_vectorial_method=semi_vectorial_method,
        plot_index=plot_index,
        **wg_kwargs,
    )

    # settings = {k: clean_value(v) for k, v in mode_solver.settings.items()}

    h = get_kwargs_hash(
        n_modes=n_modes,
        **wg_kwargs,
    )
    filepath = cache / f"{h}.hdf5"

    if cache:
        cache = pathlib.Path(cache)
        cache.mkdir(exist_ok=True, parents=True)

        if overwrite or not filepath.exists():
            print(f"Writing modes to {str(filepath)!r}")
            r = mode_solver.solve()
            f = h5py.File(filepath, "w")
            f["modes"] = r["modes"]
            f["n_effs"] = r["n_effs"]
            f.close()

        else:
            print(f"Loading modes from {str(filepath)!r}")
            f = h5py.File(filepath, "r")
            mode_solver.modes = f["modes"]
            mode_solver.n_effs = f["n_effs"]

    else:
        r = mode_solver.solve()

    if plot:
        mode_solver.write_modes_to_file(logscale=logscale)
        mode_solver.plot_modes(logscale=logscale)

    # mode_solver.results = r
    return mode_solver


if __name__ == "__main__":
    import matplotlib.pylab as plt

    # test_mode_solver_semi_vectorial_te(overwrite=True)
    # test_mode_solver_semi_vectorial_te(overwrite=False)
    # test_mode_solver_semi_vectorial_tm(overwrite=True)
    # test_mode_solver_semi_vectorial_tm(overwrite=False)
    # m = mode_solver_semi(plot=True, logscale=True)

    m = mode_solver_semi(plot=True)
    plt.show()
