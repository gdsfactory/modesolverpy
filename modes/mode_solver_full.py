import pathlib
from typing import Optional, Tuple

import h5py
import numpy as np
import pytest

from modes._mode_solver_full_vectorial import ModeSolverFullyVectorial
from modes._structure import RidgeWaveguide
from modes.autoname import autoname
from modes.config import PATH, get_kwargs_hash
from modes.materials import nitride, sio2
from modes.types import Field, PathType
from modes.waveguide import waveguide


@pytest.mark.parametrize("overwrite", [True, False])
def test_mode_solver_full_vectorial(overwrite: bool) -> None:
    mode_solver = mode_solver_full(overwrite=overwrite, logscale=True, plot=True)
    # modes = mode_solver.solve()
    # neff0 = modes["n_effs"][0].real

    neff0 = mode_solver.results["n_effs"][0].real
    assert np.isclose(neff0, 2.4717079424099673), neff0


@pytest.mark.parametrize("overwrite", [True, False])
def test_mode_solver_full_vectorial_multi_clad(overwrite: bool) -> None:
    mode_solver = mode_solver_full(
        overwrite=overwrite,
        logscale=True,
        plot=False,
        clad_thickness=[50e-3, 50e-3, 0.5],
        n_clads=[sio2, nitride, sio2],
    )
    # modes = mode_solver.solve()
    # neff0 = modes["n_effs"][0].real

    neff0 = mode_solver.results["n_effs"][0].real
    assert np.isclose(neff0, 2.483481412238637), neff0


@autoname
def _full(
    n_modes: int = 2,
    plot_index: bool = True,
    **wg_kwargs,
) -> ModeSolverFullyVectorial:
    """Returns mode solver and
    can also plot the index profile.

    Args:
        n_modes: 2
        plot_index: plot index profile
        wg_kwargs: for waveguide
    """

    wg = waveguide(**wg_kwargs)

    if plot_index:
        wg.plot()

    mode_solver = ModeSolverFullyVectorial(n_modes)
    mode_solver.wg = wg
    return mode_solver


def mode_solver_full(
    n_modes: int = 2,
    overwrite: bool = False,
    plot: bool = False,
    plot_index: bool = False,
    logscale: bool = False,
    wg: Optional[RidgeWaveguide] = None,
    fields_to_write: Tuple[Field, ...] = (
        "Ex",
        "Ey",
        "Ez",
        "Hx",
        "Hy",
        "Hz",
    ),
    cache: Optional[PathType] = PATH.cache,
    **wg_kwargs,
) -> ModeSolverFullyVectorial:
    """Return full vectorial mode solver with the computed modes

    Args:
        n_modes: 2
        overwrite: whether to run again even if it finds the modes in PATH.cache
        plot: plot modes
        plot_index: plots index profile
        logscale: plots mode in logscale
        wg: waveguide
        fields_to_write: List of fields_to_write "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"

    Keyword Args:
        x_step: 0.02 grid step (um)
        y_step: 0.02 grid step (um)
        thickness: 0.22 (um)
        width: 0.5 (um)
        slab_thickness: 0 (um)
        sub_width: 2.0 related to the total simulation width
        sub_thickness: 0.5 bottom simulation margin
        clad_thickness: [0.5]  List of claddings (top simulation margin)
        n_sub: sio2 substrate index material
        n_wg: si waveguide index material
        n_clads: list of cladding materials [sio2]
        wavelength: 1.55 wavelength (um)
        angle: 90 sidewall angle (degrees)

    .. plot::
        :include-source:

        import modes as ms

        s = ms.mode_solver_full(plot=True, plot_index=True, n_modes=1, width=0.5, thickness=0.22)
        print(s.results.keys())

    """
    mode_solver = _full(n_modes=n_modes, wg=wg, plot_index=plot_index, **wg_kwargs)
    # settings = {k: clean_value(v) for k, v in mode_solver.settings.items()}
    # jsonpath = get_modes_jsonpath(mode_solver)
    # filepath = jsonpath.with_suffix(".dat")

    h = get_kwargs_hash(
        fields_to_write=fields_to_write,
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
            f["modes"] = [mode["fields"] for mode in r["modes"]]
            f["n_effs"] = r["n_effs"]
            f["mode_types"] = mode_solver._get_mode_types()
            f["fraction_te"] = mode_solver.fraction_te
            f["fraction_tm"] = mode_solver.fraction_tm
            f.close()

        else:
            print(f"Loading modes from {str(filepath)!r}")
            f = h5py.File(filepath, "r")
            mode_solver.modes = f["modes"]
            mode_solver.n_effs = f["n_effs"]
            mode_solver.mode_types = f["mode_types"]
            mode_solver.fraction_te = f["fraction_te"]
            mode_solver.fraction_tm = f["fraction_tm"]

    if plot:
        mode_solver.plot_modes(
            filepath, fields_to_write=fields_to_write, logscale=logscale
        )

    mode_solver.results = r
    return mode_solver


if __name__ == "__main__":
    import matplotlib.pylab as plt

    test_mode_solver_full_vectorial_multi_clad(overwrite=True)
    # test_mode_solver_full_vectorial(overwrite=True)
    # test_mode_solver_full_vectorial(overwrite=False)
    plt.show()
