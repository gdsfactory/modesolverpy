import json

import numpy as np
import pytest

from modes._mode_solver import get_modes_jsonpath
from modes._mode_solver_semi_vectorial import ModeSolverSemiVectorial
from modes.autoname import autoname
from modes.autoname import clean_value
from modes.waveguide import waveguide
from modes.waveguide import write_material_index


@pytest.mark.parametrize("overwrite", [True, False])
def test_mode_solver_semi_vectorial_te(overwrite):
    mode_solver = mode_solver_semi(overwrite=overwrite)
    # modes = mode_solver.solve()
    # neff0 = modes["n_effs"][0].real

    neff0 = mode_solver.results["n_effs"][0].real
    print(neff0)
    assert np.isclose(neff0, 2.507954410087166)


@pytest.mark.parametrize("overwrite", [True, False])
def test_mode_solver_semi_vectorial_tm(overwrite):
    mode_solver = mode_solver_semi(semi_vectorial_method="Ey", overwrite=overwrite)
    # modes = mode_solver.solve()
    # neff0 = modes["n_effs"][0].real

    neff0 = mode_solver.results["n_effs"][0].real
    print(neff0)
    assert np.isclose(neff0, 1.859555511265503)


@autoname
def _semi(
    n_modes=2, semi_vectorial_method="Ex", plot=False, plot_profile=False, **wg_kwargs
):
    """
    returns mode solver with mode_solver.wg
    writes waveguide material index
    use mode_solver_semi instead

    Args:
        n_modes: 2
        semi_vectorial_method: 'Ey' for TM, 'Ex' for TE
        wg_kwargs: for waveguide
    """

    wg = waveguide(**wg_kwargs)
    if plot_profile:
        write_material_index(wg)

    mode_solver = ModeSolverSemiVectorial(
        n_modes, semi_vectorial_method=semi_vectorial_method
    )
    mode_solver.wg = wg
    # modes = mode_solver.solve(wg)
    # mode_solver.write_modes_to_file("example_modes_1.dat")
    return mode_solver


def mode_solver_semi(
    n_modes=2,
    semi_vectorial_method="Ex",
    overwrite=False,
    plot=True,
    plot_profile=False,
    logscale=False,
    **wg_kwargs
):
    """
    returns semi vectorial mode solver with the computed modes

    Args:
        n_modes: 2
        overwrite: whether to run again even if it finds the modes in CONFIG.cache
        semi_vectorial_method: 'Ey' for TM, 'Ex' for TE
        x_step: 0.02
        y_step: 0.02
        wg_height: 0.22
        wg_width: 0.5
        slab_height: 0
        sub_height: 0.5
        sub_width: 2.0
        clad_height: 0.5
        n_sub: sio2
        n_wg: si
        n_clad: sio2
        wavelength: 1.55
        angle: 90.0

    .. plot::
      :include-source:

      import modes as ms

      m = ms.mode_solver_semi()
      print(m.results.keys())

    """
    mode_solver = _semi(
        n_modes=n_modes,
        semi_vectorial_method=semi_vectorial_method,
        plot=plot,
        plot_profile=plot_profile,
        **wg_kwargs
    )
    settings = {k: clean_value(v) for k, v in mode_solver.settings.items()}
    jsonpath = get_modes_jsonpath(mode_solver)
    filepath = jsonpath.with_suffix(".dat")

    if overwrite or not jsonpath.exists():
        r = mode_solver.solve()
        n_effs_real = r["n_effs"].real.tolist()
        n_effs_imag = r["n_effs"].imag.tolist()
        modes = r["modes"]
        modes_real = [mode.real.tolist() for mode in modes]
        modes_imag = [mode.imag.tolist() for mode in modes]

        d = dict(
            n_effs_real=n_effs_real,
            n_effs_imag=n_effs_imag,
            modes_real=modes_real,
            modes_imag=modes_imag,
            n_modes=len(n_effs_real),
            settings=settings,
        )

        with open(jsonpath, "w") as f:
            f.write(json.dumps(d))
        mode_solver.write_modes_to_file(filepath, plot=plot, logscale=logscale)

        r["settings"] = settings

    else:
        d = json.loads(open(jsonpath).read())
        modes_real = d["modes_real"]
        modes_imag = d["modes_imag"]
        modes = [
            np.array(np.array(mr) + 1j * np.array(mi))
            for mr, mi in zip(modes_real, modes_imag)
        ]
        n_effs_real = d["n_effs_real"]
        n_effs_imag = d["n_effs_imag"]
        n_effs = [
            np.array(np.array(mr) + 1j * np.array(mi))
            for mr, mi in zip(n_effs_real, n_effs_imag)
        ]
        r = dict(modes=modes, n_effs=n_effs)
        mode_solver.modes = r["modes"]
        mode_solver.n_effs = r["n_effs"]
        if plot:
            mode_solver.plot_modes(filepath, logscale=logscale)

    mode_solver.results = r
    return mode_solver


# def load_mode(mode_solver):
#     filepath = get_modes_filepath(mode_solver)
#     jsonpath = filepath.with_suffix(".json")
#     data = np.loadtxt(filepath, delimiter=",").T
#     return data


if __name__ == "__main__":
    import matplotlib.pylab as plt

    # test_mode_solver_semi_vectorial_te(overwrite=True)
    # test_mode_solver_semi_vectorial_te(overwrite=False)
    # test_mode_solver_semi_vectorial_tm(overwrite=True)
    # test_mode_solver_semi_vectorial_tm(overwrite=False)

    mode_solver_semi(plot=True, plot_profile=False, logscale=True)
    plt.show()
