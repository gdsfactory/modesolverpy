import json

import numpy as np
import pytest

from modes._mode_solver import get_modes_jsonpath
from modes._mode_solver_full_vectorial import ModeSolverFullyVectorial
from modes.autoname import autoname
from modes.autoname import clean_value
from modes.materials import nitride
from modes.materials import si
from modes.materials import sio2
from modes.waveguide import waveguide
from modes.waveguide import write_material_index


@pytest.mark.parametrize("overwrite", [True, False])
def test_mode_solver_full_vectorial(overwrite):
    mode_solver = mode_solver_full(overwrite=overwrite, logscale=True, plot=True)
    # modes = mode_solver.solve()
    # neff0 = modes["n_effs"][0].real

    neff0 = mode_solver.results["n_effs"][0].real
    print(neff0)
    assert np.isclose(neff0, 2.4717079424099673)


@pytest.mark.parametrize("overwrite", [True, False])
def test_mode_solver_full_vectorial_multi_clad(overwrite):
    mode_solver = mode_solver_full(
        overwrite=overwrite,
        logscale=True,
        plot=False,
        clad_height=[50e-3, 50e-3, 0.5],
        n_clads=[sio2, nitride, sio2],
    )
    # modes = mode_solver.solve()
    # neff0 = modes["n_effs"][0].real

    neff0 = mode_solver.results["n_effs"][0].real
    print(neff0)
    assert np.isclose(neff0, 2.483481412238637)


@autoname
def _full(n_modes=2, wg=None, plot=True, plot_profile=False, **wg_kwargs):
    """
    returns mode solver with mode_solver.wg
    writes waveguide material index
    use mode_solver_semi instead

    Args:
        n_modes: 2
        wg: waveguide object
        wg_kwargs: for waveguide
    """

    wg = wg or waveguide(**wg_kwargs)
    if plot_profile:
        write_material_index(wg)

    mode_solver = ModeSolverFullyVectorial(n_modes)
    mode_solver.wg = wg
    return mode_solver


def mode_solver_full(
    n_modes=2,
    overwrite=False,
    plot=False,
    plot_profile=False,
    logscale=False,
    wg=None,
    fields_to_write=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    **wg_kwargs
):
    """
    returns full vectorial mode solver with the computed modes

    Args:
        n_modes: 2
        overwrite: whether to run again even if it finds the modes in CONFIG.cache
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

      s = ms.mode_solver_full(plot=True, n_modes=1, wg_width=0.5, wg_height=0.22)
      print(s.results.keys())

    """
    mode_solver = _full(
        n_modes=n_modes, wg=wg, plot=plot, plot_profile=plot_profile, **wg_kwargs
    )
    settings = {k: clean_value(v) for k, v in mode_solver.settings.items()}
    jsonpath = get_modes_jsonpath(mode_solver)
    filepath = jsonpath.with_suffix(".dat")

    if overwrite or not jsonpath.exists():
        r = mode_solver.solve()
        n_effs_real = r["n_effs"].real.tolist()
        n_effs_imag = r["n_effs"].imag.tolist()
        modes = r["modes"]

        modes_real = [
            {k: v.real.tolist() for k, v in mode.fields.items()} for mode in modes
        ]
        modes_imag = [
            {k: v.imag.tolist() for k, v in mode.fields.items()} for mode in modes
        ]

        d = dict(
            n_effs_real=n_effs_real,
            n_effs_imag=n_effs_imag,
            modes_real=modes_real,
            modes_imag=modes_imag,
            n_modes=len(n_effs_real),
            settings=settings,
            mode_types=mode_solver._get_mode_types(),
            fraction_te=mode_solver.fraction_te,
            fraction_tm=mode_solver.fraction_tm,
        )

        with open(jsonpath, "w") as f:
            f.write(json.dumps(d))

        mode_solver.write_modes_to_file(
            filepath, plot=plot, fields_to_write=fields_to_write, logscale=logscale
        )

        r["settings"] = settings

    else:
        d = json.loads(open(jsonpath).read())
        modes_real = d["modes_real"]
        modes_imag = d["modes_imag"]
        modes_cache = [
            {k: np.array(np.array(mr[k]) + 1j * np.array(mi[k])) for k in mr.keys()}
            for mr, mi in zip(modes_real, modes_imag)
        ]
        n_effs_real = d["n_effs_real"]
        n_effs_imag = d["n_effs_imag"]
        n_effs_cache = [
            np.array(np.array(mr) + 1j * np.array(mi))
            for mr, mi in zip(n_effs_real, n_effs_imag)
        ]

        mode_solver.mode_types = mode_types = d["mode_types"]
        mode_solver.fraction_tm = fraction_tm = d["fraction_tm"]
        mode_solver.fraction_te = fraction_te = d["fraction_te"]

        r = dict(
            modes=modes_cache,
            n_effs=n_effs_cache,
            mode_types=mode_types,
            fraction_te=fraction_te,
            fraction_tm=fraction_tm,
        )
        mode_solver.modes = r["modes"]
        mode_solver.n_effs = r["n_effs"]
        if plot:
            mode_solver.plot_modes(filepath, fields_to_write=fields_to_write)

    mode_solver.results = r
    return mode_solver


if __name__ == "__main__":
    import matplotlib.pylab as plt

    test_mode_solver_full_vectorial_multi_clad(overwrite=True)
    # test_mode_solver_full_vectorial(overwrite=True)
    # test_mode_solver_full_vectorial(overwrite=False)
    plt.show()
