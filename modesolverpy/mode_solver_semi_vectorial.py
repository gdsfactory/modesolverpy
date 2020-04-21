import json

import matplotlib.pylab as plt
import numpy as np
import pytest
from modesolverpy import _analyse as anal
from modesolverpy import _mode_solver_lib as ms
from modesolverpy.autoname import autoname, clean_value
from modesolverpy.mode_solver import _ModeSolver, get_modes_jsonpath
from modesolverpy.waveguide import waveguide, write_material_index


class ModeSolverSemiVectorial(_ModeSolver):
    """
    A semi-vectorial mode solver object used to
    setup and run a mode solving simulation.

    Args:
        n_eigs (int): The number of eigen-values to solve for.
        tol (float): The precision of the eigen-value/eigen-vector
            solver.  Default is 0.001.
        boundary (str): The boundary conditions to use.
            This is a string that identifies the type of boundary conditions applied.
            The following options are available: 'A' - Hx is antisymmetric, Hy is symmetric,
            'S' - Hx is symmetric and, Hy is antisymmetric, and '0' - Hx and Hy are zero
            immediately outside of the boundary.
            The string identifies all four boundary conditions, in the order:
            North, south, east, west. For example, boundary='000A'. Default is '0000'.
        mode_profiles (bool): `True if the the mode-profiles should be found, `False`
            if only the effective indices should be found.
        initial_mode_guess (list): An initial mode guess for the modesolver.
        semi_vectorial_method (str): Either 'Ex' or 'Ey'.  If 'Ex', the mode solver
            will only find TE modes (horizontally polarised to the simulation window),
            if 'Ey', the mode solver will find TM modes (vertically polarised to the
            simulation window).
    """

    def __init__(
        self,
        n_eigs,
        tol=0.001,
        boundary="0000",
        mode_profiles=True,
        initial_mode_guess=None,
        semi_vectorial_method="Ex",
        name="mode_solver_semi_vectorial",
        wg=None,
    ):
        self._semi_vectorial_method = semi_vectorial_method
        _ModeSolver.__init__(
            self, n_eigs, tol, boundary, mode_profiles, initial_mode_guess
        )
        self.name = name
        self.wg = wg or waveguide()
        self.results = None

    def solve(self):
        """ Find the modes of a given structure.

        Returns:
            dict: The 'n_effs' key gives the effective indices
            of the modes.  The 'modes' key exists of mode
            profiles were solved for; in this case, it will
            return arrays of the mode profiles.
        """
        structure = self._structure = self.wg
        wavelength = self.wg._wl
        self._ms = ms._ModeSolverSemiVectorial(
            wavelength, structure, self._boundary, self._semi_vectorial_method
        )
        self._ms.solve(
            self._n_eigs,
            self._tol,
            self._mode_profiles,
            initial_mode_guess=self._initial_mode_guess,
        )
        self.n_effs = self._ms.neff

        r = {"n_effs": self.n_effs}

        if self._mode_profiles:
            r["modes"] = self._ms.modes
            self._ms.modes[0] = np.real(self._ms.modes[0])
            self._initial_mode_guess = np.real(self._ms.modes[0])

        self.modes = self._ms.modes

        return r

    def write_modes_to_file(self, filename="mode.dat", plot=True, analyse=True):
        """
        Writes the mode fields to a file and optionally plots them.

        Args:
            filename (str): The nominal filename to use for the saved
                data.  The suffix will be automatically be changed to
                identifiy each mode number.  Default is 'mode.dat'
            plot (bool): `True` if plots should be generates,
                otherwise `False`.  Default is `True`.
            analyse (bool): `True` if an analysis on the fundamental
                mode should be performed.  The analysis adds to the
                plot of the fundamental mode the power mode-field
                diameter (MFD) and marks it on the output, and it
                marks with a cross the maximum E-field value.
                Default is `True`.

        Returns:
            dict: A dictionary containing the effective indices
            and mode field profiles (if solved for).
        """

        for i, mode in enumerate(self._ms.modes):
            filename_mode = self._get_mode_filename(
                self._semi_vectorial_method, i, filename
            )
            self._write_mode_to_file(np.real(mode), filename_mode)
        if plot:
            self.plot_modes(filename=filename, analyse=analyse)

        return self.modes

    def plot_modes(self, filename="mode.dat", analyse=True):
        for i, mode in enumerate(self.modes):
            filename_mode = self._get_mode_filename(
                self._semi_vectorial_method, i, filename
            )

            if i == 0 and analyse:
                A, centre, sigma_2 = anal.fit_gaussian(
                    self.wg.xc, self.wg.yc, np.abs(mode)
                )
                subtitle = (
                    "E_{max} = %.3f, (x_{max}, y_{max}) = (%.3f, %.3f), MFD_{x} = %.3f, "
                    "MFD_{y} = %.3f"
                ) % (A, centre[0], centre[1], sigma_2[0], sigma_2[1])
                plt.figure()
                self._plot_mode(
                    self._semi_vectorial_method,
                    i,
                    filename_mode,
                    self.n_effs[i],
                    subtitle,
                    sigma_2[0],
                    sigma_2[1],
                    centre[0],
                    centre[1],
                    wavelength=self.wg._wl,
                )
            else:
                plt.figure()
                self._plot_mode(
                    self._semi_vectorial_method,
                    i,
                    filename_mode,
                    self.n_effs[i],
                    wavelength=self.wg._wl,
                )


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
def _semi(n_modes=2, semi_vectorial_method="Ex", **wg_kwargs):
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
    write_material_index(wg)

    mode_solver = ModeSolverSemiVectorial(
        n_modes, semi_vectorial_method=semi_vectorial_method
    )
    mode_solver.wg = wg
    # modes = mode_solver.solve(wg)
    # mode_solver.write_modes_to_file("example_modes_1.dat")
    return mode_solver


def mode_solver_semi(
    n_modes=2, semi_vectorial_method="Ex", overwrite=False, **wg_kwargs
):
    """
    returns semi vectorial mode solver with the computed modes

    Args:
        n_modes: 2
        overwrite: whether to run again even if it finds the modes in CONFIG['cache']
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

      import modesolverpy as ms

      m = ms.mode_solver_semi()
      print(m.results.keys())

    """
    mode_solver = _semi(
        n_modes=n_modes, semi_vectorial_method=semi_vectorial_method, **wg_kwargs
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
        mode_solver.write_modes_to_file(filepath)

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
        mode_solver.plot_modes(filepath)

    mode_solver.results = r
    return mode_solver


# def load_mode(mode_solver):
#     filepath = get_modes_filepath(mode_solver)
#     jsonpath = filepath.with_suffix(".json")
#     data = np.loadtxt(filepath, delimiter=",").T
#     return data


if __name__ == "__main__":
    # test_mode_solver_semi_vectorial_te(overwrite=True)
    # test_mode_solver_semi_vectorial_te(overwrite=False)
    test_mode_solver_semi_vectorial_tm(overwrite=True)
    # test_mode_solver_semi_vectorial_tm(overwrite=False)
    # mode_solver_semi()
    # plt.show()
