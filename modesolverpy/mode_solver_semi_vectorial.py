import os

import matplotlib.pylab as plt
import modesolverpy.structure as st
import numpy as np
from modesolverpy import _analyse as anal
from modesolverpy import _mode_solver_lib as ms
from modesolverpy.mode_solver import _ModeSolver


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
    ):
        self._semi_vectorial_method = semi_vectorial_method
        _ModeSolver.__init__(
            self, n_eigs, tol, boundary, mode_profiles, initial_mode_guess
        )

    @property
    def _modes_directory(self):
        modes_directory = "./modes_semi_vec/"
        if not os.path.exists(modes_directory):
            os.mkdir(modes_directory)
        _modes_directory = modes_directory
        return _modes_directory

    def _solve(self, structure, wavelength):
        self._structure = structure
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
        modes_directory = "./modes_semi_vec/"
        if not os.path.isdir(modes_directory):
            os.mkdir(modes_directory)
        filename = modes_directory + filename

        for i, mode in enumerate(self._ms.modes):
            filename_mode = self._get_mode_filename(
                self._semi_vectorial_method, i, filename
            )
            self._write_mode_to_file(np.real(mode), filename_mode)

            if plot:
                if i == 0 and analyse:
                    A, centre, sigma_2 = anal.fit_gaussian(
                        self._structure.xc, self._structure.yc, np.abs(mode)
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
                        wavelength=self._structure._wl,
                    )
                else:
                    plt.figure()
                    self._plot_mode(
                        self._semi_vectorial_method,
                        i,
                        filename_mode,
                        self.n_effs[i],
                        wavelength=self._structure._wl,
                    )

        return self.modes


def test_mode_solver_semi_vectorial():
    modes = mode_solver_semi()
    neff0 = modes["n_effs"][0].real
    assert np.isclose(neff0, 2.47422827)


def mode_solver_semi(
    x_step=0.02,
    y_step=0.02,
    wg_height=0.4,
    wg_width=0.5,
    sub_height=0.5,
    sub_width=2.0,
    clad_height=0.5,
    n_sub=1.4,
    n_wg=3.0,
    n_clad=1.0,
    film_thickness=0.5,
    wavelength=1.55,
    angle=75.0,
):

    structure = st.RidgeWaveguide(
        wavelength,
        x_step,
        y_step,
        wg_height,
        wg_width,
        sub_height,
        sub_width,
        clad_height,
        n_sub,
        n_wg,
        angle,
        n_clad,
        film_thickness,
    )

    structure.write_to_file("example_structure_1.dat")

    mode_solver = ModeSolverSemiVectorial(2, semi_vectorial_method="Ey")
    modes = mode_solver.solve(structure)
    mode_solver.write_modes_to_file("example_modes_1.dat")
    return modes


if __name__ == "__main__":
    # test_mode_solver_semi_vectorial()
    mode_solver_semi()
    plt.show()
