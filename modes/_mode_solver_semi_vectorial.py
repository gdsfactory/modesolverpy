from pathlib import PosixPath
from typing import Dict, List, Optional, Union

import matplotlib.pylab as plt
import numpy as np
from numpy import ndarray

from modes import _analyse as anal
from modes import _mode_solver_lib as ms
from modes._mode_solver import _ModeSolver
from modes.types import SemiVectorialMethod


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
        n_eigs: int,
        tol: float = 0.001,
        boundary: str = "0000",
        mode_profiles: bool = True,
        initial_mode_guess: Optional[float] = None,
        semi_vectorial_method: SemiVectorialMethod = "Ex",
        wg: None = None,
    ) -> None:
        self._semi_vectorial_method = semi_vectorial_method
        _ModeSolver.__init__(
            self, n_eigs, tol, boundary, mode_profiles, initial_mode_guess
        )
        self.name = "mode_solver_semi_vectorial"
        self.wg = wg
        self.results = None

    def solve(self) -> Dict[str, Union[ndarray, List[ndarray]]]:
        """Find the modes of a given structure.

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

    def write_modes_to_file(
        self,
        filename: PosixPath = "mode.dat",
        plot: bool = True,
        analyse: bool = True,
        logscale: bool = False,
    ) -> List[ndarray]:
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
            self.plot_modes(filename=filename, analyse=analyse, logscale=logscale)

        return self.modes

    def plot_modes(
        self,
        filename: PosixPath = "mode.dat",
        analyse: bool = True,
        logscale: bool = False,
    ) -> None:
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
                    logscale=logscale,
                )
            else:
                plt.figure()
                self._plot_mode(
                    self._semi_vectorial_method,
                    i,
                    filename_mode,
                    self.n_effs[i],
                    wavelength=self.wg._wl,
                    logscale=logscale,
                )
