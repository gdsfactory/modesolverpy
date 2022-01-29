import abc
import os
import sys
from pathlib import PosixPath
from typing import Dict, List, Optional, Union

import matplotlib.pylab as plt
import numpy as np
import tqdm
from numpy import complex128, float64, ndarray
from six import with_metaclass

from modes.config import PATH
from modes.waveguide import waveguide

plt.rc("image", cmap="coolwarm")


class _ModeSolver(with_metaclass(abc.ABCMeta)):
    def __init__(
        self,
        n_eigs: int,
        tol: float = 0.0,
        boundary: str = "0000",
        mode_profiles: bool = True,
        initial_mode_guess: Optional[float] = None,
        n_eff_guess: Optional[float] = None,
        wg: None = None,
    ) -> None:
        self._n_eigs = int(n_eigs)
        self._tol = tol
        self._boundary = boundary
        self._mode_profiles = mode_profiles
        self._initial_mode_guess = initial_mode_guess
        self._n_eff_guess = n_eff_guess

        self.n_effs = None
        self.modes = None
        self.mode_types = None
        self.overlaps = None

        self.settings = dict(n_eigs=n_eigs, boundary=boundary)
        self.wg = wg or waveguide()
        self.name = "mode_solver"

        self._path = os.path.dirname(sys.modules[__name__].__file__) + "/"

    def get_neff(self, mode: int = 0) -> float:
        return np.real(self.n_effs[mode])

    @property
    def _modes_directory(self) -> PosixPath:
        return PATH.cache

    def solve_sweep_waveguide(
        self,
        waveguides,
        sweep_param_list,
        filename="structure_n_effs.dat",
        plot=True,
        x_label="Structure number",
        fraction_mode_list=[],
    ):
        """
        Deprecated!! use sweep_waveguide.py instead!!

        Find the modes of many waveguides.

        Args:
            waveguides (list): A list of `waveguides` to find the modes
                of.
            sweep_param_list (list): A list of the parameter-sweep sweep
                that was used.  This is for plotting purposes only.
            filename (str): The nominal filename to use when saving the
                effective indices.  Defaults to 'structure_n_effs.dat'.
            plot (bool): `True` if plots should be generates,
                otherwise `False`.  Default is `True`.
            x_label (str): x-axis text to display in the plot.
            fraction_mode_list (list): A list of mode indices of the modes
                that should be included in the TE/TM mode fraction plot.
                If the list is empty, all modes will be included.  The list
                is empty by default.

        Returns:
            list: A list of the effective indices found for each structure.
        """
        n_effs = []
        mode_types = []
        fractions_te = []
        fractions_tm = []
        for wg in tqdm.tqdm(waveguides, ncols=70):
            self.wg = wg
            self.solve()
            n_effs.append(np.real(self.n_effs))
            mode_types.append(self._get_mode_types())
            fractions_te.append(self.fraction_te)
            fractions_tm.append(self.fraction_tm)

        if filename:
            self._write_n_effs_to_file(
                n_effs, self._modes_directory + filename, sweep_param_list
            )

            with open(self._modes_directory + "mode_types.dat", "w") as fs:
                header = ",".join("Mode%i" % i for i, _ in enumerate(mode_types[0]))
                fs.write("# " + header + "\n")
                for mt in mode_types:
                    txt = ",".join("%s %.2f" % pair for pair in mt)
                    fs.write(txt + "\n")

            with open(self._modes_directory + "fraction_te.dat", "w") as fs:
                header = "fraction te"
                fs.write("# param sweep," + header + "\n")
                for param, fte in zip(sweep_param_list, fractions_te):
                    txt = "%.6f," % param
                    txt += ",".join("%.2f" % f for f in fte)
                    fs.write(txt + "\n")

            with open(self._modes_directory + "fraction_tm.dat", "w") as fs:
                header = "fraction tm"
                fs.write("# param sweep," + header + "\n")
                for param, ftm in zip(sweep_param_list, fractions_tm):
                    txt = "%.6f," % param
                    txt += ",".join("%.2f" % f for f in ftm)
                    fs.write(txt + "\n")

            if plot:
                plt.figure()
                title = "$n_{eff}$ vs %s" % x_label
                y_label = "$n_{eff}$"
                self._plot_n_effs(
                    self._modes_directory + filename,
                    self._modes_directory + "fraction_te.dat",
                    x_label,
                    y_label,
                    title,
                )

                plt.figure()
                title = "TE Fraction vs %s" % x_label
                self._plot_fraction(
                    self._modes_directory + "fraction_te.dat",
                    x_label,
                    "TE Fraction [%]",
                    title,
                    fraction_mode_list,
                )

                plt.figure()
                title = "TM Fraction vs %s" % x_label
                self._plot_fraction(
                    self._modes_directory + "fraction_tm.dat",
                    x_label,
                    "TM Fraction [%]",
                    title,
                    fraction_mode_list,
                )

        return n_effs

    def solve_sweep_wavelength(
        self,
        structure,
        wavelengths,
        filename="wavelength_n_effs.dat",
        plot=True,
    ):
        """
        Deprecated!! use sweep_wavelength.py instead!!

        Solve for the effective indices of a fixed structure at
        different wavelengths.

        Args:
            structure (Slabs): The target structure to solve
                for modes.
            wavelengths (list): A list of wavelengths to sweep
                over.
            filename (str): The nominal filename to use when saving the
                effective indices.  Defaults to 'wavelength_n_effs.dat'.
            plot (bool): `True` if plots should be generates,
                otherwise `False`.  Default is `True`.

        Returns:
            list: A list of the effective indices found for each wavelength.
        """
        n_effs = []
        for w in tqdm.tqdm(wavelengths, ncols=70):
            structure.change_wavelength(w)
            self.solve(structure)
            n_effs.append(np.real(self.n_effs))

        if filename:
            self._write_n_effs_to_file(
                n_effs, self._modes_directory + filename, wavelengths
            )
            if plot:
                title = "$n_{eff}$ vs Wavelength"
                self._plot_n_effs(
                    self._modes_directory + filename,
                    self._modes_directory + "fraction_te.dat",
                    "Wavelength",
                    "n_{eff}",
                    title,
                )
                plt.ylabel("$n_{eff}$")

        return n_effs

    def solve_ng(self, structure, wavelength_step=0.01, filename="ng.dat"):
        r"""
        Solve for the group index, :math:`n_g`, of a structure at a particular
        wavelength.

        Args:
            structure (Structure): The target structure to solve
                for modes.
            wavelength_step (float): The step to take below and
                above the nominal wavelength.  This is used for
                approximating the gradient of :math:`n_\mathrm{eff}`
                at the nominal wavelength.  Default is 0.01.
            filename (str): The nominal filename to use when saving the
                effective indices.  Defaults to 'ng.dat'.

        Returns:
            list: A list of the group indices found for each mode.
        """
        wl_nom = structure._wl

        self.solve(structure)
        n_ctrs = self.n_effs

        structure.change_wavelength(wl_nom - wavelength_step)
        self.solve(structure)
        n_bcks = self.n_effs

        structure.change_wavelength(wl_nom + wavelength_step)
        self.solve(structure)
        n_frws = self.n_effs

        n_gs = []
        for n_ctr, n_bck, n_frw in zip(n_ctrs, n_bcks, n_frws):
            n_gs.append(n_ctr - wl_nom * (n_frw - n_bck) / (2 * wavelength_step))

        if filename:
            with open(self._modes_directory + filename, "w") as fs:
                fs.write("# Mode idx, Group index\n")
                for idx, n_g in enumerate(n_gs):
                    fs.write("%i,%.3f\n" % (idx, np.round(n_g.real, 3)))

        return n_gs

    def _get_mode_filename(
        self, field_name: str, mode_number: int, filename: PosixPath
    ) -> str:
        filename_prefix, filename_ext = os.path.splitext(filename)
        filename_mode = (
            filename_prefix + "_" + field_name + "_" + str(mode_number) + filename_ext
        )
        return filename_mode

    def _write_n_effs_to_file(
        self,
        n_effs: List[ndarray],
        filename: PosixPath,
        x_vals: Optional[ndarray] = None,
    ) -> List[ndarray]:
        with open(filename, "w") as fs:
            fs.write("# Sweep param, mode 1, mode 2, ...\n")
            for i, n_eff in enumerate(n_effs):
                if x_vals is not None:
                    line_start = str(x_vals[i]) + ","
                else:
                    line_start = ""
                line = ",".join([str(np.round(n, 3)) for n in n_eff])
                fs.write(line_start + line + "\n")
        return n_effs

    def _write_mode_to_file(self, mode: ndarray, filename: str) -> ndarray:
        with open(filename, "w") as fs:
            for e in mode[::-1]:
                e_str = ",".join([str(v) for v in e])
                fs.write(e_str + "\n")
        return mode

    def _plot_n_effs(
        self,
        filename_n_effs: PosixPath,
        filename_te_fractions: PosixPath,
        xlabel: str,
        ylabel: str,
        title: str,
    ) -> Dict[str, Union[str, PosixPath, int]]:
        args = {
            "titl": title,
            "xlab": xlabel,
            "ylab": ylabel,
            "filename_data": filename_n_effs,
            "filename_frac_te": filename_te_fractions,
            "filename_image": None,
            "num_modes": len(self.modes),
        }

        filename_image_prefix, _ = os.path.splitext(filename_n_effs)
        filename_image = filename_image_prefix + ".png"
        args["filename_image"] = filename_image

        data = np.loadtxt(args["filename_data"], delimiter=",").T
        plt.clf()
        plt.title(title)
        plt.xlabel(args["xlab"])
        plt.ylabel(args["ylab"])
        for i in range(args["num_modes"]):
            plt.plot(data[0], data[i + 1], "-o")
        plt.savefig(args["filename_image"])

        return args

    def _plot_fraction(
        self,
        filename_fraction: PosixPath,
        xlabel: str,
        ylabel: str,
        title: str,
        mode_list: List[int] = [],
    ) -> Dict[str, Union[str, PosixPath]]:
        if not mode_list:
            mode_list = range(len(self.modes))
        gp_mode_list = " ".join(str(idx) for idx in mode_list)

        args = {
            "titl": title,
            "xlab": xlabel,
            "ylab": ylabel,
            "filename_data": filename_fraction,
            "filename_image": None,
            "mode_list": gp_mode_list,
        }

        filename_image_prefix, _ = os.path.splitext(filename_fraction)
        filename_image = filename_image_prefix + ".png"
        args["filename_image"] = filename_image

        data = np.loadtxt(args["filename_data"], delimiter=",").T
        plt.clf()
        plt.title(title)
        plt.xlabel(args["xlab"])
        plt.ylabel(args["ylab"])
        for i, _ in enumerate(self.modes):
            plt.plot(data[0], data[i + 1], "-o")
        plt.savefig(args["filename_image"])

        return args

    def _plot_mode(
        self,
        field_name: str,
        mode_number: int,
        filename_mode: str,
        n_eff: Optional[Union[complex128, ndarray]] = None,
        subtitle: str = "",
        e2_x: Union[float, float64] = 0.0,
        e2_y: Union[float, float64] = 0.0,
        ctr_x: Union[float, float64] = 0.0,
        ctr_y: Union[float, float64] = 0.0,
        area: Optional[Union[float, float64]] = None,
        wavelength: Optional[float] = None,
        logscale: bool = False,
    ) -> Dict[str, Union[str, int, float64, float]]:
        fn = field_name[0] + "_{" + field_name[1:] + "}"
        title = r"Mode %i $|%s|$ Profile" % (mode_number, fn)
        if n_eff:
            title += r", $n_{eff}$: " + "{:.3f}".format(n_eff.real)
        if wavelength:
            title += r", $\lambda = %s " % r"{:.3f} \mu$m".format(wavelength)
        if area:
            title += ", $A_%s$: " % field_name[1] + "{:.1f}%".format(area)

        if subtitle:
            title2 = "\n$%s$" % subtitle

        args = {
            "title": title,
            "x_pts": self.wg.xc_pts,
            "y_pts": self.wg.yc_pts,
            "x_min": self.wg.xc_min,
            "x_max": self.wg.xc_max,
            "y_min": self.wg.yc_min,
            "y_max": self.wg.yc_max,
            "x_step": self.wg.x_step,
            "y_step": self.wg.y_step,
            "filename_data": filename_mode,
            "filename_image": None,
            "e2_x": e2_x,
            "e2_y": e2_y,
            "ctr_x": ctr_x,
            "ctr_y": ctr_y,
        }

        filename_image_prefix, _ = os.path.splitext(filename_mode)
        filename_image = filename_image_prefix + ".png"
        args["filename_image"] = filename_image

        heatmap = np.loadtxt(filename_mode, delimiter=",")
        if logscale:
            heatmap = 10 * np.log10(abs(heatmap))
            heatmap -= np.max(heatmap)
        plt.clf()
        plt.suptitle(title)
        if subtitle:
            plt.rcParams.update({"axes.titlesize": "small"})
            plt.title(title2)
        plt.xlabel("x")
        plt.ylabel("y")

        vmax = 0 if logscale else None
        vmin = -20 if logscale else None

        plt.imshow(
            np.flipud(heatmap),
            extent=(
                args["x_min"],
                args["x_max"],
                args["y_min"],
                args["y_max"],
            ),
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar()
        plt.savefig(filename_image)

        return args

    if __name__ == "__main__":
        pass
