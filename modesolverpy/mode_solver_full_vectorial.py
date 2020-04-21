import json
import os

import matplotlib.pylab as plt
import numpy as np
import pytest
from modesolverpy import _mode_solver_lib as ms
from modesolverpy import get_modes_jsonpath
from modesolverpy.autoname import autoname, clean_value
from modesolverpy.config import CONFIG
from modesolverpy.mode_solver import _ModeSolver
from modesolverpy.waveguide import waveguide, write_material_index


class ModeSolverFullyVectorial(_ModeSolver):
    """
    A fully-vectorial mode solver object used to
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
        initial_mode_guess (list): An initial mode guess for the modesolver.
        initial_n_eff_guess (list): An initial effective index guess for the modesolver.
    """

    def __init__(
        self,
        n_eigs,
        tol=0.001,
        boundary="0000",
        initial_mode_guess=None,
        n_eff_guess=None,
        name=None,
        wg=None,
    ):
        self.n_effs_te = None
        self.n_effs_tm = None
        self.name = name
        self.wg = wg or waveguide()
        _ModeSolver.__init__(
            self, n_eigs, tol, boundary, False, initial_mode_guess, n_eff_guess
        )

    # @property
    # def _modes_directory(self):
    #     modes_directory = "./modes_full_vec/"
    #     if not os.path.exists(modes_directory):
    #         os.mkdir(modes_directory)
    #     _modes_directory = modes_directory
    #     return _modes_directory
    @property
    def _modes_directory(self):
        return CONFIG["cache"]

    def solve(self):
        structure = self._structure = self.wg
        wavelength = self.wg._wl
        self._ms = ms._ModeSolverVectorial(wavelength, structure, self._boundary)
        self._ms.solve(
            self._n_eigs,
            self._tol,
            self._n_eff_guess,
            initial_mode_guess=self._initial_mode_guess,
        )
        self.n_effs = self._ms.neff

        r = {"n_effs": self.n_effs}
        r["modes"] = self.modes = self._ms.modes

        self.overlaps, self.fraction_te, self.fraction_tm = self._get_overlaps(
            self.modes
        )
        self.mode_types = self._get_mode_types()

        self._initial_mode_guess = None

        self.n_effs_te, self.n_effs_tm = self._sort_neffs(self._ms.neff)

        return r

    def _get_mode_types(self):
        mode_types = []
        labels = {0: "qTE", 1: "qTM", 2: "qTE/qTM"}
        for overlap in self.overlaps:
            idx = np.argmax(overlap[0:3])
            mode_types.append((labels[idx], np.round(overlap[idx], 2)))
        return mode_types

    def _sort_neffs(self, n_effs):
        mode_types = self._get_mode_types()

        n_effs_te = []
        n_effs_tm = []

        for mt, n_eff in zip(mode_types, n_effs):
            if mt[0] == "qTE":
                n_effs_te.append(n_eff)
            elif mt[0] == "qTM":
                n_effs_tm.append(n_eff)

        return n_effs_te, n_effs_tm

    def _get_overlaps(self, fields):
        mode_areas = []
        fraction_te = []
        fraction_tm = []
        for mode in self._ms.modes:
            e_fields = (mode.fields["Ex"], mode.fields["Ey"], mode.fields["Ez"])
            h_fields = (mode.fields["Hx"], mode.fields["Hy"], mode.fields["Hz"])

            areas_e = [np.sum(np.abs(e) ** 2) for e in e_fields]
            areas_e /= np.sum(areas_e)
            areas_e *= 100

            areas_h = [np.sum(np.abs(h) ** 2) for h in h_fields]
            areas_h /= np.sum(areas_h)
            areas_h *= 100

            fraction_te.append(areas_e[0] / (areas_e[0] + areas_e[1]))
            fraction_tm.append(areas_e[1] / (areas_e[0] + areas_e[1]))

            areas = areas_e.tolist()
            areas.extend(areas_h)
            mode_areas.append(areas)

        return mode_areas, fraction_te, fraction_tm

    def write_modes_to_file(
        self,
        filename="mode.dat",
        plot=True,
        fields_to_write=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    ):
        """
        Writes the mode fields to a file and optionally plots them.

        Args:
            filename (str): The nominal filename to use for the saved
                data.  The suffix will be automatically be changed to
                identifiy each field and mode number.  Default is
                'mode.dat'
            plot (bool): `True` if plots should be generates,
                otherwise `False`.  Default is `True`.
            fields_to_write (tuple): A tuple of strings where the
                strings can be 'Ex', 'Ey', 'Ez', 'Hx', 'Hy' and 'Hz'
                defining what part of the mode should be saved and
                plotted.  By default, all six components are written
                and plotted.

        Returns:
            dict: A dictionary containing the effective indices
            and mode field profiles (if solved for).
        """

        # Mode info file.
        filename_info = filename.parent / (filename.stem + f"_info.dat")
        with open(filename_info, "w") as fs:
            fs.write("# Mode idx, Mode type, % in major direction, n_eff\n")
            for i, (n_eff, (mode_type, percentage)) in enumerate(
                zip(self.n_effs, self.mode_types)
            ):
                mode_idx = str(i)
                line = "%s,%s,%.2f,%.3f" % (
                    mode_idx,
                    mode_type,
                    percentage,
                    n_eff.real,
                )
                fs.write(line + "\n")

        # Mode field plots.
        for i, (mode, areas) in enumerate(zip(self._ms.modes, self.overlaps)):
            filename_full = filename.parent / (filename.stem + f"_{i}.dat")

            for (field_name, field_profile), area in zip(mode.fields.items(), areas):
                if field_name in fields_to_write:
                    filename_mode = self._get_mode_filename(
                        field_name, i, filename_full
                    )
                    self._write_mode_to_file(np.real(field_profile), filename_mode)
                    if plot:
                        plt.figure()
                        self._plot_mode(
                            field_name,
                            i,
                            filename_mode,
                            self.n_effs[i],
                            area=area,
                            wavelength=self._structure._wl,
                        )

        return self.modes

    def plot_modes(
        self, filename, fields_to_write=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    ):
        """ works only for cached modes """
        # Mode field plots.
        for i, mode in enumerate(self.modes):
            filename_full = filename.parent / (filename.stem + f"_{i}.dat")

            for field_name, field_profile in mode.items():
                if field_name in fields_to_write:
                    filename_mode = self._get_mode_filename(
                        field_name, i, filename_full
                    )
                    plt.figure()
                    self._plot_mode(
                        field_name,
                        i,
                        filename_mode,
                        self.n_effs[i],
                        # area=area,
                        wavelength=self.wg._wl,
                    )


@pytest.mark.parametrize("overwrite", [True, False])
def test_mode_solver_full_vectorial(overwrite):
    mode_solver = mode_solver_full(overwrite=overwrite)
    # modes = mode_solver.solve()
    # neff0 = modes["n_effs"][0].real

    neff0 = mode_solver.results["n_effs"][0].real
    print(neff0)
    assert np.isclose(neff0, 2.4717079424099673)


@autoname
def _full(n_modes=2, **wg_kwargs):
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

    mode_solver = ModeSolverFullyVectorial(n_modes)
    mode_solver.wg = wg
    # modes = mode_solver.solve(wg)
    # mode_solver.write_modes_to_file("example_modes_1.dat")
    return mode_solver


def mode_solver_full(n_modes=2, overwrite=False, **wg_kwargs):
    mode_solver = _full(n_modes=n_modes, **wg_kwargs)
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
        )

        with open(jsonpath, "w") as f:
            f.write(json.dumps(d))
        mode_solver.write_modes_to_file(filepath)

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
        r = dict(modes=modes_cache, n_effs=n_effs_cache)
        mode_solver.modes = r["modes"]
        mode_solver.n_effs = r["n_effs"]
        mode_solver.plot_modes(filepath)

    mode_solver.results = r
    return mode_solver


if __name__ == "__main__":
    # test_mode_solver_full_vectorial(overwrite=True)
    test_mode_solver_full_vectorial(overwrite=False)
    plt.show()
