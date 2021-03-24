from typing import Dict, List, Union

import matplotlib.pylab as plt
import numpy as np
import pytest
import tqdm
from numpy import float64, ndarray

from modes.mode_solver_full import mode_solver_full


def sweep_wavelength(
    wavelengths: ndarray, plot: bool = True, overwrite: bool = False, **wg_kwargs
) -> Dict[str, Union[List[ndarray], List[List[float64]], List[List[float]]]]:
    """

    Solve for the effective indices of a fixed structure at
    different wavelengths.

    Args:
        wavelengths (list): list of wavelengths to sweep
        plot (bool): `True` generates plots
        overwrite: when True forces to resimulate the structure

    Returns:
        wg_kwargs: arguments for the waveguide
        results: dict of results
        resuls['n_effs']: A list of the effective indices found for each wavelength.
        resuls['fractions_te']:
    """
    n_effs = []
    fractions_te = []

    for w in tqdm.tqdm(wavelengths, ncols=70):
        ms = mode_solver_full(wavelength=w, overwrite=overwrite, **wg_kwargs)
        n_effs.append(np.real(ms.n_effs))
        fractions_te.append(ms.fraction_te)

    suffix = "_".join([f"{int(wavelengths[i]*1e3)}" for i in [0, -1]])
    suffix += f"_{len(wavelengths)}"

    filename = ms._modes_directory / f"{ms.name}_{suffix}.dat"
    filename_neffs = filename.parent / f"{filename.stem}_neffs.dat"
    filename_fraction_te = filename.parent / f"{filename.stem}_fraction_te.dat"

    ms._write_n_effs_to_file(n_effs, filename_neffs, wavelengths)

    with open(filename_fraction_te, "w") as fs:
        header = "fraction te"
        fs.write("# param sweep," + header + "\n")
        for param, fte in zip(wavelengths, fractions_te):
            txt = "%.6f," % param
            txt += ",".join("%.2f" % f for f in fte)
            fs.write(txt + "\n")

    if plot:
        title = "$n_{eff}$ vs Wavelength"
        ms._plot_n_effs(
            filename_neffs,
            filename_fraction_te,
            "Wavelength",
            "n_{eff}",
            title,
        )
        plt.ylabel("$n_{eff}$")

    results = dict(n_effs=n_effs, fractions_te=fractions_te)
    return results


@pytest.mark.parametrize("overwrite", [True, False])
def test_sweep(overwrite: bool) -> None:
    wavelengths = np.arange(1.30, 1.60, 0.1)
    r = sweep_wavelength(wavelengths=wavelengths, overwrite=overwrite)
    print(r["n_effs"][0])
    assert np.isclose(r["n_effs"][0], np.array([2.7357584, 2.22395364])).all()
    assert r


if __name__ == "__main__":
    test_sweep(overwrite=False)
    plt.show()
