from typing import Dict, List, Tuple, Union

import matplotlib.pylab as plt
import numpy as np
import pytest
import tqdm
from numpy import float64, ndarray

from modes._structure import RidgeWaveguide
from modes.mode_solver_full import mode_solver_full
from modes.waveguide import waveguide


def sweep_waveguide(
    waveguides: List[RidgeWaveguide],
    sweep_param_list: ndarray,
    plot: bool = True,
    x_label: str = "waveguide width (um)",
    fraction_mode_list: List[int] = [],
    overwrite: bool = False,
    n_modes: int = 6,
    legend: None = None,
) -> Dict[
    str,
    Union[
        List[ndarray],
        List[List[Tuple[str, float64]]],
        List[List[float64]],
        List[List[List[Union[str, float]]]],
        List[List[float]],
    ],
]:
    """
    Find the modes of many waveguides.

    Args:
        waveguides (list): A list of `waveguides` to find the modes of.
        sweep_param_list (list): parameter that we sweep (for plotting)
        plot (bool): `True`  generates plots
        x_label (str): x-axis text to display in the plot.
        fraction_mode_list (list): A list of mode indices of the modes
            that should be included in the TE/TM mode fraction plot.
            If the list is empty, all modes will be included.  The list
            is empty by default.
        overwrite: when True forces to resimulate the structure
        n_modes: number of modes to compute
        legend:

    Returns:
        list: A list of the effective indices found for each structure.

    .. plot::
        :include-source:

        import numpy as np
        import modes as ms

        widths = np.arange(0.5, 2.0, 0.5)
        wgs = [ms.waveguide(width=width) for width in widths]
        r = ms.sweep_waveguide(
            wgs, widths, x_label="waveguide width", fraction_mode_list=[1, 2],
        )
        print(r["n_effs"][0])
    """
    n_effs = []
    mode_types: List[Tuple[str, float64]] = []
    fractions_te = []
    fractions_tm = []

    for wg in tqdm.tqdm(waveguides, ncols=70):
        ms = mode_solver_full(wg=wg, overwrite=overwrite, n_modes=n_modes, plot=False)
        n_effs.append(np.real(ms.n_effs))
        mode_types.append(ms.mode_types)
        fractions_te.append(ms.fraction_te)
        fractions_tm.append(ms.fraction_tm)

    results = dict(
        n_effs=n_effs,
        mode_types=mode_types,
        fractions_te=fractions_te,
        fractions_tm=fractions_tm,
    )

    suffix = "_".join([f"{int(sweep_param_list[i]*1e3)}" for i in [0, -1]])
    suffix += f"_{len(sweep_param_list)}"

    filename = ms._modes_directory / f"{ms.name}_{suffix}.dat"
    filename_neffs = filename.parent / f"{filename.stem}_neffs.dat"
    filename_mode_types = filename.parent / f"{filename.stem}_mode_types.dat"
    filename_fraction_te = filename.parent / f"{filename.stem}_fraction_te.dat"
    filename_fraction_tm = filename.parent / f"{filename.stem}_fraction_tm.dat"

    ms._write_n_effs_to_file(n_effs, filename_neffs, sweep_param_list)

    with open(filename_mode_types, "w") as fs:
        header = ",".join(f"Mode{i}" for i in range(len(mode_types[0])))
        fs.write(f"# {header} \n")
        for mt in mode_types:
            txt = ",".join([f"{pair[0]} {pair[1]}" for pair in mt])
            fs.write(txt + "\n")

    with open(filename_fraction_te, "w") as fs:
        header = "fraction te"
        fs.write("# param sweep," + header + "\n")
        for param, fte in zip(sweep_param_list, fractions_te):
            txt = "%.6f," % param
            txt += ",".join("%.2f" % f for f in fte)
            fs.write(txt + "\n")

    with open(filename_fraction_tm, "w") as fs:
        header = "fraction tm"
        fs.write("# param sweep," + header + "\n")
        for param, ftm in zip(sweep_param_list, fractions_tm):
            txt = "%.6f," % param
            txt += ",".join("%.2f" % f for f in ftm)
            fs.write(txt + "\n")

    suffix = f"@ {waveguides[0]._wl}"
    if plot:
        plt.figure()
        title = f"$n_{{eff}}$ {suffix}"
        y_label = "$n_{eff}$"
        ms._plot_n_effs(
            filename_neffs,
            filename_fraction_te,
            x_label,
            y_label,
            title,
        )
        if legend:
            plt.legend(legend)

        plt.figure()
        title = f"TE Fraction {suffix}"
        ms._plot_fraction(
            filename_fraction_te,
            x_label,
            "TE Fraction [%]",
            title,
            fraction_mode_list,
        )
        if legend:
            plt.legend(legend)

        plt.figure()
        title = f"TM Fraction {suffix}"
        ms._plot_fraction(
            filename_fraction_tm,
            x_label,
            "TM Fraction [%]",
            title,
            fraction_mode_list,
        )
        if legend:
            plt.legend(legend)

    return results


@pytest.mark.parametrize("overwrite", [True, False])
def test_sweep(overwrite: bool) -> None:
    widths = np.arange(0.5, 2.0, 0.5)
    wgs = [waveguide(width=width) for width in widths]
    r = sweep_waveguide(
        wgs,
        widths,
        n_modes=2,
        fraction_mode_list=[1, 2],
        overwrite=overwrite,
    )
    assert np.isclose(
        r["n_effs"][0], np.array([2.47170794, 1.81238363]), atol=0.1
    ).all(), r["n_effs"][0]
    assert r


@pytest.mark.parametrize("overwrite", [True, False])
def test_sweep2(overwrite: bool) -> None:
    widths = np.arange(0.3, 1.0, 0.5)
    wgs = [waveguide(width=width) for width in widths]
    r = sweep_waveguide(
        wgs,
        widths,
        n_modes=2,
        fraction_mode_list=[1, 2],
        overwrite=overwrite,
    )
    assert np.isclose(
        r["n_effs"][0], np.array([1.84891783, 1.60477969]), atol=0.1
    ).all(), r["n_effs"][0]
    assert r


if __name__ == "__main__":
    # test_sweep2(overwrite=False)

    widths = np.arange(0.3, 1.0, 0.5)
    wgs = [waveguide(width=width) for width in widths]
    r = sweep_waveguide(
        wgs,
        widths,
        n_modes=2,
        fraction_mode_list=[1, 2],
    )
    plt.show()
