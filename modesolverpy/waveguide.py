import matplotlib.pylab as plt
import numpy as np
import opticalmaterialspy as mat
from modesolverpy.autoname import autoname
from modesolverpy.config import CONFIG
from modesolverpy.structure import RidgeWaveguide


def si(wl):
    return mat.RefractiveIndexWeb(
        "https://refractiveindex.info/?shelf=main&book=Si&page=Li-293K"
    ).n(wl)


def sio2(wl):
    return mat.SiO2().n(wl)


@autoname
def waveguide(
    x_step=0.02,
    y_step=0.02,
    wg_height=0.22,
    wg_width=0.5,
    slab_height=0,
    sub_height=0.5,
    sub_width=2.0,
    clad_height=0.5,
    n_sub=sio2,
    n_wg=si,
    n_clad=sio2,
    wavelength=1.55,
    angle=90.0,
):
    """ returns a waveguide structure
    """
    if callable(n_sub):
        n_sub = n_sub(wavelength)
    if callable(n_clad):
        n_clad = n_clad(wavelength)
    if callable(n_wg):
        n_wg = n_wg(wavelength)

    film_thickness = wg_height
    wg_height = film_thickness - slab_height

    structure = RidgeWaveguide(
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
    return structure


def get_waveguide_filepath(waveguide):
    return CONFIG["cache"] / f"{waveguide.name}.dat"


def write_material_index(waveguide, filepath=None):
    filepath = filepath or get_waveguide_filepath(waveguide)
    waveguide.write_to_file(filepath)


def waveguide_array():
    pass


def test_waveguide_name():
    wg1 = waveguide(angle=80, wg_width=0.5)
    wg2 = waveguide(wg_width=0.5, angle=80)
    assert (
        wg1.name == wg2.name
    ), f"{wg1} and {wg2} waveguides have the same settings and should have the same name"


def test_waveguide_material_index():
    wg = waveguide()
    n = wg.n
    sx, sy = np.shape(n)
    n_wg = wg.n[sx // 2][sy // 2]
    assert n_wg == si(wg._wl)


if __name__ == "__main__":
    wg = waveguide(wg_width=0.5, angle=80, n_wg=si)
    # test_waveguide_material_index()
    write_material_index(wg)
    print(wg)
    plt.show()
