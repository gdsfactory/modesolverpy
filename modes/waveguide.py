import matplotlib.pylab as plt
import numpy as np

from modes._structure import RidgeWaveguide
from modes._structure import WgArray
from modes.autoname import autoname
from modes.config import CONFIG
from modes.materials import nitride
from modes.materials import si
from modes.materials import sio2


@autoname
def waveguide(
    x_step=0.02,
    y_step=0.02,
    wg_height=0.22,
    wg_width=0.5,
    slab_height=0,
    sub_height=0.5,
    sub_width=2.0,
    clad_height=[0.5],
    n_sub=sio2,
    n_wg=si,
    n_clads=[sio2],
    wavelength=1.55,
    angle=90.0,
):
    """ returns a waveguide structure

    ::

        _________________________________

                                        clad_height
               wg_width
             <---------->
              ___________    _ _ _ _ _ _
             |           |
        _____|           |____          |
                                        wg_heigth
        slab_height                     |
        _______________________ _ _ _ _ __

        sub_height
        _________________________________
        <------------------------------->
                     sub_width


    To define a waveguide we need to define:

    - the material functions or refractive indices of box, waveguide and clad
    - height of each material
    - x and y_steps for discretizing the structure
    - sidewall angle
    - wavelength that can be used in case the refractive index are a function of the wavelength

    Where all units are in um

    .. plot::
        :include-source:

        import modes as ms

        wg = ms.waveguide(wg_width=0.5, wg_height=0.22, slab_height=0.09, angle=80)
        ms.write_material_index(wg)

    """
    n_wg = n_wg(wavelength) if callable(n_wg) else n_wg
    n_sub = n_sub(wavelength) if callable(n_sub) else n_sub
    n_clad = [n_clad(wavelength) if callable(n_clad) else n_clad for n_clad in n_clads]

    film_thickness = wg_height
    wg_height = film_thickness - slab_height

    return RidgeWaveguide(
        wavelength=wavelength,
        x_step=x_step,
        y_step=y_step,
        wg_height=wg_height,
        wg_width=wg_width,
        sub_height=sub_height,
        sub_width=sub_width,
        clad_height=clad_height,
        n_sub=n_sub,
        n_wg=n_wg,
        angle=angle,
        n_clad=n_clad,
        film_thickness=film_thickness,
    )


@autoname
def waveguide_array(
    wg_gaps,
    wg_widths,
    x_step=0.02,
    y_step=0.02,
    wg_height=0.22,
    wg_width=0.5,
    slab_height=0,
    sub_height=0.5,
    sub_width=2.0,
    clad_height=[0.5],
    n_sub=sio2,
    n_wg=si,
    n_clads=[sio2],
    wavelength=1.55,
    angle=90.0,
):
    """ returns a waveguide_array (also known as couple waveguides) ::

         __________________________________________________________

                                                                  clad_height
              wg_widths[0]  wg_gaps[0]  wg_widths[1]
              <-----------><----------><----------->   _ _ _ _ _ _
               ___________              ___________
              |           |            |           |
         _____|           |____________|           |____          |
                                                                  wg_heigth
         slab_height                                              |
         ________________________________________________ _ _ _ _ _

         sub_height
         __________________________________________________________

         <-------------------------------------------------------->
                              sub_width

    To define a waveguide we need to define

    Args:
        wg_width: of each waveguide (list)
        wg_gaps: between waveguides
        n_sub, n_wg, n_clad:the material functions or refractive indices of substrate, waveguide and top clad
        sub_height, slab_height, wg_height, clad_height: height of each material
        x_step  y_step: for discretizing the structure
        angle: (deg) sidewall angle in degrees
        wavelength: in case the refractive index is a function of the wavelength

    Where all units are in um

    .. plot::
        :include-source:

        import modes as ms

        wg_array = ms.waveguide_array(wg_gaps=[0.2], wg_widths=[0.5, 0.5], slab_height=0.09)
        ms.write_material_index(wg_array)

    """
    n_wg = n_wg(wavelength) if callable(n_wg) else n_wg
    n_sub = n_sub(wavelength) if callable(n_sub) else n_sub
    n_clad = [n_clad(wavelength) if callable(n_clad) else n_clad for n_clad in n_clads]

    film_thickness = wg_height
    wg_height = film_thickness - slab_height

    return WgArray(
        wg_widths=wg_widths,
        wg_gaps=wg_gaps,
        wavelength=wavelength,
        x_step=x_step,
        y_step=y_step,
        wg_height=wg_height,
        sub_height=sub_height,
        sub_width=sub_width,
        clad_height=clad_height,
        n_sub=n_sub,
        n_wg=n_wg,
        angle=angle,
        n_clad=n_clad,
        film_thickness=film_thickness,
    )


def get_waveguide_filepath(wg):
    return CONFIG.cache / f"{wg.name}.dat"


def write_material_index(wg, filepath=None):
    """ writes the waveguide refractive index into filepath"""
    filepath = filepath or get_waveguide_filepath(wg)
    wg.write_to_file(filepath)


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


def test_waveguide_array_material_index():
    wg = waveguide_array(wg_gaps=[0.2], wg_widths=[0.5] * 2)
    n = wg.n
    sx, sy = np.shape(n)
    n_wg = wg.n[sx // 2][sy // 2]
    assert n_wg == sio2(wg._wl)


if __name__ == "__main__":
    wg = waveguide(
        wg_width=0.5,
        angle=80,
        n_wg=si,
        clad_height=[50e-3, 50e-3, 0.5],
        n_clads=[sio2, nitride, sio2],
    )
    # wg = waveguide_array(wg_widths=[0.5] * 2, wg_gaps=[0.2], slab_height=0.09)
    # print(wg)
    # test_waveguide_material_index()
    # test_waveguide_array_material_index()
    write_material_index(wg)
    plt.show()
