from collections.abc import Iterable
from typing import Callable, List, Optional, Union

import matplotlib.pylab as plt
import numpy as np

from modes._structure import RidgeWaveguide, WgArray
from modes.autoname import autoname
from modes.config import PATH
from modes.materials import nitride, si, sio2

WaveguideType = Union[RidgeWaveguide, WgArray]


@autoname
def waveguide(
    x_step: float = 0.02,
    y_step: float = 0.02,
    thickness: float = 0.22,
    width: float = 0.5,
    slab_thickness: float = 0,
    sub_thickness: float = 0.5,
    sub_width: float = 2.0,
    clad_thickness: List[float] = [0.5],
    n_sub: Union[Callable, float] = sio2,
    n_wg: Union[Callable, float] = si,
    n_slab: Optional[Union[Callable, float]] = None,
    n_clads: List[Union[Callable, float]] = [sio2],
    wavelength: float = 1.55,
    angle: float = 90.0,
) -> RidgeWaveguide:
    """Return waveguide structure

    Args:
        x_step: x grid step (um)
        y_step: y grid step (um)
        thickness: waveguide thickness (um)
        width: 0.5 (um)
        slab_thickness: 0 (um)
        sub_width: 2.0 related to the total simulation width (um)
        sub_thickness: 0.5 bottom simulation margin (um)
        clad_thickness: [0.5]  List of claddings (top simulation margin)
        n_sub: substrate index material
        n_wg: core waveguide index material
        n_slab: optional slab index. Defaults to n_wg.
        n_clads: list of cladding refractive index or function [sio2]
        wavelength: 1.55 wavelength (um)
        angle: 90 sidewall angle with respect to normal (degrees)

    .. code::

        _________________________________

                                        clad_thickness
               width
             <---------->
              ___________    _ _ _ _ _ _
             |           |
        _____|           |____          |
                                        thickness
        slab_thickness                     |
        _______________________ _ _ _ _ __

        sub_thickness
        _________________________________
        <------------------------------->
                     sub_width


    To define a waveguide we need to define:

    - the material functions or refractive indices of box, waveguide and clad
    - thickness of each material
    - x and y_steps for structure grid
    - sidewall angle (degrees)
    - wavelength that can be used in case the refractive index are a function of the wavelength

    Where all units are in um

    .. plot::
        :include-source:

        import modes as ms

        wg = ms.waveguide(width=0.5, thickness=0.22, slab_thickness=0.09, angle=80)
        ms.write_material_index(wg)

    """

    if not isinstance(n_clads, Iterable):
        raise ValueError(f"nclads not Iterable, got {n_clads}")
    if not isinstance(clad_thickness, Iterable):
        raise ValueError(f"clad_thickness not Iterable, got {clad_thickness}")
    n_wg = n_wg(wavelength) if callable(n_wg) else n_wg
    n_sub = n_sub(wavelength) if callable(n_sub) else n_sub
    n_clad = [n_clad(wavelength) if callable(n_clad) else n_clad for n_clad in n_clads]

    n_slab = n_slab or n_wg
    n_slab = n_slab(wavelength) if callable(n_slab) else n_slab

    film_thickness = thickness
    thickness = film_thickness - slab_thickness

    return RidgeWaveguide(
        wavelength=wavelength,
        x_step=x_step,
        y_step=y_step,
        thickness=thickness,
        width=width,
        sub_thickness=sub_thickness,
        sub_width=sub_width,
        clad_thickness=clad_thickness,
        n_sub=n_sub,
        n_wg=n_wg,
        n_slab=n_slab,
        angle=angle,
        n_clad=n_clad,
        film_thickness=film_thickness,
    )


@autoname
def waveguide_array(
    wg_gaps: List[float],
    widths: List[float],
    x_step: float = 0.02,
    y_step: float = 0.02,
    thickness: float = 0.22,
    slab_thickness: int = 0,
    sub_thickness: float = 0.5,
    sub_width: float = 2.0,
    clad_thickness: List[float] = [0.5],
    n_sub: Callable = sio2,
    n_wg: Callable = si,
    n_slab: Optional[Union[Callable, float]] = None,
    n_clads: List[Callable] = [sio2],
    wavelength: float = 1.55,
    angle: float = 90.0,
) -> WgArray:
    """Returns a evanescent coupled waveguides

    .. code::
         __________________________________________________________

                                                                  clad_thickness
                widths[0]   wg_gaps[0]  widths[1]
              <-----------><----------><----------->   _ _ _ _ _ _
               ___________              ___________               |
              |           |            |           |              |
         _____|           |____________|           |____          |
                                                                  thickness
         slab_thickness                                           |
         ________________________________________________ _ _ _ _ |

         sub_thickness
         __________________________________________________________

         <-------------------------------------------------------->
                              sub_width

    To define a waveguide we need to define

    Args:
        wg_gaps: between waveguides
        widths: of each waveguide (list)
        x_step: grid x step (um)
        y_step: grid y step(um)
        n_sub: substrate refractive index value or function(wavelength)
        n_wg: waveguide refractive index value or function(wavelength)
        n_clads: waveguide refractive index value or function(wavelength)
        n_slab: optional slab index. Defaults to n_wg.
        slab_thickness: slab thickness (um)
        sub_thickness: substrate thickness (um)
        clad_thickness: cladding thickness (um)
        wavelength: in um
        angle: sidewall angle in degrees

    Where all units are in um

    .. plot::
        :include-source:

        import modes as ms

        wg_array = ms.waveguide_array(wg_gaps=[0.2], widths=[0.5, 0.5], slab_thickness=0.09)
        ms.write_material_index(wg_array)

    """
    n_wg = n_wg(wavelength) if callable(n_wg) else n_wg
    n_sub = n_sub(wavelength) if callable(n_sub) else n_sub
    n_clad = [n_clad(wavelength) if callable(n_clad) else n_clad for n_clad in n_clads]

    n_slab = n_slab or n_wg
    n_slab = n_slab(wavelength) if callable(n_slab) else n_slab

    film_thickness = thickness
    thickness = film_thickness - slab_thickness

    return WgArray(
        widths=widths,
        wg_gaps=wg_gaps,
        wavelength=wavelength,
        x_step=x_step,
        y_step=y_step,
        thickness=thickness,
        sub_thickness=sub_thickness,
        sub_width=sub_width,
        clad_thickness=clad_thickness,
        n_sub=n_sub,
        n_wg=n_wg,
        n_slab=n_slab,
        angle=angle,
        n_clad=n_clad,
        film_thickness=film_thickness,
    )


def get_waveguide_filepath(wg):
    return PATH.cache / f"{wg.name}.dat"


def write_material_index(wg, filepath=None):
    """writes the waveguide refractive index into filepath"""
    filepath = filepath or get_waveguide_filepath(wg)
    wg.write_to_file(filepath)


def test_waveguide_name() -> None:
    wg1 = waveguide(angle=80, width=0.5)
    wg2 = waveguide(width=0.5, angle=80)
    assert wg1.name == wg2.name, (
        f"{wg1} and {wg2} waveguides have the same settings and should have the same"
        " name"
    )


def test_waveguide_material_index() -> None:
    wg = waveguide()
    n = wg.n
    sx, sy = np.shape(n)
    n_wg = wg.n[sx // 2][sy // 2]
    assert n_wg == si(wg._wl)


def test_waveguide_array_material_index() -> None:
    wg = waveguide_array(wg_gaps=[0.2], widths=[0.5] * 2)
    n = wg.n
    sx, sy = np.shape(n)
    n_wg = wg.n[sx // 2][sy // 2]
    assert n_wg == sio2(wg._wl)


if __name__ == "__main__":
    wg = waveguide(
        width=0.5,
        angle=80,
        n_wg=si,
        clad_thickness=[50e-3, 50e-3, 0.5],
        n_clads=[sio2, nitride, sio2],
    )
    # wg = waveguide_array(widths=[0.5] * 2, wg_gaps=[0.2], slab_thickness=0.09)
    # print(wg)
    # test_waveguide_material_index()
    # test_waveguide_array_material_index()
    write_material_index(wg)
    plt.show()
