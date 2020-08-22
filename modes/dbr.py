import EMpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interpolate

from modes.neff import neff
from modes.sweep_waveguide import sweep_waveguide
from modes.waveguide import waveguide

Layer = EMpy.utils.Layer  # shortcut


def dbr_period(w0=0.5, wavelength=1.55):
    """ returns period (um)
    """
    assert 2 > w0 > 0.1
    neff0 = neff(wg_width=w0, wavelength=wavelength)
    return wavelength / 2 / neff0


wg_widths0 = np.arange(0.2, 2.0, 0.01)

# TE0 effective index vs wg_width for 220nm wg_height
n0 = np.array(
    [
        1.50260303,
        1.50262444,
        1.57468689,
        1.65965899,
        1.65966473,
        1.65964752,
        1.75315899,
        1.84891628,
        1.84891613,
        1.84892559,
        1.94121384,
        2.02656345,
        2.02657261,
        2.02655917,
        2.1034659,
        2.17175979,
        2.17176343,
        2.17175779,
        2.28518501,
        2.28519689,
        2.28517322,
        2.28521261,
        2.37339684,
        2.37340128,
        2.37338809,
        2.37340185,
        2.44264057,
        2.44265644,
        2.4426231,
        2.44264416,
        2.49773698,
        2.49772708,
        2.49773331,
        2.49772371,
        2.54218889,
        2.54215512,
        2.5421784,
        2.54218477,
        2.5784884,
        2.57849722,
        2.57850223,
        2.57848611,
        2.608552,
        2.60855026,
        2.60855627,
        2.60854771,
        2.63368,
        2.63368999,
        2.63368875,
        2.63368405,
        2.65492409,
        2.6549179,
        2.65492469,
        2.65491673,
        2.67302164,
        2.67301302,
        2.67301648,
        2.67301616,
        2.68856682,
        2.68857275,
        2.6885656,
        2.68856815,
        2.70203781,
        2.70202449,
        2.70201673,
        2.70202397,
        2.71375747,
        2.71374598,
        2.71374934,
        2.71374844,
        2.72402633,
        2.72401838,
        2.72403155,
        2.72403163,
        2.73307705,
        2.73307525,
        2.73307669,
        2.73308254,
        2.74110033,
        2.74110561,
        2.74110286,
        2.74109853,
        2.7482411,
        2.74824153,
        2.74824283,
        2.74824566,
        2.75463286,
        2.75463087,
        2.75463025,
        2.75463291,
        2.76036615,
        2.76036728,
        2.76036772,
        2.76036507,
        2.76552747,
        2.76553716,
        2.76553148,
        2.76552892,
        2.77019971,
        2.77020276,
        2.77019994,
        2.77020439,
        2.77443901,
        2.77443525,
        2.77444001,
        2.77443529,
        2.77829513,
        2.77829456,
        2.77829595,
        2.77829388,
        2.78181937,
        2.78181029,
        2.78181328,
        2.78181992,
        2.78503488,
        2.78503359,
        2.78503334,
        2.78503418,
        2.78798655,
        2.78798803,
        2.78798267,
        2.78798298,
        2.79069994,
        2.79070043,
        2.79070115,
        2.79070318,
        2.79320745,
        2.79320453,
        2.79321066,
        2.79320962,
        2.79551268,
        2.79551141,
        2.79551358,
        2.79551504,
        2.79765285,
        2.79764919,
        2.79765227,
        2.79765354,
        2.79963169,
        2.79962852,
        2.79962969,
        2.79962833,
        2.8014626,
        2.80146706,
        2.80146471,
        2.80146191,
        2.8031671,
        2.80316644,
        2.80316686,
        2.80316552,
        2.80474725,
        2.80474647,
        2.80474525,
        2.80474941,
        2.8062144,
        2.80621054,
        2.80621134,
        2.80621389,
        2.80757677,
        2.80757443,
        2.80757317,
        2.807576,
        2.80884062,
        2.80884102,
        2.80884147,
        2.80883682,
        2.81001151,
        2.81000791,
        2.81001223,
        2.81000946,
        2.81109277,
        2.81108922,
        2.81109009,
        2.81109141,
        2.81245535,
        2.81245505,
        2.81245507,
        2.81245537,
        2.81334852,
        2.81334898,
    ]
)


def dbr_spectrum(
    wg_width1, wg_width2, n_periods, period=None, wavelength=1.55, **wg_args
):
    """ returns the DBR spectrum

    Args:
        w0:
        dw:
        n_periods:
        period (um):
        wavelength: 1.55

    wg_args:
        d: length
        wg_height=0.22,
        wg_width=0.5,
        slab_height=0,
        sub_height=0.5,
        sub_width=2.0,
        clad_height=[0.5],
        n_sub=sio2,
        n_wg=si,
        n_clads=[sio2],
        angle=90.0,

    Args:

            period
        <------------->
                _______
        _______|

        w0-dw/2  w0+dw/2  ...  n times
        _______
               |_______
    """

    w0 = (wg_width1 + wg_width2) / 2
    dw = abs(wg_width1 - wg_width2)

    w1 = w0 - dw / 2
    w2 = w0 + dw / 2
    period = period or dbr_period(w0, wavelength)

    if len(wg_args) > 0:
        wg_widths = np.arange(0.2, 2.0, 0.01)
        wgs = [
            waveguide(wg_width=wg_width, wavelength=wavelength)
            for wg_width in wg_widths
        ]
        r = sweep_waveguide(wgs, wg_widths, n_modes=1)
        n_effs = np.squeeze(r["n_effs"])
    else:
        wg_widths = wg_widths0
        n_effs = n0

    fn = interpolate.interp1d(wg_widths, n_effs)

    m1 = EMpy.materials.IsotropicMaterial(
        "m1", EMpy.materials.RefractiveIndex(n0_const=fn(w1))
    )
    m2 = EMpy.materials.IsotropicMaterial(
        "m2", EMpy.materials.RefractiveIndex(n0_const=fn(w2))
    )

    teeth1 = Layer(m1, period * 1e-6 / 2)
    teeth2 = Layer(m2, period * 1e-6 / 2)
    print(period)

    iso_layers = EMpy.utils.Multilayer()
    for i in range(n_periods):
        iso_layers.append(teeth1)
        iso_layers.append(teeth2)

    wls = np.linspace(1.4e-6, 1.7e-6, 100)
    theta_inc = 0
    solution_iso = EMpy.transfer_matrix.IsotropicTransferMatrix(
        iso_layers, theta_inc
    ).solve(wls)

    plt.figure()
    plt.plot(
        wls,
        solution_iso.Rs,
        wls,
        solution_iso.Ts,
        wls,
        solution_iso.Rp,
        wls,
        solution_iso.Tp,
    )
    plt.title("isotropic")
    plt.show()


if __name__ == "__main__":
    c = dbr_spectrum(
        wg_width1=0.4, wg_width2=0.6, n_periods=100, period=1.55 / 2 / 2.44
    )
