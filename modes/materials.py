"""
We have different materials available thanks to the [materialspy](https://opticalmaterialspy.readthedocs.io/en/latest/index.html) module

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import modes as ms

   wavelengths = np.linspace(1.3, 1.6, 10)
   nsi = [ms.materials.si(w) for w in wavelengths]

   plt.plot(wavelengths, nsi)
   plt.xlabel('wavelength (nm)')
   plt.ylabel('Refractive index')
   plt.title('Silicon refractive index')

"""
import opticalmaterialspy as mat


def si(wl):
    return mat.RefractiveIndexWeb(
        "https://refractiveindex.info/?shelf=main&book=Si&page=Li-293K"
    ).n(wl)


def sio2(wl):
    return mat.SiO2().n(wl)


def air(wl):
    return mat.Air().n(wl)


def nitride(wl):
    return mat.RefractiveIndexWeb(
        "https://refractiveindex.info/?shelf=main&book=Si3N4&page=Luke"
    ).n(wl)


if __name__ == "__main__":
    print(nitride(1.3))
    print(si(1.55))
