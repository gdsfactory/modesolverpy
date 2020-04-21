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
