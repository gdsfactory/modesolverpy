"""Types for modes """
from pathlib import PosixPath
from typing import Callable, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from scipy.interpolate import RectBivariateSpline

PathType = Union[str, PosixPath]

Field = Literal[
    "Ex",
    "Ey",
    "Ez",
    "Hx",
    "Hy",
    "Hz",
]

SemiVectorialMethod = Literal[
    "Ex",
    "Ey",
]

# cmap_default = 'viridis'
cmap_default = "RdBu"


class TypedArray(np.ndarray):
    """based on https://github.com/samuelcolvin/pydantic/issues/380"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return np.array(val, dtype=cls.inner_type)


class ArrayMeta(type):
    def __getitem__(self, t):
        return type("Array", (TypedArray,), {"inner_type": t})


class Array(np.ndarray, metaclass=ArrayMeta):
    pass


class Mode(BaseModel):
    """Mode object.

    Args:
        mode_number:
        wavelength: um
        neff: effective index
        ng: group index
        fraction_te:
        fraction_tm:
        effective_area:
        E: Ex, Ey, Ez
        H: Hx, Hy, Hz
        eps:
        y:
        z:

    """

    mode_number: int
    wavelength: float
    neff: float
    ng: Optional[float] = None
    fraction_te: Optional[float] = None
    fraction_tm: Optional[float] = None
    effective_area: Optional[float] = None
    E: Optional[Array[float]] = None
    H: Optional[Array[float]] = None
    eps: Optional[Array[float]] = None
    y: Optional[Array[float]] = None
    z: Optional[Array[float]] = None

    @property
    def Ex(self):
        return self.E[:, :, 0]

    @property
    def Ey(self):
        return self.E[:, :, 1]

    @property
    def Ez(self):
        return self.E[:, :, 2]

    @property
    def Hx(self):
        return self.H[:, :, 0]

    @property
    def Hy(self):
        return self.H[:, :, 1]

    @property
    def Hz(self):
        return self.H[:, :, 2]

    def get_overlaps(self) -> Tuple[float, float, float]:
        e_fields = (self.Ex, self.Ey, self.Ez)
        h_fields = (self.Hx, self.Hy, self.Hz)

        areas_e = [np.sum(np.abs(e) ** 2) for e in e_fields]
        areas_e /= np.sum(areas_e)
        areas_e *= 100

        areas_h = [np.sum(np.abs(h) ** 2) for h in h_fields]
        areas_h /= np.sum(areas_h)
        areas_h *= 100

        fraction_te = areas_e[0] / (areas_e[0] + areas_e[1])
        fraction_tm = areas_e[1] / (areas_e[0] + areas_e[1])

        areas = areas_e.tolist()
        areas.extend(areas_h)
        mode_area = areas

        return mode_area, fraction_te, fraction_tm

    def __repr__(self):
        return f"Mode{self.mode_number}"

    def E_grid_interp(self, y_arr, z_arr, index):
        """
        Creates new attributes with scipy.interpolate.RectBivariateSpline objects
        that can be used to interpolate the field on a new regular grid

        Args:
            y_grid (np.array): y values where to evaluate, in increasing array
            z_grid (np.array): z values where to evaluate, in increasing array
            index: 0: x, 1: y, 2: z
        """

        if index not in [0, 1, 2]:
            raise ValueError(f"index = {index} needs to be (0: x, 1: y, 2: z)")

        return np.flip(
            RectBivariateSpline(self.y, self.z, np.real(self.E[:, :, 0, index]))(
                y_arr, z_arr, grid=True
            )
            + (
                1j
                * RectBivariateSpline(self.y, self.z, np.imag(self.E[:, :, 0, index]))(
                    y_arr, z_arr, grid=True
                )
            )
        )

    def Ex_grid_interp(self, y_arr, z_arr):
        return self.E_grid_interp(y_arr=y_arr, z_arr=z_arr, index=0)

    def Ey_grid_interp(self, y_arr, z_arr):
        return self.E_grid_interp(y_arr=y_arr, z_arr=z_arr, index=1)

    def Ez_grid_interp(self, y_arr, z_arr):
        return self.E_grid_interp(y_arr=y_arr, z_arr=z_arr, index=2)

    def H_grid_interp(self, y_arr, z_arr, index=0):
        """
        Creates new attributes with scipy.interpolate.RectBivariateSpline objects
        that can be used to interpolate the field on a new regular grid

        Args:
            y_grid (np.array): y values where to evaluate, in increasing array
            z_grid (np.array): z values where to evaluate, in increasing array
            index: 0: x, 1: y, 2: z
        """
        if index not in [0, 1, 2]:
            raise ValueError(f"index = {index} needs to be (0: x, 1: y, 2: z)")
        return np.flip(
            RectBivariateSpline(self.y, self.z, np.real(self.H[:, :, 0, index]))(
                y_arr, z_arr, grid=True
            )
            + (
                1j
                * RectBivariateSpline(self.y, self.z, np.imag(self.H[:, :, 0, index]))(
                    y_arr, z_arr, grid=True
                )
            )
        )

    def Hx_grid_interp(self, y_arr, z_arr):
        return self.H_grid_interp(y_arr=y_arr, z_arr=z_arr, index=0)

    def Hy_grid_interp(self, y_arr, z_arr):
        return self.H_grid_interp(y_arr=y_arr, z_arr=z_arr, index=1)

    def Hz_grid_interp(self, y_arr, z_arr):
        return self.H_grid_interp(y_arr=y_arr, z_arr=z_arr, index=2)

    def plot_eps(
        self,
        cmap: str = "binary",
        origin="lower",
        logscale: bool = False,
        show: bool = True,
    ):
        """plot index profle"""
        plt.imshow(
            self.eps ** 0.5,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
        )
        plt.title("index profile")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_e(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
    ):
        """Plot Electric field module."""
        E = self.E / abs(max(self.E.min(), self.E.max(), key=abs)) if scale else self.E
        Eabs = np.sqrt(
            np.multiply(E[:, :, 2], E[:, :, 2])
            + np.multiply(E[:, :, 1], E[:, :, 1])
            + np.multiply(E[:, :, 0], E[:, :, 0])
        )
        ep = abs(Eabs)
        ep = 10 * np.log10(ep) if logscale else ep
        plt.imshow(
            ep.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=0 if scale else ep.min(),
            vmax=1 if scale else ep.max(),
        )
        plt.title("$|E|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_ex(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ):
        # E = self.E / abs(max(self.E.min(), self.E.max(), key=abs)) if scale else self.E
        ex = self.Ex
        ex = 10 * np.log10(np.abs(ex)) if logscale else operation(ex)
        plt.imshow(
            ex.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else ex.min(),
            vmax=1 if scale else ex.max(),
        )
        plt.title("{}($E_x$)".format(operation))
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_ey(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ):
        E = self.E / abs(max(self.E.min(), self.E.max(), key=abs)) if scale else self.E
        ey = E[:, :, 0, 1]
        ey = 10 * np.log10(np.abs(ey)) if logscale else operation(ey)
        plt.imshow(
            ey.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else ey.min(),
            vmax=1 if scale else ey.max(),
        )
        plt.title("{}($E_y$)".format(operation))
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_ez(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ):
        E = self.E / abs(max(self.E.min(), self.E.max(), key=abs)) if scale else self.E
        ez = E[:, :, 0, 2]
        ez = 10 * np.log10(ez) if logscale else operation(ez)
        plt.imshow(
            ez.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else ez.min(),
            vmax=1 if scale else ez.max(),
        )
        plt.title("{}($E_z$)".format(operation))
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_e_all(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ):
        plt.figure(figsize=(16, 10), dpi=100)

        plt.subplot(2, 3, 1)
        self.plot_ex(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 2)
        self.plot_ey(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 3)
        self.plot_ez(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 4)
        self.plot_e(show=False, scale=scale)

        plt.subplot(2, 3, 5)
        self.plot_eps(show=False)

        plt.tight_layout()
        plt.show()

    def plot_h(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
    ):
        H = self.H / abs(max(self.H.min(), self.H.max(), key=abs)) if scale else self.H
        Habs = np.sqrt(
            np.multiply(H[:, :, 0, 2], H[:, :, 0, 2])
            + np.multiply(H[:, :, 0, 1], H[:, :, 0, 1])
            + np.multiply(H[:, :, 0, 0], H[:, :, 0, 0])
        )
        hp = abs(Habs)
        hp = 10 * np.log10(hp) if logscale else hp
        plt.imshow(
            hp.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=0 if scale else hp.min(),
            vmax=1 if scale else hp.max(),
        )
        plt.title("$|H|$")
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_hx(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ):
        H = self.H / abs(max(self.H.min(), self.H.max(), key=abs)) if scale else self.H
        hx = H[:, :, 0, 0]
        hx = 10 * np.log10(np.abs(hx)) if logscale else operation(hx)
        plt.imshow(
            hx.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else hx.min(),
            vmax=1 if scale else hx.max(),
        )
        plt.title("{}($H_x$)".format(operation))
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_hy(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ):
        H = self.H / abs(max(self.H.min(), self.H.max(), key=abs)) if scale else self.H
        hy = H[:, :, 0, 1]
        hy = 10 * np.log10(np.abs(hy)) if logscale else operation(hy)
        plt.imshow(
            hy.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=-1 if scale else hy.min(),
            vmax=1 if scale else hy.max(),
        )
        plt.title("{}($H_y$)".format(operation))
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_hz(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ):
        H = self.H / abs(max(self.H.min(), self.H.max(), key=abs)) if scale else self.H
        hz = abs(H[:, :, 0, 2])
        hz = 10 * np.log10(hz) if logscale else operation(hz)
        plt.imshow(
            hz.T,
            cmap=cmap,
            origin=origin,
            aspect="auto",
            extent=[np.min(self.y), np.max(self.y), np.min(self.z), np.max(self.z)],
            vmin=0 if scale else hz.min(),
            vmax=1 if scale else hz.max(),
        )
        plt.title("{}($H_z$)".format(operation))
        plt.ylabel("z-axis")
        plt.xlabel("y-axis")
        plt.colorbar()
        if show:
            plt.show()

    def plot_h_all(
        self,
        cmap: str = cmap_default,
        origin="lower",
        logscale: bool = False,
        show: bool = True,
        scale: bool = False,
        operation: Callable = np.real,
    ):
        plt.figure(figsize=(16, 10), dpi=100)

        plt.subplot(2, 3, 1)
        self.plot_hx(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 2)
        self.plot_hy(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 3)
        self.plot_hz(show=False, scale=scale, cmap=cmap, operation=operation)

        plt.subplot(2, 3, 4)
        self.plot_h(show=False, scale=scale)

        plt.subplot(2, 3, 5)
        self.plot_eps(show=False)

        plt.tight_layout()
        plt.show()
