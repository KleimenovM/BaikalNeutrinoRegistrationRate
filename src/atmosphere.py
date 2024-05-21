import os

import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


class Atmosphere:
    """
    With the given predicted flux points
    this class provides an extrapolates function
    f = f(energy, zenith_angle),
    energy from 1 GeV to 1 PeV
    zenith_angle from 90 to 180 degrees
    """
    def __init__(self):
        self.lg_energy: np.ndarray = np.linspace(0, 6, 1000)  # lg(E / 1GeV)
        self.energy: np.ndarray = 10 ** self.lg_energy  # GeV
        self.given_lg_energy: np.ndarray = None
        self.given_energy: np.ndarray = None
        self.angles: np.ndarray = None
        self.cosines: np.ndarray = None
        self.data: np.ndarray = None
        self.rbc: RectBivariateSpline = None

        script_dir = os.path.dirname(__file__)
        self.get_data(script_dir + "/data/atmosphere/atmospheric_nu_fluxes.txt")
        self.set_approximation()

    def get_data(self, filename: str):
        """
        Get predicted bartol flux points as a function of energy and zenith angle
        :param filename: a path to the file
        """
        values = np.loadtxt(filename, unpack=True)  # get columns
        self.given_energy = values[0, 1:]  # energy in GeV
        self.given_lg_energy = np.log10(self.given_energy)
        self.cosines = values[1:, 0]
        self.data = values[1:, 1:] * 1e4 / self.given_energy
        pass

    def set_approximation(self):
        extrapolated = np.zeros([self.cosines.size, self.lg_energy.size])
        for i, cosine in enumerate(self.cosines):
            pf_i = np.polyfit(self.given_lg_energy, np.log10(self.data[i, :]), deg=3)
            approximation_f = np.poly1d(pf_i)
            extrapolated[i, :] = approximation_f(self.lg_energy)

        self.rbc = RectBivariateSpline(self.cosines, self.lg_energy, extrapolated)
        pass

    def flux(self, lg_energy, zenith_angle):
        """
        Return a differential spectral atmospheric flux for the given energy and zenith angle
        :param lg_energy: energy grid
        :param zenith_angle: zenith angle grid coincident to energy grid
        :return: differential spectral atmospheric flux grid
        """
        # use the given spectrum up to 1e6 GeV
        pass_band = lg_energy < self.lg_energy[-1]
        cosine = -np.cos(zenith_angle)
        return 10**self.rbc.ev(cosine, lg_energy) * pass_band

    def estimate_error(self) -> float:
        """
        Return relative error of the polynomial interpolation
        :return: float error value > 0
        """
        xg, yg = np.meshgrid(self.cosines, self.given_lg_energy, indexing='ij')
        data_int = 10 ** self.rbc.ev(xg, yg)
        return np.std(data_int / self.data - 1)


if __name__ == '__main__':
    print("Not for direct use")
