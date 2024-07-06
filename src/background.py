import os
import abc
import numpy as np

from scipy.interpolate import RectBivariateSpline


class Background(abc.ABC):
    """
    An abstract class for background calculation
    """
    @abc.abstractmethod
    def flux(self, lg_energy, zenith_angle):
        """
        Abstract method for background class
        :param lg_energy: (np.ndarray) available energies, lg(E/GeV)
        :param zenith_angle: (float) given zenith angle (rad)
        :return: (np.array) flux distribution for the given zenith angle and energy range
        """
        return .0


class AstrophysicalBackground(Background):
    """
    Based on various measured results (see Troitsky, 2023 arXiv:2112.09611)
    allows to calculate
    """
    def __init__(self, method: str = "nu_mu 2019"):
        # for clarification see table 3 of arXiv:2112.09611
        names_dict = {"HESE 2020": 0, "IceCube Cascades 2020": 1, "MESE 2014": 2, "Inelasticity 2018": 3,
                      "nu_mu 2016": 4, "nu_mu 2019": 5, "ANTARES 2019": 6}
        f0 = [2.12, 1.66, 2.06, 2.04, 0.90, 1.44, 1.5]  # 1e-18 GeV-1 cm-2 s-1 sr-1
        gamma = [2.87, 2.53, 2.46, 2.62, 2.13, 2.28, 2.3]
        f0_ers = [[0.49, -0.54], [0.25, -0.27], [0.4, -0.3], [0.23, -0.21], [0.30, -0.27], [0.25, -0.24], [1.0, -1.0]]
        g_ers = [[0.20, -0.19], [0.07, -0.07], [0.12, -0.12], [0.07, -0.07], [0.13, -0.13], [0.08, -0.09], [0.4, -0.4]]

        index = names_dict.get(method)
        self.f0 = f0[index] * 1e-18 * 1e4  # to GeV-1 m-2 s-1 sr-1
        self.gamma = gamma[index]
        self.f0_err = [f0_ers[index][0] * 1e-18 * 1e4, f0_ers[index][1] * 1e-18 * 1e4]
        self.gamma_err = g_ers[index]

    def flux(self, lg_energy, *args):
        return self.f0 * 10 ** (-self.gamma * (lg_energy - 5))

    def err_flux(self, lg_energy, *args):
        """
        Constructs the 1-sigma error boundary around the measured astrophysical flux
        :param lg_energy: (np.ndarray) lg(E/GeV)
        :param args: redundant parameters
        :return: (2 np.ndarrays) lower and higher boundaries
        """
        h1 = (self.f0 + self.f0_err[0]) * 10 ** (-(self.gamma + self.gamma_err[1]) * (lg_energy - 5))
        h2 = (self.f0 + self.f0_err[1]) * 10 ** (-(self.gamma + self.gamma_err[0]) * (lg_energy - 5))
        h3 = (self.f0 + self.f0_err[1]) * 10 ** (-(self.gamma + self.gamma_err[1]) * (lg_energy - 5))
        h4 = (self.f0 + self.f0_err[0]) * 10 ** (-(self.gamma + self.gamma_err[0]) * (lg_energy - 5))
        low = np.min(np.array([h1, h2, h3, h4]), axis=0)
        high = np.max(np.array([h1, h2, h3, h4]), axis=0)
        return low, high


class Atmosphere(Background):
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
