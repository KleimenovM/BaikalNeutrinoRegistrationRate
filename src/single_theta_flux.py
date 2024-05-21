import numpy as np
from scipy.interpolate import interp1d

from src.tools import smart_division
from src.telescope import Telescope
from src.transmission_function import TransmissionFunction


class SingleThetaFlux:
    def __init__(self, initial_flux: np.ndarray, zenith: float,
                 telescope: Telescope, tf: TransmissionFunction,
                 mid_border: float = 1., low_border: float = -1.):
        # telescope
        self.telescope = telescope

        # source
        self.initial_flux = initial_flux
        self.zenith_angle = zenith

        # Earth
        self.tf = tf
        self.mid_border = mid_border
        self.low_border = low_border

    @staticmethod
    def attenuation_extrapolation(lg_energy: np.ndarray, spectra_ratio: np.ndarray, t00: float):
        """
            Extrapolates attenuated spectrum to the region where nuFate cannot perform calculations
            :param lg_energy: lg of neutrino energy
            :param spectra_ratio: relative attenuated spectrum
            :param t00: low-energy zero attenuation border
            :return: extrapolation function: e -> relative flux at (1e^{t00}, 1e3)
            """
        # estimate derivatives
        t0, t1, t2 = lg_energy[0:3]
        f0, f1, f2 = np.log10(spectra_ratio[0:3])

        g0 = f0
        g1 = (f1 - f0) / (t1 - t0)
        g2 = (f2 - 2 * f1 + f0) / (t1 - t0) ** 2

        g0 *= (t0 - t00) ** (-4)
        g1 *= (t0 - t00) ** (-3)
        g2 *= (t0 - t00) ** (-2)

        # find a, b, c parameters
        matrix = np.array([[3, -2, 0.5],
                           [-8 * t0 + 2 * t00, 5 * t0 - t00, -t0],
                           [6 * t0 ** 2 - 4 * t0 * t00 + t00 ** 2, -3 * t0 ** 2 + t0 * t00, t0 ** 2 / 2]])
        g_vector = np.array([g0, g1, g2])

        abc_vector = matrix.dot(g_vector)
        a, b, c = abc_vector

        # set extrapolation_function
        def extrapolation_function(e):
            t = np.log10(e)
            return 10 ** ((t - t00) ** 2 * (a * t ** 2 + b * t + c))

        return extrapolation_function

    @staticmethod
    def united_parts_interpolation(e1, e2, e3, f1, f2, f3):
        """
        Unites three separate relative spectrum parts into one wide spectrum
        :param e1: low energy (zero attenuation)
        :param e2: middle energy (extrapolation region)
        :param e3: high energy (nuFate calculations)
        :param f1: identical unit value (no attenuation)
        :param f2: extrapolated flux
        :param f3: nuFate calculations
        :return: extrapolation function: e (in 1e-1, 1e10) -> relative flux
        """
        e = np.hstack([e1, e2, e3])
        f = np.hstack([f1, f2, f3])
        return interp1d(e, f)

    def convoluted_flux(self):
        """
        Step 1. Convolution -> Getting relative flux after transmission of the Earth
        """
        convoluted = self.tf.convolution(self.zenith_angle, self.initial_flux)
        return smart_division(convoluted, self.initial_flux)

    def interpolated_flux(self, relative_flux: np.ndarray):
        """
        Step 2. Low energy extrapolation
        :param relative_flux:
        :return:
        """
        # Step 2. Extrapolation
        de = self.tf.lg_energy[1] - self.tf.lg_energy[0]
        e_mid = 10 ** (np.arange(self.mid_border, self.tf.lg_energy[0], de))
        mid_e_extrapolation = self.attenuation_extrapolation(self.tf.lg_energy, relative_flux, t00=self.mid_border)

        e_low = 10 ** (np.arange(self.low_border, self.mid_border, de))
        low_e_extrapolation = np.ones([e_low.size])

        total_relative_spectrum = self.united_parts_interpolation(e1=e_low, f1=low_e_extrapolation,
                                                                  e2=e_mid, f2=mid_e_extrapolation(e_mid),
                                                                  e3=self.tf.energy, f3=relative_flux)

        return total_relative_spectrum

    def eff_area_step(self, total_relative_spectrum_function):
        # 3.1. Data generation from interpolated values
        lg_energy = self.telescope.lg_energy
        energy = self.telescope.energy
        relative_spectrum = total_relative_spectrum_function(energy)

        # 3.2. Multiplication with effective area
        effective_area = self.telescope.effective_area(np.cos(self.zenith_angle), lg_energy)

        return relative_spectrum * effective_area

    def calculate(self):
        """
        Full relative spectrum for a single zenith angle computation
        """
        conv_f = self.convoluted_flux()
        interp_f = self.interpolated_flux(conv_f)
        final_spectrum = self.eff_area_step(interp_f)
        return final_spectrum


if __name__ == '__main__':
    print("Not for direct use!")
