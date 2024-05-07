import os
import numpy as np


class TransmissionFunction:
    """
    This class takes a matrix, calculated with nuFate, from data_mod.npy
    and interpolates it in order to simplify the usage of nuFate package

    To use the interpolation, there is the -convolution- function which
    results into attenuated flux given an initial flux and zenith angle
    """

    def __init__(self, nuFate_method: int):
        """
        :param nuFate_method: (int) 0 -> cascades only, 1 -> secondaries included, 2-> tau neutrino
        """
        # load data.npy file
        script_dir = os.path.dirname(__file__)
        folder = "/data/transmission_matrix/"
        table_names = ["data_mod_0.npy", "data_mod_1.npy", "data_mod_2.npy"]
        self.table_data = np.load(script_dir + folder + table_names[nuFate_method])

        # energies and zenith angle boundaries
        theta_min, theta_max, self.m = np.pi/2, np.pi, 180
        lg_e_min, lg_e_max, self.n = 3, 10, 200

        # angle and log_energy axis
        self.angles = np.linspace(theta_min, theta_max, self.m)
        self.lg_energy = np.linspace(lg_e_min, lg_e_max, self.n)
        self.energy = 10 ** self.lg_energy

    def angle_interpolated_matrix(self, angle: float):
        """
        Perform a linear interpolation along the -angle- axis
        :param angle: (float) an input value
        :return: (np.ndarray) - a 2D transmission matrix for a point (energy_in, energy_out)
        """
        if angle <= np.pi/2:        # particle comes from the upper hemisphere
            return np.eye(self.n)   # no attenuation

        transmission_table = self.table_data

        j = self.angles[angle > self.angles].size - 1
        t = (angle - self.angles[j]) / (self.angles[j + 1] - self.angles[j])

        transmission_matrix = transmission_table[j] * (1 - t) + transmission_table[j + 1] * t
        return transmission_matrix

    def convolution(self, angle, input_spectrum):
        """
        Perform the input spectrum convolution with the Earth's transmission function
        :param angle: zenith angle
        :param input_spectrum: numpy array (self.n) - flux on angle & energy dependence
        :return: numpy array (self.n) - attenuated flux on energy dependence
        """
        if angle <= np.pi/2:
            return input_spectrum

        angle_transmission = self.angle_interpolated_matrix(angle)
        product = np.dot(input_spectrum, angle_transmission)

        return product


if __name__ == '__main__':
    print("Not for direct use")
