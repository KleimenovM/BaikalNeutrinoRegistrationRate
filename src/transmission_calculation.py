# Earth Transmission Function (ETF) calculation
# Allowed energies from 10^3 to 10^10 GeV
# This file creates a data.npy file which contains points of the ETF

# Attention! It takes several minutes to execute this file!

import numpy as np
import ROOT as rt

from nuFATE.cascade import get_eigs as get_eigs1
from nuFATE.cascade_secs import get_eigs as get_eigs2
import nuFATE.earth as earth

from tools import extrapolating_spline

# Avogadro's number
N_A = 6.0221415e23


class TransmissionCalculator:
    def __init__(self, nuFateMethod: int = 1, flavor: int = 2, width=1.0):
        lg_min, lg_max, self.n = 3, 10, 200
        self.d_lg_e = (lg_max - lg_min) / self.n
        self.lg_e = np.linspace(3, 10, 200)
        self.energy = 10 ** self.lg_e

        self.width = width

        z_angle_min, z_angle_max, self.m = np.pi / 2, np.pi, 180
        self.z_angles = np.linspace(z_angle_min, z_angle_max, self.m)

        self.hist_names = ["no_secs", "emu_secs", "tau_secs"]
        self.nuFate_method = nuFateMethod
        self.flavor = flavor
        return

    def set_delta_function(self, e):
        """
        Returns a delta-spectrum with peak at e
        :param e: delta function parameter
        :return: np.ndarray -- delta spectrum (_____|_______)
        """
        log10_e = np.log10(e)
        delta_position = np.all([self.lg_e - self.d_lg_e / 2 <= log10_e, self.lg_e + self.d_lg_e / 2 > log10_e],
                                axis=0)
        return delta_position / self.width

    def prepare_root_hist(self):
        root_hist = rt.TH3F(self.hist_names[self.nuFate_method],
                            self.hist_names[self.nuFate_method],
                            self.m - 1, self.z_angles,
                            self.n - 1, self.lg_e,
                            self.n - 1, self.lg_e)

        root_hist.GetXaxis().SetTitle("zenith angle, rad")
        root_hist.GetYaxis().SetTitle("lg(E_in / GeV)")
        root_hist.GetZaxis().SetTitle("lg(E_out / GeV)")
        return root_hist

    @staticmethod
    def get_att_value_no_secs(w, v, ci, energy_nodes, zenith, E, phi_in, absolute=False):
        """
        This function calculates attenuation value with the use of NuFATE methods
        :param w: transmission equation eigenvalues - 1
        :param v: transmission equation eigenvalues - 2
        :param ci: transmission equation solution
        :param energy_nodes: energy bins log_10(GeV)
        :param zenith: zenith angle (rad, > pi/2)
        :param E: energy to interpolate
        :param phi_in: inner flux
        :param absolute: marker to fix whether the resulting flux is relative or absolute
        :return: attenuated flux (relative or absolute) at energy E
        """
        t = earth.get_t_earth(zenith) * N_A  # g/ cm^2
        if absolute:
            phi_sol = np.dot(v, (ci * np.exp(w * t))) * energy_nodes ** (-2)  # this is the attenuated flux
        else:
            phi_sol = np.dot(v, (ci * np.exp(w * t))) / phi_in  # this is phi/phi_initial, i.e. the relative attenuation
        return extrapolating_spline(E, energy_nodes, phi_sol, if_delta=absolute)

    @staticmethod
    def get_att_value_secs(w, v, ci, energy_nodes, zenith, E, phi_in, if_tau=False, absolute=False):
        """
        This function calculates attenuation value with the use of NuFATE methods
        :param w: transmission equation eigenvalues - 1
        :param v: transmission equation eigenvalues - 2
        :param ci: transmission equation solution
        :param energy_nodes: energy bins log_10(GeV)
        :param zenith: zenith angle (rad, > pi/2)
        :param E: energy to interpolate
        :param phi_in: inner flux
        :param if_tau: marker to return the flux for the given particle or for tau-particle
        :param absolute: marker to fix whether the resulting flux is relative or absolute
        :return: attenuated flux (relative or absolute) at energy E
        """
        t = earth.get_t_earth(zenith) * N_A  # g / cm^2
        if absolute:
            e_nd = np.hstack([energy_nodes, energy_nodes])
            phi_sol = np.dot(v, (ci * np.exp(w * t))) * e_nd ** (-2)  # this is the attenuated flux
        else:
            phi_sol = np.dot(v, (ci * np.exp(w * t))) / phi_in  # this is phi/phi_0, i.e. the relative attenuation

        if if_tau:
            phi_sol1 = phi_sol[200:400]  # the tau bit.
            return extrapolating_spline(E, energy_nodes, phi_sol1, if_delta=absolute)

        phi_sol1 = phi_sol[0:200]  # the non-tau bit.
        return extrapolating_spline(E, energy_nodes, phi_sol1, if_delta=absolute)

    def get_eigs(self, gamma, pure_spectrum: bool = True):
        if self.nuFate_method == 0:
            return get_eigs1(self.flavor, gamma, "nuFATE/NuFATECrossSections.h5", pure_spectrum=True)

        return get_eigs2(self.flavor, gamma, "nuFATE/NuFATECrossSections.h5", pure_spectrum=True)

    def get_att_values(self, w, v, ci, energy_nodes, zenith, phi_in):
        if self.nuFate_method == 0:
            return self.get_att_value_no_secs(w, v, ci, energy_nodes, zenith, self.energy, phi_in, True)
        if self.nuFate_method == 1:
            return self.get_att_value_secs(w, v, ci, energy_nodes, zenith, self.energy, phi_in, True, True)
        if self.nuFate_method == 2:
            return self.get_att_value_secs(w, v, ci, energy_nodes, zenith, self.energy, phi_in, False, True)
        else:
            return .0

    @staticmethod
    def save_npy(filename: str, data: np.ndarray):
        """
        Saves the attenuation matrix in a numpy file
        :param filename: "folder/filename.npy"
        :param data: list of 3D numpy arrays
        :return:
        """
        np.save(filename, data)
        return

    @staticmethod
    def save_root(filename, data, names):
        """
        Saves the attenuation matrix in a root file as a TH3F
        :param names: titles for histograms
        :param filename: "folder/filename.root"
        :param data: 3D numpy matrix
        :return:
        """

        file = rt.TFile.Open(filename, "RECREATE")
        for k in range(len(data)):
            file.WriteObject(data[k], names[k])

        file.Close()
        return

    def run(self, if_npy: bool, if_root: bool):
        # numpy matrix
        att_matrix = np.zeros([self.m, self.n, self.n])

        # root histogram
        root_hist = self.prepare_root_hist()

        for i in range(self.n):  # split by energy
            # create a delta_function
            gamma = self.set_delta_function(self.energy[i])

            w, v, ci, energy_nodes, phi_0 = self.get_eigs(gamma)

            for j in range(self.m):  # split by angle
                if j % 30 == 0:
                    print(f"{i}-{j}")
                z_j = self.z_angles[j]

                attenuation_matrix = self.get_att_values(w, v, ci, energy_nodes, z_j, phi_0)

                # FILL THE MATRIX
                if if_npy:
                    att_matrix[j, i] = attenuation_matrix

                # FILL ROOT HISTOGRAMS
                if if_root:
                    for k in range(self.n):  # split by energy
                        root_hist.Fill(z_j, self.lg_e[i], self.lg_e[k], attenuation_matrix[k])

        filename = f"data/transmission_matrix/data_mod_{self.nuFate_method}"

        if if_npy:
            # save as a .npy file
            self.save_npy(f"{filename}.npy", att_matrix)

        if if_root:
            # save as a .root file
            self.save_root(f"{filename}.root", root_hist, self.hist_names[self.nuFate_method])

        return


if __name__ == '__main__':
    transmission_calculator = TransmissionCalculator(nuFateMethod=2, flavor=2)
    transmission_calculator.run(if_root=False, if_npy=True)
