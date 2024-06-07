# Neutrino through Earth propagation
# Source class description

import numpy as np
import pandas as pd

from src.tools import deg_to_rad, hours_to_rad


class Source:
    """
    This class describes a typical point-like stellar neutrino source
    """

    def __init__(self, name: str, declination_angle: float, right_ascension_time: float,
                 k0: float, gamma: float, e_cut=None, beta=None):
        """
        Set a source
        :param name: (str) source title
        :param declination_angle: (float) declination, radians
        :param right_ascension_time: (float) right ascension, hours
        :param k0: spectral normalization constant, 1e-11 TeV-1 s-1 cm-2
        :param gamma: spectral index, > 0
        :param e_cut: cut energy, GeV
        :param beta: exponential factor, > 0
        """
        # position parameter
        self.name = name
        self.delta = declination_angle  # rad
        self.declination = np.round(np.rad2deg(self.delta), 2)
        self.right_ascension = right_ascension_time
        self.ra = hours_to_rad(self.right_ascension)

        # spectrum parameters
        self.k0 = k0  # 1e-11 TeV-1 s-1 cm-2
        self.gamma = gamma  # spectrum attenuation coefficient

        # cut-off parameters
        self.e_cut = e_cut  # GeV
        self.beta = beta  # GeV

    def info(self):
        """
        Get information about the source: its name, declination and spectral parameters
        :return:
        """
        print(f"{self.name}, declination: {self.delta}")
        print(f"k0 = {self.k0} GeV-1 s-1 m-2, gamma = {self.gamma}")
        if not np.isnan(self.e_cut) and not np.isnan(self.beta):
            print(f"e_cut: {self.e_cut} GeV, beta: {self.beta}")
        pass

    def flux_on_energy_function(self, energy, *args):
        """
        Get the standard flux (in GeV-1 cm-2 s-1) for the source
        :param energy: (float or np.ndarray) - input energy, GeV
        :return: (float or np.ndarray) flux  - GeV-1 cm-2 s-1
        """
        # just a polynomial spectrum
        if np.isnan(self.e_cut) or np.isnan(self.beta):
            return self.k0 * (energy / 1000) ** (-self.gamma)

        # exponential cutoff
        return self.k0 * (energy / 1000) ** (-self.gamma) * np.exp(-(energy / self.e_cut) ** self.beta)


def set_a_source(line) -> Source:
    return Source(name=line['Source'],
                  declination_angle=deg_to_rad([line['delta']]),  # deg to rad
                  right_ascension_time=line['RA'],
                  k0=line['k0'] * 1e-11 * 1e4 * 1e-3,  # TeV-1 s-1 cm-2 to GeV-1 s-1 m-2
                  gamma=line['gamma'],
                  e_cut=line['e_cut'] * 1e3,  # TeV to GeV
                  beta=line['beta'])


def get_sources(filename: str) -> list[Source]:
    """
    This function provides parameters of neutrino sources
    k0 (TeV-1 cm-2 s-1) [into standard * 1e4]
    gamma (no dim)
    e_cut (TeV)
    beta (no dim)
    :param filename:
    :return:
    """
    data = pd.read_csv(filename, sep=',')

    sources = []
    for i in range(data.shape[0]):
        line_i = data.T[i].loc[['Source', 'delta', 'RA', 'k0', 'gamma', 'e_cut', 'beta']]
        source_i: Source = set_a_source(line_i)
        sources.append(source_i)

    return sources


if __name__ == '__main__':
    print("Not for direct use")
