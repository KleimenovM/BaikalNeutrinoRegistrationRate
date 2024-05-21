# Neutrino through Earth propagation
# Telescope class description
import numpy as np

import ROOT as rt
import os
from scipy.interpolate import RegularGridInterpolator

from src.source import Source
from src.tools import sph_coord, rot_matrix, deg_to_rad


class Telescope:
    """
    This class describes a typical neutrino telescope
    """

    def __init__(self, name: str, latitude: float,
                 lg_energy: np.ndarray, ef_area_table: np.ndarray, brd_angle_cosine=None, cosines=None):
        """
        Set a telescope
        :param name:                (str) telescope title
        :param latitude:            (float) telescope latitude, rad
        :param lg_energy:           (np.ndarray) lg energy table
        :param ef_area_table:       (np.ndarray) effective area table
        :param brd_angle_cosine:    *optional*, (float) border visibility height cos(90° - zenith angle)
        :param cosines:              (list) zenith angles grid for effective area
        """
        self.name = name
        self.phi = latitude
        self.ef_area_table = ef_area_table
        self.lg_energy = lg_energy
        self.energy = 10**self.lg_energy
        self.ef_area = None
        self.cosines = cosines

        if brd_angle_cosine is None:
            self.brd_cosine = 0.0  # z_brd = pi/2
        else:
            self.brd_cosine = brd_angle_cosine

        if self.cosines:
            self.complex_effective_area()
        else:
            self.simple_effective_area()

    def get_orbit_parametrization(self, s_delta: float, s_alpha: float, angular_precision: int):
        """
        Get the splitting of the source's orbit
        :param s_delta: source's declination (rad)
        :param s_alpha: source's right ascension (rad)
        :param angular_precision: (int) number of splits
        :return: 3D-vector (x, y, z), source's height, source's azimuth
        """
        psi = s_alpha + np.linspace(0, 2 * np.pi, angular_precision)  # parametrization
        vec = sph_coord(r=1, theta=s_delta, phi=psi)
        rm = rot_matrix(self.phi)
        vec = rm.dot(vec)
        theta = np.arccos(vec[2])
        return vec, theta, psi

    def source_available_time(self, source: Source):
        """
        Get the source available time above the border height (standard height 0° <-> zenith angle 90°)
        :param source: (Source) point-like object of the standard class
        :return: (float) visible time and period ratio
        """
        m = 1000
        vec, theta, psi = self.get_orbit_parametrization(s_delta=source.delta,
                                                         s_alpha=source.ra,
                                                         angular_precision=m)
        theta_good = theta > self.brd_cosine
        return np.sum(theta_good) / m

    def simple_effective_area(self, angular_precision: int = 180):
        """
        Set only energy-dependent effective area 
        :param angular_precision: (int) number of formal zenith angle splits for interpolation
        :return: an effective area function (cosine, lg_energy) -> lg_ef_area
        """
        value = self.ef_area_table  # ef_area value

        zenith_cosine_parametrization = np.linspace(-1, 1, angular_precision)
        ef_area_parametrization = np.zeros([angular_precision, value.size])

        # simple method: if zenith angle cosine < border cosine, ef. area = 0
        for i, z_cos in enumerate(zenith_cosine_parametrization):
            if z_cos < self.brd_cosine:
                ef_area_parametrization[i] = value

        xy = (zenith_cosine_parametrization, self.lg_energy)

        self.ef_area = RegularGridInterpolator(xy, np.log10(ef_area_parametrization + 1e-30), method='linear')
        return

    def complex_effective_area(self):
        """
        Set energy and zenith-angle dependent effective area
        :return: an effective area function (cos(z), lg(e)) -> lg(ef_area)
        """
        self.cosines = np.array(self.cosines)

        xy = (self.cosines, self.lg_energy)

        self.ef_area = RegularGridInterpolator(xy, np.log10(self.ef_area_table + 1e-30), method='linear',
                                               bounds_error=False, fill_value=0.0)
        return

    def effective_area(self, cosine, energy):
        xv = np.array(np.meshgrid(cosine, energy, indexing='ij')).T
        return 10 ** self.ef_area(xv).T


class RootTelescopeConstructor:
    def __init__(self, telescope_name, hist_name):
        script_dir = os.path.dirname(__file__)
        self.path = script_dir + "/data/telescopes/" + telescope_name + "/"
        self.hist_name = hist_name

        # check if there is an info.txt file
        assert os.path.exists(self.path + "info.txt"), "no info.txt file"

        self.name: str = ""
        self.latitude: float = 0.0
        self.cosines: list[float] = []
        self.filenames: list[str] = []

        self.simple = self.analyze_info()

    def analyze_info(self):
        file = open(self.path + "info.txt")
        lines = file.readlines()

        self.name = lines[0].strip()
        self.latitude = deg_to_rad([int(v) for v in lines[1].strip().split(',')])

        for line in lines[3:]:
            line = line.strip().split('\t')
            self.cosines.append(float(line[0]))
            self.filenames.append(line[1])

        if len(self.cosines) == 1:
            return True
        return False

    def get_simple_telescope(self) -> Telescope:
        f = rt.TFile(self.path + self.filenames[0], "read")
        hist = f.Get(self.hist_name)
        n = len(hist)

        # low_end, width, value
        lg_energy: np.ndarray = np.zeros(n)
        data: np.ndarray = np.zeros(n)

        for i in range(n):
            lg_energy[i] = hist.GetBinLowEdge(i) + 1/2 * hist.GetBinWidth(i)  # middle of the bin
            data[i] = hist.GetBinContent(i)  # bin average value

        return Telescope(name=self.name,
                         latitude=self.latitude,
                         lg_energy=lg_energy,
                         ef_area_table=data,
                         brd_angle_cosine=self.cosines[0])

    def get_complex_telescope(self) -> Telescope:
        lg_energy = []
        data = []

        for i, filename in enumerate(self.filenames):
            file = rt.TFile(self.path + filename, "read")
            hist = file.Get(self.hist_name)
            n = len(hist)

            if i == 0:
                e_i = np.zeros(n-1)
                for j in range(n-1):
                    e_i[j] = hist.GetBinLowEdge(j) + 1/2 * hist.GetBinWidth(j)

                lg_energy = e_i

            data_i = np.zeros(n-1)
            for j in range(n-1):
                data_i[j] = hist.GetBinContent(j)  # bin average value

            data.append(data_i)

        data = np.array(data)

        # changing data for appropriate interpolation
        m = len(self.cosines)
        n = len(lg_energy)
        new_cosines = np.zeros(m+1)
        new_data = np.zeros([m+1, n])

        cos_half_width = (self.cosines[1] - self.cosines[0]) / 2
        new_cosines[0] = self.cosines[0] - cos_half_width
        new_cosines[1:] = np.array(self.cosines) + cos_half_width

        new_data[0, :] = (3 * data[0, :] + data[1, :]) / 4
        new_data[1:-1, :] = (data[:-1, :] + data[1:, :]) / 2
        new_data[-1, :] = (data[-2, :] + 3 * data[-1, :]) / 4

        return Telescope(name=self.name,
                         latitude=self.latitude,
                         lg_energy=lg_energy,
                         ef_area_table=new_data,
                         cosines=new_cosines.tolist())

    def get(self):
        if self.simple:
            return self.get_simple_telescope()
        return self.get_complex_telescope()


if __name__ == '__main__':
    print("Not for direct use")
