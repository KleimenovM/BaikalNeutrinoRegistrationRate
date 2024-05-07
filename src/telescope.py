# Neutrino through Earth propagation
# Telescope class description
import numpy as np
import ROOT as rt
import os
from scipy.interpolate import RegularGridInterpolator as interp2d
from scipy.interpolate import interp1d

from src.source import Source
from src.tools import deg_to_rad, sph_coord, rot_matrix


class Telescope:
    """
    This class describes a typical neutrino telescope
    """

    def __init__(self, name: str, latitude: float, ef_area_table: np.ndarray, brd_angle=None, angles=None):
        """
        Set a telescope
        :param name: (str) telescope title
        :param latitude: (float) telescope latitude, rad
        :param ef_area_table: (np.ndarray) effective area table
        :param brd_angle: optional, (float) border visibility height (90° - zenith angle)
        :param angles: (list) zenith angles grid for effective area
        """
        self.name = name
        self.phi = latitude
        self.ef_area_table = ef_area_table
        self.lg_energy = None
        self.energy = None
        self.ef_area = None
        self.angles = angles

        if brd_angle is None:
            self.brd_angle = 0.0
        else:
            self.brd_angle = brd_angle

        if self.angles:
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
        theta = np.arcsin(vec[2])
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
        theta_good = theta < -self.brd_angle
        return np.sum(theta_good) / m

    def simple_effective_area(self, angle_precision: int = 180):
        """
        Set only energy-dependent effective area 
        :param angle_precision: (int) number of formal zenith angle splits for interpolation 
        :return: 
        """
        self.lg_energy = self.ef_area_table[0]  # + self.ef_area_table[1] / 2  # middle of the bin
        self.energy = 10 ** self.lg_energy
        value = self.ef_area_table[2]  # ef_area value

        zenith_parametrization = np.linspace(-np.pi / 2, np.pi / 2, angle_precision)
        ef_area_parametrization = np.zeros([angle_precision, value.size])

        # simple method: if zenith angle < border angle, ef. area = 0
        border_angle = self.brd_angle  # typically 0.0

        for i, z in enumerate(zenith_parametrization):
            if z > border_angle:
                ef_area_parametrization[i] = value

        xy = (zenith_parametrization, self.lg_energy)

        self.ef_area = interp2d(xy, ef_area_parametrization, method='linear')
        return

    def complex_effective_area(self):
        """
        Set energy and zenith-angle dependent effective area
        :return: 
        """
        self.lg_energy = self.ef_area_table[0]
        self.energy = 10 ** self.lg_energy
        self.angles = np.array(self.angles)

        angle_energy_data = self.ef_area_table[2:]

        n, m = self.energy.size, self.angles.size
        above_angles = np.linspace(0, np.pi/2, m)
        above_values = np.zeros([m, self.energy.size])

        mod_angles = np.hstack([above_angles, self.angles])
        mod_values = np.vstack([above_values, angle_energy_data])

        xy = (mod_angles, self.lg_energy)

        self.ef_area = interp2d(xy, mod_values + 1e-15, method='linear')
        return


class RootTelescopeConstructor:
    def __init__(self, telescope_name, hist_name):
        script_dir = os.path.dirname(__file__)
        self.path = script_dir + "/data/telescopes/" + telescope_name + "/"
        self.hist_name = hist_name

        # check if there is an info.txt file
        assert os.path.exists(self.path + "info.txt"), "no info.txt file"

        self.name: str = ""
        self.latitude: float = 0.0
        self.angles: list[float] = []
        self.filenames: list[str] = []

        self.simple = self.analyze_info()

    def analyze_info(self):
        file = open(self.path + "info.txt")
        lines = file.readlines()

        self.name = lines[0].strip()
        self.latitude = [int(v) for v in lines[1].strip().split(',')]

        for line in lines[3:]:
            line = line.strip().split('\t')
            self.angles.append(np.arccos(float(line[0])))
            self.filenames.append(line[1])

        if len(self.angles) == 1:
            return True
        return False

    def get_simple_telescope(self) -> Telescope:
        f = rt.TFile(self.path + self.filenames[0], "read")
        hist = f.Get(self.hist_name)
        n = len(hist)

        # low_end, width, value
        data: np.ndarray = np.zeros([3, n])

        for i in range(n):
            data[0, i] = hist.GetBinLowEdge(i)  # low level
            data[1, i] = hist.GetBinWidth(i)  # bin width
            data[2, i] = hist.GetBinContent(i)  # bin average value

        return Telescope(name=self.name,
                         latitude=self.latitude,
                         ef_area_table=data,
                         brd_angle=self.angles[0])

    def get_complex_telescope(self) -> Telescope:
        data = []

        for i, filename in enumerate(self.filenames):
            file = rt.TFile(self.path + filename, "read")
            hist = file.Get(self.hist_name)
            n = len(hist)

            data_i = np.zeros(n)

            if i == 0:
                e_i = np.zeros([2, n])
                for j in range(n):
                    e_i[0, j] = hist.GetBinLowEdge(j)
                    e_i[1, j] = hist.GetBinWidth(j)

                data.append(e_i[0])
                data.append(e_i[1])

            for j in range(n):
                data_i[j] = hist.GetBinContent(j)  # bin average value

            data.append(data_i)

        data = np.array(data)

        return Telescope(name=self.name,
                         latitude=self.latitude,
                         ef_area_table=data,
                         angles=self.angles)

    def get(self):
        if self.simple:
            return self.get_simple_telescope()
        return self.get_complex_telescope()


if __name__ == '__main__':
    print("Not for direct use")
