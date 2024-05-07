# Neutrino through Earth propagation
# Telescope class description
import numpy as np
import ROOT as rt
from scipy.interpolate import RegularGridInterpolator as interp2d
from scipy.interpolate import interp1d

from source import Source
from tools import deg_to_rad, sph_coord, rot_matrix


class Telescope:
    """
    This class describes a typical neutrino telescope
    """

    def __init__(self, name: str, latitude: float, ef_area_table: np.ndarray, brd_height=None, angles=None):
        """
        Set a telescope
        :param name: (str) telescope title
        :param latitude: (float) telescope latitude, rad
        :param ef_area_table: (np.ndarray) effective area table
        :param brd_height: optional, (float) border visibility height (90° - zenith angle)
        :param angles: (list) zenith angles grid for effective area
        """
        self.name = name
        self.phi = latitude
        self.ef_area_table = ef_area_table
        self.lg_energy = None
        self.energy = None
        self.ef_area = None
        self.angles = angles

        if brd_height is None:
            self.brd_angle = 0.0
        else:
            self.brd_angle = brd_height

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
        border_angle = -self.brd_angle  # typically 0.0

        for i, z in enumerate(zenith_parametrization):
            if z < border_angle:
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
        positive_angles = np.arange(np.pi / 2, 0, -np.pi / (2 * m))
        positive_values = np.zeros([m, self.energy.size])

        mod_angles = np.hstack([positive_angles, self.angles])
        mod_values = np.vstack([positive_values, angle_energy_data])

        xy = (mod_angles, self.lg_energy)

        self.ef_area = interp2d(xy, mod_values + 1e-15, method='linear')
        return


class TelescopeConstructor:
    def __init__(self):
        return


def get_simple_telescope(name: str, latitude: list, brd_angle: list,
                         filename: str, histname: str = "hnu") -> Telescope:
    """
    Returns a telescope with given name, declination and source of effective area
    :param brd_angle:
    :param name: string with the telescope's name
    :param latitude: [degrees, minutes, seconds] - telescope's latitude
    :param filename: path to the file with effective area data
    :param histname: name of the histogram with effective area data
    :return:
    """
    f = rt.TFile(filename, "read")
    hist = f.Get(histname)
    n = len(hist)

    # low_end, width, value
    data: np.ndarray = np.zeros([3, n])

    for i in range(n):
        data[0, i] = hist.GetBinLowEdge(i)  # low level
        data[1, i] = hist.GetBinWidth(i)  # bin width
        data[2, i] = hist.GetBinContent(i)  # bin average value

    return Telescope(name=name,
                     latitude=deg_to_rad(latitude),
                     ef_area_table=data,
                     brd_height=deg_to_rad(brd_angle))


def get_simple_telescope_from_txt(name: str, latitude: list, filename: str, brd_angle: list):
    """
    Returns a telescope with given name, declination and source of effective area
    :param brd_angle: effective area cut angle
    :param name: string with the telescope's name
    :param latitude: [degrees, minutes, seconds] - telescope's latitude
    :param filename: path to the file with effective area data
    :return:
    """
    lg_e, lg_area = np.loadtxt(filename, skiprows=1, unpack=True)

    lg_e_indices = np.argsort(lg_e)
    lg_e, lg_area = lg_e[lg_e_indices], lg_area[lg_e_indices]

    f_a = interp1d(lg_e, lg_area)

    lg_e_ref = np.linspace(min(lg_e), max(lg_e), 10000)
    lg_a_ref = f_a(lg_e_ref)

    return Telescope(name=name,
                     latitude=deg_to_rad(latitude),
                     ef_area_table=np.array([lg_e_ref, lg_e_ref, 10 ** lg_a_ref]),
                     brd_height=deg_to_rad(brd_angle))


def get_complex_telescope(name: str, latitude: list[int],
                          filenames: list[str], angles: list[float],
                          histname: str) -> Telescope:
    """
    Returns a telescope with given name, declination and source of effective area dependent on angle
    :param angles: list of angles corresponding with each file
    :param name: string with the telescope's name
    :param latitude: [degrees, minutes, seconds] - telescope's latitude
    :param filenames: path to the files with effective area data
    :param histname: name of the histogram with effective area data
    :return:
    """
    data = []

    for i, fn in enumerate(filenames):
        f = rt.TFile(fn, "read")
        hist = f.Get(histname)
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

    return Telescope(name=name,
                     latitude=deg_to_rad(latitude),
                     ef_area_table=data,
                     angles=angles)


def get_Baikal(folder: str, latitude: list = (51, 46), name_addition: str = "", histname: str = "") -> Telescope:
    """
    Prepare a Baikal telescope based on the standard effective area files
    :param folder: (str) location of the files
    :param latitude: (list) [51, 46] -- Baikal coordinates (51°46' N)
    :param name_addition: (str) name addition dependent line
    :param histname: (str) name of the effective area type (hnu_trigger, hnu_reco, hnu_stdcuts)
    :return: (Telescope) - Baikal-GVD telescope as an object of Telescope class
    """
    filenames = []
    angles = []
    p = 0.0
    filenames.append(f"{folder}/eff_area_{0.0}_{0.1}.root")
    angles.append(0.0)

    while p < 0.99:
        p_0_wr = np.round(p, 1)
        p_1_wr = np.round(p + .1, 1)
        filenames.append(f"{folder}/eff_area_{p_0_wr}_{p_1_wr}.root")
        angle_p = -np.mean((np.arcsin(p), np.arcsin(p + 0.1)))
        angles.append(angle_p)
        p += .1

    filenames.append(filenames[-1])
    angles.append(-np.pi / 2)

    return get_complex_telescope(name='BaikalGVD' + name_addition,
                                 latitude=latitude,
                                 filenames=filenames,
                                 angles=angles,
                                 histname=histname)


if __name__ == '__main__':
    print("Not for direct use")
