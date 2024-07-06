import numpy as np

from src.background import Background, Atmosphere, AstrophysicalBackground
from src.single_theta_flux import SingleThetaFlux
from src.source import Source
from src.source_extended import ExtendedSource
from src.telescope import Telescope
from src.transmission_function import TransmissionFunction


class BasicPointFlux:
    def __init__(self, telescope: Telescope, flux_on_energy_function,
                 tf: TransmissionFunction, atmosphere: Atmosphere,
                 astro: AstrophysicalBackground,
                 declination: float = 0.0, right_ascension: float = 0.0,
                 latitude: float = .0, longitude: float = .0,
                 area: float = 1.0, angular_precision: int = 180,
                 no_attenuation_brd: float = 1.0, spectrum_brd: float = -1.0):
        # basic parameters
        self.telescope = telescope  # selected telescope object
        self.tf = tf  # transmission function class

        # extrapolation boundaries
        self.no_attenuation_border = no_attenuation_brd
        self.spectrum_border = spectrum_brd

        # equatorial coordinates
        self.declination = declination  # source declination
        self.right_ascension = right_ascension  # source right ascension
        # galactic coordinates
        self.galactic_latitude = latitude
        self.galactic_longitude = longitude
        # trajectory parameter
        self.angular_precision = angular_precision
        self.zenith_angles = self.telescope.get_orbit_parametrization(s_delta=self.declination,
                                                                      s_alpha=self.right_ascension,
                                                                      angular_precision=self.angular_precision)[1]

        # flux and area
        self.flux_on_energy = flux_on_energy_function
        self.area = area
        d_lg_e = self.telescope.lg_energy[1] - self.telescope.lg_energy[0]
        self.de = 10 ** (self.telescope.lg_energy + d_lg_e) - self.telescope.energy  # bin widths

        # atmospheric neutrinos
        self.atmosphere = atmosphere
        # astrophysical neutrinos
        self.astro = astro

    def get_signal(self):
        ref_initial_flux = self.flux_on_energy(self.telescope.energy,
                                               self.galactic_longitude,
                                               self.galactic_latitude)
        initial_flux = self.flux_on_energy(self.tf.energy,
                                           self.galactic_longitude,
                                           self.galactic_latitude)

        total_flux = np.zeros([self.angular_precision, self.telescope.energy.size])

        for i, z_i in enumerate(self.zenith_angles):
            stf = SingleThetaFlux(initial_flux=initial_flux,
                                  zenith=z_i,
                                  telescope=self.telescope,
                                  tf=self.tf)

            relative_flux_i = stf.calculate()

            total_flux[i, :] = relative_flux_i * ref_initial_flux

        return np.mean(total_flux, axis=0) * self.de

    def get_background(self, bg_type: Background):

        total_flux = np.zeros([self.angular_precision, self.telescope.energy.size])

        for i, z_i in enumerate(self.zenith_angles):
            ref_initial_flux_i = bg_type.flux(self.telescope.lg_energy, z_i) * self.area
            initial_flux_i = bg_type.flux(self.tf.lg_energy, z_i) * self.area

            stf = SingleThetaFlux(initial_flux=initial_flux_i,
                                  zenith=z_i,
                                  telescope=self.telescope,
                                  tf=self.tf)

            relative_flux_i = stf.calculate()

            total_flux[i, :] = relative_flux_i * ref_initial_flux_i

        return np.mean(total_flux, axis=0) * self.de

    def get_atmospheric_background(self):
        return self.get_background(bg_type=self.atmosphere)

    def get_astrophysical_background(self):
        return self.get_background(bg_type=self.astro)


class PointSourceFlux:
    def __init__(self, telescope: Telescope, source: Source,
                 tf: TransmissionFunction = None,
                 atm: Atmosphere = None,
                 astro: AstrophysicalBackground = None,
                 angular_resolution=1.0, angular_precision: int = 180,
                 value: float = 1.0, nuFate_method: int = 1,
                 no_attenuation_brd: float = 1.0, spectrum_brd: float = -1.0,
                 lg_energy_border: float = None):
        # area from angular resolution
        area = np.pi * np.deg2rad(angular_resolution) ** 2

        if tf is None:
            tf = TransmissionFunction(nuFate_method=nuFate_method)

        if atm is None:
            atm = Atmosphere()

        if astro is None:
            astro = AstrophysicalBackground()

        self.bpf = BasicPointFlux(telescope=telescope, tf=tf, atmosphere=atm, astro=astro,
                                  flux_on_energy_function=source.flux_on_energy_function,
                                  declination=source.delta, right_ascension=source.ra,
                                  area=area, angular_precision=angular_precision,
                                  no_attenuation_brd=no_attenuation_brd, spectrum_brd=spectrum_brd)

        # constants
        self.year_seconds = 3600 * 24 * 365
        self.value = value

        # energy border
        if lg_energy_border:
            self.lg_e_border = telescope.lg_energy > lg_energy_border
        else:
            self.lg_e_border = 1.0

    def signal(self):
        signal = self.bpf.get_signal()
        return signal * self.year_seconds * self.value * self.lg_e_border

    def background(self):
        bg = self.bpf.get_atmospheric_background() + self.bpf.get_astrophysical_background()
        return bg * self.year_seconds * self.value * self.lg_e_border

    def total_signal(self, rnd: int = 2):
        return np.round(np.sum(self.signal()), rnd)

    def total_background(self, rnd: int = 2):
        return np.round(np.sum(self.background()), rnd)


class ExtendedSourceFlux:
    def __init__(self, telescope: Telescope, ext_source: ExtendedSource,
                 tf: TransmissionFunction = None, atm: Atmosphere = None,
                 astro: AstrophysicalBackground = None,
                 angular_precision: int = 180,
                 value: float = 1.0, nuFate_method: int = 1,
                 no_attenuation_brd: float = 1.0, spectrum_brd: float = -1.0,
                 lg_energy_border: float = None):
        # telescope
        self.telescope = telescope
        # source
        self.ext_source = ext_source
        # area initialization
        self.areas = self.ext_source.cell_areas()

        if tf is None:
            self.tf = TransmissionFunction(nuFate_method=nuFate_method)

        if atm is None:
            self.atm = Atmosphere()

        if astro is None:
            self.astro = AstrophysicalBackground()

        self.silent: bool = False

        self.angular_precision = angular_precision

        self.no_attenuation_brd = no_attenuation_brd
        self.spectrum_brd = spectrum_brd

        # constants
        self.year_seconds = 3600 * 24 * 365
        self.value = value

        # energy border
        if lg_energy_border:
            self.lg_e_border = telescope.lg_energy > lg_energy_border
        else:
            self.lg_e_border = 1.0

    def set_bpf(self, i_x, i_y):
        ll_i = self.ext_source.galactic_sl_grid[i_x, i_y]
        b_i = self.ext_source.galactic_sb_grid[i_x, i_y]
        dec_i = self.ext_source.equatorial_s_dec_grid[i_x, i_y]
        ra_i = self.ext_source.equatorial_s_ra_grid[i_x, i_y]

        # area setting
        area_ij = self.areas[i_x, i_y]

        return BasicPointFlux(telescope=self.telescope, tf=self.tf, atmosphere=self.atm, astro=self.astro,
                              flux_on_energy_function=self.ext_source.flux_on_energy_function,
                              declination=dec_i, right_ascension=ra_i,
                              latitude=b_i, longitude=ll_i,
                              area=area_ij, angular_precision=self.angular_precision,
                              no_attenuation_brd=self.no_attenuation_brd, spectrum_brd=self.spectrum_brd)

    def signal(self):
        result = np.zeros([self.ext_source.s_num, self.telescope.energy.size])

        if not self.silent:
            print("Signal")

        for i_x in range(self.ext_source.snx):
            for i_y in range(self.ext_source.sny):
                # status bar
                i = i_x * self.ext_source.sny + i_y
                if not self.silent and i % 20 == 0:
                    print(f"-{i}-", end='')

                bpf = self.set_bpf(i_x, i_y)

                result[i, :] = bpf.get_signal()

        if not self.silent:
            print("")

        return np.sum(result, axis=0) * self.year_seconds * self.value * self.lg_e_border

    def background(self, if_atm: bool = True, if_astro: bool = True):
        result = np.zeros([self.ext_source.s_num, self.telescope.energy.size])

        if not self.silent:
            print("Background")

        for i_x in range(self.ext_source.snx):
            for i_y in range(self.ext_source.sny):
                # status bar
                i = i_x * self.ext_source.sny + i_y
                if not self.silent and i % 20 == 0:
                    print(f"-{i}-", end='')

                bpf = self.set_bpf(i_x, i_y)

                result[i] = bpf.get_atmospheric_background() * if_atm + bpf.get_astrophysical_background() * if_astro

        if not self.silent:
            print("")

        return np.sum(result, axis=0) * self.year_seconds * self.value * self.lg_e_border

    def total_signal(self, rnd: int = 2):
        return np.round(np.sum(self.signal()), rnd)

    def total_background(self, rnd: int = 2):
        return np.round(np.sum(self.background()), rnd)


if __name__ == '__main__':
    print("Not for direct use")
