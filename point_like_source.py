# Evaluates average high-energy neutrino spectral attenuation
# for specific sources listed in "data/sources_table.csv"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.atmosphere import Atmosphere
from src.single_theta_flux import SingleThetaFlux
from src.source import get_sources, Source
from src.telescope import Telescope, RootTelescopeConstructor
from src.transmission_function import TransmissionFunction


# from plot_root import draw_root_hist


def one_telescope_full_cycle(source: Source, tf: TransmissionFunction, telescope: Telescope,
                             angular_precision: int = 180) -> np.ndarray:
    """
    Performs a full calculations cycle for one source and one telescope
    :param source: Source - the neutrino source
    :param tf: TransmissionFunction - the Earth transmission function
    :param telescope: Telescope - the telescope for which calculation is performed
    :param angular_precision: int - number of angles to describe the Earth's rotation
    :return: np.ndarray - registered spectrum (registration rate on energy)
    """
    zenith_angles = telescope.get_orbit_parametrization(s_delta=source.delta,
                                                        s_alpha=source.ra,
                                                        angular_precision=angular_precision)[1]

    ref_energy = telescope.energy
    d_lg_e = telescope.lg_energy[1] - telescope.lg_energy[0]
    de = 10 ** (telescope.lg_energy + d_lg_e) - ref_energy

    ref_initial_flux = source.flux_on_energy_function(ref_energy)

    initial_flux = source.flux_on_energy_function(tf.energy)

    m = angular_precision
    n = telescope.lg_energy.size

    total_flux_matrix = np.zeros([m, n])

    for i, z_i in enumerate(zenith_angles):
        stf = SingleThetaFlux(initial_flux=initial_flux,
                              zenith=z_i, telescope=telescope, tf=tf)

        total_flux_matrix[i] = stf.calculate()

    emu_at = total_flux_matrix.mean(axis=0)

    year_seconds = 3600 * 24 * 365
    multiplier = ref_initial_flux * de * year_seconds

    return emu_at * multiplier


def one_telescope_background_cycle(source: Source, tf: TransmissionFunction, telescope: Telescope,
                                   atm: Atmosphere,
                                   angular_resolution: float,
                                   angular_precision: int = 180) -> np.ndarray:
    """
    Performs a full background calculations cycle for one source and one telescope
    :param angular_resolution:  angular resolution in degrees
    :param atm:                 Atmosphere - the source of atmospheric flux spectrum
    :param source:              Source - the neutrino source
    :param tf:                  TransmissionFunction - the Earth transmission function
    :param telescope:           Telescope - the telescope for which calculation is performed
    :param angular_precision:   int - number of angles to describe the Earth's rotation
    :return: np.ndarray - registered spectrum (registration rate on energy)
    """
    zenith_angles = telescope.get_orbit_parametrization(s_delta=source.delta,
                                                        s_alpha=source.ra,
                                                        angular_precision=angular_precision)[1]

    ref_energy = telescope.energy
    ref_lg_energy = telescope.lg_energy
    d_lg_e = telescope.lg_energy[1] - telescope.lg_energy[0]
    de = 10 ** (telescope.lg_energy + d_lg_e) - ref_energy

    m = angular_precision
    n = telescope.lg_energy.size

    total_flux_matrix = np.zeros([m, n])

    mid_border, low_border = 1.0, -1.0

    for i, z_i in enumerate(zenith_angles):
        if z_i < np.pi/2:
            total_flux_matrix[i] = 0.0
        else:
            ref_initial_flux = atm.flux(ref_lg_energy, z_i)
            initial_flux = atm.flux(tf.lg_energy, z_i)

            stf = SingleThetaFlux(initial_flux, z_i, telescope, tf,
                                  mid_border=mid_border, low_border=low_border)

            total_flux_matrix[i] = stf.calculate() * ref_initial_flux

    result = total_flux_matrix.mean(axis=0)

    ds = np.pi * np.deg2rad(angular_resolution) ** 2  # a circle with the radius equal to angular resolution
    year_seconds = 3600 * 24 * 365
    result *= de * year_seconds * ds

    return result


def get_signal_and_background(source: Source, tf: TransmissionFunction, atm: Atmosphere,
                              telescope: Telescope, angular_resolution: float = 0.5, value: float = 1, rnd: int = 2,
                              if_sum: bool = True, lg_e_brd=None):
    """
    Return two registration rates (dN / dT) or dN / (dE dT)
    :param lg_e_brd:            lower energy border
    :param if_sum:              if true the function returns registration rates dN / dT
    :param rnd:                 number of decimals after comma
    :param value:               multiplication coefficient
    :param angular_resolution:  angular resolution in degrees
    :param atm:                 Atmosphere - the source of atmospheric flux spectrum
    :param source:              Source - the neutrino source
    :param tf:                  TransmissionFunction - the Earth transmission function
    :param telescope:           Telescope - the telescope for which calculation is performed
    :return: (float, float) - events rate, background rate
    """
    if lg_e_brd:
        energy_border = telescope.lg_energy > lg_e_brd
    else:
        energy_border = 1

    baikal_t_rel = one_telescope_full_cycle(source=source, tf=tf, telescope=telescope) * energy_border
    baikal_t_bg_rel = one_telescope_background_cycle(source=source, tf=tf, telescope=telescope,
                                                     atm=atm, angular_resolution=angular_resolution) * energy_border

    if if_sum:
        rate = np.round(np.sum(baikal_t_rel * value), rnd)
        rate_bg = np.round(np.sum(baikal_t_bg_rel * value), rnd)
        return rate, rate_bg

    return baikal_t_rel * value, baikal_t_bg_rel * value


def main(if_sum: bool = False,
         if_bg: bool = False,
         source_numbers=None):
    # sources from file "source_table.csv" -- potential high-energy neutrino sources
    if source_numbers is None:
        source_numbers = [x for x in range(12)]
    sources = get_sources("src/data/source_table.csv")

    # zenith-angle-dependent version
    baikal_trigger = RootTelescopeConstructor("baikal_bdt_mk_simp", "hnu_trigger").get()
    baikal_reco = RootTelescopeConstructor("baikal_bdt_mk_simp", "hnu_reco").get()
    baikal_std_cuts = RootTelescopeConstructor("baikal_bdt_mk_simp", "hnu_stdcuts").get()

    # background_model
    ac = Atmosphere()

    # Earth transmission function calculated with nuFate
    tf = TransmissionFunction(nuFate_method=1)

    # number of clusters & time
    value_t, value_r, value_c = 20 * 5, 20 * 5, 20 * 5

    # border energy
    lg_e_brd = 0  # GeV

    m = len(source_numbers)
    names = []
    results = np.zeros([m, 6])

    for i, sn in enumerate(source_numbers):
        source = sources[sn]  # take one source from the list

        r_t, r_t_bg = get_signal_and_background(source=source, tf=tf, atm=ac, telescope=baikal_trigger,
                                                angular_resolution=1, value=value_t, lg_e_brd=lg_e_brd,
                                                rnd=2, if_sum=if_sum)
        r_r, r_r_bg = get_signal_and_background(source=source, tf=tf, atm=ac, telescope=baikal_reco,
                                                angular_resolution=1, value=value_r, lg_e_brd=lg_e_brd,
                                                rnd=2, if_sum=if_sum)
        r_c, r_c_bg = get_signal_and_background(source=source, tf=tf, atm=ac, telescope=baikal_std_cuts,
                                                angular_resolution=1, value=value_c, lg_e_brd=lg_e_brd,
                                                rnd=4, if_sum=if_sum)

        # just print the total registration rates
        print(f'{source.name} & {r_t} & {r_t_bg} & {r_r} & {r_r_bg} & {r_c} & {r_c_bg}' + r' \\')

    return


if __name__ == '__main__':
    main(if_sum=True,
         if_bg=True,
         source_numbers=None)
