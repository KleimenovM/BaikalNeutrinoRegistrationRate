# Evaluates average high-energy neutrino spectral attenuation
# for extended sources (especially for the Galactic ridge
import matplotlib.pyplot as plt
import numpy as np

from src.atmosphere import Atmosphere

from src.single_theta_flux import SingleThetaFlux
from src.source_extended import ExtendedSource, galactic_center
from src.telescope import Telescope, RootTelescopeConstructor
from src.transmission_function import TransmissionFunction


def one_source_full_cycle(ext_source: ExtendedSource, tf: TransmissionFunction, telescope: Telescope,
                          angular_precision: int = 180, silent: bool = False) -> np.ndarray:
    """

    :param silent:
    :param ext_source:
    :param tf:
    :param telescope:
    :param angular_precision:
    :return:
    """

    ref_energy = telescope.energy
    d_lg_e = telescope.lg_energy[1] - telescope.lg_energy[0]
    de = 10 ** (telescope.lg_energy + d_lg_e) - ref_energy

    total_flux_partition = np.zeros([ext_source.s_num, angular_precision, ref_energy.size])

    for i_x in range(ext_source.snx):
        for i_y in range(ext_source.sny):
            i = i_x * ext_source.sny + i_y
            if not silent and i % 20 == 0:
                print(f"-{i}-", end='')

            ll_i, b_i = ext_source.galactic_sl_grid[i_x, i_y], ext_source.galactic_sb_grid[i_x, i_y]

            # reference flux (for telescope effective area calculations)
            ref_initial_flux_i = ext_source.flux_on_energy_function(energy=ref_energy, ll=ll_i, b=b_i)
            # flux for transmission calculation via nuFATE
            initial_flux_i = ext_source.flux_on_energy_function(energy=tf.energy, ll=ll_i, b=b_i)

            dec_i, alpha_i = ext_source.equatorial_s_dec_grid[i_x, i_y], ext_source.equatorial_s_ra_grid[i_x, i_y]
            deltas = telescope.get_orbit_parametrization(s_delta=dec_i,
                                                         s_alpha=alpha_i,
                                                         angular_precision=angular_precision)[1]

            for j in range(angular_precision):
                stf = SingleThetaFlux(initial_flux=initial_flux_i,
                                      zenith=deltas[j],
                                      telescope=telescope,
                                      tf=tf)

                relative_flux_ij = stf.calculate()

                total_flux_partition[i, j, :] = relative_flux_ij * ref_initial_flux_i

    if not silent:
        print('')

    total_flux_emu = np.mean(total_flux_partition, axis=1)
    year_seconds = 3600 * 24 * 365
    multiplier = de * year_seconds

    return np.sum(total_flux_emu, axis=0) * multiplier


def background_full_cycle(ext_source: ExtendedSource, tf: TransmissionFunction, telescope: Telescope,
                          atm: Atmosphere,
                          angular_precision: int = 180, silent: bool = False) -> np.ndarray:
    """

    :param atm:
    :param angular_precision:
    :param ext_source:
    :param tf:
    :param telescope:
    :param silent:
    :return:
    """
    # ------------------------------
    # energy structures determination
    ref_energy, ref_lg_energy = telescope.energy, telescope.lg_energy
    d_lg_e = telescope.lg_energy[1] - telescope.lg_energy[0]
    de = 10 ** (telescope.lg_energy + d_lg_e) - ref_energy

    m = angular_precision
    n = telescope.lg_energy.size

    total_flux_partition = np.zeros([ext_source.s_num, m, n])

    areas = ext_source.cell_areas()

    # ----------------------------
    # atmospheric flux calculation
    for i_x in range(ext_source.snx):
        for i_y in range(ext_source.sny):
            i = i_x * ext_source.sny + i_y
            if not silent and i % 20 == 0:
                print(f"-{i}-", end='')

            dec_i, alpha_i = ext_source.equatorial_dec_grid[i_x, i_y], ext_source.equatorial_ra_grid[i_x, i_y]
            area_xy = areas[i_x, i_y]
            deltas = telescope.get_orbit_parametrization(s_delta=dec_i,
                                                         s_alpha=alpha_i,
                                                         angular_precision=angular_precision)[1]
            for j in range(angular_precision):
                z_j = deltas[j]
                ref_initial_flux_ij = atm.flux(ref_lg_energy, z_j) * area_xy
                initial_flux_ij = atm.flux(tf.lg_energy, z_j) * area_xy

                stf = SingleThetaFlux(initial_flux=initial_flux_ij,
                                      zenith=deltas[j],
                                      telescope=telescope,
                                      tf=tf)

                relative_flux_ij = stf.calculate()

                total_flux_partition[i, j, :] = relative_flux_ij * ref_initial_flux_ij

    if not silent:
        print('\n')

    total_flux_emu = np.mean(total_flux_partition, axis=1)
    year_seconds = 3600 * 24 * 365
    multiplier = de * year_seconds

    return np.sum(total_flux_emu, axis=0) * multiplier


def get_signal_and_background(ext_source: ExtendedSource, tf: TransmissionFunction,
                              ac: Atmosphere,
                              telescope: Telescope, value: float = 1, rnd: int = 2,
                              if_sum: bool = True):
    """
    Return two registration rates (dN / dT)
    :param if_sum:
    :param ext_source:
    :param rnd: number of decimals after comma
    :param value: multiplication coefficient
    :param ac: AtmosphericFluxApproximationConstructor - the source of atmospheric flux spectrum
    :param tf: TransmissionFunction - the Earth transmission function
    :param telescope: Telescope - the telescope for which calculation is performed
    :return: (float, float) - events rate, background rate
    """

    print("Signal")
    baikal_t_rel = one_source_full_cycle(ext_source=ext_source, tf=tf, telescope=telescope)
    print("Background")
    baikal_t_bg_rel = background_full_cycle(ext_source=ext_source, tf=tf, telescope=telescope, atm=ac)

    if if_sum:
        rate = np.round(np.sum(baikal_t_rel * value), rnd)
        rate_bg = np.round(np.sum(baikal_t_bg_rel * value), rnd)
        return rate, rate_bg

    return baikal_t_rel * value, baikal_t_bg_rel * value


def main(if_sum: bool = True, if_bg: bool = True, ns: list = None):

    if ns is None:
        ns = [100]

    # zenith-angle-dependent version
    baikal_trigger = RootTelescopeConstructor("baikal_bdt_mk", "hnu_trigger").get()
    baikal_reco = RootTelescopeConstructor("baikal_bdt_mk", "hnu_reco").get()
    baikal_std_cuts = RootTelescopeConstructor("baikal_bdt_mk", "hnu_stdcuts").get()

    # background model
    atm = Atmosphere()

    # Earth transmission function calculated with nuFate
    tf = TransmissionFunction(nuFate_method=1)

    value_t, value_r, value_c = 20 * 5, 20 * 5, 20 * 5

    num = []

    # galactic ridge
    for n in ns:
        ext_source = galactic_center(num=n)
        num.append(ext_source.s_num)
        ext_source.info()
        print(np.sum(ext_source.cell_areas()))
        print(ext_source.s_num)

        r_t, r_t_bg = get_signal_and_background(ext_source=ext_source, tf=tf, ac=atm, telescope=baikal_trigger,
                                                value=value_t, rnd=2, if_sum=if_sum)
        r_r, r_r_bg = get_signal_and_background(ext_source=ext_source, tf=tf, ac=atm, telescope=baikal_reco,
                                                value=value_r, rnd=2, if_sum=if_sum)
        r_c, r_c_bg = get_signal_and_background(ext_source=ext_source, tf=tf, ac=atm, telescope=baikal_std_cuts,
                                                value=value_c, rnd=3, if_sum=if_sum)

        if if_sum and if_bg:
            # just print the total registration rates
            print(f'{ext_source.name} & {r_t} & {r_t_bg} & {r_r} & {r_r_bg} & {r_c} & {r_c_bg}' + r' \\')

        elif if_sum:
            print(f'{ext_source.name} & {r_t} & {r_r} & {r_c}' + r' \\')

        print(f'Galactic ridge & {r_t} & {r_t_bg} & {r_r} & {r_r_bg} & {r_c} & {r_c_bg}' + r' \\')

    return


if __name__ == '__main__':
    main(if_sum=True, if_bg=True, ns=[100])
