# Evaluates average high-energy neutrino spectral attenuation
# for specific sources listed in "data/sources_table.csv"
import numpy as np
import pandas as pd

from src.background import Atmosphere, AstrophysicalBackground
from src.single_source_flux import PointSourceFlux
from src.source import get_sources
from src.telescope import RootTelescopeConstructor
from src.transmission_function import TransmissionFunction


def point_like_source_plot(if_bg: bool = False,
                           source_numbers: list = None,
                           lg_e_brd: float = 0.0,
                           angular_resolution: float = 0.5):
    # sources from file "source_table.csv" -- potential high-energy neutrino sources
    sources = get_sources("../data/source_table.csv")

    telescope_name = "baikal_bdt_mk"

    # transmission function initialization
    tf = TransmissionFunction(nuFate_method=1)

    # atmoshperic flux initialization
    atm = Atmosphere()
    # astrophysical bg flux initialization
    astro = AstrophysicalBackground()

    # zenith-angle-dependent version
    baikal_trigger = RootTelescopeConstructor(telescope_name, "hnu_trigger").get()
    baikal_reco = RootTelescopeConstructor(telescope_name, "hnu_reco").get()
    baikal_std_cuts = RootTelescopeConstructor(telescope_name, "hnu_stdcuts").get()
    baikal_prev_cuts = RootTelescopeConstructor("baikal_2023_new", "hnu_stdcuts").get()

    # number of clusters & time
    value_t, value_r, value_c = 20 * 5, 20 * 5, 20 * 5

    if source_numbers is None:
        source_numbers = [x for x in range(12)]

    if if_bg:
        print(r'Sources & RR, trig & BG, trig & RR, reco & BG, reco & RR, cuts & BG, cuts & RR/BG \\')

    rnd_cuts = 3

    m = len(source_numbers)
    names = []
    results = np.zeros([m, 6])

    for i, sn in enumerate(source_numbers):
        source = sources[sn]  # take one source from the list

        ssf_trigger = PointSourceFlux(telescope=baikal_trigger, tf=tf, atm=atm, value=value_t, source=source,
                                      angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

        ssf_reco = PointSourceFlux(telescope=baikal_reco, tf=tf, atm=atm, value=value_r, source=source,
                                   angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

        ssf_cuts = PointSourceFlux(telescope=baikal_std_cuts, tf=tf, atm=atm, value=value_c, source=source,
                                   angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

        ssf_cuts_prev = PointSourceFlux(telescope=baikal_prev_cuts, tf=tf, atm=atm, value=value_c, source=source,
                                        angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

        if if_bg:
            r_t, r_t_bg = ssf_trigger.total_signal(), ssf_trigger.total_background()
            r_r, r_r_bg = ssf_reco.total_signal(), ssf_reco.total_background()
            r_c, r_c_bg = ssf_cuts.total_signal(rnd_cuts), ssf_cuts.total_background(rnd_cuts)
            print(f'{source.name} & {r_t} & {r_t_bg} & {r_r} & {r_r_bg} & {r_c} & {r_c_bg} & {np.round(r_c/r_c_bg, 2)}' + r'\\')
            results[i] = np.array([r_t, r_t_bg, r_r, r_r_bg, r_c, r_c_bg])
            names.append(source.name)
        else:
            r_t = ssf_trigger.total_signal()
            r_r = ssf_reco.total_signal()
            r_c = ssf_cuts.total_signal(3)
            r_c_p = ssf_cuts_prev.total_signal(3)
            print(f'{source.name} & {r_t} & {r_r} & {r_c_p} & {r_c} & {np.round(r_c / r_c_p, 2)}' + r'\\')

    if if_bg:
        df = pd.DataFrame(results.T, columns=names)
        df.to_csv(f'results/out_bdt_{lg_e_brd}GeV_{angular_resolution}res.csv')

    return


if __name__ == '__main__':
    point_like_source_plot(if_bg=True,
                           source_numbers=None,
                           lg_e_brd=0,
                           angular_resolution=1)
