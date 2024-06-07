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


def main(if_sum: bool = False,
         if_bg: bool = False,
         source_numbers=None):
    # sources from file "source_table.csv" -- potential high-energy neutrino sources
    if source_numbers is None:
        source_numbers = [x for x in range(12)]
    sources = get_sources("src/data/source_table.csv")

    baikal_trigger = RootTelescopeConstructor("baikal_2023_new", "hnu_trigger").get()
    km3net_trigger = RootTelescopeConstructor("km3net_2019_trigger", "hnu_trigger").get()

    # background_model
    ac = Atmosphere()

    # Earth transmission function calculated with nuFate
    tf = TransmissionFunction(nuFate_method=1)

    # number of clusters & time
    value_t = 20 * 5
    value_k = 1 * 5

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
        k_t, k_t_bg = get_signal_and_background(source=source, tf=tf, atm=ac, telescope=km3net_trigger,
                                                angular_resolution=1, value=value_k, lg_e_brd=lg_e_brd,
                                                rnd=2, if_sum=if_sum)

        # just print the total registration rates
        print(f'{source.name} & {r_t} & {k_t}' + r' \\')

    return


if __name__ == '__main__':
    main(if_sum=True,
         if_bg=True,
         source_numbers=None)
