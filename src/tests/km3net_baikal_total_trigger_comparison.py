# Evaluates average high-energy neutrino spectral attenuation
# for specific sources listed in "data/sources_table.csv"
import numpy as np

from src.background import Atmosphere
from src.single_source_flux import PointSourceFlux
from src.source import get_sources
from src.telescope import RootTelescopeConstructor
from src.transmission_function import TransmissionFunction


def main(if_sum: bool = False,
         if_bg: bool = False,
         source_numbers=None):
    # sources from file "source_table.csv" -- potential high-energy neutrino sources
    if source_numbers is None:
        source_numbers = [x for x in range(12)]
    sources = get_sources("../data/source_table.csv")

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

        p1 = PointSourceFlux(source=source, tf=tf, atm=ac, telescope=baikal_trigger,
                             angular_resolution=1, value=value_t)

        r_t, r_t_bg = p1.total_signal(rnd=2), p1.total_background(rnd=2)

        p2 = PointSourceFlux(source=source, tf=tf, atm=ac, telescope=km3net_trigger,
                             angular_resolution=1, value=value_k)

        k_t, k_t_bg = p2.total_signal(rnd=2), p2.total_background(rnd=2)

        # just print the total registration rates
        print(f'{source.name} & {r_t} & {k_t}' + r' \\')

    return


if __name__ == '__main__':
    main(if_sum=True,
         if_bg=True,
         source_numbers=None)
