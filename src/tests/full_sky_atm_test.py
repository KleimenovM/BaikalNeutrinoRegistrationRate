import numpy as np
import matplotlib.pyplot as plt

from src.atmosphere import Atmosphere
from hammer_aitoff import plot_galactic_plane, plot_the_sources, plot_grid
from point_like_source import one_telescope_background_cycle
from src.source import Source
from src.telescope import RootTelescopeConstructor
from src.transmission_function import TransmissionFunction


def homogenous_fibonacci_distribution_on_sphere(num_pts: int):
    # the source of the distribution
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075
    # see this article also (in russian)
    # https://habr.com/ru/articles/460643/
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    return phi, theta


def organize_sources(num_pts, random: bool = True, if_plot: bool = False):
    if random:
        phi = np.arccos(2 * np.random.random(num_pts) - 1)
        theta = 2 * np.pi * np.random.random(num_pts)
    else:
        phi, theta = homogenous_fibonacci_distribution_on_sphere(num_pts)

    dec, ra = np.pi / 2 - phi, theta / np.pi * 12.0 % 24.0

    sources = []

    for i in range(num_pts):
        s = Source(name=str(i),
                   declination_angle=dec[i],
                   right_ascension_time=ra[i],
                   k0=1e-9, gamma=2.0)
        sources.append(s)

    if if_plot:
        plot_grid()
        plot_galactic_plane()
        plot_the_sources(sources, the_sources=False)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return sources


def full_sky():
    n_s = 30
    angular_resolution = (4 / n_s) ** 0.5 * 180 / np.pi
    sources = organize_sources(n_s, random=False)

    ef_area_folder = "data/eff_area_new"

    # zenith-angle-dependent version
    baikal_trigger = RootTelescopeConstructor("baikal_bdt_mk", "hnu_trigger").get()
    baikal_reco = RootTelescopeConstructor("baikal_bdt_mk", "hnu_reco").get()
    baikal_std_cuts = RootTelescopeConstructor("baikal_bdt_mk", "hnu_stdcuts").get()

    atm = Atmosphere()

    # Earth transmission function calculated with nuFate
    tf = TransmissionFunction(nuFate_method=1)

    value_t, value_r, value_c = 1, 1, 1

    t_res, r_res, c_res = .0, .0, .0

    for i_s, s in enumerate(sources):
        if i_s % 10 == 0:
            print(i_s)

        t_res += np.sum(one_telescope_background_cycle(source=s, tf=tf, telescope=baikal_trigger,
                                                       atm=atm, angular_resolution=angular_resolution) * value_t)

        r_res += np.sum(one_telescope_background_cycle(source=s, tf=tf, telescope=baikal_reco,
                                                       atm=atm, angular_resolution=angular_resolution) * value_r)

        c_res += np.sum(one_telescope_background_cycle(source=s, tf=tf, telescope=baikal_std_cuts,
                                                       atm=atm, angular_resolution=angular_resolution) * value_c)

    print(t_res, r_res, c_res)

    return


if __name__ == '__main__':
    full_sky()
