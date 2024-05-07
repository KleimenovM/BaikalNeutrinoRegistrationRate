import numpy as np
import matplotlib.pyplot as plt

from src.telescope import RootTelescopeConstructor, Telescope


def check_ef_area():

    baikal: Telescope = RootTelescopeConstructor("baikal_bdt_mk", "hnu_stdcuts").get()

    lg_energy_range = baikal.lg_energy
    energy_range = baikal.energy

    a = 2*np.pi/3

    xv = np.array(np.meshgrid(a, lg_energy_range, indexing='ij')).T
    f_xv = baikal.ef_area(xv).T[0]

    plt.scatter(energy_range, f_xv)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return


if __name__ == '__main__':
    check_ef_area()
