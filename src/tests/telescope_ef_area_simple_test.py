import numpy as np
import matplotlib.pyplot as plt

from src.telescope import RootTelescopeConstructor


def check_ef_area():

    baikal = RootTelescopeConstructor("baikal_2021", "hnu").get()

    lg_energy_range = baikal.lg_energy
    energy_range = baikal.energy

    a = np.cos(2*np.pi/3 + 0.5)

    print(baikal.brd_cosine)
    print(a)

    f_xv = baikal.effective_area(a, lg_energy_range)

    plt.scatter(energy_range, f_xv)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return


if __name__ == '__main__':
    check_ef_area()
