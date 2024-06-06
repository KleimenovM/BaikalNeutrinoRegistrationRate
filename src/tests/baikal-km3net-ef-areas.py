import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from src.telescope import RootTelescopeConstructor, Telescope


def compare_ef_areas():
    km3net = RootTelescopeConstructor("km3net_2019_trigger", "hnu_trigger").get()
    baikal = RootTelescopeConstructor("baikal_2023_new_simp", "hnu_trigger").get()

    km3net = RootTelescopeConstructor("km3net_2023_reco", "hnu_reco").get()
    baikal = RootTelescopeConstructor("baikal_2023_new_simp", "hnu_reco").get()

    energy_range_1 = km3net.lg_energy
    energy_range_2 = baikal.lg_energy

    lower, upper = max(energy_range_1[0], energy_range_2[0]), min(energy_range_1[-1], energy_range_2[-1])

    n = 100
    lg_e = np.linspace(lower, upper, n)

    ef_area1 = km3net.effective_area(np.array([-0.8]), lg_e)[0]
    ef_area2 = baikal.effective_area(np.array([-0.8]), lg_e)[0] * 20

    return lg_e, ef_area1 / 3, ef_area2


def plot_it(e, ef_area, ef_area2):
    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 2])

    # subplot 1
    plt.subplot(gs[0])
    plt.title(r"Reconstruction-level effective areas comparison", fontsize=14)
    plt.step(e, ef_area, where='pre', linewidth=2, label='KM3NeT')
    plt.step(e, ef_area2, where='pre', linewidth=2, label='Baikal-GVD')
    plt.xscale('log')
    plt.xlim(min(e) * 1.15, 0.1 * max(e))
    plt.ylim(1e-5, 5e3)
    plt.yscale('log')
    plt.ylabel("$A_{eff},~m^2$", fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(color='lightgray', linestyle='dashed')

    # subplot 2
    plt.subplot(gs[1])
    plt.ylabel(r"$\mathrm{\dfrac{KM3NeT}{Baikal-GVD}}$", fontsize=14)
    plt.step(e, ef_area / ef_area2, linewidth=2, color='#090')
    plt.plot(e, np.ones(e.size), color='black', linestyle='dashed')

    plt.xscale('log')
    plt.xlim(min(e) * 1.15, 0.1 * max(e))
    plt.ylim(0, 3)
    plt.tick_params(labelsize=14)
    plt.grid(color='lightgray', linestyle='dashed')
    plt.xlabel("E, GeV", fontsize=14)

    plt.tight_layout()

    return


def main():
    lg_e, ef1, ef2 = compare_ef_areas()
    plot_it(10**lg_e, ef1, ef2)
    plt.show()
    return


if __name__ == '__main__':
    main()
