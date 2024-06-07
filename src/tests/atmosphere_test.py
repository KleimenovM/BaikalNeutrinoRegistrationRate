import numpy as np
from matplotlib import pyplot as plt

from src.atmosphere import Atmosphere


def test_atmosphere():
    ac = Atmosphere()
    lg_e, given_lg_e = ac.lg_energy, ac.given_lg_energy
    cosines = ac.cosines

    m = cosines.size
    color_values = np.zeros([m, 3])
    color_values[:, 0] = np.linspace(0.3, 1, m)
    color_values[:, 1] = np.linspace(0.0, 0.5, m)
    color_values[:, 2] = np.linspace(0.0, 0.5, m)

    plt.figure(figsize=(8, 6))
    deg = 3
    for i, c in enumerate(cosines):
        plt.plot(10 ** lg_e, 10 ** ac.rbc.ev(c, lg_e) * (10 ** lg_e) ** deg / 1e4,
                 alpha=.5, color=color_values[i], label=r'$\cos\theta=$' + str(c), linestyle='solid')
        plt.scatter(10 ** given_lg_e, ac.data[i, :] * (10 ** given_lg_e) ** deg / 1e4, color=color_values[i])
    plt.xlim(10 ** lg_e.min(), 10 ** lg_e.max())
    # plt.ylim(1e-3, 1e-1)

    plt.xlabel(r"$E,~\mathrm{GeV}$", fontsize=14)
    plt.ylabel(r"$E^{3}\dfrac{d\Phi}{dE\,d\Omega},~\mathrm{m^{-2}\,s^{-1}\,GeV^{2}\,srad^{-1}}$", fontsize=14)

    plt.tick_params(labelsize=14)

    plt.yscale('log')
    plt.xscale('log')

    plt.grid(color='lightgray', linestyle='dashed')
    plt.legend(fontsize=14)

    plt.tight_layout()
    # plt.savefig("results/atm_flux.png")
    plt.show()
    print(f"Error: {np.round(100 * ac.estimate_error(), 1)} %")


if __name__ == '__main__':
    test_atmosphere()
