import numpy as np
from matplotlib import pyplot as plt

from src.background import Atmosphere, AstrophysicalBackground


def test_atmosphere():
    ac = Atmosphere()
    lg_e, given_lg_e = ac.lg_energy, ac.given_lg_energy
    cosines = ac.cosines

    m = cosines.size
    color_values = np.zeros([m, 3])
    color_values[:, 0] = np.linspace(0.3, 1, m)
    color_values[:, 1] = np.linspace(0.0, 0.5, m)
    color_values[:, 2] = np.linspace(0.0, 0.5, m)

    plt.figure(figsize=(9, 5))
    deg = 2
    for i, c in enumerate(cosines):
        plt.plot(10 ** lg_e, 10 ** ac.rbc.ev(c, lg_e) * (10 ** lg_e) ** deg / 1e4,
                 alpha=.5, color=color_values[i], label=r'$\cos\theta=$' + str(c), linestyle='solid')
        plt.scatter(10 ** given_lg_e, ac.data[i, :] * (10 ** given_lg_e) ** deg / 1e4, color=color_values[i])

    print(f"Error: {np.round(100 * ac.estimate_error(), 1)} %")


def astro():
    astro_b = AstrophysicalBackground()
    lg_e = np.linspace(3, 8, 100)
    e = 10**lg_e
    mean = astro_b.flux(lg_e)
    low, high = astro_b.err_flux(lg_e)

    plt.plot(e, e**2 * mean / 1e4, color='royalblue', linewidth=2, label=r"$IceCube~astro~\nu_\mu$")
    plt.fill_between(e, e**2 * low / 1e4, e**2 * high / 1e4, color='lightblue', alpha=1.0)
    return


def main():
    test_atmosphere()
    astro()

    plt.xlabel(r"$E,~\mathrm{GeV}$", fontsize=14)
    plt.ylabel(r"$E^{2}\dfrac{d\Phi}{dE\,d\Omega},~\mathrm{cm^{-2}\,s^{-1}\,GeV\,sr^{-1}}$", fontsize=14)

    plt.tick_params(labelsize=14)

    plt.yscale('log')
    plt.xscale('log')

    plt.grid(color='lightgray', linestyle='dashed')
    plt.legend(fontsize=14)

    plt.xlim(1, 10**7)

    plt.tight_layout()
    plt.savefig("pictures/atm_flux.png")
    plt.savefig("pictures/atm_flux.pdf")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(color='lightgray', linestyle='dashed')
    plt.show()
    return


if __name__ == '__main__':
    main()
