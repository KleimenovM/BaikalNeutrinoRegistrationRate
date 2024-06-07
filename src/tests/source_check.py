import numpy as np
import matplotlib.pyplot as plt

from src.source import get_sources

Colors = ['cyan', 'blue', 'orange', 'green', 'green', 'violet',
          'blue', 'blue', 'blue', 'black', 'red', 'red']

Linestyles = ['solid', 'solid', 'dashdot', 'solid', 'dotted', 'dotted',
              'dotted', 'solid', 'dashed', 'solid', 'solid', 'dashed']


def check_sources():
    sources = get_sources("../data/source_table.csv")

    energy = 10 ** np.linspace(2, 6, 100)

    plt.figure(figsize=(6, 10))

    n = 3
    lw = 2

    for i, s in enumerate(sources):
        if i < 6:
            plt.subplot(n, 1, 1)
            plt.plot(energy, energy ** 2 * s.flux_on_energy_function(energy) * 1e-7,
                     label=s.name, linewidth=lw, color=Colors[i], linestyle=Linestyles[i])
        elif i < 9:
            plt.subplot(n, 1, 2)
            plt.plot(energy, energy ** 2 * s.flux_on_energy_function(energy) * 1e-7,
                     label=s.name, linewidth=lw, color=Colors[i], linestyle=Linestyles[i])
        else:
            plt.subplot(n, 1, 3)
            plt.plot(energy, energy ** 2 * s.flux_on_energy_function(energy) * 1e-7,
                     label=s.name, linewidth=lw, color=Colors[i], linestyle=Linestyles[i])

    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.legend(fontsize=12)

        plt.xscale("log")
        plt.yscale("log")

        plt.ylim(1e-15, 1e-10)
        plt.xlim(1e2, 1e6)

        plt.ylabel(r"$E^2\Phi_\nu\ [TeV\ cm^{-2}\ s^{-1}]$", fontsize=12)
        if i > 1:
            plt.xlabel("E, GeV", fontsize=12)

        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.grid(color="gray", linestyle='dashed')

    plt.savefig("pictures/spectra.pdf")
    plt.show()

    return


if __name__ == '__main__':
    check_sources()
