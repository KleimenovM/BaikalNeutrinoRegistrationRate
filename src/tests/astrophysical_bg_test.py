import numpy as np
import matplotlib.pyplot as plt

from src.background import AstrophysicalBackground


def main():
    astro_b = AstrophysicalBackground()

    lg_e = np.linspace(3, 7, 100)
    plt.plot(10**lg_e, 10**(2 * lg_e) * astro_b.flux(lg_e))
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(color='lightgray', linestyle='dashed')
    plt.show()
    return


if __name__ == '__main__':
    main()
