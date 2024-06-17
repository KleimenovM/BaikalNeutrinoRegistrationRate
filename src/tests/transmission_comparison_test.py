import matplotlib.pyplot as plt
import numpy as np

from src.nuFATE.cascade_secs import get_eigs as get_eigs2
from src.tools import smart_division
from src.transmission_calculation import TransmissionCalculator
from src.transmission_function import TransmissionFunction


def check_transmission_function():
    angles = np.linspace(np.pi / 2, np.pi, 180)
    energy = 10 ** np.linspace(3, 10, 200)

    gamma = 2
    flavor = 2
    spectrum = energy ** (-gamma)

    tf = TransmissionFunction(nuFate_method=1)

    if_delta_marker = False

    tc = TransmissionCalculator(nuFateMethod=1)
    w2, v2, ci2, energy_nodes2, phi_02 = get_eigs2(flavor, gamma,
                                                   tc.nuFate_hdf5_path,
                                                   pure_spectrum=False)

    plt.figure(figsize=(12, 6))
    for j in range(0, len(angles), 10):
        a_j = angles[j]
        att_with_regeneration = tc.get_att_value_secs(w2, v2, ci2,
                                                      energy_nodes2, a_j,
                                                      energy, phi_02, if_tau=False, absolute=if_delta_marker)

        att_tabular = smart_division(tf.convolution(a_j, spectrum), spectrum)

        plt.subplot(1, 2, 1)

        if j == 0:
            plt.plot(energy, att_with_regeneration, color='royalblue',
                     alpha=.5, label=r"$\nu$FATE")

            plt.plot(energy, att_tabular, color='red',
                     alpha=.5, label=r"transmission matrix")
        else:
            plt.plot(energy, att_with_regeneration, color='royalblue', alpha=.5)

            plt.plot(energy, att_tabular, color='red', alpha=.5)

        plt.subplot(1, 2, 2)
        plt.plot(energy, att_tabular / att_with_regeneration - 1, color='royalblue', alpha=.8)

    # TOP plot
    plt.subplot(1, 2, 1)
    plt.xscale('log')
    plt.xlabel('E, GeV', fontsize=14)
    plt.ylabel(r'$k = \phi/\phi_0$', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(color='lightgray', linestyle='dashed')

    plt.subplot(1, 2, 2)
    plt.tick_params(labelsize=14)
    plt.xscale('log')
    plt.xlabel('E, GeV', fontsize=14)
    plt.ylabel(r'$k_{TM}/k_{\nu FATE} - 1$', fontsize=14)
    plt.grid(color='lightgray', linestyle='dashed')

    plt.tight_layout()
    plt.savefig("pictures/TM_error.pdf")
    plt.savefig("pictures/TM_error.png")
    plt.show()

    return


if __name__ == '__main__':
    check_transmission_function()
