import numpy as np
import matplotlib.pyplot as plt

from src.transmission_function import TransmissionFunction


def check_transmission_matrix(nuFate_method):
    tf = TransmissionFunction(nuFate_method=nuFate_method)

    a = 11 * np.pi / 12

    m = tf.angle_interpolated_matrix(a)

    plt.pcolormesh(tf.lg_energy, tf.lg_energy, np.log10(m), vmin=-14)
    plt.colorbar()
    plt.show()

    return


if __name__ == '__main__':
    check_transmission_matrix(nuFate_method=1)
