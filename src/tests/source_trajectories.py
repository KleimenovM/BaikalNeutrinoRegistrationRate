# Neutrino through Earth propagation
# Parametrize the source daily movement

import numpy as np
import matplotlib.pyplot as plt

from src.telescope import RootTelescopeConstructor, Telescope
from src.source import get_sources
from src.tools import deg_to_rad, sph_coord, rot_matrix


def plot_a_sphere(axis: plt.axis, rotation_angle: float):
    """
    Creates a sphere with parallels and meridians on it
    @param axis: an axis on which the sphere is to be drawn
    @param rotation_angle: an angle (to vertical direction) at which the sphere is to be turned
    @return:
    """
    # meridians
    shift = deg_to_rad([20])
    cur_angle = 0
    n = 100
    theta = np.linspace(-np.pi / 2, np.pi / 2 + 2 * np.pi / n, n)

    rm = rot_matrix(rotation_angle)

    while cur_angle < 2 * np.pi:
        vec = np.zeros([3, n])
        vec[0], vec[1], vec[2] = sph_coord(r=1, theta=theta, phi=np.array([cur_angle]))

        vec = np.dot(rm, vec)

        axis.plot(vec[0], vec[1], vec[2], color='lightgray')
        cur_angle += shift

    # parallels
    shift = deg_to_rad([15])
    cur_angle = -np.pi / 2
    phi = np.linspace(0, 2 * np.pi, n)

    while cur_angle < np.pi / 2:
        vec = np.zeros([3, n])

        vec[0], vec[1], vec[2] = sph_coord(r=1, theta=cur_angle, phi=phi)

        vec = np.dot(rm, vec)

        if abs(cur_angle) < 1e-3:
            axis.plot(vec[0], vec[1], vec[2], color='black')
        else:
            axis.plot(vec[0], vec[1], vec[2], color='lightgray')

        cur_angle += shift

    m1 = 10
    vec_vert = np.zeros([3, m1])
    vec_vert[2] = np.linspace(-1.1, 1.1, m1)
    axis.plot(vec_vert[0], vec_vert[1], vec_vert[2], color='red', linewidth=2, linestyle='dashed')

    vec = np.zeros([3, n])
    vec[0], vec[1], vec[2] = sph_coord(r=1, theta=0, phi=np.linspace(0, 2 * np.pi, 100))
    axis.plot(vec[0], vec[1], vec[2], color='red')

    vec_vert2 = np.dot(rm, vec_vert)
    axis.plot(vec_vert2[0], vec_vert2[1], vec_vert2[2], color='black', linewidth=2, linestyle='dashed')

    axis.scatter(0, 0, color='black')

    return


if __name__ == '__main__':
    source_numbers = [0, 1, 2, 3, 5, 6, 9, 10]
    remove_label = [0, 0, 0, 1, 0, 1, 0, 1]

    Colors = ['cyan', 'blue', 'orange', 'green', 'violet',
              'blue', 'black', 'red']

    Linestyles = ["solid", "solid", "dashdot", "solid", "dotted", "dotted", "solid", "dashed"]

    sources = get_sources("../data/source_table.csv")

    t: Telescope = RootTelescopeConstructor("baikal_2021", "hnu").get()

    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    plot_a_sphere(ax, t.phi)

    ax.axis('scaled')
    # ax.axis('off')

    ax2.plot((0, 24), (0, 0), color='black')
    ax2.set_xlabel(r'$t,\ h$', fontsize=14)
    ax2.set_ylabel(r'$\theta,\ deg$', fontsize=14)

    psi_sample = np.linspace(0, 2 * np.pi, 90)

    for i, n in enumerate(source_numbers):
        s_i = sources[n]
        s_i.info()

        vec1, theta1 = t.get_orbit_parametrization(s_delta=s_i.delta,
                                                   s_alpha=s_i.ra,
                                                   angular_precision=90)[:2]

        name = s_i.name
        if remove_label[i] == 1:
            name = s_i.name[:-3]

        ax.scatter(vec1[0], vec1[1], vec1[2], color=Colors[i])

        ax2.plot(psi_sample / (2 * np.pi) * 24, np.rad2deg(theta1),
                 color=Colors[i], linewidth=2, label=name, linestyle=Linestyles[i])

    ax2.grid(color='gray', linestyle='dashed')
    plt.legend(fontsize=12)
    ax2.invert_yaxis()
    ax2.tick_params(labelsize=14)
    plt.ylim(180, 0)
    plt.xlim(0, 24)

    ax.legend()
    plt.tight_layout()
    plt.savefig("pictures/trajectories.png")
    plt.show()
