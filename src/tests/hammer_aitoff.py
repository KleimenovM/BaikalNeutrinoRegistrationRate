import numpy as np
import matplotlib.pyplot as plt

# from source_extended import ExtendedSource, galactic_center
from src.source import get_sources
from src.tools import rad_to_hours, hours_to_rad, galactic2equatorial


def coordinates_conversion(delta, phi):
    """
    Convert 2nd equatorial coordinates to Hammer-Aitoff projection of the whole sky
    :param delta: (float or np.ndarray) declination, rad
    :param phi: (float or np.ndarray) right ascension, rad
    :return: (float or np.ndarray) x, (float or np.ndarray) y
    """
    # See https://en.wikipedia.org/wiki/Hammer_projection
    delta = hours_to_rad(delta)
    phi = np.deg2rad(phi)
    denominator = np.sqrt(1 + np.cos(phi) * np.cos(delta / 2))
    x = 2 ** 3 / 2 * np.cos(phi) * np.sin(delta / 2) / denominator
    y = 2 ** 0.5 * np.sin(phi) / denominator
    return x, y


def plot_grid(delta_step=30, phi_step=4):
    """
    Draw the Hammer-Aitoff grid on the canvas
    :param delta_step: (float) distance between declination values, deg
    :param phi_step: (float) distance between right ascension values, hours
    :return:
    """
    phi_brd = [-90, 90]
    lambda_brd = [-12, 12]

    # draw parallels
    d = phi_brd[0]
    ph = np.linspace(lambda_brd[0], lambda_brd[1], 100)
    while d <= phi_brd[1]:
        x, y = coordinates_conversion(ph, d)
        plt.plot(x, y, color='gray', alpha=.5)

        # text
        x_t, y_t = coordinates_conversion(-12, d)
        if d < 0:
            y_t -= .14
            x_t -= .2
        if d == 0:
            y_t -= .07
            x_t -= .1
        plt.text(x_t - 0.2, y_t + .02, str(d) + r'$^\circ$', fontsize=14)

        d += delta_step

    # draw meridians
    d = lambda_brd[0]
    phi = np.linspace(phi_brd[0], phi_brd[1], 100)
    while d <= lambda_brd[1]:
        x, y = coordinates_conversion(d, phi)
        plt.plot(x, y, color='gray', alpha=.5)

        # text
        x_t, y_t = coordinates_conversion(d, 0)
        plt.text(x_t + .05, y_t - .14, str(12 - d) + r'$^h$', fontsize=14)

        d += phi_step

    x1, y1 = coordinates_conversion(-12, phi)
    x2, y2 = coordinates_conversion(12, phi)
    plt.fill_betweenx(y2, x1, x2, color='#009', alpha=.1)

    return


def plot_galactic_plane(legend: bool = False):
    """
    Draw the galactic plane line in the Hammer-Aitoff projection
    :param legend: (bool) if true, appears in the final legend
    :return:
    """
    # Galactic plane
    m = 1000
    points_ll, points_b = np.zeros([m, 1]), np.zeros([m, 1])
    points_ll[:, 0] = np.linspace(0, 2 * np.pi, m)

    d, ra = galactic2equatorial(points_ll, points_b)
    x, y = coordinates_conversion(12 - rad_to_hours(ra), np.rad2deg(d))

    x, y = x.ravel(), y.ravel()

    # Compute the difference between successive t2 values
    diffs = np.append(np.diff(x), 0)

    # Find the differences that are greater than pi
    discont_indices = np.abs(diffs) > np.pi

    # Set those t2 values to NaN
    x[discont_indices] = np.nan

    if legend:
        plt.plot(x, y, color='black', alpha=.8, label='Galactic plane')
    else:
        plt.plot(x, y, color='black', alpha=.8)

    return


def plot_galactic_center(legend: bool = False):
    """
    Draw the galactic center in Hammer-Aitoff projection
    :param legend: (bool) if true, appears in the final legend
    :return:
    """
    # Galactic center
    source: ExtendedSource = galactic_center(num=1000)
    source.info()

    s_d_c, s_ra_c = np.rad2deg(source.equatorial_s_dec_grid), rad_to_hours(source.equatorial_s_ra_grid)
    s_xs, s_ys = coordinates_conversion(12 - s_ra_c, s_d_c)

    xs, ys = np.ravel(s_xs), np.ravel(s_ys)

    if legend:
        plt.scatter(xs, ys, 3, color='orange', alpha=0.2, label='Galactic center')
    else:
        plt.scatter(xs, ys, 3, color='orange', alpha=0.2)

    source.cell_areas()

    return


def get_color(index: int):
    """
    Get a HEX color from a defined palette
    :param: index (int)
    """
    colors = ["ff595e", "ff924c", "c5ca30", "8ac926", "36949d", "1982c4", "4267ac", "565aa0", "6a4c93"]
    m = len(colors)
    return "#" + colors[5 * index % m]


def plot_the_sources(sources,
                     legend: bool = False,
                     the_sources: bool = True):
    """
    Draw the given sources in Hammer-Aitoff projection
    :param sources: (list[Source]) list of the sources to depict
    :param legend: (bool) if true, appears in the final legend
    :param the_sources: (bool) if true, one puts specific names
    :return:
    """
    for i, s in enumerate(sources):
        if i > 0 and s.name[:-3] == sources[i - 1].name[:-3] and the_sources:
            continue
        # ONLY EXTRAGALACTIC
        # if s.name != "TXS 0506+056 (1)" and s.name != "NGC 1068":
        #     continue
        # print(i)

        if i < len(sources) - 1 and s.name[:-3] == sources[i + 1].name[:-3] and the_sources:
            name = s.name[:-3]
        else:
            name = s.name
        x, y = coordinates_conversion(12 - s.right_ascension, s.declination)
        # print(x, y)
        if legend:
            plt.scatter(x, y, label=name, color=get_color(i))
        else:
            plt.scatter(x, y, color=get_color(i))
        print(s.name)
        if s.name == "Vela Jr" or s.name == "NGC 1068":
            plt.text(x - 0.15, y - 0.15, name, fontsize=12)
        elif s.name == "MGRO J1908+06 (1)" or s.name == "Vela X":
            plt.text(x + 0.1, y - 0.03, name, fontsize=12)
        else:
            plt.text(x - 0., y + 0.05, name, fontsize=12)
    return


def draw_hammer_aitoff_sources(filename: str = "data/source_table.csv"):
    plt.figure(figsize=(10, 6))

    plot_grid()
    sources = get_sources(filename)

    for i, s in enumerate(sources):
        if i > 0 and s.name[:-3] == sources[i - 1].name[:-3]:
            continue
        if i < len(sources) - 1 and s.name[:-3] == sources[i + 1].name[:-3]:
            name = s.name[:-3]
        else:
            name = s.name
        x, y = coordinates_conversion(12 - s.right_ascension, s.declination)

        plt.scatter(x, y, label=name)
        if s.name == "Vela Jr":
            plt.text(x - 0.2, y - 0.15, name, fontsize=12)
        else:
            plt.text(x - 0.2, y + 0.05, name, fontsize=12)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return


def draw_hammer_aitoff_ext():
    plt.figure(figsize=(10, 6))

    plot_grid()
    plot_galactic_plane(legend=True)
    # plot_galactic_center()

    sources = get_sources("../data/source_table.csv")
    plot_the_sources(sources)

    plt.axis('off')
    plt.tight_layout()
    plt.legend(fontsize=12)
    title = "results/all"
    # plt.savefig(title + ".pdf")
    # plt.savefig(title + ".png")
    plt.show()

    return


if __name__ == '__main__':
    # draw_hammer_aitoff_sources()
    draw_hammer_aitoff_ext()
