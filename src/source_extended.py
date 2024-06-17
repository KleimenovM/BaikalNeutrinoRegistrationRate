import numpy as np
from src.tools import galactic2equatorial, simple_gal2eq, sph_coord, vec_product, rot_matrix


class ExtendedSource:
    """
    This class describes a typical extended stellar neutrino source
    Created expecially for the Galactic ridge flux calculation
    """

    def __init__(self, name: str, center_longitude, center_latitude,
                 spectrum_function, longitude_loc=None, latitude_loc=None,
                 num=100):
        self.name = name

        # positioning parameters
        self.ll = center_longitude  # longitude (l), rad
        self.b = center_latitude  # latitude (b), rad

        # localization area
        if longitude_loc:
            self.l_loc = longitude_loc
        else:
            self.l_loc = 2 * np.pi
        if latitude_loc:
            self.b_loc = latitude_loc
        else:
            self.b_loc = 2 * np.pi

        # spectrum function
        self.spectrum = spectrum_function

        # grid knots parameters
        self.num = num
        self.nx = num
        self.ny = 1
        # grid cell parameters
        self.snx = self.nx - 1
        self.sny = 1
        self.s_num = self.snx * self.sny

        gal_grid, eq_grid, s_gal_grid, s_eq_grid = self.single_rectangular_sampling()

        # knots grid
        self.galactic_l_grid, self.galactic_b_grid = gal_grid[0], gal_grid[1]
        self.equatorial_dec_grid, self.equatorial_ra_grid = eq_grid[0], eq_grid[1]
        # cell grid
        self.galactic_sl_grid, self.galactic_sb_grid = s_gal_grid[0], s_gal_grid[1]
        self.equatorial_s_dec_grid, self.equatorial_s_ra_grid = s_eq_grid[0], s_eq_grid[1]

    def info(self):
        """
        Get the brief information about the source
        :return:
        """
        print(f"{self.name}")
        print(f"pos = {simple_gal2eq(self.ll, self.b)}")
        print(f"split: in longitude {self.snx}, in latitude {self.sny}, total {self.s_num}")
        pass

    def flux_on_energy_function(self, energy, ll, b):
        """
        Get the flux on spectrum function
        :param energy: energy, GeV
        :param ll: source's longitude (rad)
        :param b: source's latitude (rad)
        :return: flux (np.array, size = energy.size)
        """
        return self.spectrum(energy, ll, b) / self.s_num

    @staticmethod
    def find_middle(matrix):
        return (matrix[1:, 1:] + matrix[:-1, :-1]) / 2.0

    def single_rectangular_sampling(self):
        """
        Split the given region with a homogenous grid of equal sources
        Form the knot-grid (for area definition) and cell-grid (for sources setting)
        :return: knot galactic grid, knot equatorial grid, cell_galactic_grid, cell_equatorial_grid
        """
        total_area = self.l_loc * self.b_loc  # rad^2
        single_cell_area = total_area / self.num  # rad^2
        cell_size = np.sqrt(single_cell_area)  # rad, angular size of a squared cell

        self.nx = int(self.l_loc // cell_size) + 1
        self.ny = int(self.b_loc // cell_size) + 1

        self.num = self.nx * self.ny

        # knot grid
        l_split = self.ll + np.linspace(-self.l_loc / 2, self.l_loc / 2, self.nx)
        b_split = self.b + np.linspace(-self.b_loc / 2, self.b_loc / 2, self.ny)

        ll_grid, b_grid = np.meshgrid(l_split, b_split, indexing='ij')
        dec_grid, ra_grid = galactic2equatorial(ll_grid, b_grid)

        # source_grid (middles of the squares)
        self.snx, self.sny = max(self.nx - 1, 1), max(self.ny - 1, 1)
        self.s_num = self.snx * self.sny

        sl_grid, sb_grid = self.find_middle(ll_grid), self.find_middle(b_grid)
        s_dec_grid, s_ra_grid = galactic2equatorial(sl_grid, sb_grid)

        return ([ll_grid, b_grid], [dec_grid, ra_grid],
                [sl_grid, sb_grid], [s_dec_grid, s_ra_grid])

    def cell_areas(self):
        """
        Get the area of each cell in equatorial coordinates
        :return:
        """
        positions = sph_coord(1.0, self.equatorial_dec_grid, self.equatorial_ra_grid)

        # [1, 2]
        # [3, 4]

        pos1 = positions[:, :-1, :-1]
        pos2 = positions[:, 1:, :-1]
        pos3 = positions[:, :-1, 1:]
        pos4 = positions[:, 1:, 1:]

        # independence on rotation checked 05.03.24

        vp1 = vec_product(pos3 - pos1, pos2 - pos1)  # upper triangle 123
        vp2 = vec_product(pos3 - pos4, pos2 - pos4)  # lower triangle 234

        return 1 / 2 * (np.sqrt(np.sum(vp1 ** 2, axis=0)) + np.sqrt(np.sum(vp2 ** 2, axis=0)))


def galactic_center(num=100):
    ll, b = .0, .0  # rad
    l_loc, b_loc = np.deg2rad(60), np.deg2rad(4)  # [-30, 30] & [-2, 2]
    gamma_val = 2.5
    k0_val = 1.2e-9  # GeV-1 s-1 m-2

    def galactic_spectrum_function(e, *args):
        return k0_val * (e / 1000) ** (-gamma_val)

    return ExtendedSource(name='Galactic ridge',
                          center_longitude=ll, center_latitude=b,
                          spectrum_function=galactic_spectrum_function,
                          longitude_loc=l_loc, latitude_loc=b_loc,
                          num=num)


if __name__ == '__main__':
    print("Not for direct use")