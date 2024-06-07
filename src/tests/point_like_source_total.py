# Evaluates average high-energy neutrino spectral attenuation
# for specific sources listed in "data/sources_table.csv"
from src.atmosphere import Atmosphere
from src.root_plotter import RootPlotter
from src.single_source_flux import PointSourceFlux
from src.source import get_sources
from src.telescope import RootTelescopeConstructor
from src.transmission_function import TransmissionFunction


def point_like_source_plot(if_bg: bool = False,
                           source_numbers: list = None,
                           lg_e_brd: float = 0.0,
                           angular_resolution: float = 0.5):
    # sources from file "source_table.csv" -- potential high-energy neutrino sources
    sources = get_sources("../data/source_table.csv")

    telescope_name = "baikal_bdt_mk"

    # transmission function initialization
    tf = TransmissionFunction(nuFate_method=1)

    # atmoshperic flux initialization
    atm = Atmosphere()

    # zenith-angle-dependent version
    baikal_trigger = RootTelescopeConstructor(telescope_name, "hnu_trigger").get()
    baikal_reco = RootTelescopeConstructor(telescope_name, "hnu_reco").get()
    baikal_std_cuts = RootTelescopeConstructor(telescope_name, "hnu_stdcuts").get()

    # number of clusters & time
    value_t, value_r, value_c = 20 * 5, 20 * 5, 20 * 5

    if source_numbers is None:
        source_numbers = [x for x in range(12)]

    if if_bg:
        print(r'Sources & RR, trig & BG, trig & RR, reco & BG, reco & RR, cuts & BG, cuts \\')

    for sn in source_numbers:
        source = sources[sn]  # take one source from the list

        ssf_trigger = PointSourceFlux(telescope=baikal_trigger, tf=tf, atm=atm, value=value_t, source=source,
                                      angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

        ssf_reco = PointSourceFlux(telescope=baikal_reco, tf=tf, atm=atm, value=value_r, source=source,
                                   angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

        ssf_cuts = PointSourceFlux(telescope=baikal_std_cuts, tf=tf, atm=atm, value=value_c, source=source,
                                   angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

        if if_bg:
            r_t, r_t_bg = ssf_trigger.total_signal(), ssf_trigger.total_background()
            r_r, r_r_bg = ssf_reco.total_signal(), ssf_reco.total_background()
            r_c, r_c_bg = ssf_cuts.total_signal(), ssf_cuts.total_background()
            print(f'{source.name} & {r_t} & {r_t_bg} & {r_r} & {r_r_bg} & {r_c} & {r_c_bg} ' + r'\\')
        else:
            r_t = ssf_trigger.total_signal()
            r_c = ssf_reco.total_signal()
            r_r = ssf_cuts.total_signal()
            print(f'{source.name} & {r_t} & {r_r} & {r_c} & ' + r'\\')

    return


if __name__ == '__main__':
    point_like_source_plot(if_bg=True,
                           source_numbers=None,
                           angular_resolution=1.0)
