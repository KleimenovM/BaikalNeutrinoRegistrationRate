# Evaluates average high-energy neutrino spectral attenuation
# for specific sources listed in "data/sources_table.csv"
from src.background import Atmosphere
from src.root_plotter import RootPlotter
from src.single_source_flux import PointSourceFlux
from src.source import get_sources
from src.telescope import RootTelescopeConstructor
from src.transmission_function import TransmissionFunction


def point_like_source_plot(if_bg: bool = False,
                           source_number: int = 0,
                           lg_e_brd: float = 0.0,
                           angular_resolution: float = 0.5):
    # sources from file "source_table.csv" -- potential high-energy neutrino sources
    sources = get_sources("../data/source_table.csv")

    # transmission function initialization
    tf = TransmissionFunction(nuFate_method=1)

    # atmoshperic flux initialization
    atm = Atmosphere()

    telescope_name = "baikal_bdt_mk"

    # zenith-angle-dependent version
    baikal_trigger = RootTelescopeConstructor(telescope_name, "hnu_trigger").get()
    baikal_reco = RootTelescopeConstructor(telescope_name, "hnu_reco").get()
    baikal_std_cuts = RootTelescopeConstructor(telescope_name, "hnu_stdcuts").get()

    # number of clusters & time
    value_t, value_r, value_c = 20 * 5, 20 * 5, 20 * 5

    source = sources[source_number]  # take one source from the list

    ssf_trigger = PointSourceFlux(telescope=baikal_trigger, tf=tf, atm=atm, value=value_t, source=source,
                                  angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

    ssf_reco = PointSourceFlux(telescope=baikal_reco, tf=tf, atm=atm, value=value_r, source=source,
                               angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

    ssf_cuts = PointSourceFlux(telescope=baikal_std_cuts, tf=tf, atm=atm, value=value_c, source=source,
                               angular_resolution=angular_resolution, lg_energy_border=lg_e_brd)

    r_t, r_t_bg = ssf_trigger.signal(), ssf_trigger.background()
    r_r, r_r_bg = ssf_reco.signal(), ssf_reco.background()
    r_c, r_c_bg = ssf_cuts.signal(), ssf_cuts.background()

    rtp = RootPlotter(title=source.name, if_bg=if_bg)

    basic_line = "Baikal-GVD MC, 20 clusters, 5 yr"

    if if_bg:
        rtp.add_hist(baikal_trigger.energy, r_t_bg, "", "trig(BG)")
        rtp.add_hist(baikal_trigger.energy, r_t, basic_line + ", trigger", "trig")
        rtp.add_hist(baikal_reco.energy, r_r_bg, "", "reco(BG)")
        rtp.add_hist(baikal_reco.energy, r_r, basic_line + ", reco", "reco")
        rtp.add_hist(baikal_std_cuts.energy, r_c_bg, "", "cuts(BG)")
        rtp.add_hist(baikal_std_cuts.energy, r_c, basic_line + ", cuts", "cuts")
    else:
        rtp.add_hist(baikal_trigger.energy, r_t, basic_line + ", trigger", "trig")
        rtp.add_hist(baikal_reco.energy, r_r, basic_line + ", reco", "reco")
        rtp.add_hist(baikal_std_cuts.energy, r_c, basic_line + ", cuts", "cuts")

    rtp.draw()
    input("To finish enter any symbol: ")
    return


if __name__ == '__main__':
    point_like_source_plot(if_bg=True, source_number=0, angular_resolution=1)
