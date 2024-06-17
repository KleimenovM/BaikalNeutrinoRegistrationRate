# Evaluates average high-energy neutrino spectral attenuation
# for specific sources listed in "data/sources_table.csv"
from src.root_plotter import RootPlotter
from src.single_source_flux import ExtendedSourceFlux
from src.source_extended import galactic_center
from src.telescope import RootTelescopeConstructor


def extended_source_plot(if_bg: bool = False,
                         lg_e_brd: float = 0.0,
                         num: int = 100):
    # telescope
    telescope_name = "baikal_bdt_mk"

    # extended source
    source = galactic_center(num)

    # zenith-angle-dependent version
    baikal_trigger = RootTelescopeConstructor(telescope_name, "hnu_trigger").get()
    baikal_reco = RootTelescopeConstructor(telescope_name, "hnu_reco").get()
    baikal_std_cuts = RootTelescopeConstructor(telescope_name, "hnu_stdcuts").get()

    # number of clusters & time
    value_t, value_r, value_c = 20 * 5, 20 * 5, 20 * 5

    ssf_trigger = ExtendedSourceFlux(telescope=baikal_trigger, value=value_t, ext_source=source,
                                     lg_energy_border=lg_e_brd)

    ssf_reco = ExtendedSourceFlux(telescope=baikal_reco, value=value_t, ext_source=source,
                                  lg_energy_border=lg_e_brd)

    ssf_cuts = ExtendedSourceFlux(telescope=baikal_std_cuts, value=value_c, ext_source=source,
                                  lg_energy_border=lg_e_brd)

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
    extended_source_plot(if_bg=True, num=20)
