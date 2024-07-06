# Evaluates average high-energy neutrino spectral attenuation
# for specific sources listed in "data/sources_table.csv"
import ROOT as rt

from src.root_plotter import RootPlotter
from src.single_source_flux import ExtendedSourceFlux
from src.source_extended import galactic_center
from src.telescope import RootTelescopeConstructor, Telescope


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
    values = [20 * 5, 20 * 5, 20 * 5]
    telescopes = [baikal_trigger, baikal_reco, baikal_std_cuts]

    def get_signal_and_background(i: int):
        t1 = ExtendedSourceFlux(telescope=telescopes[i], value=values[i], ext_source=source, lg_energy_border=lg_e_brd)
        return (t1.signal(),
                t1.background(if_atm=True, if_astro=False),
                t1.background(if_astro=True, if_atm=False))

    # r_t, r_t_atm, r_t_astro = get_signal_and_background(0)
    # r_r, r_r_atm, r_r_astro = get_signal_and_background(1)
    r_c, r_c_atm, r_c_astro = get_signal_and_background(2)

    rtp = RootPlotter(title=source.name, if_bg=True)

    rtp.colors = [rt.kBlue, rt.kOrange-3, rt.kRed]

    rtp.fill = [3001, 3001, 3002]

    basic_line = "Baikal-GVD MC, 20 clusters, 5 yr"

    # rtp.add_hist(baikal_trigger.energy, r_t_bg, "", "trig(BG)")
    # rtp.add_hist(baikal_trigger.energy, r_t, basic_line + ", trigger", "trig")
    # rtp.add_hist(baikal_reco.energy, r_r_atm, "", "reco(atm)")
    # rtp.add_hist(baikal_reco.energy, r_r_astro, "", "reco(astro)")
    # rtp.add_hist(baikal_reco.energy, r_r, basic_line + ", reco", "reco")
    rtp.add_hist(baikal_std_cuts.energy, r_c, basic_line + ", cuts", "cuts")
    rtp.add_hist(baikal_std_cuts.energy, r_c_atm, "", "cuts(atm)")
    rtp.add_hist(baikal_std_cuts.energy, r_c_astro, "", "cuts(astro)")

    rtp.draw()
    input("To finish enter any symbol: ")
    return


if __name__ == '__main__':
    extended_source_plot(num=16)
