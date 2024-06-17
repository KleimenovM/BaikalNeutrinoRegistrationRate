# Evaluates average high-energy neutrino spectral attenuation
# for specific sources listed in "data/sources_table.csv"
from src.root_plotter import RootPlotter
from src.single_source_flux import ExtendedSourceFlux
from src.source_extended import galactic_center
from src.telescope import RootTelescopeConstructor


def extended_source_total(if_bg: bool = False,
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
    baikal_prev_cuts = RootTelescopeConstructor("baikal_2023_new", "hnu_stdcuts").get()

    # number of clusters & time
    value_t, value_r, value_c = 20 * 5, 20 * 5, 20 * 5

    ssf_trigger = ExtendedSourceFlux(telescope=baikal_trigger, value=value_t, ext_source=source,
                                     lg_energy_border=lg_e_brd)

    ssf_reco = ExtendedSourceFlux(telescope=baikal_reco, value=value_t, ext_source=source,
                                  lg_energy_border=lg_e_brd)

    ssf_cuts = ExtendedSourceFlux(telescope=baikal_std_cuts, value=value_c, ext_source=source,
                                  lg_energy_border=lg_e_brd)

    ssf_cuts_prev = ExtendedSourceFlux(telescope=baikal_prev_cuts, value=value_c, ext_source=source,
                                       lg_energy_border=lg_e_brd)

    if if_bg:
        r_t, r_t_bg = ssf_trigger.total_signal(), ssf_trigger.total_background()
        r_r, r_r_bg = ssf_reco.total_signal(), ssf_reco.total_background()
        r_c, r_c_bg = ssf_cuts.total_signal(), ssf_cuts.total_background()
        print(f'{source.name} & {r_t} & {r_t_bg} & {r_r} & {r_r_bg} & {r_c} & {r_c_bg} ' + r'\\')
    else:
        r_t = ssf_trigger.total_signal()
        r_r = ssf_reco.total_signal()
        r_c = ssf_cuts.total_signal()
        r_c_p = ssf_cuts_prev.total_signal()
        print(f'{source.name} & {r_t} & {r_r} & {r_c} & {r_c_p} ' + r'\\')

    return


if __name__ == '__main__':
    extended_source_total(if_bg=True, num=100, lg_e_brd=2)
