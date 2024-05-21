import numpy as np
import ROOT as rt

from src.telescope import RootTelescopeConstructor


def draw_eff_area():

    baikal = RootTelescopeConstructor("baikal_bdt_mk", "hnu_stdcuts").get()

    m, n = 200, baikal.lg_energy.size

    a = np.linspace(np.pi/2 + 1e-5, np.pi, m)
    lg_e, e = baikal.lg_energy, baikal.energy

    f_xv = baikal.effective_area(np.cos(a), lg_e) + 1e-9

    print(f_xv)

    canvas = rt.TCanvas("c", "c", 800, 600)
    canvas.SetLeftMargin(.1)
    canvas.SetBottomMargin(.1)
    canvas.SetRightMargin(.18)

    hist = rt.TH2F("Title", "Reconstruction single cluster effective area, MC", n-1, e, m-1, np.rad2deg(a))

    for i, a_i in enumerate(a):
        for j, e_j in enumerate(e):
            hist.Fill(e_j, np.rad2deg(a_i), f_xv[i, j])

    size = .04

    axes = hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis()
    labels = ["E, GeV", "#theta, deg", "A_{ef}, m^{2}"]

    for i, axis in enumerate(axes):
        axis.SetTitle(labels[i])
        axis.SetTitleSize(size)
        axis.SetLabelSize(size)
        axis.SetTitleOffset(1.2)

    rt.gStyle.SetOptStat(0)
    # canvas.SetGrayscale()
    hist.Draw("colz")
    canvas.SetLogx()
    canvas.SetLogz()

    input("Enter any symbol to quit: ")

    return


if __name__ == '__main__':
    draw_eff_area()
