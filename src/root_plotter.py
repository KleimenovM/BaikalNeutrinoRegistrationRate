import ROOT as rt
import numpy as np


class RootPlotter:
    def __init__(self, if_narrow: bool = False, if_bg: bool = False, title=""):
        self.width = 800
        self.height = 600 - 200 * if_narrow

        self.if_narrow = if_narrow
        self.if_bg = if_bg
        self.fs = 24 - 4 * if_narrow

        self.title = title

        self.canvas = rt.TCanvas("", "c", self.width, self.height)
        self.set_main_canvas()

        self.hists = []
        self.names = []
        self.subscripts = []

        self.index = 0

        self.xtitle = "E, GeV"
        self.ytitle = "events per 5 years per cluster"

        self.fill = [3654, 3645, 3001]
        self.colors = [602, 633, 419]
        self.set_color_palette()

    def set_main_canvas(self):
        self.canvas.SetTopMargin(.16 - .03 * self.if_narrow)
        self.canvas.SetLeftMargin(.13)
        self.canvas.SetBottomMargin(.13 + .03 * self.if_narrow)
        self.canvas.SetRightMargin(.05)
        self.canvas.SetLogx()
        if self.if_bg:
            self.canvas.SetLogy()
        rt.gStyle.SetTitleAlign(11)
        rt.gStyle.SetTitleX(.99)
        rt.gStyle.SetOptStat(0)
        return

    def add_hist(self, x: np.ndarray, y: np.ndarray,
                 title: str, subscript: str = "", x0=None):
        """
        Creates a ROOT.TH1F from the given data
        :param x: (np.ndarray) -- energy bin values
        :param y: (np.ndarray) -- flux bin values
        :param title: (str) -- name of the histogram
        :param subscript:
        :param x0: optional, (np.ndarray) -- x_axis bin values
        :return: (rt.TH1F) -- root histogram
        """
        if x0 is None:
            lg_x = np.log10(x)
            d_lg_x = lg_x[1] - lg_x[0]
            x0 = 10**(lg_x + 1/2 * d_lg_x)

        m = x0.size

        n = y.size
        hist = rt.TH1F(title, title, m - 1, x0)

        for i in range(n):
            hist.Fill(x[i], y[i])

        hist.SetTitleOffset(0.2)

        size = .04 * (1 + self.if_narrow / 3)

        hist.GetXaxis().SetTitle(self.xtitle)
        hist.GetXaxis().SetTitleOffset(1.5 - .2 * self.if_narrow)
        hist.GetXaxis().SetLabelOffset(0.015)
        hist.GetXaxis().SetTitleSize(size)
        hist.GetXaxis().SetLabelSize(size)

        hist.GetYaxis().SetTitle(self.ytitle)
        hist.GetYaxis().SetTitleOffset(1.5 - .5 * self.if_narrow)
        hist.GetYaxis().SetTitleSize(size)
        hist.GetYaxis().SetLabelSize(size)

        hist.SetLineWidth(2)
        hist.SetLineColor(self.colors[self.index])

        hist.SetFillStyle(self.fill[self.index])
        hist.SetFillColorAlpha(self.colors[self.index], .9)

        self.hists.append(hist)
        self.names.append(title)
        self.subscripts.append(subscript)
        self.index += 1
        return

    @staticmethod
    def add_text(font_size: int = 22, align_left: bool = False):
        """
        Basic module for adding text
        :param font_size: (int) font size, pt
        :param align_left: (bool) if true, aligns left, else -- right
        :return: (ROOT.TLatex) text object
        """
        text = rt.TLatex(.5, .5, "")
        text.SetTextFont(43)
        text.SetTextSize(font_size)
        if align_left:
            text.SetTextAlign(11)
        else:
            text.SetTextAlign(22)
        return text

    def set_color_palette(self):
        if self.if_bg:
            sh = 1
            self.fill = [0, 3554, 0, 3545, 0, 3002]
            self.colors = [602 + sh, 602, 633 + sh, 633, 419 + sh, 419]
        return

    def draw(self, caption_pos: str = "right"):
        n = len(self.hists)
        for i in range(n):
            self.hists[i].Draw("same hist")

            integral_i = np.round(self.hists[i].Integral(), 2)

            if caption_pos == 'left':
                x_0 = .17
            else:
                x_0 = .67 - .04 * self.if_bg

            y_0, dy = .75 + .02 * self.if_bg, .08 * (1 + self.if_narrow)

            text = self.add_text(self.fs - 2 * self.if_bg, align_left=True)
            text.SetTextColor(self.colors[i])
            text.DrawLatexNDC(x_0, y_0 - i * dy,
                              'R_{' + self.subscripts[i] + '} = ' + str(integral_i) + ' #frac{counts}{5 years}')

            # right-corner commentary
            if not self.if_narrow:
                text = self.add_text(self.fs, align_left=True)
                text.SetTextColor(self.colors[i])
                text.DrawLatexNDC(.35, .985 - .050 * i * (1 - 0.5 * self.if_bg), self.names[i])

        text = self.add_text(self.fs + 2, align_left=False)
        text.DrawLatexNDC(.15, .95, self.title)
        self.canvas.Update()
        input("Enter any symbol to exit ")
        return


if __name__ == '__main__':
    print("Not for direct use")

