# Baikal Neutrino Registration Rate project
Author: Maksim Kleimenov

Last update: 07/24

## Brief description

Baikal neutrino registration rate project is created in order to make primitive estimations for the expected neutrino registration rates considering the following factors:
* Source visibility
* Neutrino flux attenuation in the Earth
* Effective area of the detector

This project also is designed to model the background flux from atmospheric neutrinos (in this version, 06/24 only conventional neutrinos are taken into consideration)

In the *src/tests* folder one will find examples of how to use the implemented classes and get neutrino registration rates estimation

The most important files in *src/tests* folder are:
* *point_like_source_plot.py* which generates a ROOT histogram of an expected registration rate spectral density for both signal (astrophysical neutrinos) and background (atmospheric neutrinos) 
* *point_like_source_total.py* which calculates total registration rates (events / year)
* *extended_source_plot.py* which uses the implemented classes to plot an expected registration rate spectrum for an extended source (galactic ridge)
* *extended_source_total.py* which calculates total flux from the extended source
* *full_sky_bg_test.py* which calculates the expected number of events from atmospheric and astrophysical neutrinos from the whole sky
