from src.telescope import RootTelescopeConstructor


def first_simple_test():
    telescope_name = "baikal_2021"
    rtc = RootTelescopeConstructor(telescope_name, "hnu")
    rtc.get()

    telescope_name2 = "baikal_bdt_mk"
    rtc2 = RootTelescopeConstructor(telescope_name2, "hnu_trigger")
    rtc2.get()
    return


if __name__ == '__main__':
    first_simple_test()
