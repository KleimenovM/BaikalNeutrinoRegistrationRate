### Transmission matrices

Transmission matrices are computational results of the Transmission Matrix calculation with nuFATE:

see `src/transmission_calculation.py`

```
transmission_calculator = TransmissionCalculator(nuFateMethod=#, flavor=##)`
transmission_calculator.run(if_root=False, if_npy=True)
```

Flavors:

0 - e, 1 - mu, 2 - tau

Methods:

0 - no secondaries;

1 - with secondaries for e and mu

2 - with secondaries for tau

-----

The files `data_mod_{f}.npy` are numpy tables storing the transmission matrix for the energy grid [lg(E/GeV), deg] as described below for flavor `f` calculated with nuFateMethod `1` for `f=0,1` and `2` for `f=2`.

```
lg_min, lg_max, self.n = 3, 10, 200
self.d_lg_e = (lg_max - lg_min) / self.n
self.lg_e = np.linspace(3, 10, 200)
self.energy = 10 ** self.lg_e

z_angle_min, z_angle_max, self.m = np.pi / 2, np.pi, 180
self.z_angles = np.linspace(z_angle_min, z_angle_max, self.m)
```
