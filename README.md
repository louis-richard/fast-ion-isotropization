# Fast Ion Isotropization by Current Sheet Scattering in Magnetic Reconnection Jets
[![GitHub license](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE) [![LASP](https://img.shields.io/badge/datasets-MMS_SDC-orange.svg)](https://lasp.colorado.edu/mms/sdc/)

Code for the paper [Fast Ion Isotropization by Current Sheet Scattering in Magnetic Reconnection Jets](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.131.115201)

## Abstract
We present a statistical analysis of ion distributions in magnetic reconnection jets using data from the Magnetospheric Multiscale spacecraft. Compared with the quiet plasma in which the jet propagates, we often find anisotropic and non-Maxwellian ion distributions in the plasma jets. We observe magnetic field fluctuations associated with unstable ion distributions, but the wave amplitudes are not large enough to scatter ions during the observed travel time of the jet. We estimate that the phase-space diffusion due to chaotic and quasiadiabatic ion motion in the current sheet is sufficiently fast to be the primary process leading to isotropization.


## Requirements
- A [`requirements.txt`](./requirements.txt) file is available at the root of this repository, specifying the
 required packages for our analysis.

- Routines specific to this study [`ionaniso`](./IonAniso) is
  pip-installable: from the [`IonAniso`](./IonAniso) folder run `pip
  install .`

## Reproducing our results
Ion temperature anisotropies and non-Maxwellianities are calculted using `compile.py`

```console
foo@bar:~$ python3.10 compile.py -h
Load IGRF coefficients ...
usage: Calculate the temperature anisotropy, non-Maxwellianity, etc. in the dataset of reconnection jets. [-h] [--start START] [-a AVERAGE]

options:
  -h, --help            show this help message and exit
  --start START, -s START
                        Index of the first time interval to compute
  -a AVERAGE, --average AVERAGE
                        Number of ion VDFs to use in averaging

```

To compile the dataset used in the paper run

```bash
python3.10 compile.py -s 0 -a 3
```

Figures in the paper can be reproduced using `plot_distribution.py`, `plot_flapping.
py`, and `plot_tau.py`.


To reproduce Figure 1 run `plot_distribution.py`

```bash
python3.10 plot_distribution.py -a 3
```

To reproduce Figure 2 run `plot_flapping.py`

```bash
python3.10 plot_flapping.py -a 3
```

To reproduce Figure 3 run `plot_tau.py`

```bash
python3.10 plot_tau.py -a 3
```

## Citation

If you found this code and findings useful in your research, please consider citing:

```bibtex
@ARTICLE{2023PhRvL.131k5201R,
       author = {{Richard}, Louis and {Khotyaintsev}, Yuri V. and {Graham}, Daniel B. and {Vaivads}, Andris and {Gershman}, Daniel J. and {Russell}, Christopher T.},
        title = "{Fast Ion Isotropization by Current Sheet Scattering in Magnetic Reconnection Jets}",
      journal = {\prl},
     keywords = {Physics - Space Physics},
         year = 2023,
        month = sep,
       volume = {131},
       number = {11},
          eid = {115201},
        pages = {115201},
          doi = {10.1103/PhysRevLett.131.115201},
archivePrefix = {arXiv},
       eprint = {2301.10139},
 primaryClass = {physics.space-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023PhRvL.131k5201R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```



## Acknowledgement
We thank the entire MMS team and instrument PIs for data access and support. All of the data used
 in this paper are publicly available from the MMS Science Data Center https://lasp.colorado.edu
 /mms/sdc/. Data analysis was performed using the pyrfu analysis package available at https://github.com/louis-richard/irfu-python. This work is supported by the SNSA grant 139/18.