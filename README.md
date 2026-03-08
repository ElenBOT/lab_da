# LAB Data Analysis
General-purpose laboratory data analysis toolkit with native support for Labber log files.

# download
```git bash
git clone https://github.com/ElenBOT/lab_da.git
```

# Features
* dataholder: container for 1D and 2D data, provide name and unit attribute to keep it informative, provide cutting and slicing method (slice 1D data from 2D data, cut data with given value).
* labberreader: Provide object to read raw data from .hdf5 file of labber log file, also comes with `auto_xyz` function to read 1D and 2D data.
