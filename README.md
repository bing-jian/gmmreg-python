# gmmreg-python
Python implementation of "Robust Point Set Registration Using Gaussian Mixture Models" by Jian &amp; Vemuri, PAMI'11.


Please note that we have another [github repo](https://github.com/bing-jian/gmmreg) that contains C++ implementation and 
more info about this work.

### Build and Test
Run the following steps to download, install and test the gmmreg python package:
  ```Shell
  git clone https://github.com/bing-jian/gmmreg-python.git
  cd gmmreg-python/src
  python setup.py install --user
  cd ../data
  python ../demo.py ./fish_partial.ini
  ```
If the gmmreg package was successfully installed, the last command should give a point set matching result like the image below:
<p align="center"> 
<img src="images/fish_partial_matching.png" width=640> 
</p>


### Citing

When using this code in a scientific publication, please cite 
```
@article{Jian&Vemuri_pami11,
  author  = {Bing Jian and Baba C. Vemuri},
  title   = {Robust Point Set Registration Using {Gaussian} Mixture Models},
  journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
  year = {2011},
  volume = {33},
  number = {8},
  pages = {1633-1645},
  url = {https://github.com/bing-jian/gmmreg/},
}
```

### Acknowledgement

We thank [Shen-Chi Chen](https://github.com/schen119) for helping the migration to Python 3.
