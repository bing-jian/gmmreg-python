# gmmreg-python
Python implementation of "Robust Point Set Registration Using Gaussian Mixture Models" by Jian &amp; Vemuri, PAMI'11

Run the following steps to download, install and demo the library:
  ```Shell
  git clone https://github.com/bing-jian/gmmreg-python.git
  cd gmmreg-python/src
  python setup.py install
  cd ../data
  python ../demo.py ./fish_partial.ini
  ```
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
