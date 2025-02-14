# Farrell-Lab

## Installation
1) Download Miniconda or Anaconda Navigator
* Miniconda: https://docs.anaconda.com/miniconda/miniconda-install/
* Anaconda Navigator: https://www.anaconda.com/download
  * Navigator provides a GUI and a large suite of packages/applications, but takes up much more disk space
  
2) Open Anaconda Prompt terminal window
  
4) Allow package installations from conda-forge channel
```
conda config –add channels conda-forge
conda config –set channel_priority strict
```

5) Set the current directory to the location where you want to download the code folder
```
cd [PATH_TO_PARENT_FOLDER]
```
* e.g. ```cd C:\Users\Amanda Schott\Documents\Data```

6) Install ```git``` package, clone the GitHub repository to a new folder on your computer, then set the current directory to the new folder with all the code files
```
conda install git
git clone https://github.com/fear-the-kraken/Farrell-Lab [FOLDER_NAME]
cd [FOLDER_NAME]
```

7) Install all necessary packages by trial and error
```
python import_packages.py
```
* When you encounter a ```ModuleNotFoundError```, perform the following steps:
  * Try installing the missing module using conda
    * ```conda install [MODULE_NAME]```
  * If the above results in a ```PackagesNotFoundError```, install the module using ```pip```
    * ```pip install [MODULE_NAME]```
  * In most cases, ```MODULE_NAME``` will be the same as the package name in the error message. However, the following exceptions apply:
    * Module ```sklearn``` must be installed as ```scikit-learn```
    * Module ```PyQt5``` must be installed as ```pyqt```
    * Module ```open_ephys``` must be installed as ```open-ephys-python-tools```
* Repeat the above steps until ```import_packages.py``` runs successfully

8) Run the application!
```
python hippos.py
```
