# Farrell-Lab

## Installation
1) Download Miniconda or Anaconda Navigator
* Miniconda: https://docs.anaconda.com/miniconda/miniconda-install/
* Anaconda Navigator: https://www.anaconda.com/download
  * Navigator provides a GUI and a large suite of packages/applications, but takes up much more disk space
  
2) Open Anaconda Prompt terminal window
  
3) Allow package installations from conda-forge channel
```
conda config –add channels conda-forge
conda config –set channel_priority strict
```

4) Set the current directory to the location where you want to download the code folder
```
cd [PATH_TO_PARENT_FOLDER]
```
* e.g. ```cd C:\Users\Amanda Schott\Documents\Data```

5) Install ```git``` package, clone the GitHub repository to a new folder on your computer, then set the current directory to the new folder with all the code files
```
conda install git
git clone https://github.com/fear-the-kraken/Farrell-Lab [FOLDER_NAME]
cd [FOLDER_NAME]
```

6) Create a new Anaconda environment running Python 3.9 for the Toothy application, then activate your new environment
```
conda create -n <myenv> python=3.9
conda activate <myenv>
```

7) Run the following lines (one at a time) to install the necessary packages in your new Anaconda environment
   * Note that ```conda install``` is used when possible, ```pip install``` is used otherwise
```
pip install neo==0.13.1
conda install numpy=1.26.0
conda install pandas=2.1.1
conda install matplotlib=3.8.0
conda install scipy=1.11.4
conda install seaborn=0.13.0
pip install ipython==8.12.3
conda install scikit-learn=1.2.2
conda install pyqt=5.15.10
pip install open-ephys-python-tools==0.1.10
conda install probeinterface=0.2.23
```

8) Run the ```import_packages.py``` file to ensure that all required packages are installed
```
python import_packages.py
```
* If you encounter a ```ModuleNotFoundError```, perform the following steps:
  * Try installing the missing module using conda
    * ```conda install [MODULE_NAME]```
  * If the above results in a ```PackagesNotFoundError```, install the module using ```pip```
    * ```pip install [MODULE_NAME]```
  * In most cases, ```MODULE_NAME``` will be the same as the package name in the error message. However, the following exceptions apply:
    * Module ```sklearn``` must be installed as ```scikit-learn```
    * Module ```PyQt5``` must be installed as ```pyqt```
    * Module ```open_ephys``` must be installed as ```open-ephys-python-tools```
* Repeat the above steps until ```import_packages.py``` runs successfully

9) Run the application!
```
python hippos.py
```
