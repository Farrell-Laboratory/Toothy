# Farrell-Lab

## Installation
1) Download Miniconda or Anaconda Navigator
* Miniconda: https://docs.anaconda.com/miniconda/miniconda-install/
* Anaconda Navigator: https://www.anaconda.com/download
  * Navigator provides a GUI and a large suite of packages/applications, but takes up much more disk space

2) Download the Toothy ZIP file from GitHub and move the folder to the desired location

3) Open an Anaconda Prompt terminal window and set the current directory to the Toothy folder
```
cd [PATH_TO_TOOTHY_FOLDER]
```
* e.g. ```cd C:\Users\Amanda Schott\Documents\Data\Toothy-main```

6) Create a new Anaconda environment for the Toothy application using the provided ```environment.yml``` file, then activate your new environment
```
conda env create -f environment.yml
conda activate toothy_gui_env
```

7) Run the application!
```
python toothy.py
```
