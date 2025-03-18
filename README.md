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

## Getting Started

#### Set your "Base Folders"
The "Base Folders" window allows you to set default folders and files for data analysis.

1) Raw Data
* Select the directory where your raw data files are stored. The default location is the Toothy folder itself; updating this location is optional but convenient for selecting raw recording data for initial processing.

3) Probe Files
* Select the directory where your probe configuration files will be stored. The application automatically creates a ```probe_configs``` directory within the Toothy folder and creates a ```demo_probe_config.json``` file as an example probe object.

4) Default Probe
* If your recordings tend to use the same probe, you can optionally select a default probe configuration file that will be automatically loaded during the data processing phase.

5) Default Parameters
* Select the .txt file containing the parameter values that you want to use for data processing. The application automatically generates a ```default_params.txt``` file with reasonable initial values, which can be changed in the next step.
