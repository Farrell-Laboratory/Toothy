# Toothy: a graphical user interface for curating dentate spikes

# Installation

Requires Python 3.11.

Create a conda environment with Python 3.11. For example, here we create an environment called "toothy-env":

```
conda create -n toothy-env python=3.11
```

Activate the environment:

```
conda activate toothy-env
```

Install toothy using pip:

```bash
pip install toothy
```

When in your environment, you should be able to launch the Toothy GUI with just the command `toothy`:

```bash
toothy
```

On first launch, Toothy creates `toothy_paths.txt`, `probe_configs/`, and `toothy_default_parameters.json` in your **current working directory**.

# Getting Started

### Set the "Base Folders"

The "Base Folders" window allows users to set default folders and files for data analysis. This information is stored in a `toothy_paths.txt` file, which is automatically generated upon startup in the directory from which you run Toothy.

**Raw Data Folder:** Select the directory where your raw data files are stored; the default location is the Toothy folder itself. Updating this location is optional but convenient when selecting raw recording data for initial processing.

**Probe Configuration Folder:** Select the directory where your probe configuration files will be stored. The application automatically creates a `probe_configs` directory within the Toothy folder and creates a `demo_probe_config.json` file as an example probe object.

**Default Probe File:** If your recordings tend to use the same probe, you can optionally select a default probe configuration file that will be automatically loaded during the data processing phase. If this field is blank (the default state), probes will be loaded manually for each recording.

**Parameter File:** Select the TXT file containing the parameter values that you want to use for data processing. The application automatically generates a `default_params.txt` file with reasonable initial values, which can be changed in the next step. To generate a new parameter file with default values, press the button.

---

### Set the analysis parameters

The "Parameters" window allows users to view and edit the parameters used for data processing and analysis, which are stored in the TXT file specified in the previous step.

A short description of each parameter can be displayed by hovering over its label, and changes can be saved either to the current parameter file or as a new TXT file.

# Probe Creation

The probe designer uses the `probeinterface` Python package to create a software representation of the electrode geometry and channel mapping of specific neural probes, which is stored as a JSON file. The **"Build"** window can be used to create a probe completely from scratch by specifying number of channels and electrode geometry, while the **"Paste"** window accepts input lists of x and y-coordinates.

**"Build"**

1. Set the number of channels and the number of shanks. For multi-shank probes, the number of channels per shank and the shank spacing (um) must also be specified.
2. Set the electrode geometry as a Linear/Edge configuration (one electrode column per shank), a Polytrode configuration (2+ columns per shank), or a Tetrode configuration (groups of 4 closely spaced electrodes).
3. Set the electrode spacing for the specified probe configuration.

- Inter-electrode spacing: distance between electrodes along the shank (*linear/polytrode*)
- Intra-electrode spacing: distance between electrodes across the shank (*polytrode*)
- Inter-site spacing: distance between tetrode recording sites along the shank (*tetrode*)
- Intra-site spacing: distance between the most lateral (X) and vertical (Y) electrodes within a single recording site (*tetrode*)
- Tip offset: distance between the tip of the shank and the deepest electrode (*linear/polytrode*) or recording site (*tetrode*). For polytrodes, this parameter can be set individually for each column.

1. Set the electrode contact shape (circles, squares, or rectangles) and size (area/radius/width/height).

**"Paste"**

1. Input lists of comma-separated x-coordinates and y-coordinates corresponding to each channel.
2. Set the shank ID for each channel by entering another list into the text field, or by pressing the "Set..." button to manually map each unique x-coordinate to a shank ID.
3. Set the electrode contact shape and size.

**Channel Mapping:** set the device indices for mapping the contact indices of the probe to the logical channel indices of the recording device; this depends on the wiring of the particular probe and headstage. Data may be entered as comma-separated values ("Text field") or as values in a table column ("Table")

- If the "Channel Mapping" box is left unchecked, Toothy will assume that the Nth electrode contact corresponds to the Nth row of raw data
- When creating a probe from x and y data, the "Use coordinates" button enables automatic mapping of the contact indices to the physical channel positions. For instance, device index 0 is the index of the shallowest contact (maximum y-value) on the leftmost electrode column (minimum x-value) in the inputted lists of x and y-coordinates

**Actions**

When all necessary probe parameters have been supplied, the **Generate** button (bottom row) will create a `probeinterface.Probe` object and launch a pop-up window with a visual representation of the probe. This external plot can interactively display the contact indices, device indices, and shank IDs over each channel to ensure that the configuration is correct.

Other Buttons

**Load:** load existing probe configuration file into the "Build" or "Paste" window

**Plot:** view the current probe in the external plotting window

**Save:** save the current probe as a JSON configuration file

**Clear:** reset all probe parameters to their default states

# Data Ingestion

The "Raw Data" window provides a pipeline for loading raw recordings, assigning probes, pre-processing the data, and saving the files in a new `processed_data` folder. Users can also set the analysis parameters for a given recording by expanding the "Settings" panel.

### Loading Data from a Supported Recording System

 Toothy supports automatic data loading from the following acquisition systems:

- **NeuroNexus:** data source must contain a `.xdat.json` metadata file
- **OpenEphys:** data source must contain a `structure.oebin` metadata file
- **Neuralynx:** data source must contain unique `.ncs` files for each channel

: select raw data directory from a supported recording system

: select raw data file in a supported format (see below)

### Loading Data from a File

To analyze electrophysiology signals from a non-supported recording system, Toothy can also load 2-dimensional data arrays (channels x timepoints) from `.npy` and `.mat` files. Since these files lack contextual metadata, the user must provide information about the recording into a popup window.

**Data Array:** label data dimensions and channel ordering

- Specify whether data rows represent channels or time points
- Specify whether the data channels are organized from shallowest to deepest (or vice versa)

**Recording:** set key recording parameters

- Set the recording sampling rate (Hz); the recording duration is automatically calculated using the number of time points
- Set the SI units (uV, mV, V, or kV) of the data

---

### Assigning Probes

After the raw data source is loaded, each data row must be mapped to a unique probe channel. The central panel shows all probes currently associated with the recording, and a dynamically updated text box (right) displays the total number of probe channels along with the number of data rows; these values must be identical in order to proceed.

**Assign probes to the recording:**  
Load button: load an existing probe configuration file  
Create button: open the probe designer window to make a new probe  

: duplicate assigned probe and add to the recording

: remove assigned probe from the recording

**Set indexing mode:**  
Contiguous rows: each probe is assigned to a block of N consecutive rows in the data array  
Alternating rows: each probe is assigned to N rows evenly distributed throughout the array

**View probe assignments**  
View button: display a table with all data rows and their corresponding probe IDs

- For unassigned rows, the "Probe" column is left blank

# Analyzing the Recording

The  "Data Analysis" window manages event channel selection and DS classification for processed recordings, which can be selected using the file button.

For a valid recording folder, the window will display dropdown menus allowing the user to select a specific probe and shank for analysis; the example recording has one probe and three individual shanks

"Select event channels": launches the main analysis GUI for visualizing recording data, determining optimal event channels, and curating event datasets.

"Classify dentate spikes": launches the DS classification GUI for estimating CSDs and identifying DS1 vs DS2 dentate spikes. This option is enabled when the user saves an optimal DS channel and dataset via the main analysis GUI.

# Selecting Event Channels

The channel selection window contains numerous interactive features for analyzing hippocampal recordings, with the main goal of determining the optimal LFP channels for dentate spikes, sharp-wave ripples, and theta frequency band power (indicating the hippocampal fissure).

 

## General Controls

The central plot shows the LFP signal for each channel on a given shank in the selected probe, which can be toggled using the lists in the top right hand corner. The plot initially shows a 2 second viewing window in the middle of the recording, which can be moved and scaled using the above sliders.

**Navigation:** the *main slider (purple)* controls the position of the viewing window, allowing users to quickly scroll through the recording

- *Left and right arrow keys:* shift the viewing window back and forth by 25%, allowing users to incrementally step through the data

**Scaling:** the *secondary sliders (blue)* control the width, height, and data amplitude of the viewing window

- *X slider:* adjusts the time range of the viewing window to zoom in/out of the recording
- *Y slider:* adjusts the height of the central plot to zoom in/out on LFP channels
- *Z slider:* adjusts the amplitude of each LFP to flatten or magnify the signal

**Live CSD Plotting:** a *span selector (red box)* is used to select a time interval for calculating a current source density (CSD) plot

(1) Click and drag the mouse across the central plot to visually select the desired time range  

(2) Press the `Enter` key to estimate the CSD, displaying the resulting heatmap over the selected LFPs

## The Recording Tab

The "Recording" tab in the settings sidebar contains generally useful widgets for navigating, cleaning, and taking notes on the current recording.

**Jump To:** centers the viewing window at a specific position, allowing users to quickly jump between events of interest

- *Time:* jump to the given time point (s)
- *Index:* jump to the given recording index

*To copy a time point or index to the clipboard, right-click the central plot and select "Copy time" or "Copy index" in the popup menu*

**Noise Channels:** designates channels as "clean" (default) or "noise" (unsuitable for event detection). Noisy channels are shown as flat gray lines on the central plot, and their data is ignored when normalizing channel data, calculating CSDs, plotting frequency band power, etc.

- Set Channel as Noise: select the target channel item in the dropdown menu, then click the green arrow button to move the channel to the "noise" list
- Set Channel as Clean: select the target channel item in the "noise" list, then click the "Restore channel(s)" button to reclassify the channel as "clean"

*Users can also right-click the LFP channel in the central plot and select "Mark as noise" or "Mark as clean" in the popup menu*

**NOTES:** built-in documentation that links the text input field to a `notes.txt` file in the recording folder.

- The GUI automatically loads the contents of the text file on startup, and the button writes the current content of the text field to disk

## The Events Tab

### Frequency Band Plots

Frequency band plots display the relative power in the theta (~~6-10 Hz), ripple (~~120-180 Hz), and gamma (~25-55 Hz; ~60-100 Hz) frequency bands across all shank channels. The Y-axes of the frequency plots align with the central plot for cross-referencing, and the current event channels (see below) are marked by color-coded lines and dynamically updated.

- "Show freq. band power" button: toggles the visibility of the frequency band plots
- Designated "noise" channels appear as blank spaces and are not used in normalization

### Event Boxes

Event boxes are the central hub for setting event channels and analyzing DS and SPW-R datasets.

**Event Channel Assignment:** users can set each event channel through the *channel input* at the top of the corresponding event box. The LFP signals are color-coded to reflect the current event channels for DSs (red), SPW-Rs (green), and theta power (blue), and the central plot displays DS and SPW-R events detected on the specified channel.

button: resets the event channel to its initial value

---

**Viewing Events:** DSs and SPW-Rs detected on the current event channels are marked by solid red and green vertical lines on the central plot. Dotted lines are used for events manually added by the user, and dashed lines represent detected events manually deleted by the user.

button: toggle visibility of event markers on the central plot

**← →**   buttons: move the viewing window to the next (→) or previous (←) event from the current position

"Show deleted events" option: toggle visibility of user-deleted events on the central plot

---

**Editing Events:** users may curate DS and SPW-R datasets by manually adding or removing event instances

*Add an Event:* manually insert a DS or SPW-R at time point *t*  

(1) Check the "Add" box for the desired event type  

(2) Double-click the mouse on the central plot, as close as possible to time point *t*

*Delete an Event:* delete all DS and SPW-R events within a given time span  

(1) Click and drag the mouse horizontally across the central plot to surround the target event markers  

(2) Press the `Backspace` key to delete all visible events within the selected window

*Restore an Event:* return previously deleted DS and SPW-R events to their respective datasets  

(1) Check the "Show deleted events" box for the desired event type(s)  

(2) Click and drag the mouse to surround the target deleted event markers  

(3) Press the `Spacebar` to restore all deleted events within the selected window

*Permanently Erase an Event:* delete all event information so that it cannot be restored  

(1) Click and drag the mouse to surround the target event markers  

(2) Press the `Escape` key to erase all visible events within the selected window

## Event Analysis Popups

For more detailed analysis of DSs and SPW-Rs, users can open event-specific GUIs from the "Events" tab by pressing the *"View DS"* or the *"View ripples"* button. These windows will be initialized with the current event channel as the "primary" channel, allowing users to review individual events (**Single Event Mode**, left) or compare mean event waveforms with other channels (**Average Mode**, right).

**Static parameter distributions**  

The top row of the GUI displays three statistical subplots comparing events across all channels, with data points color-coded by magnitude for clarity.  

(1) Event count: number of events detected on each channel  

(2) Event amplitude: peak amplitudes of DS waveforms or sharp-wave ripple envelopes  

(3A) DS height above surround: DS waveform peak heights relative to surrounding signal  

(3B) Ripple/theta power: ratios of ripple power to theta power during SPW-Rs

The *"Highlight data from current channel"* option outlines the data from the primary event channel in red for easy visual identification.

 

**Single Event Mode**

 Users navigate through the set of event waveforms on the primary channel, displayed individually on the plot. The *main slider (purple)* is used to scroll through the event dataset (in chronological order by default), and the *left and right arrow buttons* step backward or forward by one event at a time.

- Events can be reordered by any parameter in the **SORT** section of the sidebar, allowing users to inspect the waveforms at each extreme. These attributes are displayed for each event instance as a text annotation

---

**Average Mode**

 Users compare event morphology between the primary channel and other candidate channels by overlaying their mean LFP waveforms on the same plot. Candidate channels are chosen from the dropdown menu in the Add channel section of the sidebar, and the green arrow button adds the event waveform of the selected channel to the plot

- Added waveforms are plotted in a random color, which is displayed in the legend and as a data highlight in the statistical subplots
- The *Clear channels* button removes all added waveforms from the plot, and the primary channel waveform is shown ±SEM

---

**View Options**

- The *Raw* and *Filtered* plot buttons display either the "standard" LFP signal or the bandpass-filtered LFP used for event detection
- The *X slider* adjusts the size of the event window to show more/less of the surrounding signal
- The *Y slider* scales the Y-axis of the LFP plots
- The **VIEW** parameters in the sidebar control the visibility of various plot annotations
  - Thresholds: show or hide event detection thresholds (e.g. min. peak height, min. envelope height, min. ripple duration)
  - Data Features: show or hide event attributes (e.g. DS half-width/height at half-prominence, ripple envelope/duration)
  - Axes: show or hide X and Y-axes

## Saving Event Data

When all event channel inputs are set to the optimal values, pressing the Save button will save the event data for the currently loaded shank and probe. Any probe shanks without saved data are missing from the following CSV tables and represented as empty lists in the event channel file.

`theta_ripple_hil_chan_[PROBE].npy` : a nested list of [theta, SPW-R, DS] channels for each shank in the probe

`DS_DF_[PROBE]` and `SWR_DF_[PROBE]` : CSV files containing DS and SPW-R datasets for the probe

# Classifying Dentate Spikes

The DS classification window is used to estimate current source density (CSD) profiles for detected dentate spikes, followed by principal components analysis (PCA) and clustering to classify DS1 and DS2 events.

### Set the CSD Window

The central plot shows the mean LFPs for each channel surrounding DS events, using the same color-coding to label the DS/hilus channel (red), the SPW-R channel (green), and the theta/fissure channel (blue). Noisy channels are shown as flat gray lines and interpolated for CSD calculation 

 The *CSD slider* controls the range of the CSD window (cyan), which determines the channels used for CSD analysis

- The default CSD window spans from the hilus to the fissure

### Set the CSD Parameters

Probe Settings: spatial and electrical properties of the current source  

CSD Mode: parameters for calculating and filtering CSDs  

Clustering Algorithm: parameters for clustering analysis (K-means or DBSCAN)  

****Additional details are available in the main Parameter Window*

**Calculate:** estimate CSDs using the `icsd` Python module  

**Save:** save CSDs and classifications to disk

   