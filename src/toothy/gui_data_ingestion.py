"""
Toothy 

Purpose: Data ingestion (input data selection and data processing pipeline)

Authors: Amanda Schott, Kathleen Esfahany

Last updated: 2025-10-08

TODO
- Replace icons for the probe selection widget black and white circle
- TODO review probe_hanlder
- TODO implement view_data_assignments better: should open nicely, should be read-only, etc.
"""
# Import standard libraries
import os
import shutil
from pathlib import Path
import time as timelib
import pickle

# Import third-party libraries
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
import h5py
import probeinterface as prif
from PyQt5 import QtWidgets, QtCore, QtGui

# Import custom modules
from . import QSS
from . import pyfx
from . import qparam
from . import helpers_io as h_io
from . import hdf5_dataframe as h5df
from . import gui_items as gi
from . import data_processing as dp
from .probe_handler import ProbeObjectPopup
from . import resources_rc
from . import resources_v2

##############################################
##############################################
######     Section 1: Worker Objects    ######
##############################################
##############################################

class ArrayReaderWorker(QtCore.QObject):
    """
    Reads data from NumPy and MATLAB files and returns data array(s) (and optional metadata) in dictionaries via emitted signals.
    
    - Checks for 2-dimensional arrays in the file (potential "data") and special keywords relating to sampling rate or units (potential "metadata").
    - Produces errors if files cannot be loaded or no 2-dimensional arrays are found.
    - Subsequent functions outside this class handle selection of data variable (in cases where multiple 2d arrays are found) and metadata entry/confirmation.
    """
    # Define PyQt signals to communicate with the main thread
    progress_signal = QtCore.pyqtSignal(str)            # Progress signal used to update spinner text throughout loading process (later connected to RawRecordingSelectionPopup's SpinnerWindow's "report_progress_string" slot)
    data_signal = QtCore.pyqtSignal(str, dict, dict)    # Emits (filepath, data dictionary, metadata dictionary) at the end; later connected to "enter_array_metadata" which is the final storage/resting place of the data and metadata before it goes into the data extraction stage
    error_signal = QtCore.pyqtSignal(str, str)          # Emits (filepath, error message) if loading fails
    finished = QtCore.pyqtSignal()                      # Signal to indicate that the worker has finished its task
    
    def __init__(self, file_path):
        """
        Initialize path to input data file
        """
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        """
        Load input data; if successful, emit signal "data_signal" with dictionary of LFP data array(s) and metadata
        """
        # Extract file name and extension to use in the progress message
        file_path = self.file_path # Copy the file path
        file_name = os.path.basename(file_path) # Extract the file name ("basename" removes any preceding directory and only leaves the last component, i.e. recording.npy)
        file_ext = os.path.splitext(file_path)[-1][1:] # Get the file extension (e.g., .npy, .mat) ("splittext" splits a pathname into a pair (root, ext) where ext includes the leading dot) and index to remove the leading dot
        
        # Emit a progress signal indicating the start of file loading with the file extension in uppercase
        display_ext_dict = {'npy':'NumPy', 'mat':'MATLAB'}
        self.progress_signal.emit(f'Loading {display_ext_dict[file_ext]} file ...')
        
        error_msg = ''
        try:
            # Load NumPy file
            if file_ext == 'npy':
                file_data = np.load(file_path, allow_pickle=True)

            # Load MATLAB file
            elif file_ext == 'mat':
                file_data = sio.loadmat(file_path, squeeze_me=True) # squeeze_me=True removes dimensions of size 1 from the array 
            
            # If the loaded data is a NumPy n-dimensional array, add it to a dictionary with key 'data'
            if isinstance(file_data, np.ndarray):
                
                if file_data.ndim == 2: # Check if the data is 2-dimensional
                    data_dict = {"data" : np.array(file_data)}
                    metadata_dict = {'fs':None, 'units':None}
                else:
                    error_msg = 'Error: Data must be a 2-dimensional array.'

            # If the loaded data is a dictionary, parse it to extract data arrays and metadata
            elif isinstance(file_data, dict):
                self.progress_signal.emit('Parsing data dictionary...')

                try:  # get data array(s) (required) and SR/unit metadata (optional)
                    data_dict, metadata_dict = self.read_data_from_dict(file_data)
                    if len(data_dict) == 0:
                        error_msg = f'Error: No 2-dimensional data arrays found in "{file_name}".'
                except: 
                    error_msg = f'Error: Could not parse data from "{file_name}".'
            else: 
                error_msg = 'Error: File must contain data array or dictionary.'

        except: 
            error_msg = f'Error: Unable to load "{file_name}".'
        
        # Emit signals to return data or error message
        if error_msg == '': # No errors, return data by emitting signals
            self.data_signal.emit(str(file_path), data_dict, metadata_dict)
            self.progress_signal.emit('Done!')
        else: # Errors occurred, emit error signal
            self.error_signal.emit(str(file_path), error_msg)

        # Emit a finished signal to indicate that the worker has completed its task
        self.finished.emit()

    def read_data_from_dict(self, input_data_dict):
        """
        Parse imported data dictionary to find keys with 2D array values (for LFP data) and keys for metadata (sampling rate/data SI units).
        """
        # Initialize two dictionaries: one for data arrays and one for metadata
        data_dict = {}
        metadata_dict = {'fs':None, 'units':None}

        # Define keywords for sampling rate and units
        fs_keys = ['fs', 'sr', 'sampling_rate', 'sample_rate', 'sampling_freq', 'sample_freq', 'sampling_frequency', 'sample_frequency']
        unit_keys = ['unit', 'units']

        # Iterate through the input dictionary to find 2D arrays and metadata
        for k, v in input_data_dict.items():

            # Skip MATLAB metadata fields (i.e. __header__, __version__, etc.)
            if k.startswith('__'): 
                continue

            # Look for sampling rate and units keys (note: if there are multiple matches, the last one will overwrite previous ones)
            if k.lower() in fs_keys:
                metadata_dict['fs'] = float(v)
            elif k.lower() in unit_keys:
                metadata_dict['units'] = str(v)

            else:
                if not hasattr(v, '__iter__'): # True for iterables (arrays, lists, strings), False for ints, floats
                    continue
                if len(v) == 0: # If empty iterable, skip
                    continue
                if np.array(v).ndim != 2: # If not 2D array, skip
                    continue
                data_dict[k] = v # Add key and value if array is 2D. The selection of which key/value pair will be processed occurs later in the pipeline.
        return data_dict, metadata_dict
        
        
class ExtractorWorker(QtCore.QObject):
    """
    Returns spikeinterface recording object and timestamps (if available) for the selected recording.

    Note that while recordings with time can have the timestamps re-extracted at any point using the recording object, we explicitly return time (as an array or as None) here so the user is informed about the status of if their recording has timestamps.
    """
    # Define custom signals
    progress_signal = QtCore.pyqtSignal(str)
    data_signal = QtCore.pyqtSignal(object, object)
    error_signal = QtCore.pyqtSignal(str, str)
    finished = QtCore.pyqtSignal()
    
    def __init__(self, filepath, extractor_special_input_dict):
        """
        Initialize recording filepath (required) and raw data (optional; only if previously loaded in the case of NumPy or MATLAB array files).
        
        extractor_special_input_dict contains three keys:
        - data_array: LFP array from a NumPy or MATLAB file upload
        - metadata_dict: dictionary with metadata from a NumPy or MATLAB file upload; can be empty
        - electrical_series_path: selected ElectricalSeries path (key) from NWB file with multiple ElectricalSeries
        """
        super().__init__()
        self.filepath = filepath

        # Fill a dictionary with special input for NumPy/MATLAB uploads and some NWB recordings (values are None/empty data structures for other types of recordings)
        # Note: dict.get() retrieves the value associated with the specified key; if the key does not exist, it returns None
        self.extractor_special_input_dict = {
            'data_array': extractor_special_input_dict.get('data_array'),
            'metadata': extractor_special_input_dict.get('metadata', {}),
            'electrical_series_path': extractor_special_input_dict.get('electrical_series_path')
        }
    
    def run(self):
        """
        Get recording object (and timestamps, if available).
        """
        data_type = dp.get_data_format(self.filepath)
        self.progress_signal.emit(f'Getting {data_type} extractor ...')

        # Get the spikeinterface recording (and timestamps, if available; otherwise None)
        try:
            recording, time = dp.get_extractor(self.filepath, data_type, **self.extractor_special_input_dict)
        # Failed to load extractor object
        except Exception as e: 
            self.error_signal.emit(str(self.filepath), f'Error: {str(e)}')
        # Emit valid extractor
        else:   
            self.data_signal.emit(recording, time)
            self.progress_signal.emit('Done!')
        finally:
            self.finished.emit()
    

class DataWorker(QtCore.QObject):
    """ 
    Handles data imports, preprocessing, and exports in processing pipeline
    """
    progress_signal = QtCore.pyqtSignal(str)
    data_signal = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    
    def __init__(self, RECORDING = None, TIME = None, PROBE_GROUP = None, PARAMS = None, OUTPUT_DIR = None, parent = None):
        """
        Initialize data (recording, time, probe, parameters, and output directory)
        """
        super().__init__(parent)

        self.RECORDING = RECORDING          # spikeinterface recording extractor object
        self.TIME = TIME                    # array or None
        self.PROBE_GROUP = PROBE_GROUP      # probeinterface ProbeGroup object
        self.PARAMS = PARAMS                # parameter dictionary
        self.OUTPUT_DIR = OUTPUT_DIR        # location of toothy output files (str)
        
    def quit_thread(self, error_msg, ff=None):
        """
        Terminate pipeline upon data processing error.
        """
        # Close HDF5 datasets
        if ff is not None:
            ff.close()  
        self.error_signal.emit(error_msg)
        self.finished.emit()
    
    def run(self, **kwargs):
        """
        Run the data processing pipeline to load, downsample, bandpass filter, and perform event detection on LFP traces.

        Steps:
        1. Creates "toothy" output folder
        2. Loads recording object (from spikeinterface extractor)
        3. Slices recording to include only LFP channels
        4. Connects probe object (from probeinterface) to the recording
        """
        # Create the "toothy" folder for output
        toothy_output_dir = kwargs.get('save_ddir', self.OUTPUT_DIR)
        if not os.path.isdir(toothy_output_dir):
            os.mkdir(toothy_output_dir)

        # Get core data objects (recording, time)
        # Note: While we could pull the time array (recording.get_times()) here at the "process data" stage, it makes more sense to pull during the extraction/file selection stage so that the user is aware of if timestamps are available or not before processing.
        recording = kwargs.get('recording', self.RECORDING)
        time = kwargs.get("time", self.TIME)

        # Get metadata about the recording
        recording_metadata_dict = dp.get_metadata_from_recording(recording)
        recording_fs, recording_n_samples, recording_n_lfp_channels = recording_metadata_dict['fs'], recording_metadata_dict['n_samples'], recording_metadata_dict['n_lfp_channels']

        # Slice the recording to include only the LFP channels. Note that this respects the channel order in the list of IDs and re-orders channels as needed.
        lfp_recording = recording.select_channels(recording_metadata_dict["lfp_channel_ids"])

        # If there is no timestamp array, make one for internal use only. Users will only receive indices at the end.
        # TODO implement
        if type(time) != np.ndarray and time == None:
            print("Running DataWorker. Creating artificial time.")
            # Note: np.linspace takes args (start, stop, n_samples), where start/stop can be floats and stop is inclusive. Example: linspace(0, 1.5, 3) = [0, 0.75, 1.5]
            total_est_time_s = recording_n_samples/recording_fs # Estimate the total time of the recording
            time = np.linspace(0, total_est_time_s, recording_n_samples)
            recording.set_times(time) 

        # Get the probeinterface "probe_group" object and set it for the spikeinterface "recording" object's probe group
        probe_group = kwargs.get('probe_group', self.PROBE_GROUP) 
        lfp_recording.set_probegroup(probe_group, in_place=True)
        n_probes = len(probe_group.probes) # probe_group.probes is a list with probe objects; take the length to get the number of probes
        
        # Get the global device channel indices (an array of length n_lfp_channels)
            # For each channel, provides a zero-based index for its physical order in the probe group, i.e. a channel mapping
            # For example: 
                # a simple linear array and probe will just have [0, 1, 2, ...]
                # a linear probe with a channel map will have [29, 4, 15, ...]
                # Multiple shanks/probes will also be represented as [probe1_ch_idxs..., probe_2_ch_idxs, ...]
        global_dev_idx = probe_group.get_global_device_channel_indices()['device_channel_indices']
        
        # Get the processing parameters (set by user)
        PARAMS = kwargs.get('PARAMS', self.PARAMS)

        # Look at the desired time range 
        # TODO: allow for start/end *indices* instead of times (for recs. w/o time) ##### 
        # TODO: In the GUI, the start/end defaults need to reflect the actual recording. bug: currently automatically starts at 0
        # TODO: In the GUI, the params menu needs to actually take values, right now weirdly stuck at 100s of seconds
        # TODO: test that the time range selected is truly only the one loaded.
        t_start_s, t_end_s = PARAMS['time_range']   # Selected time points between which the data will be processed

        if t_start_s == 0: # Cheap fix for now; HACK to make the slice start at the actual t0 and not the default 0
            slice_start = time[0]
        else:
            slice_start = t_start_s
        if t_end_s == -1: # Cheap fix for now; HACK to make the slice end at the actual tf.
            slice_end = time[-1]
        else:
            slice_end = t_end_s

        # Slice the recording again by time to include only the desired range
        lfp_recording_trimmed = lfp_recording.time_slice(start_time = slice_start, end_time = slice_end)
        n_samples_initial = lfp_recording_trimmed.get_num_samples()
        
        # Get the downsampled dataset size
        ds_factor = int(PARAMS["lfp_decimation_factor"])
        target_fs = recording_fs / float(ds_factor)
        PARAMS["lfp_fs"] = target_fs
        # Get the number of samples after downsampling
        # Example: If the selection is 100k samples at 10kHz, the downsampled selection at 1kHz would be (factor of 10) 10k samples
        # Exact calculation:
            # a signal with <n_samples_initial> samples is indexed (0) through (n_samples_initial - 1)
            # Decimating by a factor d means every d^th index sample is kept after index 0: 1d, 2d, 3d, etc.
            # Thus the total samples kept (n_samples_post_processing) must be <= the final index (n_samples_initial - 1)
            # Represent the above as: n_samples_post_processing * d <= n_samples_initial - 1
            # We want n_samples_post_processing to be an int, so: n_samples_post_processing = floor((n_samples_initial - 1) / d)
            # Finally, we add 1 to account for 0 (the first index), so: n_samples_post_processing = floor((n_samples_initial - 1) / d) + 1
        n_samples_post_processing = int(np.floor(((n_samples_initial - 1)/ds_factor)) + 1)

        # TODO edit KW
        # We use the recording object that was already sliced by channel/time 
        KW = {"recording": lfp_recording_trimmed}
        
        #########################
        #####      HDF5     #####
        #########################

        # HDF5 notes
            # Group: analogous to a folder/directory; can hold other groups/datasets
                # [file/group].create_group(<str_name>) to create a group
                # file[<str_path>] to access a group, file.keys() to see groups/datasets
            # Dataset: NumPy array-like structure that can store scalars, vectors, matrices, etc.; can be chunked/partially read without loading entirely into memory
                # [file/group].create_dataset(<str_name>, data = ...) to create a dataset
                # [file/group][<str_name>][:] to load the entire dataset
            # Attributes: metadata attached to groups or datasets
                # [file/group/dataset].attrs[<key>] = <value> to create an attribute
                # [^].attrs to access attrs
                # [^].[<key>] to access specific key/value pair

        # Create HDF5 file to store LFP data from each probe
            # Note: This creates a file at the "toothy_output_dir" with the name "DATA.hdf5"
            # Opens the file in "write mode" (parameter "w")
            # "track_order = True" ensures the order in which files are added is preserved
        data_hdf5_file = h5py.File(Path(toothy_output_dir, 'DATA.hdf5'), 'w', track_order =True) # TODO change the file name to be more descriptive... maybe "toothy-processing-data"

        # Create attributes of the file for the sampling rate of the input data (recording) and the intended downsampling sampling rate
        # FS is the recording, LFP_fs is the downsampled? i think, yes TODO make names more descriptive
        data_hdf5_file.attrs['fs'] = recording_fs 
        data_hdf5_file.attrs['lfp_fs'] = target_fs
        data_hdf5_file.attrs['lfp_downsample_factor'] = ds_factor

        ##################################
        #####     LOAD TIMESTAMPS    #####
        ##################################

        # Create HDF5 dataset for (downsampled) timestamps. 
        # Note: if ds_factor is 1, this operation returns the same array as the original.
        lfp_time_trimmed = lfp_recording_trimmed.get_times() # TODO This works because we set_times() above, once we remove that, we will need to change this
        time_downsampled = lfp_time_trimmed[::ds_factor] # Take every n^th sample (where n is the ds_factor)
        lfp_time = data_hdf5_file.create_dataset("lfp_time", data = time_downsampled)

        ######################################
        #####     PREPARE TO LOAD LFP    #####
        ######################################

        # Make HDF5 datasets with the channel mapping for each probe
        ch_idx_lists = [] # Each element will be a list containing the channel mapping for that probe
        h5_datasets = []  # Each element will be an HDF5 dataset object containing a matrix of (n_channels, n_samples) for a given probe

        # Iterate over probes to pull channel maps and initialize HDF5 datasets
        for i_probe in range(n_probes):
            # Create an HDF5 group (name is the probe index)
            h5_probe_group = data_hdf5_file.create_group(str(i_probe), track_order = True)

            # Create an "LFP" group inside the probe group
            h5_probe_lfp_group = h5_probe_group.create_group('LFP', track_order = True)

            # Get the indices of channels on the current probe
            idx = np.where(lfp_recording_trimmed.get_channel_groups() == i_probe)[0]

            # Index the "global" device indices to get the channel map for this probe
            probe_ch_idx_list = global_dev_idx[idx]
            n_channels_on_probe = len(probe_ch_idx_list)

            # Create a dataset in the probe/LFP group called "raw" with shape channels x samples
            probe_h5_dataset = h5_probe_lfp_group.create_dataset('raw', (n_channels_on_probe, n_samples_post_processing), dtype='float32')

            # Also create datasets for the filtered signals
            # TODO
            # h5_probe_lfp_group.create_dataset('raw', (n_channels_on_probe, n_samples_post_processing), dtype='float32')
            
            # Store the channel mapping and the probe HDF5 dataset for the current probe in lists that we will iterate over later
            ch_idx_lists.append(probe_ch_idx_list)
            h5_datasets.append(probe_h5_dataset)

        #############################################
        #####     LOADING & DOWNSAMPLING LFP    #####
        #############################################

        # Iterate over probes
        for probe_i in range(n_probes):

            # Get a list of indices (in the recording's channel list) of the LFP channels for this probe
            probe_ch_idx_list = ch_idx_lists[probe_i]
            n_channels_on_probe = len(probe_ch_idx_list)

            # Get the HDF5 dataset matrix for this probe to store the LFP (shape: channels x samples)
            probe_h5_dataset  = h5_datasets[probe_i]
            
            # Iterate over the channels on the probe
            for i, ch_idx in enumerate(probe_ch_idx_list):
                try:
                    progress_txt = f"Extracting LFP for channel {i} of {n_channels_on_probe}<br>on Probe {probe_i + 1} of {n_probes}..."
                    self.progress_signal.emit(progress_txt)

                    # Get the channel ID
                    ch_id = lfp_recording_trimmed.channel_ids[ch_idx]

                    # Get the LFP for this channel
                    # Loads as (samples, channels); use .T to transpose to (channels, samples), then index the first element to grab the row associated with this singular channel
                    ch_lfp_arr_uv = lfp_recording_trimmed.get_traces(channel_ids=[ch_id], return_scaled=True).T[0]

                    # Loads as uV, convert to mV
                    ch_lfp_arr_mV = ch_lfp_arr_uv / 1000. 
                    
                    # Add the LFP to the HDF5 dataset row i (downsample first if needed via decimation)
                    if ds_factor == 1:
                        # No downsampling needed
                        probe_h5_dataset[i] = ch_lfp_arr_mV
                    else:
                        # Decimate using an 8th-order Chebyshev Type 1 IIR filter
                        ch_lfp_arr_mV_downsampled = scipy.signal.decimate(ch_lfp_arr_mV, q = ds_factor, n = 8, ftype = "iir", zero_phase = True)
                        probe_h5_dataset[i] = ch_lfp_arr_mV_downsampled
                except Exception as e:
                    self.quit_thread("") # TODO
                    return
                    # example from old code
                    # self.quit_thread(f'Downsampling Error: {str(e)}', ff=data_hdf5_file, KW=KW)
                    # return
        
        ####################################
        #####    BANDPASS FILTERING    #####
        ####################################
        # Bandpass filtering notes
            # For each band, run a Butterworth bandpass filter
            # Calculate the Hilbert transform of the filtered signal
            # The Hilbert takes a real signal x(t) and produces a complex signal (called the analytic signal) z(t) = x(t) + i * y(t) where y(t) is a Hilbert transform of x(t)
            # The instantaneous amplitude (also called the envelope) is the magnitude of the analytic signal
            # Calculate the instantaneous amplitude by taking the absolute value of the Hilbert tranform
            # Then, calculate power by squaring the amplitude and take a mean value for the channel

        # Iterate over probes
        for iprb, probe in enumerate(probe_group.probes):

            hdr = f'ANALYZING PROBE {iprb+1} / {n_probes}<br>'

            shank_ids = np.array(probe.shank_ids, dtype='int') # TODO what is this for?

            # Get the LFP data loaded earlier (downsampled but unfiltered)
            lfp_unfiltered = data_hdf5_file[str(iprb)]['LFP']['raw']
            
            # Get the number of channels on the probe
            n_channels_on_probe = len(lfp_unfiltered)

            # Initialize a dictionary to hold filtered signals
            # TODO initialize HDF5 matrices for each band
            LFP_dict = {}
            
            # For each band of interest, initialize a numpy array of ones (length = # of samples in the unfiltered LFP; sample count does not change anymore)
            bands = [
                "theta",
                "slow_gamma",
                "fast_gamma",
                "ds",
                "swr"
            ]
            for band in bands:
                LFP_dict[band] = np.ones(lfp_unfiltered.shape, dtype='float32')
                data_hdf5_file[str(iprb)]["LFP"].create_dataset(band, (n_channels_on_probe, n_samples_post_processing), dtype='float32')

            # Iterate over channels
            for i in range(n_channels_on_probe):
                # Emit progress message
                self.progress_signal.emit(hdr + f'<br>Bandpass filtering signals ... {i}')

                # Get the unfiltered LFP
                ch_lfp = lfp_unfiltered[i]

                # Filter for each band
                for band in bands:
                    param_key = band
                    if band == "ds" or band == "swr":
                        param_key += "_freq" # TODO fix?

                    #TODO put into HDF5 not this
                    LFP_dict[band][i] = dp.butter_bandpass_filter(ch_lfp, *PARAMS[param_key], lfp_fs = target_fs)
                    # filtered_ch_lfp = dp.butter_bandpass_filter(ch_lfp, *PARAMS[param_key], lfp_fs = target_fs)
                    # data_hdf5_file[str(iprb)]["LFP"][band][:] = filtered_ch_lfp

                    # Save the power calc for this given CHANNEL AND BAND

            # Get the standard deviation for each channel
            std_dict = {}
            for band, filtered_lfp_matrix in LFP_dict.items():
                std_dict[band] = np.std(filtered_lfp_matrix, axis = 1)
                std_dict[f'norm_{band}'] = pyfx.Normalize(std_dict[band])
            
            # try:
            #....
            # except Exception as e:
            #     self.quit_thread(f'Filtering Error: {str(e)}', ff=data_hdf5_file)
            #     return
            
            # Store the filtered signal for the dentate spike (DS) and sharp-wave ripple (SPW-R) bands for easy adjustment of event-detection parameters in later steps
            data_hdf5_file[str(iprb)]['LFP']['ds'][:] = np.array(LFP_dict['ds'])
            data_hdf5_file[str(iprb)]['LFP']['swr'][:] = np.array(LFP_dict['swr'])
            
            # Store the power calculation 
            data_hdf5_file[str(iprb)].create_group("band_power")
            data_hdf5_file[str(iprb)]["band_power"].attrs.update(std_dict)

            STD = pd.DataFrame(std_dict)
            h5df.save_df(data_hdf5_file.filename, f'/{iprb}/STD', STD)
        
        # could keep in the same loop, sure
            ####################################
            #####      EVENT DETECTION     #####
            ####################################
            
            # Initial ripple and DS detection for each channel
            progress_txt = hdr + f'<br>Detecting DSs and ripples on channel<br>%s / {n_channels_on_probe} ...'
            SWR_DF, DS_DF, SWR_THRES, DS_THRES = None, None, None, None
            for i,(swr_d, ds_d) in enumerate(zip(LFP_dict['swr'], LFP_dict['ds'])):
                self.progress_signal.emit(progress_txt % (i+1))
                try:  # detect sharp-wave ripples
                    SWR_DF, SWR_THRES = dp.detect_channel('swr', i, swr_d, lfp_time[:],
                                                       DF=SWR_DF, THRES=SWR_THRES, 
                                                       pprint=False, **PARAMS)
                except Exception as e:
                    self.quit_thread(f'Ripple Detection Error: {str(e)}', ff=data_hdf5_file)
                    return
                try:  # detect DSs
                    DS_DF, DS_THRES = dp.detect_channel('ds', i, ds_d, lfp_time[:], 
                                                        DF=DS_DF, THRES=DS_THRES, 
                                                        pprint=False, **PARAMS) 
                    
                except Exception as e:
                    self.quit_thread(f'DS Detection Error: {str(e)}', ff=data_hdf5_file)
                    return
            
            # add status columns for later curation
            for DF in [SWR_DF, DS_DF]:
                if DF.size == 0: DF.loc[0] = np.nan
                DF['status'] = 1 # 1=auto-detected; 2=added by user; -1=removed by user
                DF['is_valid'] = 1 # valid events are either auto-detected and not user-removed OR user-added
                tups = [(ch, shank_ids[ch]) for ch in np.unique(DF.index.values)]
                DF['shank'] = 0
                for ch,shkID in tups:
                    DF.loc[ch,'shank'] = shkID
            h5df.save_df(data_hdf5_file.filename, f'/{iprb}/ALL_SWR', SWR_DF)
            h5df.save_df(data_hdf5_file.filename, f'/{iprb}/ALL_DS', DS_DF)
            
            for ek,THRES in [('swr',SWR_THRES),('ds',DS_THRES)]:
                THRES_DF = pd.DataFrame(THRES).T  # save threshold magnitudes
                h5df.save_df(data_hdf5_file.filename, f'/{iprb}/{ek.upper()}_THRES', THRES_DF)
            
            # initialize noise train # TODO what is this
            data_hdf5_file[str(iprb)]['NOISE'] = np.zeros(n_channels_on_probe, dtype='int')
        
        # for iprb, probe in enumerate(probe_group.probes):
        #     hdr = f'ANALYZING PROBE {iprb+1} / {n_probes}<br>'

        #     shank_ids = np.array(probe.shank_ids, dtype='int')

        #     # Get the raw/unfiltered LFP data stored earlier
        #     lfp_unfiltered = data_hdf5_file[str(iprb)]['LFP']['raw']
            
        #     # Get the number of channels
        #     n_channels = len(lfp_unfiltered)

        #     # Bandpass filter LFP and get mean amplitude per channel

        #     # Initialize a dictionary to hold filtered signals
        #     LFP_dict = {}

        #     # For each band of interest, initialize a numpy array of ones (the same size as the )
        #     for k in ['theta', 'slow_gamma', 'fast_gamma', 'ds', 'swr']:
        #         LFP_dict[k] = np.ones(lfp_unfiltered.shape, dtype='float32')
            
        #     # Emit progress message
        #     self.progress_signal.emit(hdr + '<br>Bandpass filtering signals ...')

        #     filter_kwargs = {'lfp_fs': target_fs, 'axis': 1}
        #     try:
        #         # Split the array into sections
        #         arr_list = np.array_split(lfp_unfiltered, int(np.ceil(n_samples_post_processing / ichunk)), axis=1)
        #         p = 0

        #         # Go section by section and filter the data - needs to happen, downsampling or not.
        #         # TODO do this with padding on the edges to avoid edge effects...
        #         for i, yarr in enumerate(arr_list):
        #             q = p + yarr.shape[1]
        #             LFP_dict['theta'][:,p:q] = dp.butter_bandpass_filter(yarr, *PARAMS['theta'], **filter_kwargs)
        #             LFP_dict['slow_gamma'][:,p:q] = dp.butter_bandpass_filter(yarr, *PARAMS['slow_gamma'], **filter_kwargs)
        #             LFP_dict['fast_gamma'][:,p:q] = dp.butter_bandpass_filter(yarr, *PARAMS['fast_gamma'], **filter_kwargs)
        #             LFP_dict['swr'][:,p:q] = dp.butter_bandpass_filter(yarr, *PARAMS['swr_freq'], **filter_kwargs)
        #             LFP_dict['ds'][:,p:q] = dp.butter_bandpass_filter(yarr, *PARAMS['ds_freq'], **filter_kwargs)
        #             p += yarr.shape[1]
        #         std_dict = {}
        #         for k,arr in LFP_dict.items():
        #             std_dict[k] = np.std(arr, axis=1)
        #             std_dict[f'norm_{k}'] = pyfx.Normalize(std_dict[k])
        #     except Exception as e:
        #         self.quit_thread(f'Filtering Error: {str(e)}', ff=data_hdf5_file)
        #         return
            
        #     # Store the filtered signal for the dentate spike (DS) and sharp-wave ripple (SPW-R) bands for easy adjustment of event-detection parameters in later steps
        #     data_hdf5_file[str(iprb)]['LFP']['ds'] = np.array(LFP_dict['ds'])
        #     data_hdf5_file[str(iprb)]['LFP']['swr'] = np.array(LFP_dict['swr'])
            
        #     STD = pd.DataFrame(std_dict)
        #     STD.to_hdf(data_hdf5_file.filename, key=f'/{iprb}/STD')
            
            
        #     ####################################
        #     #####      EVENT DETECTION     #####
        #     ####################################
            
        #     # lfp_time = lfp_time_resampled #TODO remove
        #     # lfp_time =lfp_time_resampled# Replace lfp_time with the one made in more chunks...
            
        #     # Initial ripple and DS detection for each channel
        #     progress_txt = hdr + f'<br>Detecting DSs and ripples on channel<br>%s / {n_channels} ...'
        #     SWR_DF, DS_DF, SWR_THRES, DS_THRES = None, None, None, None
        #     for i,(swr_d, ds_d) in enumerate(zip(LFP_dict['swr'], LFP_dict['ds'])):
        #         self.progress_signal.emit(progress_txt % (i+1))
        #         try:  # detect sharp-wave ripples
        #             SWR_DF, SWR_THRES = dp.detect_channel('swr', i, swr_d, lfp_time[:],
        #                                                DF=SWR_DF, THRES=SWR_THRES, 
        #                                                pprint=False, **PARAMS)
        #         except Exception as e:
        #             self.quit_thread(f'Ripple Detection Error: {str(e)}', ff=data_hdf5_file)
        #             return
        #         try:  # detect DSs
        #             DS_DF, DS_THRES = dp.detect_channel('ds', i, ds_d, lfp_time[:], 
        #                                                 DF=DS_DF, THRES=DS_THRES, 
        #                                                 pprint=False, **PARAMS) 
                    
        #         except Exception as e:
        #             self.quit_thread(f'DS Detection Error: {str(e)}', ff=data_hdf5_file)
        #             return
            
        #     # add status columns for later curation
        #     for DF in [SWR_DF, DS_DF]:
        #         if DF.size == 0: DF.loc[0] = np.nan
        #         DF['status'] = 1 # 1=auto-detected; 2=added by user; -1=removed by user
        #         DF['is_valid'] = 1 # valid events are either auto-detected and not user-removed OR user-added
        #         tups = [(ch, shank_ids[ch]) for ch in np.unique(DF.index.values)]
        #         DF['shank'] = 0
        #         for ch,shkID in tups:
        #             DF.loc[ch,'shank'] = shkID
        #     SWR_DF.to_hdf(data_hdf5_file.filename, key=f'/{iprb}/ALL_SWR')
        #     DS_DF.to_hdf(data_hdf5_file.filename, key=f'/{iprb}/ALL_DS')
            
        #     for ek,THRES in [('swr',SWR_THRES),('ds',DS_THRES)]:
        #         THRES_DF = pd.DataFrame(THRES).T  # save threshold magnitudes
        #         THRES_DF.to_hdf(data_hdf5_file.filename, key=f'/{iprb}/{ek.upper()}_THRES')
            
        #     # initialize noise train # TODO what is this
        #     data_hdf5_file[str(iprb)]['NOISE'] = np.zeros(n_channels, dtype='int')
            
        ##########################################
        #####    SAVE PROBE/PARAM SETTINGS   #####
        ##########################################
            
        # initialize event channel dictionary with [None,None,None]
        _ = h_io.init_event_channels(toothy_output_dir, probes=probe_group.probes, psave=True)
        
        # Save params and info file in recording folder
        # TODO this is where the pkl is saved. 
        # It is read later in both channel_selection_gui and in the ds_classification_gui
        # I think we should instead save it as a txt file so it can be easily read. it can also be doubly saved within an h5py file?
        # TODO it should also be saved later during ds classification
        param_path = Path(toothy_output_dir, pyfx.unique_fname(toothy_output_dir, 'params.pkl')) # TODO where is this file read? Also should we save it as a TXT file so it can be re-used?
        with open(Path(param_path), 'wb') as f:
            pickle.dump(PARAMS, f)
        
        # write probe group to file
        probegroup_path = Path(toothy_output_dir, pyfx.unique_fname(toothy_output_dir, 'probe_group'))
        prif.write_probeinterface(probegroup_path, probe_group)
        # Close the file
        data_hdf5_file.close()
        self.data_signal.emit()
        self.progress_signal.emit('Done!')
        timelib.sleep(1)
        self.finished.emit()

####################################################################
####################################################################
######     Section 2: Widgets for loading data and probes     ######
####################################################################
####################################################################
        

class RawRecordingSelectionWidget(gi.FileSelectionWidget):
    """
    Selection and validation of an input data file path.
    Subclass of FileSelectionWidget.

    - "select_filepath" launches a file dialog (restricted to supported file types)
    - "update_filepath" validates the file is a supported type, updates the displayed file path on the GUI, and emits a status signal (if True, drives forward the next step of file extraction)
    - "enter_array_metadata" is run after "update_filepath" emits "read_array_signal" (which calls "load_array_worker", which runs ArrayReaderWorker, which emits to "load_array_finished_slot", which calls "enter_array_metadata")
    - "enter_array_metadata" stores the data and metadata from the array file as an instance variable, which are referenced directly (rather than using emitted signals)
    """
    
    # Create a signal for broadcasting if an array (NumPy or MATLAB file type) is selected -> leads to ArrayReaderWorker running to extract data immediately
    read_array_signal = QtCore.pyqtSignal(str)

    def __init__(self, title='', parent=None):
        super().__init__(title=title, parent=parent)
        self.icon_btn.hide() # Hide icon button
    
    def select_filepath(self):
        """
        Launch file dialog for raw recording selection, filter unsupported extensions.
        """
        init_ddir = self.get_init_ddir() # (function from parent class)
        supported_extensions = [x[1] for x in dp.supported_formats.values()]

        # Create a modal QFileDialog for the user to select a file; return the file path (as type str)
        file_path = h_io.select_raw_recording_file(supported_extensions, init_ddir, parent=self)

        # If a file is not selected above (i.e. the file dialog is closed via hitting "Cancel" or the X button), file_path will be an empty string
        # If a file is selected (non-empty strings evaluate as True), update the data array, metadata, and filepath
        if file_path:
            # Initialize variables for the data array and metadata dictionary
            self.data_array, self.meta = None, {}

            # Send the file path string to "update_filepath" to validate it and (if an array) extract the array
            # "update_filepath" also emits a signal ("signal") connected to a slot "recording_file_path_updated" that triggers the next step of the overall GUI flow
            self.update_filepath(file_path)

        # If a file is not selected, we don't update the data_array/metadata or filepath
        
    def update_filepath(self, file_path, x=None):
        """
        Handle selection of a filepath. Overwrites function in parent class to add handling for NumPy and MATLAB arrays.
        """
        
        if x is not None:
            validation_status = x
            
        if x is None: # (default)
            validation_status = self.validate_path(file_path) # True, False, or "NPY or MAT"

        if validation_status == "NPY or MAT":   # User selected a NPY or MAT file
            self.read_array_signal.emit(file_path)  # Emit signal for reading a NumPy/MATLAB array; connected to "load_array_worker" slot which begins ArrayReaderWorker
            return                              # Exit the function; downstream functions called by "read_array_signal" signal will eventually call "update_filepath" again

        else:
            self.le.setText(file_path)                  # Update QLineEdit widget defined in parent class 
            self.update_status(validation_status)   # Use parent class "update_status" to change the QLineEdit box edge from red to green
            self.signal.emit(self.VALID_PPATH)      # Defined in parent class; most important signal emitted by this class; True or False to indicate if a valid file has been selected 
    
    def enter_array_metadata(self, file_path, data_array, metadata_dict):
        """
        Prompt user to enter contextual metadata for NPY/MAT files.
        If file contained special keys, those values will be populated.
        Important: assigns data_array and metadata_dict to class variables, which are referenced by the next step (extraction).
        """
        # Launch dialog windows
        file_name = os.path.basename(file_path)
        dialog_window = RawArrayPopup(data_array.shape, **metadata_dict, filename=file_name)

        # If the user hits "accept", assign values to instance variables
        if dialog_window.exec():
            self.data_array = np.array(data_array) # Ensure data is stored as a NumPy array
            self.metadata_dict = {
                'n_channels' : dialog_window.n_channels, 
                'n_samples' : dialog_window.n_samples,
                'fs' : dialog_window.fs_spinbox.value(),
                'units' : dialog_window.units_combobox.currentText()
            }

            # Finally, call "update_filepath" with x=True to move the workflow on to the next part (extraction)
            # Concludes the journey from file selection and array processing for NPY/MAT files
            self.update_filepath(file_path, True)
                
    def validate_path(self, path):
        """
        Check if selected data path exists and is a supported file type.
        This is largely achieved prior to this point by using a QFileDialog filered to the supported file types, but worth having as a backup.

        Returns:
            True if the path exists and is a supported data format.
            False if the path doesn't exist, is a directory, or is an unsupported file format.
            "NPY or MAT" if the path points to a NumPy or MATLAB file.
        """
        # Check if the path exists and is a file (not a directory)
        if not os.path.exists(path) or os.path.isdir(path):
            return False
        
        # "get_data_format" either returns a string (one of the six supported formats), or raises an exception
        try: 
            data_format = dp.get_data_format(path)
        except:
            return False
        
        # Special return for NumPy or MATLAB files to prompt further array processing
        if data_format in ['NPY','MAT']:
            return "NPY or MAT"
    
        # For all other supported formats, just return True
        else:
            return True
        

class RawArrayPopup(QtWidgets.QDialog):
    """
    Interface for user-provided metadata for NPY/MAT recordings
    """
    supported_units = ['uV', 'mV', 'V', 'kV']
    
    def __init__(self, data_shape, fs=None, units=None, filename='', parent=None):
        super().__init__(parent)
        assert len(data_shape) == 2, 'Data array must be 2-dimensional.'

        self.setWindowTitle("Specify recording properties")

        # Remove the "?" help button
        gi.remove_help_button(self)

        # Set icon for upper left corner and toolbar
        self.setWindowIcon(QtGui.QIcon(':/resources/logo.png'))

        # Define variables used in gen_layout
        self.filename = filename

        self.gen_layout(data_shape, fs, units)
        self.connect_signals()
        
    def gen_layout(self, data_shape, fs, units):
        """
        Set up layout
        """
        # Display the information in a row
        file_name_label = QtWidgets.QLabel(f"<code>{self.filename}</code>")
        file_name_label.setStyleSheet('QLabel {background-color : white; padding : 4px;}')
        file_name_title_label = QtWidgets.QLabel("<b>File name: </b>")
        self.file_name_row = pyfx.get_widget_container('h', file_name_title_label, file_name_label, stretch_factors=[2,0], widget='widget')
        
        # Assign rows/cols to samples/channels depending on which is smaller
        n_rows, n_cols = data_shape
        if n_rows > n_cols: # rows = samples, columns = channels
            self.n_samples, self.n_channels = data_shape
            row_lbl, col_lbl = ['samples', 'channels']
        else: # rows = channels, columns = samples
            self.n_channels, self.n_samples = data_shape
            row_lbl, col_lbl = ['channels', 'samples']

        # Display the information in a row
        dimension_label_str = f'<code>{n_rows} {row_lbl} x {n_cols} {col_lbl}</code>'
        dimension_label = QtWidgets.QLabel(dimension_label_str)
        dimension_label.setStyleSheet('QLabel {background-color : white; padding : 4px;}')
        dimension_title_label = QtWidgets.QLabel('<b>Data dimensions:</b>')
        self.dimension_row = pyfx.get_widget_container('h', dimension_title_label, dimension_label, stretch_factors=[2,0], widget='widget')
        
        # Create labeled ComboBox for user to input units
        self.units_widget = QtWidgets.QWidget()
        self.units_layout = QtWidgets.QHBoxLayout(self.units_widget)
        self.units_layout.setContentsMargins(0,0,0,0)
        self.units_layout.addWidget(QtWidgets.QLabel("<b>Units:</b>"))
        self.units_layout.addStretch()
        self.units_combobox = QtWidgets.QComboBox()
        self.units_layout.addWidget(self.units_combobox)
        self.units_combobox.addItems(self.supported_units)
        
        # If a unit was provided, set the input box to match
        if units in self.supported_units:
            self.units_combobox.setCurrentText(units)

        # Create a labeled SpinBox for user to input sampling rate
        self.fs_widget = QtWidgets.QWidget()
        self.fs_layout = QtWidgets.QHBoxLayout(self.fs_widget)
        self.fs_layout.setContentsMargins(0,0,0,0)
        self.fs_layout.addWidget(QtWidgets.QLabel("<b>Sampling rate:</b>"))
        self.fs_layout.addStretch()
        self.fs_spinbox = QtWidgets.QDoubleSpinBox()
        self.fs_spinbox.setMinimum(1)
        self.fs_spinbox.setMaximum(9999999999)
        self.fs_spinbox.setSuffix(" Hz")
        self.fs_spinbox.setDecimals(5)
        self.fs_layout.addWidget(self.fs_spinbox)

        # Create a labeled QWidget for displaying the computed duration
        self.dur_widget = QtWidgets.QWidget()
        self.dur_layout = QtWidgets.QHBoxLayout(self.dur_widget)
        self.dur_layout.setContentsMargins(0,0,0,0)
        self.dur_layout.addWidget(QtWidgets.QLabel("<i>Computed duration:</i>"))
        self.dur_layout.addStretch()
        self.dur_qlabel = QtWidgets.QLabel()
        self.dur_layout.addWidget(self.dur_qlabel)

        # If a sampling rate was provided, fill in the sampling rate and computed duration
        if fs is not None:
            self.fs_spinbox.setValue(fs)
            computed_dur = self.n_samples / fs
            computed_dur_str = str(np.round(computed_dur, 5)) + " s"
            self.dur_qlabel.setText(computed_dur_str)

        # Stack all the labeled widgets
        v_layout = pyfx.get_widget_container('v', self.units_widget, self.fs_widget, self.dur_widget)
        v_layout.setSpacing(10)
        
        # Create action buttons
        self.button_box = QtWidgets.QWidget()
        button_box_layout = QtWidgets.QHBoxLayout(self.button_box)
        self.confirm_meta_btn = QtWidgets.QPushButton('Continue')
        self.close_btn = QtWidgets.QPushButton('Cancel')
        button_box_layout.addWidget(self.close_btn)
        button_box_layout.addWidget(self.confirm_meta_btn)
        
        # Add the widgets to a QVBoxLayout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.file_name_row)
        self.layout.addWidget(self.dimension_row)
        self.layout.addWidget(pyfx.DividerLine(lw=1, mlw=1))
        self.layout.addLayout(v_layout)
        self.layout.addWidget(self.button_box)
    
    def connect_signals(self):
        """
        Connect GUI inputs
        """
        # Changes in the sampling rate should also update the "computed duration"
        self.fs_spinbox.valueChanged.connect(lambda x: self.update_fs_dur(x))

        # Close the QDialog    
        self.confirm_meta_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)
        
    def update_fs_dur(self, val):
        """
        Update duration from sampling rate
        """
        computed_dur = self.n_samples / self.fs_spinbox.value()
        computed_dur_str = str(np.round(computed_dur, 5)) + " s"
        self.dur_qlabel.setText(computed_dur_str)
        

class ProbeRow(QtWidgets.QWidget):
    """
    Interactive widget representation of a probe object within the "probe assignment" section of the GUI

    Note: this does not use probeinterface
    """
    def __init__(self, probe, n_recording_lfp_channels, start_row, mode):
        super().__init__()

        # Probe object (from probeinterfqace)
        self.probe = probe

        # Number of channels on the probe
        self.n_probe_channels = probe.get_contact_count()

        # Number of LFP channels in the recording
        self.n_recording_lfp_channels = n_recording_lfp_channels

        # Number of copies of the given probe that would fully encompass the recording's LFP channels
        # Used for mode == 1 (alternating rows)
        self.n_probes_alternating = int(self.n_recording_lfp_channels / self.n_probe_channels)
        
        # Set up the layout for the probe
        self.gen_layout()

        # Sets the variable "self.ROWS" according to the start index and mode (contiguous vs. alternating)
        self.get_rows(start_row, mode)
        
    def gen_layout(self):
        """
        Set up layout
        """
        # Create a QPushButton with two icons (white circle and black circle) to mimic a radio button
        # Selecting this button in the GUI allows the user to copy or delete a probe 
        self.btn = QtWidgets.QPushButton()
        self.btn.setCheckable(True)
        self.btn.setChecked(True)
        self.btn.setFixedSize(20,20)
        self.btn.setFlat(False)
        probe_button_ss = """
            QPushButton {
                border : none;
                image : url(:/icons/white_circle.png);
                outline : none;
            }
            
            QPushButton:checked {
                image : url(:/icons/black_circle.png);
            }
        """
        self.btn.setStyleSheet(probe_button_ss)

        # QLabel with probe information
        # Example:
            # (in bold) probe_name
            # Data row indices: start_idx-end_idx
        self.probe_label = QtWidgets.QLabel()
        self.probe_label_fmt = '<b>{a}</b><br>Data row indices: {b}' # Later, "a" becomes the probe name and "b" specifies the channel index range encompassed by the probe
        probe_label_ss = """
            QLabel {
                background-color: white;
                border: 1px solid gray;
                border-radius: 5px;
                padding: 5px 10px; /* top/bottom, left/right*/
            }
        """
        self.probe_label.setStyleSheet(probe_label_ss)
        
        # Make a widget to hold the action buttons
        self.probe_action_button_container = QtWidgets.QWidget()
        policy = self.probe_action_button_container.sizePolicy()
        policy.setRetainSizeWhenHidden(True) # Maintain the horizontal space when hidden so the layout doesn't shift around when the row is selected
        self.probe_action_button_container.setSizePolicy(policy)
        
        # Assign a layout to hold the buttons
        button_layout = QtWidgets.QHBoxLayout(self.probe_action_button_container)
        button_layout.setContentsMargins(0,0,0,0)
        button_layout.setSpacing(0)

        # Define action buttons for the probe: copy, delete, edit, and save
        toolbtns = [QtWidgets.QToolButton(), QtWidgets.QToolButton()]
        self.copy_btn, self.delete_btn = toolbtns
        
        # Set icons and style 
        # TODO change icons
        self.copy_btn.setIcon(QtGui.QIcon(":/icons/copy.png"))
        self.delete_btn.setIcon(QtGui.QIcon(":/resources/clear-selection.png"))
        for btn in toolbtns:
            btn.setIconSize(QtCore.QSize(20,20))
            btn.setAutoRaise(True) # Makes the button icon appear flat without a box around it
            button_layout.addWidget(btn)

        # Show/hide the action buttons depending on if that probe is selected with the left-side button
        self.btn.toggled.connect(lambda status: self.probe_action_button_container.setVisible(status))
        
        # Horizontal layout to have the button, label, and action buttons
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.btn, stretch = 0)
        self.layout.addWidget(self.probe_label, stretch = 2) # Only the QLabel will expand to fill the space; the buttons will keep their original shapes
        self.layout.addWidget(self.probe_action_button_container, stretch = 0)
        
    def get_rows(self, start_row, mode):
        """
        Map probe channels to a subset of LFP channels.
        """
        # Contiguous rows (i.e. N consecutive channels in a row from starting index: 1, 2, 3, etc.)
        if mode == 0:
            self.ROWS = np.arange(0, self.n_probe_channels) + start_row
            txt = f'{self.ROWS[0]}-{self.ROWS[-1]}'

        # Alternating rows (i.e. N indices distributed evenly across M probes)
        # Requirement: N is evenly divisible by M; all probes have the same number of contacts C equal to N/M
        # Note: This requirement is enforced in other functions; this function just assigns rows assuming requirements are met TODO check
        # Examples:
            # 100 LFP channels; 2 probes with 50 contacts each; first probe has indices 0, 2, 4, ...; second probe has 1, 3, 5, ...
            # 90 LFP channels; 3 probes with 30 contacts each; first probe has indices 0, 3, 6, ...; second has 1, 4, 7, ...; third has 2, 5, 8, ...
        elif mode == 1:  # M indices distributed evenly across M*N total rows
            # Generate a sequence (0 through n_probe_channels), then multiply by the number of probes that will be spanning the space to space them out, and add a start_row offset
            self.ROWS = (np.arange(0, self.n_probe_channels) * self.n_probes_alternating) + start_row
            txt = f'{self.ROWS[0]}-{self.ROWS[-1]}, intervals of {self.n_probes_alternating}'
        
        self.probe_label.setText(self.probe_label_fmt.format(a=self.probe.name, b=txt))


class ProbeAssignmentWidget(QtWidgets.QWidget):
    """
    Assigns LFP data channels to specific probes.

    Adds ProbeRow widgets serially until all channels are assigned.

    Probes assigned to "contiguous" (0) or "alternating" (1) rows; default is contiguous.
    """
    # Define custom signal connected to slot in the main GUI that checks for completed probe assignment 
    check_signal = QtCore.pyqtSignal()

    def __init__(self, n_lfp_channels):
        """
        Initialize rows based on the number of LFP channels in the recording.
        """
        super().__init__()
        self.MODE = 0 # default mode is "contiguous"
        self.n_rows = n_lfp_channels
        self.remaining_row_array = np.arange(self.n_rows)
        self.row_assignment_df = pd.DataFrame({'Row Index':np.arange(self.n_rows), 'Probe Index(es)':''})
        self.probe_button_group = QtWidgets.QButtonGroup() # (not a visual widget; ensures only one button in the group is selected at a time)

        # Set up layout and connect signals/slots
        self.gen_layout()
        self.connect_signals()

        # Set status variable (read by main GUI to determine when to move to the next step)
        self.probe_assignment_complete = False
            
    def gen_layout(self):
        """
        Set up layout.
        """
        # Top-level vertical layout for the whole widget
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        # Row 1: Title for the section
        assign_probe_label = QtWidgets.QLabel('<b>Assign channels to probe(s):</b>')

        # Row 2: Text displaying the channels assigned vs. remaining
        self.data_txt_fmt = (f"<b><code>{self.n_rows} LFP channels: "
                             "<font color='green'>%s assigned</font>, "
                             "<font color=%s>%s remaining</font></code></b>")
        self.data_lbl = QtWidgets.QLabel(self.data_txt_fmt % (0, "red", self.n_rows))

        # Row 3: "Load" and "Create" buttons for adding probes
        self.load_prb_btn = QtWidgets.QPushButton('Load probe')
        self.create_prb_btn = QtWidgets.QPushButton('Create/edit probe')
        self.view_assignments_btn = QtWidgets.QPushButton('View row-to-probe assignment table')
        btn_list = [self.load_prb_btn, self.create_prb_btn, self.view_assignments_btn]
        button_ss = """
            QPushButton {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px 10px;         /* padding inside button (top/bottom, left/right) */
                margin: 0;
            }
            QPushButton:hover {
                background-color: #f0f0f0; /* light gray on hover */
            }
            QPushButton:pressed {
                background-color: #e0e0e0; /* darker gray when pressed */
            }
        """
        self.add_probe_button_layout = QtWidgets.QHBoxLayout()
        self.add_probe_button_layout.setContentsMargins(0,0,0,0)
        self.add_probe_button_layout.setSpacing(3)  
        for btn in btn_list:
            btn.setStyleSheet(button_ss)      
            self.add_probe_button_layout.addWidget(btn)
        self.add_probe_button_layout.addStretch() # Keep the buttons left-aligned by adding a space to the rest of the row

        # Row 4: Container for probe objects (displayed as styled QPushButtons within a QFrame)
        self.qframe = QtWidgets.QFrame()
        self.qframe.setFrameShape(QtWidgets.QFrame.Panel)
        self.qframe.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.qframe.setLineWidth(2)
        self.qframe.setStyleSheet("QFrame {background-color: white;}")
        qframe_layout = QtWidgets.QVBoxLayout(self.qframe)  # Assign a vertical layout to the QFrame
        qframe_layout.setSpacing(10)                        # Since we only add one element to qframe_layout, the spacing is not too important
        self.probe_row_layout = QtWidgets.QVBoxLayout()     # Vertical layout to which the actual probe rows will be added
        qframe_layout.addLayout(self.probe_row_layout)      # Add to the qframe_layout
        self.zero_probes_added_label = QtWidgets.QLabel("<i>To start, load or create a probe.</i>")
        self.zero_probes_added_label.setVisible(True)
        qframe_layout.addWidget(self.zero_probes_added_label)

        # Row 5: Row indexing mode (contiguous/serial blocks vs. alternating)
        row_indexing_settings_layout = QtWidgets.QHBoxLayout()
        row_indexing_settings_layout.setSpacing(5)
        assign_lbl = QtWidgets.QLabel('Row indexing mode (for multi-probe recordings):')
        self.contiguous_radio_btn = QtWidgets.QRadioButton('Contiguous')
        self.alternating_radio_btn = QtWidgets.QRadioButton('Alternating')
        row_indexing_settings_layout.addWidget(assign_lbl)
        row_indexing_settings_layout.addWidget(self.contiguous_radio_btn)
        row_indexing_settings_layout.addWidget(self.alternating_radio_btn)
        row_indexing_settings_layout.addStretch()   # Keep the buttons left-aligned by adding a space to the rest of the row
        self.contiguous_radio_btn.setChecked(True)  # Default setting is contiguous

        # Add widgets and layouts to a vertical layout
        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.vertical_layout.addWidget(assign_probe_label)              # "Assign channels to probe(s)" label
        self.vertical_layout.addWidget(self.data_lbl)                   # Channels assigned vs. remaining label
        self.vertical_layout.addLayout(self.add_probe_button_layout)    # Buttons
        self.vertical_layout.addWidget(self.qframe)                     # QFrame to display probes
        self.vertical_layout.addLayout(row_indexing_settings_layout)    # Contiguous vs. alternating row indexing

        # Add layout with all the labels/buttons/frames to the top-level VBoxLayout
        self.layout.addLayout(self.vertical_layout)
                
    def connect_signals(self):
        """
        Connect GUI button signals to slots.
        """
        self.load_prb_btn.clicked.connect(self.load_probe_from_file)
        self.create_prb_btn.clicked.connect(self.design_probe)
        self.view_assignments_btn.clicked.connect(self.view_data_assignments)
        self.contiguous_radio_btn.toggled.connect(self.switch_index_mode) # Connect the "contiguous" button to the "switch_index_mode" slot; toggled = True, unchecked = False
    
    def view_data_assignments(self):
        """
        Show mapping between each LFP channel row index and the assigned probe index.
        """
        # Create a table widget
        row_to_probe_table_widget = QtWidgets.QTableWidget()

        # Make the table read-only
        row_to_probe_table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # Get properties of the DataFrame and use them to arrange the table
        column_labels = self.row_assignment_df.columns
        n_rows = len(self.row_assignment_df.index.values)
        n_columns = len(column_labels)
        row_to_probe_table_widget.setRowCount(n_rows)
        row_to_probe_table_widget.setColumnCount(n_columns)
        row_to_probe_table_widget.setHorizontalHeaderLabels(column_labels)

        # Assign, but hide, the vertical header labels since the "row index" is basically the same as the index
        row_to_probe_table_widget.setVerticalHeaderLabels(self.row_assignment_df.index.values.astype(str))
        row_to_probe_table_widget.verticalHeader().hide()
        
        # Add DataFrame items to table
        for row in range(n_rows):
            for col in range(n_columns):
                item = QtWidgets.QTableWidgetItem(str(self.row_assignment_df.iat[row, col]))
                row_to_probe_table_widget.setItem(row, col, item)
        
        # Make the two columns equal in size, but keep the rows tight
        row_to_probe_table_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        row_to_probe_table_widget.resizeRowsToContents()

        # Make a pop-up window; given it a title, icon, and remove the help button
        popup_window = QtWidgets.QDialog()
        popup_window.setWindowTitle("Channel to probe assignment map")
        popup_window.setWindowIcon(QtGui.QIcon(':/resources/logo.png'))
        gi.remove_help_button(popup_window)

        # Add the table widget to the pop-up window
        popup_window.setLayout(QtWidgets.QVBoxLayout())
        popup_window.layout().addWidget(QtWidgets.QLabel("<b>LFP channel row index to probe index</b>"))
        popup_window.layout().addWidget(row_to_probe_table_widget)

        # Set the pop-up window size to wrap the table horizontally     
        popup_window.adjustSize()
        popup_window.setMinimumHeight(600)

        # Show the pop-up window
        popup_window.exec_()

    def switch_index_mode(self, contiguous_btn_status):
        """
        Assign probes to contiguous blocks or distributed rows of data.
        """
        if contiguous_btn_status:
            self.MODE = 0 # Contiguous
        else:
            self.MODE = 1 # Alternating

        # Get the list of probes currently added
        probe_row_object_list = pyfx.layout_items(self.probe_row_layout)

        # Reset the "remaining_row_array" to span the full set of LFP channels
        self.remaining_row_array = np.arange(self.n_rows)

        start_row = 0
        for i, probe in enumerate(probe_row_object_list):
            probe.get_rows(start_row, self.MODE)
            if self.MODE == 0:
                start_row = probe.ROWS[-1] + 1 # If in contiguous mode, the next probe will start at after this probe's block of channels
            elif self.MODE == 1:
                start_row += 1 # If in alternating mode, the next probe will start after this probe's first channel

            # Remove the current probe's rows from the "remaining_row_array"
            self.remaining_row_array = np.setdiff1d(self.remaining_row_array, probe.ROWS)
        
        self.check_assignments()
        
    def add_probe_row(self, probe):
        """
        Add new probe to collection. Used for loading, creating, and copying probes.

        probe: probeinterface.probe.Probe object
        """
        n_probe_channels = probe.get_contact_count()
        probe_row_object_list = pyfx.layout_items(self.probe_row_layout)

        # Check 1: Enough remaining rows to assign to probe channels
        try:
            assert n_probe_channels <= len(self.remaining_row_array)
        except AssertionError:
            msg = f'Cannot map {n_probe_channels}-channel probe to {len(self.remaining_row_array)} remaining data rows'
            gi.MsgboxError(msg, parent=self).exec()
            return
        
        # Check 2: If in alternating mode, the number of channels on the probe to be added must: 
            # (1) be able to evenly divide the number of channels in the recording
            # (2) be equal to the number of channels on any other existing probes already added
        if self.MODE == 1:
            if len(probe_row_object_list) == 0: # No probes added yet
                try: 
                    assert (self.n_rows % n_probe_channels) == 0 # True if evenly divisible, False if not
                except AssertionError:
                    msg = "Alternating indexing requires the LFP channel count to be evenly divisible by the probe channel count."
                    gi.MsgboxError(msg, parent=self).exec()
                    return
            else: # Existing probes; compare to see if the number of channels across probes matches
                probe_channel_counts = [probe.n_probe_channels for probe in probe_row_object_list] + [n_probe_channels]
                try:
                    assert len(np.unique(probe_channel_counts)) <= 1
                except AssertionError:
                    msg = 'Alternating indexing requires all probes to be the same size.'
                    gi.MsgboxError(msg, parent=self).exec()
                    return
        
        # If the checks above pass, get the start row for the new probe based on the previously added probes and assignment mode.
        start_row = 0
        if len(probe_row_object_list) > 0:
            last_added_probe_rows = probe_row_object_list[-1].ROWS # Get the rows assigned to the previously-added probe
            if self.MODE == 0:
                start_row = last_added_probe_rows[-1] + 1   # Begin a block after the last probe
            elif self.MODE == 1:
                start_row = last_added_probe_rows[0] + 1    # Begin alternating one spot after the last probe

        # Create a new ProbeRow object for the added probe
        probe_row = ProbeRow(probe = probe, n_recording_lfp_channels = self.n_rows, start_row = start_row, mode = self.MODE)
        self.probe_button_group.addButton(probe_row.btn) # Add the button to the button group to ensure only one probe can be selected at a time

        # Connect the action buttons to slots
        probe_row.copy_btn.clicked.connect(lambda: self.copy_probe_row(probe_row)) 
        probe_row.delete_btn.clicked.connect(lambda: self.del_probe_row(probe_row))
        
        self.probe_row_layout.addWidget(probe_row) # TODO explain 

        # Update the "remaining_row_array" by removing all the rows claimed by this new probe
        self.remaining_row_array = np.setdiff1d(self.remaining_row_array, probe_row.ROWS)

        # Check the overall status of the probe assignment process now that a new probe has been added
        self.check_assignments()        

    def copy_probe_row(self, probe_row):
        """
        Duplicate an assigned probe.
        """
        # Copy probe configuration to new probe object, add as row.
        selected_probe = probe_row.probe
        new_probe = selected_probe.copy()
        new_probe.annotate(**dict(selected_probe.annotations))
        new_probe.set_shank_ids(np.array(selected_probe.shank_ids))
        new_probe.set_contact_ids(np.array(selected_probe.contact_ids))
        new_probe.set_device_channel_indices(np.array(selected_probe.device_channel_indices))
        self.add_probe_row(new_probe)

    def del_probe_row(self, probe_row):
        """
        Remove assigned probe from collection.
        Re-calculate the array of remaining rows.
        """
        # Get the list of added probes
        probe_row_object_list = pyfx.layout_items(self.probe_row_layout)

        # Position of probe object to be deleted
        idx = probe_row_object_list.index(probe_row)
        
        # Remove the probe button from the button group
        self.probe_button_group.removeButton(probe_row.btn)

        # Remove the widget from the layout
        self.probe_row_layout.removeWidget(probe_row)

        # Remove the widget from the parent/UI and delete
        probe_row.setParent(None)
        probe_row.deleteLater()
        
        # Get an updated list of added probes
        updated_probe_row_object_list = pyfx.layout_items(self.probe_row_layout)

        # Reset the remaining row array to include all LFP channels
        self.remaining_row_array = np.arange(self.n_rows)

        # Iterate over the probes to update the "remaining row array" (removing the assigned rows for each probe)
        for i, probe in enumerate(updated_probe_row_object_list):
            # (Visual change) Select the row above the deleted row
            if i == max(idx-1, 0): 
                probe.btn.setChecked(True)

            # Probes above the deleted row can be kept the same
            if i < idx:
                pass # Continue on to the "remaining_row_array" update without modifying the probe rows
            
            # Otherwise, they need their rows adjusted
            else:
                if i == 0:
                    start_row = 0
                else:
                    # Example: row index 3 was deleted. At i = 3 in the updated list, we get the probe at i = 2
                    previous_probe_rows = updated_probe_row_object_list[i-1].ROWS
                    if self.MODE == 0:
                        start_row = previous_probe_rows[-1] + 1 # Start a block after the previous probe
                    elif self.MODE == 1:
                        start_row = previous_probe_rows[0] + 1 

                # Assign new rows to the probe
                probe.get_rows(start_row, self.MODE)

            # Key step for all probes: Update the remaining probe rows
            self.remaining_row_array = np.setdiff1d(self.remaining_row_array, probe.ROWS)

        # Check overall probe assigment status now that a probe has been removed
        self.check_assignments()

    def load_probe_from_file(self):
        """
        Load probe object from saved file, add to collection.
        """
        probe, _ = h_io.select_load_probe_file(parent=self) # Returns probe (probeinterface object) and filepath ("_", str)

        if probe is None: # No probe was selected, exit function
            return
        else:
            self.add_probe_row(probe)
    
    def design_probe(self):
        """
        Launch probe designer popup
        """
        probe_popup = ProbeObjectPopup()
        probe_popup.setModal(True)
        probe_popup.accept_btn.setVisible(True)
        probe_popup.accept_btn.setText('CHOOSE PROBE')
        res = probe_popup.exec()
        if res:
            probe = probe_popup.probe_widget.probe
            self.add_probe_row(probe)

    def check_assignments(self):
        """
        Check for valid assignment upon probe addition/deletion/reindexing.
        Emits signal connected to a main GUI slot that checks for a completed probe assignment and can trigger the next step in the workflow.
        """
        # Get a list of probe(s) widgets
        probe_row_object_list = pyfx.layout_items(self.probe_row_layout)

        # If there are no probes added, the rest of the check is not necessary. Reset the UI and exit function.
        if len(probe_row_object_list) == 0: 
            self.zero_probes_added_label.setVisible(True)                       # Display message about loading/creating probe in the box where probes are displayed once added
            self.alternating_radio_btn.setEnabled(True)                         # Allow for mode switching
            self.data_lbl.setText(self.data_txt_fmt % (0, "red", self.n_rows))  # Update the text which says how many channels are left to assign
            self.probe_assignment_complete = False
            self.check_signal.emit()
            return
        else:
            self.zero_probes_added_label.setVisible(False)
        
        # Multi-probe indexing: Disable "alternating" option when conditions aren't satisfied
        # Alternating row indexing requires probes to all have the same number of channels
        all_probes_equal_ch_count = len(np.unique([probe_widget.n_probe_channels for probe_widget in probe_row_object_list])) <= 1 # 0 if no probes, 1 if all probes have the same number of channels
        if all_probes_equal_ch_count:
            # In addition, the number of channels in the recording must be evenly divisible by the number of channels on the probe
            probe_ch_count = probe_row_object_list[0].n_probe_channels
            recording_ch_evenly_divisible = (self.n_rows % probe_ch_count) == 0 # True if evenly divisible, False if not
            self.alternating_radio_btn.setEnabled(recording_ch_evenly_divisible)
        else:
            self.alternating_radio_btn.setEnabled(False)
        
        # Make a dictionary mapping the data row index to the probe index
        data_row_idx_to_probe_idx_dict = {}
        for data_row_idx in np.arange(self.n_rows): # Iterate over each data row 
            # For each data row index
            # Iterate over the probe widgets
            # For each probe, get the array with its assigned rows
            # If "data_row_idx" is assigned to the probe, add the probe index to a list
            # If a given row_idx is not assigned to any probe yet, the list will be empty
            data_row_idx_to_probe_idx_dict[data_row_idx] = [probe_idx for probe_idx, probe_widget in enumerate(probe_row_object_list) if data_row_idx in probe_widget.ROWS]
        
        # Probe assignment is valid if each data row is matched to exactly one probe
        matches = [len(probe_assignment_list) == 1 for probe_assignment_list in data_row_idx_to_probe_idx_dict.values()]
        n_valid_rows = sum(matches) # True = 1, False = 0
        self.probe_assignment_complete = bool(n_valid_rows == self.n_rows) # Probe assignment is complete when all channels have been assigned
        
        # Update the label listing the number of assigned vs. remaining channels
        n_rows_assigned = n_valid_rows
        n_rows_remaining = self.n_rows - n_rows_assigned
        if n_rows_remaining == 0:
            text_color = "green"
        else:
            text_color = "red"
        self.data_lbl.setText(self.data_txt_fmt % (n_rows_assigned, text_color, n_rows_remaining))

        # Update the DataFrame (pairs row indices with probe indices)
        probe_assignment_strings = [', '.join(np.array(probe_idx_list, dtype=str)) for probe_idx_list in data_row_idx_to_probe_idx_dict.values()]
        row_idx_list = data_row_idx_to_probe_idx_dict.keys()
        self.row_assignment_df = pd.DataFrame({'Row Index': row_idx_list, 'Probe Index(es)':probe_assignment_strings})

        # Emit signal connected to main GUI slot that can trigger the next step in the pipeline (if self.probe_assignment_complete is True)
        self.check_signal.emit()


class ParameterSettingWidget(QtWidgets.QWidget):
    """
    Widget for setting processing parameters.

    Note that much of this code is similar or identical to that in "gui_set_parameters.py"
    """
    # Define custom signal connected to slot in the main GUI that checks for completed parameter setting
    parameter_setting_status = QtCore.pyqtSignal()

    def __init__(self, param_file_path):
        super().__init__()

        self.current_param_file_path = param_file_path
    
        # Set up layout and connect signals/slots
        self.gen_layout()
        self.connect_signals()

        # Set status variable (read by main GUI to determine when to move to the next step)
        self.parameter_setting_complete = False
            
    def gen_layout(self):
        """
        Set up layout.
        """
        # Top-level vertical layout for the whole widget
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        # Row 1: Title for the section
        set_parameters_label = QtWidgets.QLabel('<b>Set processing parameters:</b>')

        # Row 2: Text displaying the channels assigned vs. remaining
        # Define stylesheet for QLineEdit
        le_ss = """
        QLineEdit {
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 0px;
        }
        """
        # Create a QLineEdit for displaying the current parameter file path 
        # Note: We use QLineEdit instead of QLabel because QLineEdit automatically displays the end of the string when the width of the window is less than the string, while QLabel shows the beginning.
        self.path_display_le = QtWidgets.QLineEdit(self.current_param_file_path) #HEREO
        self.path_display_le.setTextMargins(0,4,0,4) # Margins for left, top, right, bottom
        self.path_display_le.setReadOnly(True)
        self.path_display_le.setStyleSheet(le_ss)

        # Create title and button to go around QLineEdit
        subhead_qlabel_a = QtWidgets.QLabel(f'Parameter file:')
        self.paramfile_btn = gi.make_icon_button(QtGui.QIcon(f":/resources/load-file.png"))
        self.paramfile_autofill_btn = gi.make_icon_button(QtGui.QIcon(f":/resources/magic-wand.png"))
        self.paramfile_btn.setToolTip("Select a file")  # Add a hint to the "load file" buttons
        self.paramfile_autofill_btn.setToolTip("Generate a new parameter file with Toothy's default values") # Add a hint to the "autofill" parameter button
        row_elements = [self.path_display_le, self.paramfile_btn, self.paramfile_autofill_btn]
        
        # Create a horizontal row with the QLineEdit and icon buttons
        row_layout = pyfx.get_widget_container('h', *row_elements, spacing = 10)

        # Get a widget with the QLabel positioned above the QLineEdit/QPushButtons row from above
        paramfile_row_widget = pyfx.get_widget_container('v', subhead_qlabel_a, row_layout, spacing = 5, widget='widget')

        # Add widgets and layouts to a vertical layout
        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.vertical_layout.addWidget(set_parameters_label)
        self.vertical_layout.addWidget(paramfile_row_widget)
        subhead_qlabel_b = QtWidgets.QLabel("File values:")     
        self.vertical_layout.addWidget(subhead_qlabel_b)
        self.instructions_qlabel = QtWidgets.QLabel("<i>Hover over parameter name for a description</i>")
        self.vertical_layout.addWidget(self.instructions_qlabel)

        # Style the subheadings
        subhead_labels = [subhead_qlabel_a, subhead_qlabel_b]
        for label in subhead_labels:
            label.setStyleSheet("color: #0A2B30; font-weight: bold;")

        # Set a "filler widget" to the frame
        self.filler_widget = QtWidgets.QWidget()
        self.filler_widget.setStyleSheet("QWidget {background-color: white;}")
        layout = QtWidgets.QVBoxLayout(self.filler_widget)
        layout.addWidget(QtWidgets.QLabel("<i>Select a file to view and edit parameter values.</i>"))
        self.filler_widget.setLayout(layout)
        self.vertical_layout.addWidget(self.filler_widget)
        
        # Hide instructions for interacting with parameters since none are shown yet
        self.instructions_qlabel.setVisible(False)

        # If there is a parameter file, create a ParameterWidget and add it to frame
        # Note that we create a filler widget first since "initialize_parameter_widget" removes it
        if os.path.isfile(self.current_param_file_path):
            self.initialize_parameter_widget()
    
        # Add layout with all the labels/buttons/frames to the top-level VBoxLayout
        self.layout.addLayout(self.vertical_layout)
             
    def connect_signals(self):
        """
        Connect GUI button signals to slots.
        """
        # Parameters file buttons
        self.paramfile_btn.clicked.connect(self.choose_param_file)
        self.paramfile_autofill_btn.clicked.connect(self.generate_default_param_file)

    def initialize_parameter_widget(self):
        """
        Load a parameter dictionary and use it to instantiate a ParameterWidget. Add the ParameterWidget to the QDialog in the QScroll widget.
        """
        # Get the parameter dictionary
        param_dict = qparam.read_param_file(self.current_param_file_path)[0]
            
        # Initialize parameter input widget (ParameterWidget)
        self.main_widget = qparam.ParameterWidget(param_dict, mode = "data_processing")

        self.main_widget.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)

        self.main_widget.setContentsMargins(0,0,15,0)
        
        # Get the current parameter values as a dictionary
        self.PARAMS_DICT, _ = self.main_widget.get_param_dict_from_gui()

        # Copy the "original values"
        self.PARAMS_DICT_ORIG = dict(self.PARAMS_DICT)

        # Connect signal to slot # TODO
        # self.main_widget.update_signal.connect(self.update_slot)

        # Show instructions for interacting with parameters
        self.instructions_qlabel.setVisible(True)
        
        self.vertical_layout.removeWidget(self.filler_widget)
        self.vertical_layout.addWidget(self.main_widget)

        # TODO maybe make this some kind of check for parameter compatibility
        # # Make the button for confirming a file active
        # self.exit_btn.setEnabled(True)

    def file_selection_update(self):
        """
        Called when a file selection is made. Cases:
        1. Saving changes to an open file TODO update
        2. Loading a file from the "Select a file" button
        3. Generating (and saving) a new file with default parameters

        Updates the displayed filepath, displayed GUI values, internal variables representing the parameters, and toggles the appropriate buttons.

        Does not update the "toothy paths"; this is achieved with the "confirm/exit" button.
        """
        # Update the path displayed in the QLineEdit widget
        self.path_display_le.setText(self.current_param_file_path)

        # Read the values in the file
        new_param_dict = qparam.read_param_file(self.current_param_file_path)[0]

        # Display the updated values in the GUI
        self.main_widget.update_gui_from_param_dict(new_param_dict)

        # Set the current parameter values
        self.PARAMS_DICT = new_param_dict
        
        # Update the "original values" dictionary
        self.PARAMS_DICT_ORIG = dict(self.PARAMS_DICT)

        # # Disable the "save" and "reset" buttons, enable the "confirm/exit" button
        # self.save_btn.setEnabled(False)
        # self.reset_btn.setEnabled(False)
        # self.exit_btn.setEnabled(True)

    def choose_param_file(self):
        """
        Slot for parameter file selection button.
        Select parameter file from file explorer and trigger function to update the GUI.
        """
        # Get the current path
        current_path = self.current_param_file_path 

        # Open a file dialog and allow the user to load a file
        # Note: param_dict is None if invalid; otherwise, can be used without further checking of valid types
        param_dict, file_path = h_io.select_load_param_file(init_path=current_path, parent=self)

        # If the parameter file is valid (invalid files are returned as None), update 
        if param_dict is not None:
            # Make note of the current file path; if it is empty, we will need to initialize a parameter widget
            previous_file_path = self.current_param_file_path

            # Update the file path internally
            self.current_param_file_path = str(file_path)

            # Create a parameter widget if needed
            if previous_file_path == "":
                self.initialize_parameter_widget()
                
            # Propagate widger changes using the updated parameter file path
            self.file_selection_update()

        # If param_dict is None, the dictionary was invalid. We do not make any updates.
    
    def generate_default_param_file(self):
        """
        Slot for default parameter file generation button.
        Opens a file dialog for user to select a path to which Toothy will save a new parameter file with default values.
        """
        # Get a dictionary of default parameters
        toothy_default_parameter_dict = qparam.get_toothy_default_parameter_dict()

        # Save the dictionary as a JSON file; open a file dialog over the current QDialog with the provided title; returns path to saved file
        file_path = h_io.select_save_param_file(
            parameter_dictionary = toothy_default_parameter_dict, 
            initial_dir = os.getcwd(),
            initial_file_name="toothy_default_parameters.json",
            title = 'Save default parameter file', 
            parent = self
        )

        # If the file was saved, update
        if file_path:
            # Make note of the current file path; if it is empty, we will need to initialize a parameter widget
            previous_file_path = self.current_param_file_path

            # Update the file path internally
            self.current_param_file_path = str(file_path)

            # Create a parameter widget if needed
            if previous_file_path == "":
                self.initialize_parameter_widget()

            # Propagate widger changes using the updated parameter file path
            self.file_selection_update()
    
    def check_assignments(self):
        """
        Check for valid assignment upon probe addition/deletion/reindexing.
        Emits signal connected to a main GUI slot that checks for a completed probe assignment and can trigger the next step in the workflow.
        """
        #...
        # Emit signal connected to main GUI slot that can trigger the next step in the pipeline (if self.probe_assignment_complete is True)
        # self.check_signal.emit()
        pass

###############################################
###############################################
######     Section 3: Main Interface     ######
###############################################
###############################################

class InputDataSelectionPopup(QtWidgets.QDialog):
    """
    Data processing GUI for input recording selection and probe assignment.
    
    QDialog widget (presents a top-level window used to collect a response from the user.)
    """
    output_dir = None # Used in toothy.py to set the initial folder for loading processed data in the Channel Selection step
    
    def __init__(self):
        super().__init__()

        # Set the window title (changes as the user proceeds through the workflow of this QDialog)
        self.setWindowTitle('Step 1: Data Ingestion')

        # Set icon for upper left corner and toolbar
        self.setWindowIcon(QtGui.QIcon(':/resources/logo.png'))

        # Remove the "?" help button
        gi.remove_help_button(self)

        # Make the QDialog background white
        ss = "QDialog {background-color : white;}"
        self.setStyleSheet(ss)
        
        # TODO remove
        # print("breaks here i guess?")
        # self.PARAMS = h_io.read_params() # Returns a dictionary of validated default parameters

        # Get the initial directories
        input_data_dir_path, probe_config_dir_path, probe_file_path, param_file_path  = h_io.get_toothy_paths()
        self.default_probe_file = probe_file_path
        self.param_file_path = param_file_path
        
        # Set up the window; creates "self.fsw" referenced two lines down
        self.gen_layout()

        # Connect all the signals and slots
        self.connect_signals()

        # Update the file selection widget with the default directory for input data
        # Note: this will display the directory in the QLineEdit, but keep the outline red, since validation will fail on a path that is a directory rather than a file
        self.fsw.update_filepath(input_data_dir_path)

        # Hide downstream sections of the workflow
        self.probe_frame.hide()
        self.output_dir_frame.hide()
        
    def gen_layout(self):
        """
        Set up layout and add widgets for input recording selection.
        """
        # Create a "content widget" with a QVBoxLayout (linear vertical layout)
        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(0)
        
        ###########################################
        ###########################################
        #####   INPUT DATA SOURCE SELECTION   #####
        ###########################################
        ###########################################
    
        # Create a QFrame widget for the input data source section
        self.recording_selection_frame = QtWidgets.QFrame()

        # Set stylesheet for the input data QFrame to have a light gray background
        qframe_ss = 'QFrame {background-color : #E8E8E8;}'
        self.recording_selection_frame.setStyleSheet(qframe_ss)

        ##### FILE SELECTION WIDGET #####
        # Get a nested QVBoxLayout to add file selection widget
            # Interwidget takes a parent widget and orientation ('h' or 'v') and returns a layout of a child widget added to the parent widget
            # Send our parent QGroupBox to interwidget
            # Applies a vertical layout to our parent QGroupBox ("interlayout")
            # Adds a QWidget ("interwidget") with a vertical layout (QVBoxLayout, specified by the "v" parameter) to the parent QGroupBox
            # Returns a tuple of (interlayout, interwidget, layout), where layout is the interwidget's layout
            # We index into the tuple and just take "layout"
        fs_vbox = pyfx.InterWidgets(self.recording_selection_frame, 'v')[2] # QVBoxLayout

        # Create file selection widget (instance of gui_items.FileSelectionWidget)
        self.fsw = RawRecordingSelectionWidget(title='<b>Select a recording:</b>')

        # Add file selecton widget to layout
        fs_vbox.addWidget(self.fsw)

        ##### LISTING SUPPORTED FILE FORMATS #####
        # Add a label for the heading "Supported file formats:"
        header_label = QtWidgets.QLabel("<i>Supported file formats:</i>")
        header_label.setContentsMargins(0, 10, 0, 0)    # Add space to the top margin (left, top, right, bottom)
        self.fsw.vlay.addWidget(header_label)           # Add header label to the bottom of the directory selection widget's layout
        
        # Add labels listing the supported data formats
        self.data_format_layout = QtWidgets.QVBoxLayout()
        self.data_format_layout.setSpacing(3) # Reduce row spacing
        for k, (file_format, file_ext) in dp.supported_formats.items():
            format_label = QtWidgets.QLabel(f'• {file_format} ({file_ext})') # Bullet point with the format then the file extension in parenthesis
            format_label.setContentsMargins(25, 0, 0, 0)    # Add space to the left margin (left, top, right, bottom) to appear indented
            self.data_format_layout.addWidget(format_label) # Add label to vertical layout
        self.fsw.vlay.addLayout(self.data_format_layout)    # Add data format layout to the bottom of the directory selection widget's layout
                
        ##### DISPLAYING RESULT OF SUCCESSFULLY SELECTING & EXTRACTING FILE #####
        # Create a QFrame widget for displaying result of file selection and make the background white
        self.metadata_frame = QtWidgets.QFrame() 
        ss = 'QFrame {background-color : white;}'
        self.metadata_frame.setStyleSheet(ss)
        self.metadata_frame_layout = pyfx.InterWidgets(self.metadata_frame, 'v')[2] # Get a QVBoxLayout of a child widget in the QGroupBox

        # File loaded message
        self.file_loaded_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.file_loaded_label.setStyleSheet("color: darkgreen;")
        self.metadata_frame_layout.addWidget(self.file_loaded_label)

        # Filename
        self.file_name_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.metadata_frame_layout.addWidget(self.file_name_label)

        # Recording metadata
        self.recording_metadata_header = QtWidgets.QLabel("<b>Recording properties:</b>")
        self.metadata_frame_layout.addWidget(self.recording_metadata_header)
        self.metadata_display_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.metadata_display_label.setContentsMargins(25, 0, 0, 0)  # Add padding to the left so it looks indented
        # self.metadata_display_label.setFont(QtGui.QFont("Courier New") # Text needs to be monospace for alignment; use <code> tags later instead of setting the font for monospacing
        self.metadata_frame_layout.addWidget(self.metadata_display_label)

        # Time metadata
        self.time_metadata_header = QtWidgets.QLabel("<b>Timestamps properties:</b>")
        self.metadata_frame_layout.addWidget(self.time_metadata_header)
        self.time_metadata_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.time_metadata_label.setContentsMargins(25, 0, 0, 0)  # Add padding to the left so it looks indented
        self.metadata_frame_layout.addWidget(self.time_metadata_label)

        # Hide the section to start; make visible later in "extractor_finished_slot"
        self.metadata_frame.hide() 

        ##### DISPLAYING RESULT OF EXTRACTION ISSUE #####
        # Create a QFrame widget for displaying result of file selection and make the background white
        self.extraction_error_frame = QtWidgets.QFrame() 
        ss = 'QFrame {background-color : white;}'
        self.extraction_error_frame.setStyleSheet(ss)
        self.extraction_error_frame_layout = pyfx.InterWidgets(self.extraction_error_frame, 'v')[2] # Get a QVBoxLayout of a child widget in the QGroupBox

        # File loaded message
        self.file_issue_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.file_issue_label.setStyleSheet("color: #991F08;")
        self.extraction_error_frame_layout.addWidget(self.file_issue_label)

        # Filename
        self.file_name_label_copy = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.extraction_error_frame_layout.addWidget(self.file_name_label_copy) # Same label as used for regular metadata

        # Hide the section to start; make visible later in "extractor_error_slot"
        self.extraction_error_frame.hide() 

        ########################################
        ########################################
        #####   PROBE ASSIGNMENT SECTION   #####
        ########################################
        ########################################

        self.probe_frame = QtWidgets.QFrame()
        self.probe_frame.setStyleSheet("QFrame {background-color: #E8E8E8}")
        self.probe_vbox = pyfx.InterWidgets(self.probe_frame, 'v')[2]
        self.paw = None

        #################################
        #################################
        #####   PARAMETER SETTING   #####
        #################################
        #################################

        # Create a QFrame widget for the parameter setting section 
        self.parameters_frame = QtWidgets.QFrame()
        self.parameters_frame.setStyleSheet("QFrame {background-color: white;}")
        self.parameters_frame_layout = pyfx.InterWidgets(self.parameters_frame, 'v')[2] # Get a QVBoxLayout of a child widget in the QGroupBox
        self.parameters_frame.setVisible(False)
        
        ############################## QUARANTINE ########################################################################################

        ### settings widget
        self.settings_w = QtWidgets.QWidget()
        settings_vlay = QtWidgets.QVBoxLayout(self.settings_w)
        settings_vlay.setContentsMargins(0,0,0,0)
        
        # Initialize main parameter input widget, embed in scroll area
        # self.params_widget = qparam.ParameterWidget(params=dict(self.PARAMS), mode='data_processing', parent=self)
        # self.qscroll = QtWidgets.QScrollArea()
        # self.qscroll.horizontalScrollBar().hide()
        # self.qscroll.setWidgetResizable(True)
        # self.qscroll.setWidget(self.params_widget)
        # qh = pyfx.ScreenRect(perc_height=0.25, keep_aspect=False).height()
        # self.qscroll.setMaximumHeight(qh)
        # self.qscroll.hide()

        # self.parameters_frame_layout.addWidget(self.params_widget)

        # create settings button to show/hide param widgets --> Rename to parameters for consistency TODO3
        # self.params_bar = QtWidgets.QPushButton('Confirm processing parameters')
        # self.params_bar.setCheckable(True)
        # self.params_bar.setFocusPolicy(QtCore.Qt.NoFocus)
        # self.params_bar.setStyleSheet(pyfx.dict2ss(QSS.EXPAND_PARAMS_BTN))
        # self.params_bar.hide()
        # create icon to indicate warnings about 1 or more parameter inputs
        # self.params_warning_btn = QtWidgets.QPushButton()
        # self.params_warning_btn.setObjectName('params_warning')
        # self.params_warning_btn.setCheckable(True)
        # self.params_warning_btn.setFlat(True)
        # self.params_warning_btn.setFixedSize(25,25)
        # self.params_warning_btn.setEnabled(False)
        # self.params_warning_btn.setStyleSheet(pyfx.dict2ss(QSS.ICON_BTN))
        # self.params_warning_btn.setFlat(True)
        # self.params_warning_btn.hide()
        
        # Bar to put settings/base folders/metadata icon (two down, 1 to go)
        # bbar = QtWidgets.QHBoxLayout()
        # bbar.addWidget(self.params_warning_btn, stretch=0)
        # bbar.addWidget(self.params_bar, stretch=2)

        # settings_vlay.addLayout(bbar)
        
        # self.settings_container = QtWidgets.QSplitter()
        # self.settings_container.addWidget(self.qscroll)
        # settings_vlay.addWidget(self.settings_container)
        ############################## QUARANTINE ########################################################################################

        
        ##### OUTPUT DIRECTORY LOCATION #####
        save_lbl = QtWidgets.QLabel('<b>Select output directory:</b>')
        self.save_le = QtWidgets.QLineEdit()
        self.save_le.setReadOnly(True)
        self.select_output_dir_btn = QtWidgets.QPushButton() # Button to launch QFileDialog
        self.select_output_dir_btn.setIcon(QtGui.QIcon(':/resources/folder.png'))
        self.select_output_dir_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        # Make a QFrame with the QLabdel, QLineEdit, and folder QPushButton
        self.output_dir_frame = pyfx.get_widget_container('h', save_lbl, self.save_le, self.select_output_dir_btn, spacing = 5, widget='frame')

        ##### Bottom row action buttons #####
        # The set of buttons showing changes as the user moves through the workflow.
        bottom_row_layout = QtWidgets.QHBoxLayout()
        bottom_row_layout.setContentsMargins(0, 10, 0, 0)
        button_ss = """
            QPushButton {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 8px 14px;         /* padding inside button (top/bottom, left/right) */
                font-weight: bold;
                margin: 0;
            }
            QPushButton:hover {
                background-color: #f0f0f0; /* light gray on hover */
            }
            QPushButton:pressed {
                background-color: #e0e0e0; /* darker gray when pressed */
            }
        """

        # Button for starting probe assignment step (hidden and disabled to start)
        self.probe_map_btn = QtWidgets.QPushButton('Map LFP channels to probe(s)')
        self.probe_map_btn.setVisible(False)
        self.probe_map_btn.setEnabled(False)

        # Button for going back to file selection step from probe assignment step (hidden to start)
        self.back_to_fs_btn = QtWidgets.QPushButton('Back to file selection')
        self.back_to_fs_btn.setVisible(False)

        # Button for triggering parameter setting step
        self.set_params_btn = QtWidgets.QPushButton('Set processing parameters')
        self.set_params_btn.setVisible(False)
        self.set_params_btn.setEnabled(False)

        # Button for going back to probe assignment step from parameter setting step (hidden to start)
        self.back_to_pa_btn = QtWidgets.QPushButton('Back to probe assignment')
        self.back_to_pa_btn.setVisible(False)

        # Button for triggering output folder selection step
        self.set_output_dir_step_btn = QtWidgets.QPushButton('Set output directory')
        self.set_output_dir_step_btn.setVisible(False)
        self.set_output_dir_step_btn.setEnabled(False) # Start off disabled; enable if all parameters work well for processing
        
        # Button for triggering the data processing pipeline (hidden and disabled to start)
        self.pipeline_btn = QtWidgets.QPushButton('Process data!')
        self.pipeline_btn.setVisible(False)
        self.pipeline_btn.setEnabled(False)

        # Add buttons to QHBoxLayout
        for button in [self.probe_map_btn, self.back_to_fs_btn, self.set_params_btn, self.back_to_pa_btn, self.set_output_dir_step_btn, self.pipeline_btn]:
            button.setStyleSheet(button_ss)
            bottom_row_layout.addWidget(button)

        ##### ADD WIDGETS TO CENTRAL LAYOUT #####
        self.content_layout.addWidget(self.recording_selection_frame)    # File selection frame (top section)
        self.content_layout.addWidget(self.metadata_frame)               # Metadata frame (hidden; appears under file selection after user selects a file if extraction is successful)
        self.content_layout.addWidget(self.extraction_error_frame)       # Error message frame (hidden; appears under file selection after user selects a file if an error occurs)
        self.content_layout.addWidget(self.probe_frame)
        self.content_layout.addWidget(self.parameters_frame)
        self.content_layout.addWidget(self.settings_w)
        self.content_layout.addWidget(self.output_dir_frame)
        self.content_layout.addLayout(bottom_row_layout)                              # Bottom buttons (Map to probes, Back, Process data)
        self.content_layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)

        ##### Create scroll area #####
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.content_widget)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)  # Remove border
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)  # No horizontal scroll
        
        # Create the main dialog layout (assign to the QDialog)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.scroll)
        
        # Set dialog size constraints
        screen_height = QtWidgets.QApplication.primaryScreen().availableGeometry().height()
        self.max_dialog_height = int(screen_height * 0.85)  # Use 85% of screen height at most, otherwise enable scroll bar if more space is needed
        self.setMaximumHeight(self.max_dialog_height)
        self.setMinimumWidth(650)
        
        # "loading" spinner animation # TODO clean up
        self.spinner_window = gi.SpinnerWindow(self)
        self.spinner_window.spinner.setInnerRadius(25)
        self.spinner_window.spinner.setNumberOfLines(10)
        #self.spinner_window.layout.setContentsMargins(5,5,5,5)
        self.spinner_window.layout.setSpacing(0)
        #self.spinner_window.adjust_labelSize(lw=2.5, lh=0.65, ww=3)
    
    def resize_window_scroll_widget(self):
        # Necessary for reducing window size after content is removed
        # However, sometimes this line causes the window to shrink too much
        # Note that this function should be called before "resize_window_content_widget"
        self.scroll.adjustSize()
 
    def resize_window_content_widget(self):
        """
        Resize the window to display as much content as possible (when content is added)
        """
        # Necessary for growing window after content is added
        QtCore.QCoreApplication.processEvents() # Process any pending events (critical for scrolling to work correctly)
        content_height = self.content_widget.height() # Get the height of the inner content
        window_height = np.min([content_height, self.max_dialog_height]) # The new window height should be the smaller of the content and the max allowable window height
        self.resize(self.width(), window_height)

    def connect_signals(self):
        """
        Connect GUI inputs
        """
        # File selection and extraction
        self.fsw.ppath_btn.clicked.connect(self.fsw.select_filepath)    # Connect folder icon button (ppath_btn) clicked signal to "select_filepath" slot (launches file selection widget)
        self.fsw.signal.connect(self.recording_file_path_updated)       # Connect FileSelectionWidget output signal "signal" to "recording_file_path_updated" slot, which launches ExtractorWorker
        self.fsw.read_array_signal.connect(self.load_array_worker)      # Connect signal "read_array_signal" to "load_array_worker" slot, which processed NumPy and MATLAB files so that the user enters/confirms the data array and metadata for those files

        # Move from file selection to the probe mapping workflow
        self.probe_map_btn.clicked.connect(self.start_probe_assignment)

        # Move from probe mapping to parameter setting workflow
        self.set_params_btn.clicked.connect(self.start_param_selection)

        # Not sure what this is, TODO
        # self.params_widget.warning_update_signal.connect(self.update_param_warnings)
        
        # Move from parameter setting to choosing an output directory
        self.set_output_dir_step_btn.clicked.connect(self.start_output_dir_selection)   # Bottom row button for moving to the output directory selection step
        self.select_output_dir_btn.clicked.connect(self.set_output_dir)                 # Folder button for launching QFileDialog

        # Go back to a previous step
        self.back_to_fs_btn.clicked.connect(self.back_to_file_selection)
        self.back_to_pa_btn.clicked.connect(self.back_to_probe_assignment)
        
        # Start the data processing pipeline
        self.pipeline_btn.clicked.connect(self.pipeline_worker)

    def create_worker_thread(self):
        """
        Creates a parallel worker thread for long-running processing steps.
        - Generates a worker thread (QThread).
        - Adds worker object (self.worker_object) to the worker thread (self.worker_object a QObject of either ArrayReaderWorker, ExtractorWorker, or DataWorker).
        - Connects worker thread's "started" signal to worker object's "run" slot. For each worker object, "run" has been defined to contain the important processing.
        - Connects worker object "progress_signal" (which emits strings) to the spinner window slot to show progress messages.
        - Connects worker thread "finished" signal to slots that d
        """
        # Create a QThread (worker thread)
        self.worker_thread = QtCore.QThread()
        self.worker_thread.setStackSize(16 * 1024 * 1024)

        # Move the worker object to the worker thread
        self.worker_object.moveToThread(self.worker_thread)

        # Connect the started signal for the worker thread to the worker object's "run" slot
        # When the thread starts, it will also run the worker object
        self.worker_thread.started.connect(self.worker_object.run)

        # Connect the worker object's progress signal to the spinner window's "report_progress_string" slot
        self.worker_object.progress_signal.connect(self.spinner_window.report_progress_string)

        # Connect the worker thread's finished signal to several slots
        self.worker_object.finished.connect(self.worker_thread.quit)        # Exit event loop
        self.worker_thread.finished.connect(self.worker_object.deleteLater) # Tell PyQt to delete the worker object when the event loop ends
        self.worker_thread.finished.connect(self.worker_thread.deleteLater) # Tell PyQt to delete the worker thread when the event loop ends
        self.worker_thread.finished.connect(self.finished_slot)             # Run "finished_slot" which stops the progress spinner and sets the worker object/thread to None
    
    def start_qthread(self):
        """ Start worker thread """
        # Seems to happen right after clicking "open" in file dialog TODO

        # Create worker thread self.worker_thread (also moves worker object to the worker thread)
        self.create_worker_thread()

        # Start the progress spinner
        self.spinner_window.start_spinner()

        # Start the worker thread
        self.worker_thread.start()
    
    @QtCore.pyqtSlot()
    def finished_slot(self):
        """ Worker thread stopped """
        self.spinner_window.stop_spinner() # stop "loading" spinner icon
        self.worker_object = None
        self.worker_thread = None
        
    def recording_file_path_updated(self, status):
        """ 
        Slot called when "signal" is emitted from the RawRecordingSelectionWidget after a file is selected.
        - status: "signal" emits a True/False to indicate if the selected file is valid 
        """

        # If True (path is valid), update the text in the QLineEdit widget and run the ExtractorWorker
        if status:
            ppath = self.fsw.le.text()
            data_type = dp.get_data_format(ppath)
            self.extractor_worker(ppath, data_type) # Load new recording
        else:
            # Set the status of the "Map to probe(s)" button to match "status"
            self.probe_map_btn.setEnabled(status)
            data_type = None # TODO review here
            self.recording = None # delete previous recording (if any)
            # self.params_widget.time_range.set_duration(np.inf)
            # self.params_widget.lfp_decimation_factor.set_fs(None)
    
    #################################
    #####   ArrayReaderWorker   #####
    #################################

    def load_array_worker(self, ppath):
        """
        Read raw data array into memory. Slot called  
        """
        self.worker_object = ArrayReaderWorker(ppath)                           # Initialize ArrayReaderWorker QObject
        self.worker_object.data_signal.connect(self.load_array_finished_slot)   # Connect ArrayReaderWorker signal "data_signal" to slot "load_array_finished_slot"
        self.worker_object.error_signal.connect(self.load_array_error_slot)     # Connect ArrayReaderWorker signal "error_signal" to slot "load_array_error_slot"
        self.start_qthread()                                                    # Create worker objects and worker thread, add worker object to worker thread, start progress spinner window, and start worker thread
        
    # Decorate with pyqtSlot to ensure function can be connected as a slot to the signal before thread is started    
    @QtCore.pyqtSlot(str, dict, dict)
    def load_array_finished_slot(self, file_path, data_dict, metadata_dict):
        """
        Pass valid arrays and initial metadata to the user.
        Slot function connected to "data_signal" signal of class ArrayReaderWorker which emits the file path, data dictionary, and metadata dictionary.
        If there is more than one 2D array, the user chooses the key of the one they want to use.
        Final line calls a function in the file selection widget for the user to confirm metadata values.
        """
        # If the file contains multiple 2-dimensional arrays, the user will choose one.
        if len(data_dict) > 1:
            # Create a QDialog popup for with buttons for the user to select a key
            lbl = '<u>Select LFP dataset from available keys:<br><br></u><i>(hover over key for array dimensions)<br></i>'
            key_selection_dialog = gi.ButtonPopup(*data_dict.keys(), label=lbl, title="LFP key selection", parent=self)

            # For each key button, add hover text with the dimensions (rows by columns) (appears when user hovers over button with mouse) 
            for b in key_selection_dialog.btns:
                nrows, ncols = data_dict[b.text()].shape
                b.setToolTip(f'{nrows} rows x {ncols} columns')
            
            # User selects a LFP dataset key
            if key_selection_dialog.exec(): 
                data_array = data_dict[key_selection_dialog.result]
            
            # User closes the selection window without selecting a key
            else:
                gi.MsgboxError('Aborted LFP dataset selection.', parent=self).exec() # Show error message
                return # End slot and return to data selection GUI
        
        # If the file contains only one 2-dimensional dataset, it is automatically selected.
        else: 
            data_array = list(data_dict.values())[0]

        # Lastly, the user will manually confirm sampling rate, units, etc.
        # In this function, the data/metadata is then stored within "fsw" variables which are referenced during file extraction.
        self.fsw.enter_array_metadata(file_path, data_array, metadata_dict)
        
    @QtCore.pyqtSlot(str, str)
    def load_array_error_slot(self, ppath, error_msg):
        """ Handle invalid NPY/MAT files """
        gi.MsgboxError(error_msg, parent=self).exec()
        self.fsw.update_filepath(ppath, False)

    #####################################
    #####   NWB Electrical Series   #####
    #####################################

    def get_nwb_potential_lfp_es(self, filepath):
        """
        Returns a list of potential LFP ElectricalSeries dataset names in NWB file 
        Searches only the *acquisition* group for series with an "electrodes" key in them.

        The NWB ElectricalSeries class is used to store extracellular recordings.

        NWB file structure:
            HDF5 file > acquisition > series (multiple, typically only one is an ElectricalSeries)
        """
        potential_lfp_es = []

        # Get ElectricalSeries names
        with h5py.File(filepath, 'r') as f:
            # Get the group associated with the "acquisition" key
            acquisition_group = f.get('acquisition')

            # If the group exists, make a list of the keys
            if acquisition_group:
                series_names = list(acquisition_group.keys())
            else: 
                series_names = []
            
            # For each key, get the associated group and check if there is an "electrodes" key
            # Another way to check for this would be to look at the attributes and see if the "neurodata_type" is "ElectricalSeries"
            # ^ dict(f["aquisition"][series_name].attrs)["neurodata_type"] == "ElectricalSeries"
            for series_name in series_names:
                series_keys = list(f["acquisition"][series_name].keys())
                if "electrodes" in series_keys:
                    potential_lfp_es.append(series_name)

        return potential_lfp_es
    
    def select_nwb_es(self, ppath):
        """
        Identify LFP dataset in NWB file. 
        - Typically, the LFP is stored in an ElectricalSeries object within the "acquisition" section.
        - If only one ElectricalSeries is found, it is automatically selected.
        - Otherwise, a QDialog prompts the user to choose which series is the LFP to analyze.
        """
        # Try getting names of available ElectricalSeries in NWB file
        try:
            eseries = self.get_nwb_potential_lfp_es(ppath)
        
        # File missing or corrupted
        except Exception as e:
            gi.MsgboxError(f'Error: {str(e)}', parent=self).exec()
            return False
        
        # No analyzable data
        if len(eseries) == 0:   
            gi.MsgboxError('NWB file contains no acquisition group ElectricalSeries with valid electrodes.', parent=self).exec()
            return False
        
        # If there's only one ElectricalSeries, return the only available dataset
        elif len(eseries) == 1: 
            return eseries[0]
        
        # If there are multiple, ask the user to select a target dataset
        elif len(eseries) > 1:  
            window_title = "ElectricaSeries Selection"
            instructions = 'Select an ElectricalSeries dataset:'
            dlg = gi.ButtonPopup(*eseries, label=instructions, title=window_title, parent=self)
            if dlg.exec():
                return str(dlg.result)
            else:
                gi.MsgboxError('Aborted NWB dataset selection.', parent=self).exec()
                return False
            
    ################################
    #####   Extractor Worker   #####
    ################################
    def extractor_worker(self, file_path, data_type):
        """
        Instantiate spikeinterface Extractor for loading recordings.
        """
        # Initialize data_array and metadata (only filled in for NumPy/MATLAB; otherwise kept as None/empty when passed to ExtractorWorker)
        data_array, metadata = None, {}

        # Get data_array and metadata from NumPy or MATLAB files; stored in the FileSelectionWidget by "enter_array_metadata", called after ArrayReaderWorker runs
        if data_type in ['NPY','MAT']:
            data_array, metadata = self.fsw.data_array, self.fsw.metadata_dict

            # Transpose the array if needed to ensure rows = samples and columns = channels
                # .shape returns (rows, columns)
                # if the number of rows is equal to the number of channels, transpose so that the rows = samples and columns = channels
            if data_array.shape[0] == metadata['nch']: 
                data_array = data_array.T
        
        # For NWB files, note the path to the ElectricalSeries name
        electrical_series_path = None
        if data_type == 'NWB':
            es_name = self.select_nwb_es(file_path)
            if es_name == False:
                self.fsw.update_filepath(file_path, False)
                return
            electrical_series_path = f'acquisition/{es_name}'
        
        # Make a dictionary where each variable name is a key with the variable as its value
        extractor_special_input_dict = dict(data_array=data_array, metadata=metadata, electrical_series_path=electrical_series_path)
        
        # Create extractor worker. Send the file path and any available data (see "extractor_input_kwargs").
        self.worker_object = ExtractorWorker(file_path, extractor_special_input_dict)

        # Connect ExtractorWorker signals to slots in this class
            # "data_signal" emits two objects (recording, time) to "extractor_finished_slot" which assigns them to class variables self.recording and self.time
            # "error_signal" sends a filepath and error message to "extractor_error_slot" which displays the error mesaage.
        self.worker_object.data_signal.connect(self.extractor_finished_slot)
        self.worker_object.error_signal.connect(self.extractor_error_slot)

        # Create a worker thread, add self.worker_object (ExtractorWorker) to it, and start the worker thread 
        self.start_qthread()
    
    @QtCore.pyqtSlot(object, object)
    def extractor_finished_slot(self, recording, time):
        """
        Load recording data into pipeline.
        """
        # Receive recording and time from ExtractorWorker
        self.recording = recording      # ExtractorWorker emits recording; received here by "extractor_finished_slot" and assigned to self.recording
        self.time = time                # ExtractorWorker also emits time; received here and assigned to self.time
        
        # Get a dictionary of metadata from the recording
        recording_metadata = dp.get_metadata_from_recording(self.recording)
        n_segments = self.recording.get_annotation("n_segments")

        ##### Update the metadata QLabel, then show the metadata section QFrame #####
        # First, update all the labels with the relevant text
        file_path = self.fsw.le.text()
        file_name = os.path.basename(file_path)
        data_type = dp.get_data_format(file_path)
        file_type, file_ext = dp.supported_formats[data_type]

        # Line 1: {file type} successfully loaded (in green)
        if data_type == "Neuralynx":
            file_loaded_str = f"<b>{file_type}</b> files successfully loaded."
        else:
            file_loaded_str = f"<b>{file_type}</b> file successfully loaded."
        self.file_loaded_label.setText(file_loaded_str)
        
        # Line 2: File name: {file name}
        if data_type == "Neuralynx":
            folder_name = os.path.dirname(file_path).split("/")[-1]
            file_name_str = f"<b>Folder name (all .ncs files loaded):</b> {folder_name}"
        else:
            file_name_str = f"<b>File name:</b> {file_name}"
        self.file_name_label.setText(file_name_str)

        # Line 3: Recording properties: number of channels, number of samples, and sampling rate
        display_properties = {
            "N segments": f"{n_segments} (concatenated into one)",
            "N channels (total)": recording_metadata["n_total_channels"],
            "N channels (LFP)": recording_metadata["n_lfp_channels"],
            "N samples": recording_metadata["n_samples"],
            "Sampling rate (Hz)": recording_metadata['fs']
        }
        recording_properties_str = ""
        for i, (label, val) in enumerate(display_properties.items()):
            if len(recording_properties_str) > 0:
                recording_properties_str += "\n" # Add a newline for all rows except the first

            # Skip showing the number of segments if the recording is monosegment
            if i == 0 and n_segments == 1:
                continue

            recording_properties_str += f"{f'{label}:'.ljust(25)}{val}" 
        
        # Introducing html tags (like <code>) ignores newline and whitespace characters, so we'll also add <pre> to respect whitespace
        # Important: <pre> needs to go inside <code> tags to avoid introducing large newline spaces
        self.metadata_display_label.setText("<code><pre>"+recording_properties_str+"</code>")

        # Line 4: Time properties: start time, end time, duration
        if self.time is not None:
            t_start = self.time[0]
            t_end = self.time[-1]
            display_properties = {
                "Start time": np.round(t_start, 3),
                "End time": np.round(t_end, 3),
                "Duration (s)": np.round(t_end - t_start, 3),
            }
            time_properties_str = ""
            for label, val in display_properties.items():
                if len(time_properties_str) > 0:
                    time_properties_str += "\n" # Add a newline for all rows except the first
                time_properties_str += f"{f'{label}:'.ljust(25)}{val}"
            self.time_metadata_label.setText("<code><pre>"+time_properties_str+"</code>")
        else: # time is None
            self.time_metadata_label.setText("<i>Timestamps not available. Toothy will return sample indices for each event.<i>")
        
        # Make the metadata frames visible
        self.metadata_frame.setVisible(True)
        self.extraction_error_frame.setVisible(False) # Hide the "error" section, since extraction was successful

        # Enable the "Map to probe(s)" button to match "status"
        self.probe_map_btn.setEnabled(True)
        self.probe_map_btn.setVisible(True)
        
        # After showing frames/buttons, resize the window
        self.resize_window_content_widget()
        pyfx.center_window_upward(self)
           
        # Update parameter widgets with the recording's metadata
        # self.params_widget.lfp_decimation_factor.set_fs(float(recording_metadata['fs']))       # Updates lfp_fs param widget "fs" variable with the Fs from the recording (for comparison with desired downsample Fs)
        # self.params_widget.time_range.set_duration(self.recording.get_duration())   # Updates the duration with the duration from the recording TODO2 check
        # self.params_widget.lfp_decimation_factor.set_fs(self.PARAMS["lfp_decimation_factor"])

        ## JUST CHANGED
        # self.processing_options_widget.update_factor(self.PARAMS["lfp_decimation_factor"])
        # self.processing_options_widget.update_input_fs(recording_metadata["fs"])
        ###
        
        # TODO also update the start/end times with the recording metadata
        # Make the next part of the workflow (choosing a processing option for downsampling) visible and send metadata to it to update the suggested action
    
    # def update_target_fs(self, payload):
    #     """Updates the params widget with the params picked in this section of the GUI"""
    #     self.lfp_decimation_factor = payload["factor"]
    #     self.params_widget.lfp_decimation_factor.update_param(self.lfp_decimation_factor)
    #     new_params, _ = self.params_widget.get_param_dict_from_gui()
    #     self.params_widget.update_gui_from_param_dict(new_params)

    @QtCore.pyqtSlot(str, str)
    def extractor_error_slot(self, file_path, error_msg):
        """
        Handle data extraction errors
        """
        gi.MsgboxError(error_msg, parent=self).exec()
        self.fsw.update_filepath(file_path, False)

        # Update metadata box
        self.metadata_frame.setVisible(False)           # Hide the metadata box if extraction failed
        self.extraction_error_frame.setVisible(True)    # Show the error box
        file_name = os.path.basename(file_path)
        data_type = dp.get_data_format(file_path)
        file_type, file_ext = dp.supported_formats[data_type]

        # Line 1: Error loading {file type} file. (in red)
        file_loaded_str = f"Error loading <b>{file_type}</b> file."
        self.file_issue_label.setText(file_loaded_str)
        
        # Line 2: File name: {file name}
        if data_type == "Neuralynx":
            folder_name = os.path.dirname(file_path).split("/")[-1]
            file_name_str = f"<b>Folder name (all .ncs files loaded):</b> {folder_name}"
        else:
            file_name_str = f"<b>File name:</b> {file_name}"
        self.file_name_label_copy.setText(file_name_str)

    #################################
    #####   Probe Assignment   ######
    ################################# 
    def start_probe_assignment(self):
        """
        Initiate probe mapping phase of data ingestion process.
        """
        # Get the number of LFP channels
        n_lfp_channels = dp.get_metadata_from_recording(self.recording)["n_lfp_channels"]

        # Initialize probe mapping box
        self.paw = ProbeAssignmentWidget(n_lfp_channels)        # Create QWidget for probe assignment
        self.probe_vbox.addWidget(self.paw)                     # Add widget to layout in the probe QFrame
        self.probe_frame.setVisible(True)                       # Make the parent QFrame visible

        # Connect custom "check_signal" signal from Probe Assignment Widget to this class' "check_probe_assignment_status" slot which checks for a valid probe assignment to enable the next step
        self.paw.check_signal.connect(self.check_probe_assignment_status) 
        
        # If there is a user-specified preferred probe, add it to the Probe Assignment Widget if it has an appropriate number of channels
        dflt_probe = h_io.read_probe_file(self.default_probe_file)
        if (dflt_probe is not None) and (dflt_probe.get_contact_count() <= n_lfp_channels):
            self.paw.add_probe_row(dflt_probe)

        # Show/hide relevant buttons
        self.back_to_fs_btn.setVisible(True)
        self.set_params_btn.setVisible(True)
        self.probe_map_btn.setVisible(False)                    # Hide the "Map LFP channels to probe(s)" button (which would otherwise be next to the bottom row buttons)
        
        # Resize the window after adding content
        self.resize_window_content_widget()
        pyfx.center_window_upward(self)

        # Disable data loading
        self.recording_selection_frame.setEnabled(False)
        
    def assemble_probe_group(self):
        """
        Return probeinterface ProbeGroup object, assembled from row elements added to the Probe Assignment Widget.
        Called in the processing pipeline to create a ProbeGroup to pass to the spikeinterface recording object.
        """
        PROBE_GROUP = prif.ProbeGroup()
        items = pyfx.layout_items(self.paw.probe_row_layout)
        for i,item in enumerate(items):
            prb  = item.probe
            rows = item.ROWS  # group of rows belonging to this probe
            # reorder assigned rows by device indices
            sorted_rows = [rows[dvi] for dvi in prb.device_channel_indices]
            prb.set_contact_ids(rows)
            # device_indices * nprobes + start_row = sorted_rows
            prb.set_device_channel_indices(sorted_rows)
            if i > 0:  # make sure probe boundaries do not overlap
                xmax = max(PROBE_GROUP.probes[-1].contact_positions[:,0])
                cur_xmin = min(prb.contact_positions[:,0])
                prb.contact_positions[:,0] += (xmax - cur_xmin + 1)
            PROBE_GROUP.add_probe(item.probe)
        return PROBE_GROUP

    def check_probe_assignment_status(self):
        """
        Check for valid probe assignment (i.e. if all LFP channels in the data are accounted for in the probe group). 

        If valid, enable the next step (selection of parameters for data processing). 
        """
        # Check the Probe Assignment Widget ("paw") exists. If it doesn't, the user hasn't reached this part of the workflow.
        paw_exists = self.paw is not None
        if paw_exists:
            # Check that the probe assignment is complete
            if self.paw.probe_assignment_complete:
                self.set_params_btn.setEnabled(True)

                # Resize the window to show the new elements
                # Note: Need to call this function twice for it to work properly here
                self.resize_window_content_widget()
                self.resize_window_content_widget()
                pyfx.center_window_upward(self)
            
            # If probe assignment is not complete, don't show the next step
            else:
                self.set_params_btn.setEnabled(False)
                # Note: do not use .resize_window() here, produces a small scroll area.
                # Resize the window to show the new elements
                # Note: Need to call this function twice for it to work properly here
                self.resize_window_content_widget()
                self.resize_window_content_widget()
                pyfx.center_window_upward(self)
    
    ##########################################
    ######   Set Processing Parameters   #####
    ##########################################

    def start_param_selection(self):
        """
        Triggered after clicking the "Set processing parameters" button 
        """
        # Initialize widget for setting parameters (includes heading, file selection, parameter adjustment widgets)        
        self.param_widget = ParameterSettingWidget(self.param_file_path)
        self.parameters_frame_layout.addWidget(self.param_widget)
        self.parameters_frame.setVisible(True)

        # Show/hide and enable/disable the relevant buttons
        self.back_to_fs_btn.setVisible(False)   # Hide the "Back to file selection" button
        self.set_params_btn.setVisible(False)   # Hide the "Set processing parameters" button
        self.back_to_pa_btn.setVisible(True)    # Show the "Back to probe assignment" button
        self.probe_frame.setEnabled(False)      # Disable the probe assignment frame
        self.set_output_dir_step_btn.setVisible(True) # Show the "Set output directory" button

        # TODO
        parameter_status = self.param_widget.parameter_setting_complete
        self.set_output_dir_step_btn.setEnabled(True) # TODO make this based on if the params are good to go
        self.settings_w.setVisible(True) # TODO REMOVE KNE
        
        self.resize_window_content_widget()     # Resize the widget after all the changes
        pyfx.center_window_upward(self)

    def update_param_warnings(self, n, ddict):
        """ 
        Check for any input parameter warnings
        """
        warnings_exist = n > 0
        # self.params_warning_btn.setChecked(warnings_exist)
        # self.params_warning_btn.setVisible(warnings_exist)

        # After checking for warnings, re-assess status to activate or inactivate the next step #TODO this should probably point to a different function
        self.check_probe_assignment_status()

    #########################################
    ######   Select Output Directory   ######
    #########################################

    def start_output_dir_selection(self):
        # Hide the button that initialized this step
        self.set_output_dir_step_btn.setVisible(False)

        # Show the "output directory" text box and initialize it with a "toothy (n)" folder
        self.output_dir_frame.setVisible(True) 
        recording_filepath = self.recording.get_annotation('recording_filepath')
        recording_dir = os.path.dirname(recording_filepath)
        processed_data_dir = str(Path(recording_dir, pyfx.unique_fname(recording_dir, "toothy")))
        self.save_le.setText(processed_data_dir) # Set the text of the "save QLineEdit" to the "toothy" output directory path
        # Note: "save_le" text is read and the actual directory is made (and/or overwritten) in the Data Worker

        # Show and enable the "Process data!" button
        # Note: doesn't need to be conditional on anything since previous steps should have checked for parameter compliance
        self.pipeline_btn.setVisible(True)
        self.pipeline_btn.setEnabled(True)

        # Resize the widget after adding content
        self.resize_window_content_widget()     

    def set_output_dir(self):
        """
        Select location for the "toothy" output folder.
        Note that the desired directory is created later; this function only sets a variable (self.save_le) with the desired path.
        """
        # Read the directory path specified in the "save_le" QLineEdit
        initial_selection_dir_path = os.path.dirname(self.save_le.text())

        # QFileDialog
        output_dir_path = h_io.select_directory(init_dir = initial_selection_dir_path, title = "Set output directory", parent = self)
        dir_name = os.path.basename(output_dir_path)

        # If the path already exists, check to see if there are items inside.
        if os.path.isdir(output_dir_path):
            try:
                dir_items = len(os.listdir(output_dir_path))
            except:
                dir_items = 0
            
            # If there are items inside, ask the user if they want to overwrite the folder.
            if dir_items > 0:
                suffix = '' if dir_items == 1 else 's'
                msg = f'Folder "{dir_name}" contains {dir_items} item{suffix}.'
                sub_msg = 'Overwrite existing directory?'

                # Create a QMessageBox dialog
                mb_dlg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, 'Confirm overwrite', msg, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                mb_dlg.setInformativeText(sub_msg)
                # If the user does not say "Yes" (i.e. they select "No" or close the window), do not actually select this directory
                if mb_dlg.exec() != QtWidgets.QMessageBox.Yes:
                    return
                # If the user says yes, continue on and update the QLineEdit 
                # Note that the existing folder is deleted later on, this function only sets a variable with the desired path
                        
        # If a folder is chosen, update the "save_le" QLineEdit
        # Otherwise (output_dir_path is "None"), do nothing, i.e. don't update the "save_le" QLineEdit
        if output_dir_path:
            self.save_le.setText(output_dir_path)

    #####################################
    ######   Processing Pipeline   ######
    #####################################

    def pipeline_worker(self):
        """
        Run processing pipeline.
        """
        # Create empty folder for processed data using the folder selected in the 
        # Note: When selecting an existing directory, the system will warn the user. However, the directory and files are not actually deleted, so we have to do that below.
        processed_data_dir_path = self.save_le.text()
        if os.path.isdir(processed_data_dir_path):
            shutil.rmtree(processed_data_dir_path)  # Delete existing directory
        if os.path.isfile(processed_data_dir_path):
            os.remove(processed_data_dir_path)  # Delete existing file
        os.makedirs(processed_data_dir_path)

        # Create probe group with all probe objects used in the recording
        self.PROBE_GROUP = self.assemble_probe_group()

        # TODO: assess if this is the right way for this to work. We should also save the params
        # Create a dictionary with updated analysis parameters from the GUI to pass to the DataWorker
        # param_dict = self.params_widget.DEFAULTS  # Create a parameter dictionary (with default Toothy values)
        # gui_param_dict = self.params_widget.get_param_dict_from_gui()[0]  # Get the parameters entered into the GUI
        
        
        param_dict = self.param_widget.main_widget.DEFAULTS  # Create a parameter dictionary (with default Toothy values)
        gui_param_dict = self.param_widget.main_widget.get_param_dict_from_gui()[0]

        param_dict.update(gui_param_dict) # Update the parameter dictionary with the GUI values

        # Create DataWorker which runs the processing pipeline (loading, downsampling, filtering, event detection)
        # Note: variables "recording" and "time" passed to DataWorker wer set in "extractor_finished_slot"
        self.worker_object = DataWorker(
            RECORDING = self.recording, 
            TIME = self.time, 
            PROBE_GROUP = self.PROBE_GROUP, 
            PARAMS = param_dict,
            OUTPUT_DIR = processed_data_dir_path
        )

        # Connect signals to slots
        self.worker_object.data_signal.connect(self.pipeline_finished_slot)
        self.worker_object.error_signal.connect(self.pipeline_error_slot)
        
        # Start the thread
        self.start_qthread()
    
    @QtCore.pyqtSlot()
    def pipeline_finished_slot(self):
        """ Worker successfully completed the processing pipeline """
        self.output_dir = str(self.save_le.text())
        msg = 'Data processing complete!<br><br>Load another recording?'
        res = gi.MsgboxSave(msg, parent=self).exec()
        if res == QtWidgets.QMessageBox.Yes:
            self.back_to_probe_assignment()
            self.back_to_file_selection()
            # TODO make it go all the way back to a reset window (i.e. clear the selected file entirely?)
        else:  # close window
            self.accept()
    
    @QtCore.pyqtSlot(str)
    def pipeline_error_slot(self, error_msg):
        """ Worker encountered an error in the processing pipeline """
        # TODO notes on why original solution resulted in several "QCoreApplication::exec: The event loop is already running" messages
        gi.MsgboxError(error_msg, parent=self).exec()
        # delete incomplete recording folder
        save_ddir = self.save_le.text()
        if os.path.isdir(save_ddir):
            folder_name = os.path.basename(save_ddir)
            filestring = ', '.join(map(lambda f: f'"{f}"', os.listdir(save_ddir)))
            print(f'"{folder_name}" contains the following file(s): {filestring}' + os.linesep)
            shutil.rmtree(save_ddir) # delete incomplete recording folder
            print(f'"{folder_name}" folder deleted!')

            # Original solution:
            # res = ''
            # while res.lower() not in ['y','n']:
            #     res = input('Delete folder? (y/n) --> ')
            # print('')
            # if res == 'y':
            #     shutil.rmtree(save_ddir) # delete incomplete recording folder
            #     print(f'"{folder_name}" folder deleted!')
    

    ############################
    #####   Back Buttons   #####
    ############################

    def back_to_file_selection(self):
        """ 
        During the probe mapping step, return to the data selection step.
        """
        # Hide:
        self.back_to_fs_btn.setVisible(False)   # "Back" button for probe mapping -> file selection
        self.set_params_btn.setVisible(False)   # "Set processing parameters" button
        self.set_params_btn.setEnabled(False)   
        self.pipeline_btn.setVisible(False)     # "Process data" button
        self.pipeline_btn.setEnabled(False)

        # Re-enable file selection and the button to begin the probe-mapping step
        self.recording_selection_frame.setEnabled(True)
        self.probe_map_btn.setVisible(True)
        self.probe_frame.setVisible(False) # Hide the probe-mapping frame

        # Delete the probe assignment widget
        self.probe_vbox.removeWidget(self.paw)
        self.paw.deleteLater()
        self.paw = None

        # Adjust the window size
        self.resize_window_scroll_widget() # Necessary for shrinking back the height
        self.resize_window_content_widget()

    def back_to_probe_assignment(self):
        """ 
        During the parameter setting step, return to the probe assignment step.
        """
        # Hide:
        self.back_to_pa_btn.setVisible(False)   # "Back" button for parameter seting -> probe assignment
        self.parameters_frame.setVisible(False) # Hide the parameter setting section
        self.pipeline_btn.setVisible(False)     # "Process data" button
        self.pipeline_btn.setEnabled(False)
        self.set_output_dir_step_btn.setVisible(False)
        self.set_output_dir_step_btn.setEnabled(False)
        self.output_dir_frame.setVisible(False)

        # Re-enable probe assignment and the button to begin the parameter-setting
        self.back_to_fs_btn.setVisible(True)    # "Back" button for probe assignment -> file selection
        self.probe_frame.setEnabled(True)
        self.set_params_btn.setVisible(True)    # "Set processing parameters" button

        # Delete the parameter setting widget
        self.parameters_frame_layout.removeWidget(self.param_widget)
        self.param_widget.deleteLater()
        self.param_widget = None

        # Adjust the window size
        self.resize_window_scroll_widget() # Necessary for shrinking back the height
        self.resize_window_content_widget()