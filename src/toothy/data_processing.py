"""
Toothy

Purpose: Raw data ingestion and pre-processing

Authors: Amanda Schott, Kathleen Esfahany

#TODO
- Check for TODOs
    - get_extractor
- Remove commented-out code
"""
# Import standard libraries
import os
from pathlib import Path
import re
import warnings

# Import third-party libraries
import numpy as np
import pandas as pd
import scipy.signal
from PyQt5 import QtWidgets, QtCore
import neo
import spikeinterface
import spikeinterface.extractors as extractors
from open_ephys.analysis import Session

# Import custom modules
from . import helpers_io as h_io

# TODO remove
# import h5py
# import json
# import time
# import warnings
# import pdb
# import pickle
# import math
# import scipy.io as so
# import probeinterface as prif
# import quantities as pq
# import pyfx
# import gui_items as gi

# Define dictionary of supported recording formats
supported_formats = {'NeuroNexus' : ['NeuroNexus', '.xdat.json'],
                     'OpenEphys'  : ['Open Ephys', '.oebin'],
                     'Neuralynx'  : ['NeuraLynx', '.ncs'],
                     'NWB'        : ['Neurodata Without Borders', '.nwb'],
                     'NPY'        : ['NumPy', '.npy'],
                     'MAT'        : ['MATLAB', '.mat']}

##############################################################################
##############################################################################
################                                              ################
################             IMPORT RAW RECORDINGS            ################
################                                              ################
##############################################################################
##############################################################################

##### Functions to validate raw data formats (NeuroNexus, OpenEphys, Neuralynx) which require specific files/directories to exist. #####
def validate_neuronexus(filepath):
    """
    Check whether filepath represents valid NeuroNexus metadata file, located in a directory with the necessary "_data.xdat" file.
    """
    # First, check that the file has the correct extension.
    # For NeuroNexus, this is ".xdat.json".
    if not filepath.endswith('.xdat.json'):
        return False
    
    # Check if there is at least one file with the correct extensions
    files = os.listdir(os.path.dirname(filepath)) # .dirname gets the full path to the parent directory of the filepath; .listdir lists all the files in tht directory
    metafiles = [f for f in files if f.endswith('.xdat.json')]
    datafiles = [f for f in files if f.endswith('_data.xdat')]
    status = len(metafiles)==1 and len(datafiles)==1
    return status

def validate_openephys(filepath):
    """
    Check whether filepath represents valid OpenEphys metadata file nested within the appropriate structure.
    """
    # Ensure the existence of a file called "structure.oebin" which indexes the binary data files
    if os.path.basename(filepath) != 'structure.oebin':
        return False
    # Get the "recording" folder in which the "structure.oebin" lives
    rec_folder = os.path.dirname(filepath)
    
    # Get the "experiment" folder in which the "recording" folder lives
    exp_folder = os.path.dirname(rec_folder)

    # Get the "node" folder in which the "experiment" folder lives
    node_folder = os.path.dirname(exp_folder)

    # Check for a "settings.xml" file in the "node" folder
    if 'settings.xml' not in os.listdir(node_folder): 
        return False
    
    # Check for a "continuous" directory in the "recording" folder
    status = os.path.isdir(Path(rec_folder, 'continuous'))
    return status

def validate_neuralynx(filepath):
    """
    Check whether filepath represents valid Neuralynx file (.ncs).
    """
    return filepath.endswith('.ncs')

def get_data_format(filepath):
    """
    Return data type for the recording at <filepath>. 
    
    Note that this function can handle invalid filepaths or non-supported filetypes, but these cases are unlikely to occur since the file selection window limits users to selecting valid files of the specified types.
    """
    try:
        if not os.path.exists(filepath):
            raise Exception(f'Filepath {filepath} does not exist.')
    except:
        raise Exception(f'Input {filepath} is invalid.')
    
    if validate_neuronexus(filepath):
        return 'NeuroNexus'
    elif validate_openephys(filepath):
        return 'OpenEphys'
    elif validate_neuralynx(filepath):
        return 'Neuralynx'
    elif filepath.endswith('.nwb'):
        return 'NWB'
    elif filepath.endswith('.npy'):
        return 'NPY'
    elif filepath.endswith('.mat'):
        return 'MAT'
    else:
        raise Exception(f'{filepath} is not a supported data file.')

##### Reading files into spikeinterface "extractor" objects #####
def get_extractor(filepath, data_format, **kwargs):
    """
    Return spikeinterface Extractor object (and a timestamps array, if available) for the given recording.

    From the spikeinterface paper: 
    "RecordingExtractor is an object representation of an extracellular recording and the associated probe configuration."

    Note for NeuroNexus, OpenEphys, and Neuralynx, we use stream_id 0 by default, but this may not work for all files.
    """
    # Unpack keyword arguments; .get() returns None if key not found
    # Note: 
        # Only NumPy/MATLAB array files will send a data_array or metadata_dict. They are then put into a spikeinterface object.
        # All the other recording types go through extraction later in this function.
    # TODO add a "time" kwarg. time can be included as a key in a matlab file. 
    # TODO otherwise, time can also be uploaded separately, after extraction, and added as recording.set_times(...)
    # TODO rename "kwargs" to match upstream function
    data_array = kwargs.get('data_array') 
    metadata   = kwargs.get('metadata', {})
    electrical_series_path = kwargs.get('electrical_series_path')
    
    ##### NumPy or MATLAB #####
    if data_format in ['NPY','MAT']:
        # Ensure the data is a NumPy array and the two required metadata elements (sampling rate and units) are present
        assert isinstance(data_array, np.ndarray)
        assert all([k in metadata for k in ['fs','units']])

        try:
            # spikeinterface.NumpyRecording is an fairly generic class to construct a SpikeInterface object from memory (rather than directly from a file) which accepts:
                # traces_list (our variable data_array), a list of arrays (multi-segment recording) or a single array (mono-segment recording)
                # sampling_frequency (our variable metadata['fs']), in Hz
                # t_starts (default None), time in seconds of the first sample of each segment
                # channel_ids (default None with linear channels assumed)
            # Note that it is not an actual extractor

            ##### Create the NumpyRecording object #####
            recording = spikeinterface.NumpyRecording(data_array, metadata['fs'])

            ##### Add "properties" to the recording (arrays with length equal to the # of channels) #####
                # "gain_to_uV" via ".set_channel_gains"
                # "offset_to_uV" via ".set_channel_offsets"
                # "channel_name" via ".set_property("channel_name", [...])"

            # Standardize voltages to microvolts (uV) by setting a gain and offset
            # Rough formula: voltage_in_uV = (raw_value * gain_to_uV) + offset_to_uV
            # Gain and offset of "recording" object start as "None"; we set them below.

            # Note: "gain" refers to factor to apply to the traces to convert them to uV
            voltage_units_to_gains = {"V": 1e6, "Volt": 1e6, "Volts": 1e6, "mV": 1e3, "uV": 1.0}
            gain_to_uV = voltage_units_to_gains[metadata['units']]
            
            # .set_channel_gains() takes a float (applied to all channels) or an array of floats (one per channel)
            # Since we want to apply the same gain to all channels, we can pass one value
            # Previous code: gain_to_uV = np.repeat(voltage_units_to_gains[metadata['units']], metadata['nch']) 
            recording.set_channel_gains(gain_to_uV)

            # Previous code: recording.set_channel_offsets(np.zeros(metadata['nch']))
            # Note: setting channel offsets is necessary or else the LFP appears flat in later stages.
            recording.set_channel_offsets(0)

            # Set the channel names to be the same as the channel IDs (which were assumed to be linear)
            channel_ids = recording.channel_ids
            recording.set_property("channel_name", channel_ids)
            
            ##### Set the time array to be None #####
            time = None
        except:
            raise Exception(f'Unable to extract {data_format} array.')
    
    ##### NeuroNexus #####
    elif data_format == 'NeuroNexus':
        try: 
            # TODO remove: NeuroNexusDebug
            ## Custom NeuroNexus extractor to extract only electrophysiology channels; leaves out auxiliary channels
            ## recording = _NeuroNexusRecordingExtractor(filepath, stream_id='0')

            #  Get an extractor with the full set of channels
            # Later, we will filter down to just electrode channels (exclude AUX, DIN, and DOUT channels).
                # - Electrode channels (start with "pri_")
                # - AUX (start with "aux_"; auxiliary analog inputs, e.g. photometry, etc.)
                # - DIN (start with "din_"; digital inputs, e.g. TTL pulses sent into the system)
                # - DOUT (start with "dout_"; digital outputs, e.g. trigger signals sent by the system)
            # To filter to just electrode channels we find the "channel names" starting with "pri_", then get the corresponding "channel IDs"

            recording = extractors.NeuroNexusRecordingExtractor(filepath)
        except:
            raise Exception('Unable to load NeuroNexus extractor.')
        
    ##### OpenEphys #####
    elif data_format == 'OpenEphys':
        try: # OpenEphys extractor
            exp_folder_path = os.path.dirname(os.path.dirname(filepath))
            experiment_name = [os.path.basename(exp_folder_path)] # list with name of experiment folder
            node_folder_path = os.path.dirname(exp_folder_path)
            recording = extractors.OpenEphysBinaryRecordingExtractor(folder_path=node_folder_path, stream_id='0', experiment_names=experiment_name)
        except:
            raise Exception('Unable to load OpenEphys extractor.')
        
    ##### Neuralynx #####
    elif data_format == 'Neuralynx':
        try: # Neuralynx extractor
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning) # Temporarily suppress warnings to avoid ("UserWarning: `exclude_filename` is deprecated and will be removed..." warning)
                folder_path = os.path.dirname(filepath) # Get the parent folder of the selected file
                exclude = [f for f in os.listdir(folder_path) if not os.path.isfile(Path(folder_path, f))] # Exclude any directories in the folder (such as the "processed_data" folder)
                recording = extractors.NeuralynxRecordingExtractor(folder_path=folder_path, stream_id='0', exclude_filename=exclude) # Note this line produces a bug about the "exclude_filename" argument, but fixing it to what the bug suggests ("exclude_filenames") breaks the line

        except Exception as e:
            print(e)
            raise Exception('Unable to load Neuralynx extractor.')
        
    ##### Neurodata Without Borders #####
    elif data_format == 'NWB':
        try: # NWB extractor
            recording = extractors.NwbRecordingExtractor(file_path=filepath, electrical_series_path=electrical_series_path)
            
            # If there is no "channel_name" property, set the channel IDs as the channel names
            if "channel_name" not in recording._properties.keys():
                channel_ids = recording.channel_ids
                recording.set_property("channel_name", channel_ids)
        except:
            raise Exception('Unable to load NWB extractor.')
    
    #### Any other data format ####
    else:
        raise Exception(f'Unsupported data format: {data_format}.')
    
    ##### Check if there are multiple segments (and concatenate them if so) ######
    n_segments = recording.get_num_segments()

    # spikeinterface has a function "concatenate_recordings([recording])" that provides a ConcatenateSegmentRecording object and strips recordings of their time data
    # We use the function to get the concatenated traces, but we create our own timestamps array
    # The spikeinterface concatenated timestamps are artificially constructed based on the # samples and an Fs value and are not accurate to the original timestamps values
    timestamps_concatenated = None
    try:
        if n_segments > 1:
            # Create timestamps using original timestamps 
            timestamps_concatenated = np.concatenate([recording.get_times(segment_index = i) for i in range(n_segments)])

            # Concatenate the recording
            combined_recording = spikeinterface.concatenate_recordings([recording])

            # Replace the "recording" variable with the concatenated object (be sure to do this *after* the timestamps creation step)
            recording = combined_recording

            # Set the times to the concatenated timestamps
            recording.set_times(timestamps_concatenated)
    except Exception as e:
        raise Exception('Error: Multi-segment recording. Unable to concatenate multiple segments.')

    ###### Get timestamps array (for non-array files) ######
    if data_format in ["NeuroNexus", "OpenEphys", "Neuralynx", "NWB"]:
        try:
            # Get the number of samples in the recording
            n_samples = recording.get_num_samples()

            # Get the timestamps from the recording
            time = recording.get_times() # time format is a NumPy ndarray

            # Check that the time array is 1D and the same length as the recording
            if time.ndim == 1 and len(time) == n_samples:
                pass
            else:
                time = None # Not sure this case ever happens, but basically time is only useful if it is aligned with the recording
        except:
            time = None # If issues are encountered when loading the times, set time to None to treat recording as if there is no time

    # Add the path and data format as annotations to the extractor object. Keys accessed later with "recording.get_annotation("recording_filepath")", etc.
    # TODO change ppath to filepath throughout codebase for this annotation (changed ppath to recording_filepath)
    recording.annotate(
        recording_filepath = filepath, 
        data_format = data_format,
        n_segments = n_segments
    ) 

    # Return the recording object and timestamps array
    return recording, time

##### Get metadata from the recording #####
def get_metadata_from_recording(recording):
    """
    Return recording parameters from extractor object

    Returns dictionary with keys:
        - fs: sampling frequency (from recording.get_sampling_frequency(); may not be fully accurate to the actual timestamps)
        - n_samples: number of samples
        - n_total_channels: total number of channels (of any type)
        - n_lfp_channels: number of LFP channels
        - lfp_channel_idxs: ordered list of LFP channel indices
            - Each value in this list corresponds to an index of a channel in channel_ids which contains LFP data
            - The order may not be meaningful and may be re-ordered/grouped by a probe configuration later
            - However, for Neuralynx files, as a starting point, the indices are re-ordered such that they correspond to increasing channel numbers (initially channels are incorrectly loaded in lexicographic order, i.e. 1, 11, 12, etc.)
            - For all other file types, no changes are made to the order
        - lfp_channel_ids: ordered list of LFP channel IDs (values indexed using lfp_channel_idxs)
    """    
    # Get the sampling frequency and the sample count
    # Note: the Fs may not be perfectly accurate to the actual timestamps
    metadata_dict = {
        'fs': recording.get_sampling_frequency(),
        'n_samples': recording.get_num_samples()
    }

    n_total_channels = recording.get_num_channels()
    
    # Get the channel names that correspond to LFP channels, then get a list of the corresponding channel IDs
    # Notes:
        # NumpyRecordings and NwbRecordingExtractor: recording.get_property("channel_name") is None unless explicitly set (as we do above)
        # NeuroNexusRecordingExtractor: "pri_" prefix indicates LFP channels (others are "aux_", "din_", etc.)
        # OpenEphysBinaryRecordingExtractor: "CH" prefix indicates LFP channels (others can be "A1_AUX", etc.)
        # NeuralynxRecordingExtractor: "CSC" prefix indicates LFP channels 
    channel_names = recording.get_property("channel_name")
    channel_names = np.array([str(name) for name in channel_names]) # Ensure all names are represented as strings. Keep in a NumPy array for multi-element indexing operations later.
    
    # Get the data format (NeuroNexus, OpenEphys, Neuralynx, NWB, NPY, or MAT) which was added to the recording as an annotation during extraction
    data_format = recording.get_annotation('data_format')

    # Regular expressions mapping each data format to how channel names look
    lfp_channel_regexp_dict = dict(
        NeuroNexus = 'pri_\d+', # NeuroNexus starts with pri_ (pri_1, pri_2, etc.)
        OpenEphys = 'CH\d+',    # OpenEphys starts with CH (CH1, CH2, etc.)
        Neuralynx = 'CSC\d+',   # Neuralynx starts with CSC (CSC1, CSC2, etc.)
        NPY = '\d+',            # NPY and MAT will be numbers (1, 2, 3, etc.)
        MAT = '\d+',
        NWB = '.*'              # NWB channels can be anything
    )
    reg_exp = lfp_channel_regexp_dict[data_format]

    # For each channel, see if there is a match to the regular expression; returns None if not
    ch_match = [*map(lambda n: re.match(reg_exp, n), channel_names)]

    # Get the channel indices in the full list of channels that correspond to *LFP* channels (i.e. where the channel name matches the regex)
    lfp_channel_idxs = np.nonzero(ch_match)[0]

    # For Neuralynx files, the channels are loaded in lexicographic order (1, 11, 12, ..., 2, 20, 21, etc.). 
    # We re-order the indices to reflect increasing channel numbers (1, 2, 3, etc.).
    # We use these indices to load the LFP traces later (i.e. recording.get_traces(channel_ids = [...])) so this reordering ensures that the LFP trace array order matches the intended channel order 
    if data_format == 'Neuralynx':
        lfp_channel_names = channel_names[lfp_channel_idxs] # Get the channel names for just the LFP channels (i.e. "CSC...")
        lfp_channel_number = [int(name.replace("CSC", "")) for name in lfp_channel_names] # Pull the numbers at the end of the channel name
        sort_idxs = np.argsort(lfp_channel_number) # Get indices that would sort them numerically
        lfp_channel_idxs = lfp_channel_idxs[sort_idxs] # Use the indices to sort the lfp_channel_idxs in order of increasing channel number

    # Get the LFP channel IDs
    lfp_channel_ids = recording.channel_ids[lfp_channel_idxs]

    # Add additional elements to the metadata dictionary
    metadata_dict.update({
        # 'total_ch': len(channel_names), # TODO cleanup
        # 'nch': len(ch_match), # this was buggy and would just give back the total anyway
        # 'ipri': ipri TODO remove
        "n_total_channels": n_total_channels,       # Number of channels in the recording file (both LFP and auxiliary)
        "n_lfp_channels": len(lfp_channel_idxs),    # Number of LFP channels in the recording
        "lfp_channel_idxs": lfp_channel_idxs,       # Indices of LFP channels in the recording file
        "lfp_channel_ids": lfp_channel_ids          # IDs of LFP channels in the recording file
    })

    return metadata_dict

##############################################################################
##############################################################################
################                                              ################
################              PROCESSING PIPELINE             ################
################                                              ################
##############################################################################
##############################################################################


# def get_chunkfunc(loading_window_s, recording_fs, recording_n_samples, target_fs = None, i_start= 0, i_end= -1):
#     """
#     Returns a function for stepping through recording in chunks of time (chunk duration given by "loading_window_s") 

#     Parameters:
#     - loading_window_s: time window (in seconds) to load
#     - recording_fs: the sampling rate of the recording
#     - recording_n_samples: the total number of samples in the recording
#     - target_fs: the target downsampled rate
#     - i_start, i_end: indices in the recording to bound the steps (i.e first step starts at i_start, last step ends at i_end)
#     """
#     # Get the (appx) number of indices in the loading window for the original recording
#     # Example: If the sampling rate is 1000Hz and the loading window is 600s, the rec_chunk_n_idxs will be 600k
#     rec_chunk_n_idxs = int(recording_fs * loading_window_s)

#     # Get the indices of the original recording to start/end the function
#     rec_i_start, rec_i_end = i_start, i_end

#     # If *not* downsampling, the chunk size (in indices) and start/end indices will be the same for the processed data
#     if target_fs is None:
#         print("Not downsampling inside chunkfunc")
#         proc_chunk_n_idxs, proc_i_start, proc_i_end = [int(x) for x in [rec_chunk_n_idxs, rec_i_start, rec_i_end]]
    
#     else:
#         # Get the (appx) number of indices in the loading window for the downsampled recording
#         # Example: If the target rate is 1000Hz and the loading window is 600s, the proc_chunk_n_idxs will be 600k (samples)
#         proc_chunk_n_idxs = int(target_fs * loading_window_s)

#         # Get the start and end indices of the recording for the downsampled recording
#         # Example: If the original recording is 2kHz and the target is 1kHz, the ds_factor is 2, and the start/end are each 1/2 of their original value
#         ds_factor = recording_fs/target_fs # value > 1
#         proc_i_start =  int(rec_i_start/ds_factor)
#         proc_i_end = int(rec_i_end/ds_factor)
#         print("PROC I END", proc_i_end)

#     # Minumum of 1s recording #CHECK -- might be minutes
#     DUR = max(recording_n_samples/recording_fs/60, 1)
    
#     # Create a function using the above parameters
#     def fx(chunk_n):
#         """
#         Return starting and ending indices for "chunk_n"-th recording chunk.
#         """
#         # Chunk indices corresponding to the original recording
#         rec_chunk_start_i = rec_i_start + (chunk_n * rec_chunk_n_idxs)
#         rec_chunk_end_i = rec_chunk_start_i + rec_chunk_n_idxs
#         rec_chunk_end_i = min(rec_chunk_end_i, rec_i_end) # Ensure the end of the chunk does not exceed the final index of the recording
#         # ii, jj = (chunk_n*rec_chunk_n_idxs+rec_i_start, chunk_n*rec_chunk_n_idxs+rec_chunk_n_idxs+rec_i_start)
#         # print("Huge check")
#         # print(ii == rec_chunk_start, jj == rec_chunk_end)

#         # Chunk indices corresponding to the processed/downsampled recording
#         proc_chunk_start_i = proc_i_start + (chunk_n * proc_chunk_n_idxs)
#         proc_chunk_end_i = proc_chunk_start_i + proc_chunk_n_idxs
#         proc_chunk_end_i = min(proc_chunk_end_i, proc_i_end) # Ensure the end of the chunk does not exceed the final index of the processed/downsampled recording length
#         # print("check 2")
#         # aa, bb = (chunk_n*proc_chunk_n_idxs+proc_i_start, chunk_n*proc_chunk_n_idxs+proc_chunk_n_idxs+proc_i_start)
#         # print(aa == proc_chunk_start_i)
#         # print(bb == proc_chunk_end_i)
#         # jj, bb = min(jj, rec_i_end), min(bb, proc_i_end)

#         # Generate a string to indicate progress
#         start_time_s = rec_chunk_start_i/recording_fs # Get the start of the chunk in seconds
#         start_time_m = start_time_s/60 # Convert to minutes
#         m_start_str = f'{int(start_time_m)}m'
#         end_time_s = rec_chunk_end_i/recording_fs # Get the end of the chunk in seconds
#         end_time_m = max(end_time_s/60, 1) # Convert to minutes (minimum value of 1)
#         m_end_str = f'{int(end_time_m)}m'
#         # Compose str. Note: ":^3" pads the string to have 3 characters, with non-padding characters centered
#         txt = f'Extracting {m_start_str:^3} - {m_end_str:^3} of {DUR:.0f}m ...'

#         return (rec_chunk_start_i, rec_chunk_end_i), (proc_chunk_start_i, proc_chunk_end_i), txt
    
#     # Return the function
#     return fx, (rec_chunk_n_idxs, proc_chunk_n_idxs)

# def load_recording_chunk(recording, ii, jj, ichan=None, **kwargs):
#     """ Read in data chunk from spikeinterface Recording object """

#     if type(ichan) == str: 
#         if ichan == "Neuralynx": # Put this first so that in casess where ichan is none, it doesnt throw an error after creating an array and then trying to compare it to a str
#             print("do we get here?")
#             channel_ids = recording.get_channel_ids()
#             print("channel ids", channel_ids)
#             channel_names_arr = recording.get_property("channel_name") # as numpy array of np.str_
#             channel_names = [str(name) for name in channel_names_arr] # convert to list of strings
#             print(channel_names)
#             paired = list(zip(channel_names, channel_ids))
#             # Sort by the numeric part of the channel name
#             paired_sorted = sorted(paired, key=lambda x: int(x[0][3:]))
#             # Unzip them back
#             sorted_channel_names, sorted_channel_ids = zip(*paired_sorted)
#             # Convert back to lists if needed
#             sorted_channel_names = list(sorted_channel_names)
#             sorted_channel_ids = list(sorted_channel_ids)
#             chids = sorted_channel_ids
#             # print("buggy channels")
#             # print(np.array(sorted_channel_ids)[[0, 7, 40, 58]], np.array(sorted_channel_names)[[0, 7, 40, 58]])
#             print("do we get here?")
#             # ichan = chids
#     # print(type(ichan))
#     # print(ichan)
#     # try:
#     #     print(ichan[0])
#     #     print(type(ichan[0]))
#     # except:
#     #     print("nah")
#     elif ichan is None:
#         print("ichan is None")
#         ichan = np.arange(recording.get_num_channels())
#     else:
#         chids = recording.get_channel_ids()[ichan]

#     # get_traces()
#         # Returns an array of shape (samples, channels) such that each column (i.e. x[:,i]) is a channel
#         # channel_ids follows the provided order; changing the order of the input arg will change the order of the columns of the returned array
#     arr = recording.get_traces(start_frame=ii, end_frame=jj, channel_ids=chids,
#                                return_scaled=True).T / 1000. #TODO look into if we shouldbe using "get_traces (return in UV instead)"
#     # TIMES!
#     # t_test = recording.get_times() # this doesnt make sense to put here in the chunking because it will always be the full length of the recording
#     # print("time?", t_test[:10], t_test.shape, sep = "\n")
#     return arr

# def load_chunk(data_format, ii, jj, ichan=None, **kwargs):
#     """
#     Return scaled, channel-mapped recording data for the given time chunk.
    
#     Parameters: #TODO look through this docstring, it was auto-generatd
#         data_format : str
#             One of 'NeuroNexus', 'OpenEphys', 'Neuralynx', 'NPY', 'MAT', or 'NWB'.
#         ii, jj : int
#             Starting and ending sample indices for the data chunk.
#         ichan : array-like, optional
#             Indices of channels to extract. If None, all channels are extracted.
#         **kwargs : dict
#             Additional arguments required for loading the data chunk,
#             depending on the data format:
#                 NeuroNexus : fid (file object), total_ch (int) or recording (Extractor)
#                 OpenEphys  : cont (OpenEphys continuous object)
#                 Neuralynx  : recording (Extractor)
#                 NPY, MAT   : recording (Extractor)
#                 NWB        : recording (Extractor)
#     """
#     # analyze recording between tstart and tend
#     if False:
#         pass # TODO remove NeuroNexusDebug
#     # if data_format == 'NeuroNexus':
#     #     assert 'fid' in kwargs, 'Missing required "fid" argument.'
#     #     assert 'total_ch' in kwargs or 'recording' in kwargs, \
#     #            'Must provide "recording" or "total_ch" argument.'
#     #     if 'total_ch' not in kwargs:
#     #         ch_names = kwargs['recording'].neo_reader.header['signal_channels']['name']
#     #         kwargs['total_ch'] = len(ch_names)
#     #     snip = load_neuronexus_chunk(ii=ii, jj=jj, ichan=ichan, **kwargs)
#     # elif data_format == 'OpenEphys':
#     #     assert 'cont' in kwargs, 'Missing required "cont" argument.'
#     #     snip = load_openephys_chunk(ii=ii, jj=jj, ichan=ichan, **kwargs)
#     # elif data_format in ['Neuralynx', 'NPY', 'MAT', 'NWB']:
#     elif data_format in ['Neuralynx', 'NPY', 'MAT', 'NWB', "NeuroNexus", "OpenEphys"]: #TODO fix

#         assert 'recording' in kwargs, 'Missing required "recording" argument.'
#         if data_format == 'Neuralynx':
#             ichan = "Neuralynx"
#         snip = load_recording_chunk(ii=ii, jj=jj, ichan=ichan, **kwargs)
#     return snip

# def resample_chunk(snip, nbins):
#     """ Scipy signal processing module """ # Used to downsample LFP in gui_data_ingestion. TODO change to decimate.
#     snip_dn = scipy.signal.resample(snip, nbins, axis=1)
#     return snip_dn


##############################################################################
##############################################################################
################                                              ################
################               SIGNAL PROCESSING              ################
################                                              ################
##############################################################################
##############################################################################

def butter_bandpass(low, high, lfp_fs, order=3):
    """
    Return filter coefficients for a given frequency cutoff and sampling rate.

    low: lower boundary (Hz)
    high: higher boundary (Hz)
    lfp_fs: sampling rate of the signal to be filtered
    order: filter order (default 3)
    """
    # Calculate the Nyquist frequency (half the sampling rate; the highest frequency that can be accurate represented at a given sampling rate)
    nyq = 0.5 * lfp_fs

    # Normalize the critical frequencies with respect to the Nyquist frequency
    low = low / nyq     
    high = high / nyq

    # Get a Butterworth, digital, second-order sections (sos) bandpass filter representation
    sos = scipy.signal.butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, low, high, lfp_fs):
    """
    Return bandpass-filtered data arrays.
    """
    sos = butter_bandpass(low, high, lfp_fs)
    filtered_signal = scipy.signal.sosfiltfilt(sos, data, padtype='odd')
    return filtered_signal


# Important, actually used in raw_data_pipeline
def detect_channel(event, i, data, lfp_time, DF=None, THRES=None, pprint=False, **PARAMS):
    """ Run ripple or DS detection for the given $data signal """
    assert event in ['swr', 'ds'], 'Event type must be "swr" or "ds".'
    if DF is None    : DF    = pd.DataFrame()
    if THRES is None : THRES = {}
    if event == 'swr':
        df, thres = h_io.get_swr_peaks(data, lfp_time, 
                                        pprint=pprint, **PARAMS)
    elif event == 'ds':
        # print(data.shape)
        # print(lfp_time.shape)
        df, thres = h_io.get_ds_peaks(data, lfp_time, 
                                       pprint=pprint, **PARAMS)
    df.set_index(np.repeat(i, len(df)), inplace=True)
    DF = pd.concat([DF, df], ignore_index=False)
    THRES[i] = thres
    return DF, THRES


##############################################################################
##############################################################################
################                                              ################
################                 MANUAL IMPORTS               ################
################                                              ################
##############################################################################
##############################################################################

# all of this seems vestigial and can be removed, except for the very bottom 

    
# Remove vestigial function; never called
# def get_idx_by_probe(probe):
#     """ Clean $probe input, return list of channel maps """
#     if probe.__class__ == prif.Probe:
#         idx_by_probe = [probe.device_channel_indices]
#     elif probe.__class__ == prif.ProbeGroup:
#         idx_by_probe = [prb.device_channel_indices for prb in probe.probes]
#     elif type(probe) in [list, np.ndarray]:
#         if type(probe) == list:
#             probe = np.array(probe)
#         if type(probe) == np.ndarray:
#             if probe.ndim == 1:
#                 idx_by_probe = [probe]
#             elif probe.ndim == 2:
#                 idx_by_probe = [x for x in probe]
#     return idx_by_probe


# Vestigial/never called
# def process_probe_data(_lfp, lfp_time, lfp_fs, PARAMS, pprint=True):
#     """
#     Filter LFPs and run event (SPW-R and DS) detection on each channel.

#     Parameters
#     ----------
#     _lfp : 2D array
#         LFP signals (channels x timepoints).
#     lfp_time : 1D array
#         Timepoints for LFP signals (seconds).
#     lfp_fs : float
#         Sampling rate for LFP signals (Hz).
#     PARAMS : dict
#         Dictionary of parameters for filtering and event detection.
#     pprint : bool, optional
#         Whether to print progress messages. The default is True.
#     """
    
#     # bandpass filter LFPs within different frequency bands
#     if pprint: print('Bandpass filtering signals ...')    
#     bp_dict = bp_filter_lfps(_lfp, lfp_fs, **PARAMS)
#     # get standard deviation (raw and normalized) for each filtered signal
#     std_dict = {k : np.std(v, axis=1) for k,v in bp_dict.items()}
#     std_dict.update({f'norm_{k}' : pyfx.Normalize(v) for k,v in std_dict.items()})
#     STD = pd.DataFrame(std_dict)
    
#     # run ripple detection on all channels
#     SWR_DF = pd.DataFrame()
#     SWR_THRES = {}
#     if pprint: print('Detecting ripples on each channel ...')
#     for ch in range(_lfp.shape[0]):
#         # sharp-wave ripples
#         swr_df, swr_thres = ephys.get_swr_peaks(bp_dict['swr'][ch], lfp_time, lfp_fs, 
#                                                 pprint=False, **PARAMS)
#         swr_df.set_index(np.repeat(ch, len(swr_df)), inplace=True)
#         SWR_DF = pd.concat([SWR_DF, swr_df], ignore_index=False)
#         SWR_THRES[ch] = swr_thres
    
#     # run DS detection on all channels
#     DS_DF = pd.DataFrame()
#     DS_THRES = {}
#     if pprint: print('Detecting dentate spikes on each channel ...')
#     for ch in range(_lfp.shape[0]):
#         # dentate spikes
#         ds_df, ds_thres = ephys.get_ds_peaks(bp_dict['ds'][ch], lfp_time, lfp_fs, 
#                                              pprint=False, **PARAMS)
#         ds_df.set_index(np.repeat(ch, len(ds_df)), inplace=True)
#         DS_DF = pd.concat([DS_DF, ds_df], ignore_index=False)
#         DS_THRES[ch] = ds_thres
#     THRESHOLDS = dict(SWR=SWR_THRES, DS=DS_THRES)
    
#     return bp_dict, STD, SWR_DF, DS_DF, THRESHOLDS

# Also never called, vestigial
# def process_all_probes(lfp_list, lfp_time, lfp_fs, PARAMS, save_ddir, pprint=True):
#     """
#     Process LFPs for each probe in dataset, save to new data folder
#     """
#     if type(lfp_list) == np.ndarray:
#         lfp_list = [lfp_list]
#     bp_dicts = {'raw':[], 'theta':[], 'slow_gamma':[], 'fast_gamma':[], 'swr':[], 'ds':[]}
#     std_dfs, swr_dfs, ds_dfs, thresholds, noise_trains = [], [], [], [], []
    
#     for i,_lfp in enumerate(lfp_list):
#         if pprint: print(f'\n#####   PROBE {i+1} / {len(lfp_list)}   #####\n')
#         bp_dict, STD, SWR_DF, DS_DF, THRESHOLDS = process_probe_data(_lfp, lfp_time, lfp_fs, 
#                                                                      PARAMS, pprint=pprint)
#         for k,l in bp_dicts.items(): l.append(bp_dict[k])
#         std_dfs.append(STD)
#         swr_dfs.append(SWR_DF)
#         ds_dfs.append(DS_DF)
#         thresholds.append(THRESHOLDS)
#         noise_trains.append(np.zeros(len(_lfp), dtype='int'))
#     ALL_STD = pd.concat(std_dfs, keys=range(len(std_dfs)), ignore_index=False)
#     ALL_SWR = pd.concat(swr_dfs, keys=range(len(swr_dfs)), ignore_index=False)
#     ALL_DS = pd.concat(ds_dfs, keys=range(len(ds_dfs)), ignore_index=False)
    
#     # save downsampled data
#     if pprint: print('Saving files ...')
#     if not os.path.isdir(save_ddir):
#         os.mkdir(save_ddir)
#     np.save(Path(save_ddir, 'lfp_time.npy'), lfp_time)
#     np.save(Path(save_ddir, 'lfp_fs.npy'), lfp_fs)
#     np.savez(Path(save_ddir, 'lfp_bp.npz'), **bp_dicts)
    
#     # save bandpass-filtered power in each channel (index)
#     ALL_STD.to_csv(Path(save_ddir, 'channel_bp_std'), index_label=False)
    
#     # save event quantifications and thresholds
#     ALL_SWR.to_csv(Path(save_ddir, 'ALL_SWR'), index_label=False)
#     ALL_DS.to_csv(Path(save_ddir, 'ALL_DS'), index_label=False)
#     np.save(Path(save_ddir, 'THRESHOLDS.npy'), thresholds)
#     # initialize noise channels
#     np.save(Path(save_ddir, 'noise_channels.npy'), noise_trains)
#     # save params and info file
#     with open(Path(save_ddir, 'params.pkl'), 'wb') as f:
#         pickle.dump(PARAMS, f)
    
#     if pprint: print('Done!' + os.linesep)

        
def validate_processed_ddir(dir_path):
    """
    Check if a directory contains the required Toothy-generated output files to proceed with the channel selection step.

    Required files:
    - probe_group
    - params.pkl
    - DATA.hdf5
    """
    try:
        files = os.listdir(dir_path)
        status = "probe_group" in files and "params.pkl" in files and "DATA.hdf5" in files
        return int(status)
    except:
        # File reading failed
        return 0
    
    
    # return 
    # if 'probe_group' not in files: 
    #     return 0
    # if 'params.pkl' not in files: 
    #     return 0
    # if 'DATA.hdf5' not in files:
    #     return 0
    # else: 
    #     return 1


def validate_classification_ddir(ddir, iprobe, ishank):
    """ Check whether directory contains required files for DS classification """
    try    : files = os.listdir(ddir)  
    except : return False
    if f'DS_DF_{iprobe}' in files:
        PROBE_DS_DF = pd.read_csv(Path(ddir, f'DS_DF_{iprobe}')).reset_index(drop=True)
        shanks = np.unique(PROBE_DS_DF['shank'].values)
        for ishk in shanks:
            DDF = PROBE_DS_DF[PROBE_DS_DF['shank']==ishk].reset_index(drop=True)
            DDF.to_csv(Path(ddir, f'DS_DF_probe{iprobe}-shank{ishk}'), index_label=False)
        os.remove(Path(ddir, f'DS_DF_{iprobe}'))
    try:
        assert f'DS_DF_probe{iprobe}-shank{ishank}' in files
        assert len(h_io.load_ds_dataset(ddir, iprobe, ishank)) > 1
        llist = h_io.load_event_channels(ddir, iprobe, ishank)
    except:
        return False
    return bool(len(llist)==3 and llist != [None,None,None])
        