"""
Raw data ingestion and pre-processing

Authors: Amanda Schott, Kathleen Esfahany
"""
# import importlib.metadata
# __version__ = importlib.metadata.version("spikeinterface")
import os
from pathlib import Path
import re
import json
import neo
import h5py
import time
import pickle
import scipy.io as so
import scipy.signal
import numpy as np
import pandas as pd
import math
from PyQt5 import QtWidgets, QtCore
from open_ephys.analysis import Session
import spikeinterface
import spikeinterface.extractors as extractors
import probeinterface as prif
import quantities as pq
import warnings
import pdb
# custom modules
import pyfx
import ephys
import gui_items as gi

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
    Check whether filepath represents valid NeuroNexus metadata file.
    """
    # First, check that the file has the correct extension.
    # For NeuroNexus, this is ".xdat.json".
    if not filepath.endswith('.xdat.json'):
        return False
    
    # TODO: rewrite and explain this logic
    files = os.listdir(os.path.dirname(filepath))
    metafiles = [f for f in files if f.endswith('.xdat.json')]
    datafiles = [f for f in files if f.endswith('_data.xdat')]
    # TODO, add some kind of timestamps thing?
    x = len(metafiles)==1 and len(datafiles)==1
    return x

def validate_openephys(filepath):
    """ Check whether filepath represents valid OpenEphys metadata file """
    if os.path.basename(filepath) != 'structure.oebin':
        return False
    rec_folder = os.path.dirname(filepath)
    exp_folder = os.path.dirname(rec_folder)
    node_folder = os.path.dirname(exp_folder)
    if 'settings.xml' not in os.listdir(node_folder): return False
    x = os.path.isdir(Path(rec_folder, 'continuous'))
    return x

def validate_neuralynx(filepath):
    """Check whether filepath represents valid Neuralynx file (.ncs)."""
    return filepath.endswith('.ncs')
    # TODO maybe show a list of files in directory? NLX file selection is weird right now

def get_data_format(ppath):
    """ Return data type for the recording at $ppath. Supported formats include:
            Directories: NeuroNexus, OpenEphys, and Neuralynx
            Files: Neurodata Without Borders (NWB), NPY, and MAT """
    try:
        res = os.path.exists(ppath)
        if not res: raise Exception(f'Filepath {ppath} does not exist.')
    except:
        raise Exception(f'Input {ppath} is invalid.')
    
    if validate_neuronexus(ppath):
        return 'NeuroNexus'
    elif validate_openephys(ppath):
        return 'OpenEphys'
    elif validate_neuralynx(ppath):
        return 'Neuralynx'
    elif ppath.endswith('.nwb'):
        return 'NWB'
    elif ppath.endswith('.npy'):
        return 'NPY'
    elif ppath.endswith('.mat'):
        return 'MAT'
    else:
        raise Exception(f'{ppath} is not a supported data file.')

def get_nwb_eseries(ppath):
    """ Return list of valid ElectricalSeries datasets in NWB file """
    good_series = []
    # get ElectricalSeries names
    with h5py.File(ppath, 'r') as f:
        acquisition_group = f.get('acquisition')
        if acquisition_group:
            series_names = list(acquisition_group.keys())
        else: series_names = []
        for name in series_names:
            es = f["acquisition"][name]
            if "electrodes" in es:
                good_series.append(name)
    return good_series

def get_extractor(ppath, data_format, **kwargs):
    """ Return spikeinterface extractor object for the given recording """

    # Unpack keyword arguments; .get() returns None if key not found
    data_array = kwargs.get('data_array') # This is sent in for mat/npy only (then put into a spikeinterface object); all the other recordings go through extraction
    metadata   = kwargs.get('metadata', {})
    electrical_series_path = kwargs.get('electrical_series_path')
    
    ##### NumPy or MATLAB #####
    if data_format in ['NPY','MAT']:
        assert isinstance(data_array, np.ndarray) 
        assert all([k in metadata for k in ['fs','units']])

        try:
            # spikeinterface.NumpyRecording is an fairly generic class to construct a SpikeInterface object from memory (rather than directly from a file) which accepts:
                # traces_list (our variable data_array), a list of arrays (multi-segment recording) or a single array (mono-segment recording)
                # sampling_frequency (our variable metadata['fs']), in Hz
                # t_starts (default None), time in seconds of the first sample of each segment
                # channel_ids (default None with linear channels assumed)
            # Note that it is not an actual extractor
            recording = spikeinterface.NumpyRecording(data_array, metadata['fs'])

            # Standardize voltages to microvolts (uV)
            voltage_units_to_gains = {"V": 1e6, "Volt": 1e6, "Volts": 1e6, "mV": 1e3, "uV": 1.0} # TODO look into why there are three different ways to say volts

            # Commented out 9/1/25 -- seems like we can just pass one float
            gain_to_uV = np.repeat(voltage_units_to_gains[metadata['units']], metadata['nch']) 

            # .set_channel_gains() takes a float (applied to all channels) or an array of floats (one per channel)
            # gain_to_uV = voltage_units_to_gains[metadata['units']]
            recording.set_channel_gains(gain_to_uV)

            # Commented out 9/1/25 -- not clear why we need to set offsets, we dont do it for any other recordings, and we never "get_channel_offsets"
            recording.set_channel_offsets(np.zeros(metadata['nch']))
            time = None

        except:
            raise Exception(f'Unable to extract {data_format} array.')
    
    ##### NeuroNexus #####
    elif data_format == 'NeuroNexus':
        try: # custom NeuroNexus extractor for ephys channels only
            recording = _NeuroNexusRecordingExtractor(ppath, stream_id='0')
            n_samples = recording.get_num_samples()

            # Get time array
            try:
                time = recording.get_times() # NumPy ndarray
                if time.ndim == 1 and len(time) == n_samples:
                    pass
                else:
                    time = None # Not sure this case ever happens, but basically time is only useful if it is aligned with the recording
            except:
                time = None # If issues are encountered when loading the times, set time to None to treat recording as if there is no time
        except:
            raise Exception('Unable to load NeuroNexus extractor.')
        
    ##### OpenEphys #####
    elif data_format == 'OpenEphys':
        try: # OpenEphys extractor
            exp_folder = os.path.dirname(os.path.dirname(ppath))
            folder_path = os.path.dirname(exp_folder)
            experiment_names = [os.path.basename(exp_folder)]
            recording = extractors.OpenEphysBinaryRecordingExtractor(folder_path=folder_path, stream_id='0', experiment_names=experiment_names)
            n_samples = recording.get_num_samples()
            # Get time array
            try:
                time = recording.get_times() # NumPy ndarray
                if time.ndim == 1 and len(time) == n_samples:
                    pass
                else:
                    time = None # Not sure this case ever happens, but basically time is only useful if it is aligned with the recording
            except:
                time = None # If issues are encountered when loading the times, set time to None to treat recording as if there is no time
        except:
            raise Exception('Unable to load OpenEphys extractor.')
        
    ##### Neuralynx #####
    elif data_format == 'Neuralynx':
        try: # Neuralynx extractor
            folder_path = os.path.dirname(ppath) # Get the parent folder of the selected file
            exclude = [f for f in os.listdir(folder_path) if not os.path.isfile(Path(folder_path, f))] # Exclude any directories in the folder

            recording = extractors.NeuralynxRecordingExtractor(folder_path=folder_path, stream_id='0', exclude_filename=exclude) # Note this line produces a bug about the "exclude_filename" argument, but fixing it to what the bug suggests ("exclude_filenames") breaks the line
            n_samples = recording.get_num_samples()

            # Get time array
            try:
                time = recording.get_times() # NumPy ndarray
                if time.ndim == 1 and len(time) == n_samples:
                    pass
                else:
                    time = None # Not sure this case ever happens, but basically time is only useful if it is aligned with the recording
            except:
                time = None # If issues are encountered when loading the times, set time to None to treat recording as if there is no time
        except:
            raise Exception('Unable to load Neuralynx extractor.')
        
    ##### Neurodata Without Borders #####
    elif data_format == 'NWB':
        try: # NWB extractor
            recording = extractors.NwbRecordingExtractor(file_path=ppath, electrical_series_path=electrical_series_path)
            n_samples = recording.get_num_samples()
        
        except:
            eseries = get_nwb_eseries(ppath)
            if len(eseries) == 0:
                raise Exception('NWB file contains no ElectricalSeries with valid electrodes.')
            es_name = eseries[0]; es_path = f'acquisition/{es_name}'
            try:
                recording = extractors.NwbRecordingExtractor(file_path=ppath, electrical_series_path=es_path)
                n_samples = recording.get_num_samples()
                
            except:
                raise Exception('Unable to load NWB extractor.')
        
        # Get time array
        try:
            time = recording.get_times() # NumPy ndarray
            if time.ndim == 1 and len(time) == n_samples:
                pass
            else:
                time = None # Not sure this case ever happens, but basically time is only useful if it is aligned with the recording
        except:
            time = None # If issues are encountered when loading the times, set time to None to treat recording as if there is no time

    else:
        raise Exception(f'Unsupported data format {data_format}.')
    
    # Add the path and data format as annotations to the extractor object
    recording.annotate(ppath=ppath, data_format=data_format) # accessed later with keys "get_annotation("ppath")" etc

    return recording, time


class _NeuroNexusRecordingExtractor(extractors.NeuroNexusRecordingExtractor):
    """ Separate NeuroNexus electrode channels from AUX, DIN, and DOUT streams """
    
    @classmethod
    def get_neo_io_reader(cls, raw_class: str, **neo_kwargs):
        """ Adjust IO header before returning IO object """
        neo_reader = neo.rawio.NeuroNexusRawIO(**neo_kwargs)
        neo_reader.parse_header()
        neo_reader = cls.fix_header(neo_reader)
        return neo_reader
    
    @classmethod
    def fix_header(cls, neo_reader):
        """ Assign different stream IDs to ephys (0) and auxiliary (1) channels """
        signal_channels = neo_reader.header['signal_channels']
        chtypes = [x.split('_')[0] for x in signal_channels['name']]
        ch_stream_ids = np.array([str(int(x!='pri')) for x in chtypes])
        signal_channels['stream_id'] = ch_stream_ids
        
        # update header with all available signal streams
        stream_dict = {'0':'ELEC', '1':'ADC'}
        stream_ids = list(filter(lambda k: k in ch_stream_ids, stream_dict))
        signal_streams = [(stream_dict[k], k, '0') for k in stream_ids]
        _signal_stream_dtype = neo_reader.header['signal_streams'].dtype
        neo_reader.header['signal_streams'] = np.array(signal_streams, dtype=_signal_stream_dtype)
        
        # map stream IDs to signal_channel indices
        buf_slice = {sid:np.flatnonzero(ch_stream_ids==sid) for sid in stream_ids}
        neo_reader._stream_buffer_slice = buf_slice
        # blocks > segments > signals/spikes/events > list of stream dicts
        neo_reader._generate_minimal_annotations()
        return neo_reader
    
def get_openephys_session(ddir):
    """ Return top-level Session object of OpenEphys recording directory $ddir """
    session = None
    child_dir = str(ddir)
    while True:
        parent_dir = os.path.dirname(child_dir)
        if os.path.samefile(parent_dir, child_dir):
            break
        if 'settings.xml' in os.listdir(parent_dir):
            session_ddir = os.path.dirname(parent_dir)
            try:
                session = Session(session_ddir) # top-level folder 
            except OSError:
                session = Session(parent_dir)   # recording node folder
            break
        else:
            child_dir = str(parent_dir)
    return session
    
def oeNodes(session, ddir):
    """ Return Open Ephys nodes from parent $session to child recording """
    # session is first node in path
    objs = {'session' : session}
    
    def isPar(par, ddir):
        return os.path.commonpath([par]) == os.path.commonpath([par, ddir])
    # find recording node in path
    if hasattr(session, 'recordnodes'):
        for node in session.recordnodes:
            if isPar(node.directory, ddir):
                objs['node'] = node
                break
        recs = node.recordings
    else:
        recs = session.recordings
    # find recording folder with raw data files
    for recording in recs:
        if os.path.samefile(recording.directory, ddir):
            objs['recording'] = recording
            break
    objs['continuous'] = recording.continuous
    return objs

def get_meta_from_recording(recording):
    """
    Return key experiment parameters from Extractor object
    """

    metadata_dict = {
        'fs': recording.get_sampling_frequency(),
        'nsamples': recording.get_num_samples()
    }

    if recording.__class__.__name__ in ['NumpyRecording', 'NwbRecordingExtractor']:
        ch_names = recording.get_channel_ids().astype('str')
    else:
        ch_names = recording.neo_reader.header['signal_channels']['name']
    
    # Get the data format (NeuroNexus, OpenEphys, Neuralynx, NWB, NPY, or MAT) which was added to the recording as an annotation during extraction
    data_format = recording.get_annotation('data_format')

    # Regular expressions mapping each data format to how channel names look
    ddict = dict(
        NeuroNexus = 'pri_\d+', # NeuroNexus starts with pri_ (pri_1, pri_2, etc.)
        OpenEphys = 'C\d+',     # OpenEphys starts with C (C1, C2, etc.)
        Neuralynx = 'CSC\d+',   # Neuralynx starts with CSC (CSC1, CSC2, etc.)
        NPY = '\d+',            # NPY and MAT will be numbers (1, 2, 3, etc.)
        MAT = '\d+',
        NWB = '.*'              # NWB channels can be anything
    )
    reg_exp = ddict[data_format]

    # For each channel, see if there is a match to the regular expression; returns None if not
    ch_match = [*map(lambda n: re.match(reg_exp, n), ch_names)]

    # Get the indices where channels match the desired format "ipri"
    # Now ipri isn't used anywhere, only shows up in vestigial code
    # ipri = np.nonzero(ch_match)[0]

    # if data_format == 'Neuralynx':  # order 1,2,3... instead of 1,10,11...  # previously ipri waas used, but now it is only used in vestigial code... 
        # ipri = ipri[np.argsort([int(n.replace('CSC','')) for n in ch_names[ipri]])]
    # print("ipri2: ", ipri)

    # Add more elements to the metadata dictionary
        # "total_ch" is used for 
    metadata_dict.update({
        'total_ch': len(ch_names), 
        'nch': len(ch_match), 
        # 'ipri': ipri TODO remove
    })

    return metadata_dict

def get_raw_source_kwargs(recording):
    """ Return core IO object for importing raw data """
    ddict = {'fid':None, 'cont':None, 'recording':None}
    data_format = recording.get_annotation('data_format')
    raw_ddir = os.path.dirname(recording.get_annotation('ppath'))
    if data_format == 'NeuroNexus':
        # read raw data from "_data.xdat" file
        fname = [f for f in os.listdir(raw_ddir) if f.endswith('_data.xdat')][0]
        fid = open(Path(raw_ddir, fname), 'rb')
        ddict['fid'] = fid
    elif data_format == 'OpenEphys':
        # read raw data from "continuous.dat" file via OpenEphys data object
        session = get_openephys_session(raw_ddir)
        OE = oeNodes(session, raw_ddir)
        cont = OE['recording'].continuous[0]
        ddict['cont'] = cont
    elif data_format in ['Neuralynx', 'NPY', 'MAT', 'NWB']:
        # read raw data directly from extractor
        ddict['recording'] = recording
    return ddict

# Doesn't seem to be called anywhere; vestigial
# def choose_array_key(keys, txt='Select LFP data key'):
#     """ Prompt user to specify dataset key with raw LFP signals """
#     pyfx.qapp()
#     lbl = QtWidgets.QLabel(txt)
#     lbl.setAlignment(QtCore.Qt.AlignCenter)
#     qlist = QtWidgets.QListWidget()
#     qlist.addItems(keys)
#     qlist.setFocusPolicy(QtCore.Qt.NoFocus)
#     qlist.setStyleSheet('QListWidget {'
#                         'border : 4px solid lightgray;'
#                         'border-style : double;'
#                         'selection-color : white;}'
#                         'QListWidget::item {'
#                         'border : none;'
#                         'border-bottom : 2px solid lightgray;'
#                         'background-color : white;'
#                         'padding : 4px;}'
#                         'QListWidget::item:selected {'
#                         'background-color : blue;}')
#     go_btn = QtWidgets.QPushButton('Continue')
#     go_btn.setEnabled(False)
#     dlg = QtWidgets.QDialog()
#     dlg.setStyleSheet('QWidget {font-size : 15pt;}')
#     lay = QtWidgets.QVBoxLayout(dlg)
#     lay.addWidget(lbl)
#     lay.addWidget(qlist)
#     lay.addWidget(go_btn)
#     qlist.itemSelectionChanged.connect(lambda: go_btn.setEnabled(len(qlist.selectedItems())==1))
#     go_btn.clicked.connect(dlg.accept)
#     res = dlg.exec()
#     if res : return qlist.selectedItems()[0].text()
#     else   : return None

# Moved to raw_Data_pipeline since thats the only place it was used...
# def read_data_from_dict(ddict):
#     """ Parse imported data dictionary for raw LFP array (required) and
#         sampling rate/data SI units (optional) """
#     # find the real data
#     data_dict = {}
#     meta = {'fs':None, 'units':None}
#     fs_keys = ['fs', 'sr', 'sampling_rate', 'sample_rate', 'sampling_freq', 
#                'sample_freq', 'sampling_frequency', 'sample_frequency']
#     unit_keys = ['unit', 'units']
#     for k,v in ddict.items():
#         if k.startswith('__'): continue
#         if k.lower() in fs_keys:
#             meta['fs'] = float(v)
#         elif k.lower() in unit_keys:
#             meta['units'] = str(v)
#         else:
#             if not hasattr(v, '__iter__'): continue
#             if len(v) == 0: continue
#             if np.array(v).ndim != 2: continue
#             data_dict[k] = v
#     return data_dict, meta

# Apparently also not used
# def read_array_file(fpath, raise_exception=False):
#     """ Load raw data from .npy or .mat file """
#     if not os.path.exists(fpath):
#         if raise_exception:
#             raise Exception('Data file does not exist')
#         return None, None
    
#     # load data according to file extension
#     ext = os.path.splitext(fpath)[-1]
#     exception_msg = 'Invalid data file.'
#     meta = {'fs':None, 'units':None}
#     try:
#         if ext == '.npy':
#             npyfile = np.load(fpath, allow_pickle=True)
#             if isinstance(npyfile, np.ndarray):
#                 data_dict = {}
#                 if npyfile.ndim == 2:
#                     data_dict['data'] = np.array(npyfile)
#                 else:  # data array not 2-dimensional
#                     exception_msg = 'NumPy data array must be 2-dimensional.'
#                     raise Exception(exception_msg)
#             elif isinstance(npyfile, dict):
#                 data_dict, meta = read_data_from_dict(npyfile)
#                 if len(data_dict) == 0:
#                     exception_msg = 'NumPy file must contain a 2-dimensional data array.'
#                     raise Exception(exception_msg)
#             else:
#                 exception_msg = 'NumPy file must contain data array or dictionary.'
#                 raise Exception(exception_msg)
#         elif ext == '.mat':
#             matfile = so.loadmat(fpath, squeeze_me=True)
#             data_dict, meta = read_data_from_dict(matfile)
#             if len(data_dict) == 0:
#                 exception_msg = 'MAT file must contain a 2-dimensional data array.'
#                 raise Exception(exception_msg)
#         else:
#             exception_msg = 'Data file must have ".npy" or ".mat" extension.'
#     except:
#         data_dict = None
#         if raise_exception:
#             raise Exception(exception_msg)
#     return data_dict, meta


##############################################################################
##############################################################################
################                                              ################
################              PROCESSING PIPELINE             ################
################                                              ################
##############################################################################
##############################################################################

def get_rec_bounds2(NSAMPLES, FS, tstart=0, tend=-1):
    """ Convert start/end timepoints to sample indices """
    # TODO change this, currently hinges on recording starting at 0... 
    # tstart must be between 0 and 1s before last timepoint
    istart = int(round(min(max(tstart*FS,0), NSAMPLES-FS)))
    # tend must be between tstart+1s and last timepoint
    if tend == -1: iend = NSAMPLES
    else: iend = int(round(min(max(tend*FS, istart+FS), NSAMPLES)))
    return istart, iend

def get_rec_bounds(NSAMPLES, FS, tstart=0, tend=-1):
    """ Convert start/end timepoints to sample indices """
    # TODO change this, currently hinges on recording starting at 0... 
    # tstart must be between 0 and 1s before last timepoint
    istart = int(round(min(max(tstart*FS,0), NSAMPLES-FS)))
    # tend must be between tstart+1s and last timepoint
    if tend == -1: 
        iend = NSAMPLES
    else: 
        iend = int(round(min(max(tend*FS, istart+FS), NSAMPLES)))
    return istart, iend

def get_chunkfunc(loading_window_s, recording_fs, recording_n_samples, target_fs = None, i_start= 0, i_end= -1):
    """
    Returns a function for stepping through recording in chunks of time (chunk duration given by "loading_window_s") 

    Parameters:
    - loading_window_s: time window (in seconds) to load
    - recording_fs: the sampling rate of the recording
    - recording_n_samples: the total number of samples in the recording
    - target_fs: the target downsampled rate
    - i_start, i_end: indices in the recording to bound the steps (i.e first step starts at i_start, last step ends at i_end)
    """
    # Get the (appx) number of indices in the loading window for the original recording
    # Example: If the sampling rate is 1000Hz and the loading window is 600s, the rec_chunk_n_idxs will be 600k
    rec_chunk_n_idxs = int(recording_fs * loading_window_s)

    # Get the indices of the original recording to start/end the function
    rec_i_start, rec_i_end = i_start, i_end

    # If *not* downsampling, the chunk size (in indices) and start/end indices will be the same for the processed data
    if target_fs is None:
        proc_chunk_n_idxs, proc_i_start, proc_i_end = [int(x) for x in [rec_chunk_n_idxs, rec_i_start, rec_i_end]]
    
    else:
        # Get the (appx) number of indices in the loading window for the downsampled recording
        # Example: If the target rate is 1000Hz and the loading window is 600s, the proc_chunk_n_idxs will be 600k (samples)
        proc_chunk_n_idxs = int(target_fs * loading_window_s)

        # Get the start and end indices of the recording for the downsampled recording
        # Example: If the original recording is 2kHz and the target is 1kHz, the ds_factor is 2, and the start/end are each 1/2 of their original value
        ds_factor = recording_fs/target_fs # value > 1
        proc_i_start =  int(rec_i_start/ds_factor)
        proc_i_end = int(rec_i_end/ds_factor)

    # Minumum of 1s recording #CHECK -- might be minutes
    DUR = max(recording_n_samples/recording_fs/60, 1)
    
    # Create a function using the above parameters
    def fx(chunk_n):
        """
        Return starting and ending indices for "chunk_n"-th recording chunk.
        """
        # Chunk indices corresponding to the original recording
        rec_chunk_start_i = rec_i_start + (chunk_n * rec_chunk_n_idxs)
        rec_chunk_end_i = rec_chunk_start_i + rec_chunk_n_idxs
        rec_chunk_end_i = min(rec_chunk_end_i, rec_i_end) # Ensure the end of the chunk does not exceed the final index of the recording

        # Chunk indices corresponding to the processed/downsampled recording
        proc_chunk_start_i = proc_i_start + (chunk_n * proc_chunk_n_idxs)
        proc_chunk_end_i = proc_chunk_start_i + proc_chunk_n_idxs
        proc_chunk_end_i = min(proc_chunk_end_i, proc_i_end) # Ensure the end of the chunk does not exceed the final index of the processed/downsampled recording length
      
        # Generate a string to indicate progress
        start_time_s = rec_chunk_start_i/recording_fs # Get the start of the chunk in seconds
        start_time_m = start_time_s/60 # Convert to minutes
        m_start_str = f'{int(start_time_m)}m'
        end_time_s = rec_chunk_end_i/recording_fs # Get the end of the chunk in seconds
        end_time_m = max(end_time_s/60, 1) # Convert to minutes (minimum value of 1)
        m_end_str = f'{int(end_time_m)}m'
        # Compose str. Note: ":^3" pads the string to have 3 characters, with non-padding characters centered
        txt = f'Extracting {m_start_str:^3} - {m_end_str:^3} of {DUR:.0f}m ...'

        return (rec_chunk_start_i, rec_chunk_end_i), (proc_chunk_start_i, proc_chunk_end_i), txt
    
    # Return the function
    return fx, (rec_chunk_n_idxs, proc_chunk_n_idxs)

def load_neuronexus_chunk(fid, ii, jj, total_ch, ichan=None, **kwargs):
    """ Read in data chunk from NeuroNexus "_data.xdat" binary file """
    if ichan is None:
        ichan = np.arange(total_ch)
    fid.seek(int(ii * total_ch * 4))
    data_amp = np.fromfile(fid, dtype='float32', count=int((jj-ii)*total_ch))
    arr = np.reshape(data_amp, (total_ch, -1), order='F')[ichan, :] / 1000.
    return arr

def load_openephys_chunk(cont, ii, jj, ichan=None, **kwargs):
    """ Read in data chunk from OpenEphys recording object """
    arr = cont.get_samples(ii, jj, selected_channels=ichan).T / 1000.
    return arr

def load_recording_chunk(recording, ii, jj, ichan=None, **kwargs):
    """ Read in data chunk from spikeinterface Recording object """

    if type(ichan) == str: 
        if ichan == "Neuralynx": # Put this first so that in casess where ichan is none, it doesnt throw an error after creating an array and then trying to compare it to a str
            channel_ids = recording.get_channel_ids()
            channel_names_arr = recording.get_property("channel_name") # as numpy array of np.str_
            channel_names = [str(name) for name in channel_names_arr] # convert to list of strings
            paired = list(zip(channel_names, channel_ids))
            # Sort by the numeric part of the channel name
            paired_sorted = sorted(paired, key=lambda x: int(x[0][3:]))
            # Unzip them back
            sorted_channel_names, sorted_channel_ids = zip(*paired_sorted)
            # Convert back to lists if needed
            sorted_channel_names = list(sorted_channel_names)
            sorted_channel_ids = list(sorted_channel_ids)
            chids = sorted_channel_ids
    elif ichan is None:
        ichan = np.arange(recording.get_num_channels())
    else:
        chids = recording.get_channel_ids()[ichan]
    arr = recording.get_traces(start_frame=ii, end_frame=jj, channel_ids=chids,
                               return_scaled=True).T / 1000.
    return arr
    
# Not called, vestigial
# def load_array_chunk(array, ii, jj, ichan=None, **kwargs):
#     """ Return data chunk from Numpy-like array """
#     assert array.ndim == 2, 'Data must be a 2-dimensional array (channels x samples).'
#     if ichan is None:
#         ichan = np.arange(array.shape[0])
#     scalef = kwargs.get('scalef', 1.)
#     arr = array[ichan, ii:jj] * scalef
#     return arr

def load_chunk(data_format, ii, jj, ichan=None, **kwargs):
    """
    Return scaled, channel-mapped recording data for the given time chunk.
    
    Parameters:
        data_format : str
            One of 'NeuroNexus', 'OpenEphys', 'Neuralynx', 'NPY', 'MAT', or 'NWB'.
        ii, jj : int
            Starting and ending sample indices for the data chunk.
        ichan : array-like, optional
            Indices of channels to extract. If None, all channels are extracted.
        **kwargs : dict
            Additional arguments required for loading the data chunk,
            depending on the data format:
                NeuroNexus : fid (file object), total_ch (int) or recording (Extractor)
                OpenEphys  : cont (OpenEphys continuous object)
                Neuralynx  : recording (Extractor)
                NPY, MAT   : recording (Extractor)
                NWB        : recording (Extractor)
    """
    # analyze recording between tstart and tend
    if data_format == 'NeuroNexus':
        assert 'fid' in kwargs, 'Missing required "fid" argument.'
        assert 'total_ch' in kwargs or 'recording' in kwargs, \
               'Must provide "recording" or "total_ch" argument.'
        if 'total_ch' not in kwargs:
            ch_names = kwargs['recording'].neo_reader.header['signal_channels']['name']
            kwargs['total_ch'] = len(ch_names)
        snip = load_neuronexus_chunk(ii=ii, jj=jj, ichan=ichan, **kwargs)
    elif data_format == 'OpenEphys':
        assert 'cont' in kwargs, 'Missing required "cont" argument.'
        snip = load_openephys_chunk(ii=ii, jj=jj, ichan=ichan, **kwargs)
    elif data_format in ['Neuralynx', 'NPY', 'MAT', 'NWB']:
        assert 'recording' in kwargs, 'Missing required "recording" argument.'
        if data_format == 'Neuralynx':
            ichan = "Neuralynx"
        snip = load_recording_chunk(ii=ii, jj=jj, ichan=ichan, **kwargs)
    return snip

# Doesn't appear to be called anywhere
# def get_interp_factors(FS, lfp_fs):
#     """ Get upsampling and downsampling factors for interpolation """
#     lcm = math.lcm(int(FS), int(lfp_fs)) # least common multiple
#     upf = int(lcm / FS)     # expand to LCM
#     dnf = int(lcm / lfp_fs) # reduce to target size
#     return lcm, upf, dnf
    
# def downsample_chunk(snip, upf, dnf):
#     """ Downsample data chunk by interpolation and binning """
#     NCH, NTS = snip.shape
#     up_snip = np.repeat(snip, upf, axis=1)
#     j = int(np.floor(up_snip.shape[1] / dnf) * dnf)
#     shape = (-1, dnf, NCH)
#     snip_dn = np.reshape(up_snip[:, 0:j].T, shape).mean(axis=1, dtype='float32').T
#     return snip_dn

def resample_chunk(snip, nbins):
    """ Scipy signal processing module """
    snip_dn = scipy.signal.resample(snip, nbins, axis=1)
    return snip_dn

# doesn't seem to be called anywhere; vestigial
# def load_chunks(recording, load_win=600, lfp_fs=1000, ichannels=[], 
#                 update_fx=lambda txt: print(txt), **kwargs):
#     """ Import and process raw recording in chunks of $load_win seconds """
#     data_format = recording.get_annotation('data_format')
#     META = get_meta_from_recording(recording)
#     FS, NSAMPLES, TOTAL_CH = META['fs'], META['nsamples'], META['total_ch']
#     ds_factor = int(FS/lfp_fs)  # get downsampling factor
    
#     # get function for stepping through chunks of recording data
#     tstart, tend = kwargs.get('tstart', 0), kwargs.get('tend', -1)
#     iistart, iiend = get_rec_bounds(NSAMPLES, FS, tstart, tend)
#     chunkfunc, (iichunk,_) = get_chunkfunc(load_win, FS, NSAMPLES, lfp_fs=lfp_fs,
#                                            tstart=tstart, tend=tend)
#     NSAMPLES_TRUNC = int(iiend - iistart)
#     NSAMPLES_DN_TRUNC = int(NSAMPLES_TRUNC / ds_factor)
    
#     # get list of channel indices and corresponding datasets
#     if len(ichannels) == 0 or len(ichannels) > TOTAL_CH:
#         ichannels = META['ipri']
#     ichannels = np.squeeze(ichannels)
#     if ichannels.ndim == 1:
#         ichannels = [ichannels]
#     datasets = kwargs.get('datasets', [])
#     if not isinstance(datasets, list):
#         datasets = [datasets]
#     if len(datasets) != len(ichannels):
#         datasets = [np.zeros((len(ichan), NSAMPLES_DN_TRUNC), 
#                              dtype='float32') for ichan in ichannels]
    
#     # get IO object (fid, cont, or recording)
#     KW = {**get_raw_source_kwargs(recording), 'total_ch':TOTAL_CH}
#     if KW['fid'] is not None and iistart > 0:
#         KW['fid'].seek(int(iistart * TOTAL_CH * 4))
    
#     count = 0
#     while True:
#         (ii,jj),(aa,bb),txt = chunkfunc(count)
#         update_fx(txt)
#         for ichan,dset in zip(ichannels, datasets):
#             snip = load_chunk(data_format, ii=ii, jj=jj, ichan=ichan, **KW)
#             snip_dn = downsample_chunk(snip, ds_factor)
#             dset[:, aa:bb] = snip_dn
#         if jj >= iiend:
#             break
#         count += 1
#     if KW['fid'] is not None:
#         KW['fid'].close()
#     return datasets

# Doesn't seem to be called anywhere; vestigial
# def bp_filter_lfps(lfp, lfp_fs, bp_dict=None, load_win=None, **kwargs):
#     """ Bandpass filter LFP signals within fixed frequency bands """
#     # set filter cutoffs
#     theta      = kwargs.get('theta',      [6,10])
#     slow_gamma = kwargs.get('slow_gamma', [25,55])
#     fast_gamma = kwargs.get('fast_gamma', [60,100])
#     swr_freq   = kwargs.get('swr_freq',   [120,180])
#     ds_freq    = kwargs.get('ds_freq',    [5,100])
    
#     if bp_dict is None:
#         bp_dict = {'raw' : lfp}
#         for k in ['theta', 'slow_gamma', 'fast_gamma', 'swr', 'ds']:
#             bp_dict[k] = np.zeros(lfp.shape, dtype='float32')
            
#     KW = {'lfp_fs':lfp_fs, 'axis':1}
    
#     if load_win is None:
#         start_time = time.time()
#         # collect filtered LFPs in data dictionary
#         bp_dict['theta'][:]      = pyfx.butter_bandpass_filter(lfp, *theta,      **KW)
#         bp_dict['slow_gamma'][:] = pyfx.butter_bandpass_filter(lfp, *slow_gamma, **KW)
#         bp_dict['fast_gamma'][:] = pyfx.butter_bandpass_filter(lfp, *fast_gamma, **KW)
#         bp_dict['swr'][:]        = pyfx.butter_bandpass_filter(lfp, *swr_freq,   **KW)
#         bp_dict['ds'][:]         = pyfx.butter_bandpass_filter(lfp, *ds_freq,    **KW)
#         end_time = time.time()
#         print(f'outsourced whole filtering --> {end_time-start_time:.2f} s')
#     else:
#         start_time = time.time()
#         nchunks = int(np.ceil(lfp.shape[1] / lfp_fs / load_win))
#         arr_list = np.array_split(lfp, nchunks, axis=1)
#         p = 0
#         for yarr in arr_list:
#             q = p + yarr.shape[1]
#             bp_dict['theta'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *theta, **KW)
#             bp_dict['slow_gamma'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *slow_gamma, **KW)
#             bp_dict['fast_gamma'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *fast_gamma, **KW)
#             bp_dict['swr'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *swr_freq, **KW)
#             bp_dict['ds'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *ds_freq, **KW)
#             p = int(q)
#         end_time = time.time()
#         print(f'outsourced chunk filtering --> {end_time-start_time:.2f} s')
#     return bp_dict


# def bp_filter_lfp(lfp, lfp_fs, fband, load_win=None):
#     filt_lfp = pyfx.butter_bandpass_filter(lfp, *fband, lfp_fs=lfp_fs, axis=1)
#     return filt_lfp


# Important, actually used in raw_data_pipeline
def detect_channel(event, i, data, lfp_time, DF=None, THRES=None, pprint=False, **PARAMS):
    """ Run ripple or DS detection for the given $data signal """
    assert event in ['swr', 'ds'], 'Event type must be "swr" or "ds".'
    if DF is None    : DF    = pd.DataFrame()
    if THRES is None : THRES = {}
    if event == 'swr':
        df, thres = ephys.get_swr_peaks(data, lfp_time, 
                                        pprint=pprint, **PARAMS)
    elif event == 'ds':
        # print(data.shape)
        # print(lfp_time.shape)
        df, thres = ephys.get_ds_peaks(data, lfp_time, 
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

# def load_openephys_data(ddir):
#     """ Load raw data files from Open Ephys recording software """
#     # initialize Open Ephys data objects
#     session = get_openephys_session(ddir)
#     OE = oeNodes(session, ddir)
#     #node = OE['node']           # node containing the selected recording
#     recording = OE['recording']  # experimental recording object
#     continuous_list = OE['continuous']  # continuous data from 1 or more processors
#     #settings_file = str(Path(node.directory, 'settings.xml'))
#     metadata_list = recording.info['continuous']
    
#     continuous, meta = continuous_list[0], metadata_list[0]
#     if len(continuous_list) > 1:
#         print(f'WARNING: Data extracted from 1 of {len(continuous_list)} existing processors')
    
#     # load sampling rate, timestamps, and number of channels
#     fs = meta['sample_rate']
#     #num_channels = meta['num_channels']
#     tstart, tend = pyfx.Edges(continuous.timestamps)
    
#     # load channel names and bit volt conversions, find primary channels
#     ch_names, bit_volts = zip(*[(d['channel_name'], d['bit_volts']) for d in meta['channels']])
#     ipri = np.nonzero([*map(lambda n: bool(re.match('C\d+', n)), ch_names)])[0]
#     if len(ipri) > 0:
#         units = meta['channels'][ipri[0]]['units']
#     else:
#         units = None
#     iaux = np.nonzero([*map(lambda n: bool(re.match('ADC\d+', n)), ch_names)])[0]
#     idig = np.nonzero([*map(lambda n: bool(re.match('DAC\d+', n)), ch_names)])[0]
#     if ipri.size == 0 and iaux == 0 and idig == 0:
#         return
#     # load raw signals (uV)
#     first,last = pyfx.Edges(continuous.sample_numbers)
#     raw_signal_array = np.array([x*bv for x,bv in zip(continuous.get_samples(0, last-first+1).T, bit_volts)])
#     if ipri.size > 0:
#         A,B = ipri[[0,-1]] + [0,1]
#         pri_mx = raw_signal_array[A:B]
#     else: pri_mx = np.array([])
#     # look for aux channels
#     if iaux.size > 0:
#         A_AUX, B_AUX = iaux[[0,-1]] + [0,1]
#         aux_mx = raw_signal_array[A_AUX : B_AUX]
#     else: aux_mx = np.array([])
#     dig_mx=np.array([np.clip(d, 0, 1).astype('uint8') for d in raw_signal_array[idig]])
#     return (pri_mx, aux_mx, dig_mx), fs, units


# def load_neuronexus_data(ddir):
#     """ Load raw data files from NeuroNexus recording software """
#     # get raw file names
#     meta_file = [f for f in os.listdir(ddir) if f.endswith('.xdat.json')][0]
#     stem = meta_file.replace('.xdat.json', '')
#     data_file = os.path.join(ddir, stem + '_data.xdat')
    
#     # load metadata
#     with open(os.path.join(ddir, meta_file), 'rb') as f:
#         metadata = json.load(f)
#     fs             = metadata['status']['samp_freq']
#     #num_channels   = metadata['status']['signals']['pri']
#     total_channels = metadata['status']['signals']['total']
#     tstart, tend   = metadata['status']['t_range']
#     num_samples    = int(round(tend * fs)) - int(round(tstart * fs))
#     # get SI units
#     udict = {'micro-volts':'uV', 'milli-volts':'mV', 'volts':'V'}
#     units = metadata['sapiens_base']['sigUnits']['sig_units_pri']
#     units = udict.get(units, units)
    
#     # organize electrode channels by port
#     ports,ddicts = map(list, zip(*metadata['sapiens_base']['sensors_by_port'].items()))
#     #nprobes = len(ports)
#     #probe_nch = [d['num_channels'] for d in ddicts]
    
#     # separate primary and aux channels
#     ch_names = metadata['sapiens_base']['biointerface_map']['chan_name']
#     ipri = np.array([i for i,n in enumerate(ch_names) if n.startswith('pri')])
#     A,B = ipri[[0,-1]] + [0,1]
    
#     # load raw probe data
#     with open(data_file, 'rb') as fid:
#         fid.seek(0, os.SEEK_SET)
#         raw_signals = np.fromfile(fid, dtype=np.float32, count=num_samples*total_channels)
#     raw_signal_array = np.reshape(raw_signals, (num_samples, total_channels)).T #[a:b+1]#[ipri]#[0:num_channels]
#     pri_mx = raw_signal_array[A:B]
#     # look for aux channels
#     iaux = np.array([i for i,n in enumerate(ch_names) if n.startswith('aux')])
#     if iaux.size > 0:
#         A_AUX, B_AUX = iaux[[0,-1]] + [0,1]
#         aux_mx = raw_signal_array[A_AUX : B_AUX]
#     else: aux_mx = np.array([])
#     # look for digital inputs
#     idig = np.array([i for i,n in enumerate(ch_names) if n.startswith('din')])
#     dig_mx=np.array([np.clip(d, 0, 1).astype('uint8') for d in raw_signal_array[idig]])
#     return (pri_mx, aux_mx, dig_mx), fs, units#, info


# def load_ncs_file(file_path):
#     # make sure .ncs file exists
#     assert os.path.isfile(file_path) and file_path.endswith('.ncs')
    
#     HEADER_LENGTH = 16 * 1024  # 16 kilobytes of header
#     NCS_SAMPLES_PER_RECORD = 512
#     NCS_RECORD = np.dtype([('TimeStamp',       np.uint64),       # Cheetah timestamp for this record. This corresponds to
#                                                                  # the sample time for the first data point in the Samples
#                                                                  # array. This value is in microseconds.
#                            ('ChannelNumber',   np.uint32),       # The channel number for this record. This is NOT the A/D
#                                                                  # channel number
#                            ('SampleFreq',      np.uint32),       # The sampling frequency (Hz) for the data stored in the
#                                                                  # Samples Field in this record
#                            ('NumValidSamples', np.uint32),       # Number of values in Samples containing valid data
#                            ('Samples',         np.int16, NCS_SAMPLES_PER_RECORD)])  # Data points for this record. Cheetah
#                                                                                     # currently supports 512 data points per
#                                                                                     # record. At this time, the Samples
#                                                                                     # array is a [512] array.
    
#     def parse_header(raw_header):
#         """ Parse Neuralynx file header """
#         # decode header as iso-8859-1 (the spec says ASCII, but there is at least one case of 0xB5 in some headers)
#         raw_hdr = raw_header.decode('iso-8859-1')
#         hdr_lines = [line.strip() for line in raw_hdr.split('\r\n') if line != '']
#         # look for line identifiying Neuralynx file
#         if hdr_lines[0] != '######## Neuralynx Data File Header':
#             warnings.warn('Unexpected start to header: ' + hdr_lines[0])
#         # return header information as dictionary
#         tmp = [l.split() for l in hdr_lines[1:]]
#         tmp = [x + [''] if len(x)==1 else x for x in tmp]
#         header = {x[0].replace('-','') : ' '.join(x[1:]) for x in tmp}
#         return header
    
#     # read in .ncs file
#     with open(file_path, 'rb') as fid:
#         # Read the raw header data (16 kb) from the file object fid. Restores the position in the file object after reading.
#         pos = fid.tell()
#         fid.seek(0)
#         raw_header = fid.read(HEADER_LENGTH).strip(b'\0')
#         records = np.fromfile(fid, NCS_RECORD, count=-1)
#         fid.seek(pos)
#     header = parse_header(raw_header)
#     fs = records['SampleFreq'][0]                   # get sampling rate
#     bit_volts = float(header['ADBitVolts']) * 1000  # convert ADC counts to mV

#     # load data
#     D = np.array(records['Samples'].reshape(-1) * bit_volts, dtype=np.float32)
#     ts = np.linspace(0, len(D) / fs, len(D))
#     return D, ts, fs


# def load_neuralynx_data(ddir, pprint=True, use_array=True, save_array=True):
#     """ Load raw data files from Neuralynx recording software """
#     # identify and sort all .ncs files in data directory
#     flist = np.array([f for f in os.listdir(ddir) if f.endswith('.ncs')])
#     fnums = [int(f.strip('CSC').strip('.ncs')) for f in flist]
#     fpaths = [str(Path(ddir, f)) for f in flist[np.argsort(fnums)]]
#     # get number of total channels, timestamps, and sampling rate
#     nch = len(fpaths)
#     _, ts, fs = load_ncs_file(fpaths[0])
#     if pprint: 
#         print(os.linesep + '###   LOADING NEURALYNX DATA   ###' + os.linesep)
        
#     data_path = str(Path(ddir, 'DATA_ARRAY.npy'))
#     if use_array and os.path.isfile(data_path):
#         # load existing array
#         if pprint: print('Loading existing DATA_ARRAY.npy file ...')
#         pri_mx = np.load(data_path)
#     else:
#         print_progress = np.round(np.linspace(0, nch-1, 10)).astype('int')
#         # initialize data array (channels x timepoints)
#         pri_mx = np.empty((nch, len(ts)), dtype=np.float32)
#         for i,f in enumerate(fpaths):
#             if pprint and (i in print_progress):
#                 print(f'Loading NCS file {i+1}/{nch} ...')
#             pri_mx[i,:] = load_ncs_file(f)[0]
#         if save_array:
#             print('Saving data array ...')
#             np.save(data_path, pri_mx)
#     if pprint: print('Done!' + os.linesep)
#     aux_mx = np.array([])
#     dig_mx = np.array([])
#     return (pri_mx, aux_mx, dig_mx), fs, 'mV'


# Never called; seems vestigial and can be removed
# def load_raw_data(ddir, ignore_flat=False, pprint=True):
#     """ Load raw data files from Open Ephys, NeuroNexus, or Neuralynx software """
#     try:
#         files = os.listdir(ddir)
#     except:
#         raise Exception(f'Directory "{ddir}" does not exist')
#     xdat_files = [f for f in files if f.endswith('.xdat.json')]
#     # load Open Ephys data
#     if 'structure.oebin' in files:
#         if pprint: print('Loading Open Ephys raw data ...')
#         res = load_openephys_data(ddir) # removed info, added fs`
#         if not res:
#             gi.MsgboxError('Unable to load channels from Open Ephys data.').exec()
#             return
#         (pri_array, aux_array, dig_array), fs, units = res
#     # load NeuroNexus data
#     elif len(xdat_files) > 0:
#         if pprint: print('Loading NeuroNexus raw data ...')
#         (pri_array, aux_array, dig_array), fs, units = load_neuronexus_data(ddir)
#     # load Neuralynx data
#     elif len([f for f in files if f.endswith('.ncs')]) > 0:
#         (pri_array, aux_array, dig_array), fs, units = load_neuralynx_data(ddir, pprint=pprint)
#     # no valid raw data found
#     else:
#         raise Exception(f'No raw Open Ephys (.oebin), NeuroNexus (.xdat.json), or Neuralynx (.ncs) files found in directory "{ddir}"')
#     if ignore_flat:
#         dig_array = dig_array[np.nonzero(np.any(dig_array, axis=1))[0], :]
#     return (pri_array, None), fs, units
    
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

# Also seems vestigial and can be removed
# def extract_data(raw_signal_array, idx, fs=30000, lfp_fs=1000, units='uV', lfp_units='mV'):
#     """ Extract, scale, and downsample each raw signal in depth order down the probe """
#     ds_factor = int(fs / lfp_fs)  # calculate downsampling factor
#     cf = pq.Quantity(1, units).rescale(lfp_units).magnitude  # mV conversion factor
#     lfp = np.array([pyfx.Downsample(raw_signal_array[i], ds_factor)*cf for i in idx])
#     return lfp

# Looks like this is vestigial and can be removed
# def extract_data_by_probe(raw_signal_array, chMap, fs=30000, lfp_fs=1000, units='uV', lfp_units='mV'):
#     """ Get LFP array for each probe represented in $chMap """
#     idx_by_probe = get_idx_by_probe(chMap)
#     ds_factor = int(fs / lfp_fs)  # calculate downsampling factor
#     cf = pq.Quantity(1, units).rescale(lfp_units).magnitude  # uV -> mV conversion factor
#     lfp_list = []
#     for idx in idx_by_probe:
#         lfp = np.array([pyfx.Downsample(raw_signal_array[i], ds_factor)*cf for i in idx])
#         lfp_list.append(lfp)
#     return lfp_list


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


# def process_aux(aux_mx, fs, lfp_fs, save_ddir, pprint=True):
#     ds_factor = int(fs / lfp_fs)
#     for i,aux in enumerate(aux_mx):
#         if pprint: print(f'Saving AUX {i+1} / {len(aux_mx)} ...')
#         aux_dn = pyfx.Downsample(aux, ds_factor)
#         np.save(Path(save_ddir, f'AUX{i}.npy'), aux_dn)
        
        
def validate_processed_ddir(ddir):
    """ Check whether directory contains required processed data files """
    try:
        files = os.listdir(ddir)
    except:
        return 0
    if 'probe_group' not in files: 
        return 0
    if 'params.pkl' not in files: 
        return 0
    if 'DATA.hdf5' not in files: #TODO remove this part for conversion because we shouldnt need or have it
        npz_files = ['lfp_bp.npz', 'lfp_time.npy', 'lfp_fs.npy', 'ALL_DS', 'ALL_SWR']
        is_npz = all([bool(f in files) for f in npz_files])
        if is_npz:
            return 2
        else: 
            return 0
    else: return 1


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
        assert len(ephys.load_ds_dataset(ddir, iprobe, ishank)) > 1
        llist = ephys.load_event_channels(ddir, iprobe, ishank)
    except:
        return False
    return bool(len(llist)==3 and llist != [None,None,None])
        