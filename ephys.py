#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File I/O and event analysis

@author: amandaschott
"""
import os
import re
from pathlib import Path
import scipy.io as so
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from PyQt5 import QtWidgets
import pickle
import quantities as pq
import probeinterface as prif
import pdb
# custom modules
import pyfx
import qparam
import icsd


##############################################################################
##############################################################################
################                                              ################
################                FILE MANAGEMENT               ################
################                                              ################
##############################################################################
##############################################################################

###   BASE FOLDERS   ###

def base_dirs(return_keys=False):
    """ Return default data directories saved in default_folders.txt """
    # Mode 0 for paths only, 1 for keys only
    with open('default_folders.txt', 'r') as fid:
        keys,vals = zip(*[map(str.strip, l.split('=')) for l in fid.readlines()])
    if not return_keys:
        return list(vals)
    return list(zip(keys,vals))

def write_base_dirs(ddir_list):
    """ Write input directories to default_folders.txt """
    assert len(ddir_list) == 4
    keys = ['RAW_DATA','PROBE_FILES','DEFAULT_PROBE', 'DEFAULT_PARAMETERS']
    with open('default_folders.txt', 'w') as fid:
        for k,path in zip(keys, ddir_list):
            fid.write(k + ' = ' + str(path) + '\n')

def clean_base_dirs():
    """ Ensure default_folder.txt is correctly formatted with valid info """
    # check for required keys
    keys,paths = map(list, zip(*base_dirs(return_keys=True)))
    if 'RAW_DATA' in keys and os.path.isdir(paths[keys.index('RAW_DATA')]):
        raw_data_path = paths[keys.index('RAW_DATA')]
    else: raw_data_path = str(os.getcwd())
    if 'PROBE_FILES' in keys and os.path.isdir(paths[keys.index('PROBE_FILES')]):
        probe_file_path = paths[keys.index('PROBE_FILES')]
    else: probe_file_path = str(os.getcwd())
    if 'DEFAULT_PROBE' in keys and os.path.isfile(paths[keys.index('DEFAULT_PROBE')]):
        default_probe_path = paths[keys.index('DEFAULT_PROBE')]
    else: default_probe_path = ''
    if 'DEFAULT_PARAMETERS' in keys and os.path.isfile(paths[keys.index('DEFAULT_PARAMETERS')]):
        param_path = paths[keys.index('DEFAULT_PARAMETERS')]
    else:
        param_path = Path(os.getcwd(), 'default_params.txt')
        if not os.path.isfile(param_path):
            print('Parameter file not found. Creating default_params.txt file ...')
            tmp = qparam.get_original_defaults()
            _ = qparam.write_param_file(tmp, param_path)
    # validate parameter file
    ddict, invalid_params = qparam.read_param_file(param_path, return_none=False)
    if len(invalid_params) > 0:
        fname = os.path.basename(param_path)
        print((f'Replaced {len(invalid_params)} missing or invalid parameter(s)'
               f' in {fname} with default value(s).'))
        new_dict = qparam.fix_params(ddict)
        _ = qparam.write_param_file(new_dict, param_path)
    # save base folders
    llist = [raw_data_path, probe_file_path, default_probe_path, param_path]
    write_base_dirs(llist)
    
def init_default_folders():
    """ Generate an initial default_folders.txt file """
    print('Initializing application settings ...')
    default_data_folder  = os.getcwd()
    default_probe_folder = Path(os.getcwd(), 'probe_configs')
    default_probe_file   = ''
    default_param_file   = Path(os.getcwd(), 'default_params.txt')
    if not os.path.isdir(default_probe_folder):
        print('Creating "probe_config" folder with demo probe configuration file ...')
        os.makedirs(default_probe_folder)  # initialize probe folder
        prb = demo_probe()
        _ = write_probe_file(prb, Path(default_probe_folder, 'demo_probe_config.json'))
    if not os.path.isfile(default_param_file):
        print('Creating default parameter file ...')
        param_dict = qparam.get_original_defaults()
        _ = qparam.write_param_file(param_dict, default_param_file)
    llist = [default_data_folder,  # raw data folder
             default_probe_folder, # probe configuration folder
             default_probe_file,   # default probe file (optional)
             default_param_file]   # parameter file
    write_base_dirs(llist)
    print('Done!' + os.linesep)
    
###   PARAMETER SETTINGS   ###
    
def read_params():
    """ Return parameter dictionary from current param file """
    ddict = qparam.fix_params(qparam.read_param_file(base_dirs()[3], return_none=False)[0])
    return ddict

###   RECORDING NOTES/PARAMS   ###

def read_notes(filepath):
    """ Load any recording notes from .txt file """
    try:
        with open(filepath, 'r') as fid:
            notes_txt = fid.read()
    except: 
        notes_txt = ''
    return notes_txt

def write_notes(filepath, txt):
    """ Write recording notes to .txt file """
    with open(filepath, 'w') as fid:
        fid.write(str(txt))
    
def load_recording_info(ddir):
    """ Load info dictionary from processed recording """
    INFO = pd.Series(pickle.load(open(Path(ddir, 'info.pkl'), 'rb')))
    return INFO

def save_recording_info(ddir, INFO):
    """ Save recording metadata to processed data folder """
    with open(Path(ddir, 'info.pkl'), 'wb') as f:
        pickle.dump(INFO, f)

def load_recording_params(ddir):
    """ Load param dictionary from processed recording """
    PARAMS = pd.Series(pickle.load(open(Path(ddir, 'params.pkl'), 'rb')))
    return PARAMS

def save_recording_params(ddir, PARAMS):
    """ Save param values used to analyze processed data """
    PARAMS = dict(PARAMS)
    if 'RAW_DATA_FOLDER' in PARAMS.keys()       : del PARAMS['RAW_DATA_FOLDER']
    if 'PROCESSED_DATA_FOLDER' in PARAMS.keys() : del PARAMS['PROCESSED_DATA_FOLDER']
    with open(Path(ddir, 'params.pkl'), 'wb') as f:
        pickle.dump(PARAMS, f)
        
###   PROBE OBJECTS   ###

def get_probe_filepaths(ddir):
    """ List all probe files in folder $ddir """
    probe_files = []
    for f in os.listdir(str(ddir)):
        if os.path.splitext(f)[-1] not in ['.json', '.prb', '.mat']:
            continue
        tmp = read_probe_file(str(Path(ddir, f)))
        if tmp is not None:
            probe_files.append(f)
    return probe_files

def read_probe_file(fpath, raise_exception=False):
    """ Load probe configuration from .json, .mat, or .prb file """
    if not os.path.exists(fpath):
        if raise_exception:
            raise Exception('Probe file does not exist')
        return
    # load data according to file extension
    ext = os.path.splitext(fpath)[-1]
    try:
        if ext == '.json':
            prb = prif.io.read_probeinterface(fpath).probes[0]
        elif ext == '.prb':
            prb = prif.io.read_prb(fpath).probes[0]
        elif ext == '.mat':
            prb = mat2probe(fpath)
        probe = copy_probe(prb)
        # keep probe name consistent with the file name
        probe.name = os.path.splitext(os.path.basename(fpath))[0].replace('_config','')
    except:
        probe = None
        if raise_exception:
            raise Exception('Invalid probe file')
    return probe

def read_probe_group(ddir):
    """ Load probe group for a processed recording """
    probe_group = prif.io.read_probeinterface(str(Path(ddir, 'probe_group')))
    return probe_group

def mat2probe(fpath):
    """ Load probe config from .mat file """
    file = so.loadmat(fpath, squeeze_me=True)
    xy_arr = np.array([file['xcoords'], 
                       file['ycoords']]).T
    probe = prif.Probe(ndim=int(file['ndim']), 
                            name=str(file['name']))
    probe.set_contacts(xy_arr, 
                       shank_ids   = np.array(file['shankInd']), 
                       contact_ids = np.array(file['contact_ids']))
    probe.set_device_channel_indices(np.array(file['chanMap0ind']))
    return probe

def probe2mat(probe, fpath):
    """ Save probe config as .mat file"""
    chanMap = probe.device_channel_indices
    probe_dict = {'chanMap'     : np.array(chanMap + 1), 
                  'chanMap0ind' : np.array(chanMap),
                  'connected'   : np.ones_like(chanMap, dtype='int'),
                  'name'        : str(probe.name),
                  'shankInd'    : np.array(probe.shank_ids, dtype='int'),
                  'xcoords'     : np.array(probe.contact_positions[:,0]),
                  'ycoords'     : np.array(probe.contact_positions[:,1]),
                  # specific to probeinterface module
                  'ndim' : int(probe.ndim),
                  'contact_ids' : np.array(probe.contact_ids, dtype='int')}
    probe_dict['connected'][np.where(chanMap==-1)[0]] = 0
    # save file
    so.savemat(fpath, probe_dict)
    return True

def write_probe_file(probe, fpath):
    """ Write probe configuration to .json, .mat, .prb, or .csv file """
    ext = os.path.splitext(fpath)[-1]
    
    if ext == '.json':   # best for probeinterface
        prif.io.write_probeinterface(fpath, probe)
        
    elif ext == '.prb': # loses a bunch of data, but required by some systems
        probegroup = prif.ProbeGroup()
        probegroup.add_probe(probe)
        prif.io.write_prb(fpath, probegroup)
        
    elif ext == '.mat': # preserves data, not automatically handled by probeinterface
        _ = probe2mat(probe, fpath)
        
    elif ext == '.csv': # straightforward, easy to view (TBD)
        probe.to_dataframe(complete=True)
        return False
    return True

def copy_probe(probe):
    """ Return new probe object identical to the given $probe """
    probe_new = probe.copy()
    probe_new.annotate(**dict(probe.annotations))
    probe_new.set_shank_ids(np.array(probe.shank_ids))
    probe_new.set_contact_ids(np.array(probe.contact_ids))
    probe_new.set_device_channel_indices(np.array(probe.device_channel_indices))
    return probe_new

def make_probe_group(probe, n=1):
    """ For multi-probe recordings, create group of $n probes to map channels """
    nch = probe.get_contact_count()
    PG = prif.ProbeGroup()
    for i in range(n):
        prb = probe.copy()
        cids = np.array(probe.contact_ids, dtype='int') + i*nch
        dids = np.array(probe.device_channel_indices) + i*nch
        prb.set_contact_ids(cids)
        prb.set_device_channel_indices(dids)
        PG.add_probe(prb)
    return PG

def demo_probe():
    """ Return 8-channel linear probe object """
    shank = prif.generate_linear_probe(num_elec=8, ypitch=50)
    pos_adj = shank.contact_positions[::-1] + np.array([0, 50])
    shank.set_contacts(pos_adj, shapes='circle', 
                            shape_params=dict(radius=7.5))
    shank.create_auto_shape('tip', margin=20)
    shank.set_shank_ids(np.zeros(8, dtype='int'))
    probe = prif.combine_probes([shank])
    probe.set_contact_ids(np.arange(8))
    probe.set_device_channel_indices(np.array([5,2,3,0,6,7,4,1]))
    probe.annotate(**{'name':'demo_probe'})
    return probe


##############################################################################
##############################################################################
################                                              ################
################                FILE SELECTION                ################
################                                              ################
##############################################################################
##############################################################################


def select_directory(init_ddir='', title='Select directory', show_dirs_only=False, parent=None):
    """ Return selected directory name """
    opts = QtWidgets.QFileDialog.DontUseNativeDialog
    if show_dirs_only:
        opts = opts | QtWidgets.QFileDialog.ShowDirsOnly
    ddir = QtWidgets.QFileDialog.getExistingDirectory(parent, title, init_ddir, 
                                                      options=opts)
    return ddir

def select_raw_recording_file(supported_extensions, init_ppath=None,
                              title='Select raw recording file', parent=None):
    """ Return selected recording file name """
    if init_ppath is None: init_ppath = base_dirs()[0]
    # get file extension filter
    opts = QtWidgets.QFileDialog.DontUseNativeDialog
    ffilter = f'Raw files ({" ".join([*map(lambda x: "*"+x, supported_extensions)])})'
    fpath,_ = QtWidgets.QFileDialog.getOpenFileName(parent, title, init_ppath, 
                                                    ffilter, options=opts)
    return fpath

def select_load_probe_file(init_ppath=None, title='Select probe configuration file',
                           exts=['.json', '.prb', '.mat'], parent=None):
    """ Return selected probe config file name and the loaded probe object """
    if init_ppath is None: init_ppath = base_dirs()[1]
    opts = QtWidgets.QFileDialog.DontUseNativeDialog
    ffilter = f'Probe files ({" ".join([*map(lambda x: "*"+x, exts)])})'
    fpath,_ = QtWidgets.QFileDialog.getOpenFileName(parent, title, init_ppath, 
                                                    ffilter, options=opts)
    if fpath:
        probe = read_probe_file(fpath)
        return probe, fpath
    return None, None

def select_load_param_file(init_ppath=None, title='Select parameter file',
                           allow_invalid_params=False, show_msgbox=True, parent=None):
    """ Return selected param file name and the loaded param dictionary """
    if init_ppath is None: init_ppath = os.getcwd()
    opts = QtWidgets.QFileDialog.DontUseNativeDialog
    ffilter = 'Text files (*.txt)'
    fpath,_ = QtWidgets.QFileDialog.getOpenFileName(parent, title, init_ppath, 
                                                    ffilter, options=opts)
    if fpath:
        PARAMS, invalid_keys = qparam.read_param_file(fpath, return_none=False)
        if not allow_invalid_params and len(invalid_keys) > 0:
            PARAMS = None
            if show_msgbox:
                msg = ('<b><u>MISSING/INVALID PARAMETERS</u></b>:<br>'
                       f'<code>{", ".join(invalid_keys)}</code>')
                QtWidgets.QMessageBox.critical(parent, '', msg)
        return PARAMS, fpath
    return None, None

def select_save_directory(init_ddir='', init_dirname='', title='Save directory', parent=None):
    """ Return new or existing directory name """
    ddir = SaveFileDialog.run(init_ddir=init_ddir, init_fname=init_dirname, 
                              filter_txt='', exts=[''], title=title, parent=parent)
    return ddir

def select_save_probe_file(probe, init_ddir=None, 
                           title='Save probe configuration file', parent=None):
    """ Save probe object to new or existing file, return file name """
    if init_ddir is None: init_ddir = base_dirs()[1]
    init_fname = f'{probe.name}_config.json'
    fpath = SaveFileDialog.run(init_ddir=init_ddir, init_fname=init_fname, 
                               filter_txt='Probe files', exts=['.json','.mat','.prb'], 
                               title=title, parent=parent)
    if fpath:
        res = write_probe_file(probe, fpath)
        if res:
            print('Probe configuration file saved!')
            return fpath
    return None

def select_save_param_file(ddict, init_ddir=os.getcwd(), init_fname='default_params.txt',
                           title='Save parameter file', parent=None):
    """ Save param dictionary to new or existing text file, return file name """
    assert qparam.validate_params(ddict)[0], 'Invalid parameter file'
    fpath = SaveFileDialog.run(init_ddir=init_ddir, init_fname=init_fname, 
                               filter_txt='Text files', exts=['.txt'], title=title, parent=parent)
    if fpath:
        res = qparam.write_param_file(ddict, fpath)
        if res:
            print('Parameter file saved!')
            return fpath
    return None
    
    
class SaveFileDialog(QtWidgets.QFileDialog):
    """ QFileDialog with custom filters for saving directories/files """
    
    def __init__(self, filter_txt='All files', exts=[], title='Save As', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.exts = exts  # list of accepted file extensions
        # set file dialog options
        self.setModal(True)
        self.setViewMode(self.List)
        self.setAcceptMode(self.AcceptSave)
        if exts == ['']:
            self.setFileMode(self.Directory)
        else:
            self.setFileMode(self.AnyFile)
        self.setOption(self.DontUseNativeDialog, True)
        self.setOption(self.DontConfirmOverwrite, True)
        # set file extension filter
        ext_str = ' '.join([f'*{x}' for x in exts])
        if len(exts)==0 : ffilter = ''
        elif exts==[''] : ffilter = ''  # directory mode
        else            : ffilter = f'{filter_txt} ({ext_str})'
        self.setNameFilter(ffilter)
        # get "Save" button and filename input
        self.btn = self.findChild(QtWidgets.QPushButton)
        self.lineEdit = self.findChild(QtWidgets.QLineEdit)
        self.lineEdit.textChanged.connect(self.updated_filename)
    
    def updated_filename(self, fname):
        """ Enable file selection if extension is valid """
        base, ext = os.path.splitext(fname)
        if base == '' or base.startswith('.'):
            x = False
        elif len(self.exts) == 0:
            x = True
        else:
            x = bool(ext in self.exts)
        self.btn.setEnabled(x)
    
    def accept(self):
        """ Implement file validation and overwrite warnings """
        if not self.btn.isEnabled():  # invalid filename
            return
        ddir = self.directory().path()
        fname = self.lineEdit.text()
        ppath = str(Path(ddir, fname))
        if os.path.exists(ppath):
            if os.path.isdir(ppath):
                n_items = len(os.listdir(ppath))
                suffix = '' if n_items == 1 else 's'
                msg = f'Folder "{fname}" contains {n_items} item{suffix}.'
                sub_msg = 'Overwrite existing directory?'
            elif os.path.isfile(ppath):
                msg = f'File "{fname}" already exists.'
                sub_msg = 'Do you want to replace it?'
            # launch messagebox
            msgbox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, '', msg, 
                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            msgbox.setInformativeText(sub_msg)
            res = msgbox.exec()
            if res != QtWidgets.QMessageBox.Yes:
                return
        QtWidgets.QDialog.accept(self)
    
    @classmethod
    def run(cls, init_ddir='', init_fname='', filter_txt='All files', exts=[], 
            title='Save As', parent=None):
        """ Initialize file dialog with the given params, return selected file """
        pyfx.qapp()
        dlg = cls(filter_txt=filter_txt, exts=exts, title=title, parent=parent)
        if os.path.isdir(init_ddir):
            dlg.setDirectory(init_ddir)
        # initialize file name and button status
        pyfx.stealthy(dlg.lineEdit, init_fname)
        dlg.updated_filename(init_fname)
        res = dlg.exec()
        if res:
            fpath = str(Path(dlg.directory().path(), dlg.lineEdit.text()))
            return fpath
        return None
    

##############################################################################
##############################################################################
################                                              ################
################              LOAD RECORDING DATA             ################
################                                              ################
##############################################################################
##############################################################################


###   NPY/NPZ FILES   ###

def load_bp(ddir, key='', iprb=-1):
    """ Load bandpass-filtered LFP data for 1+ probes from NPZ file """
    nprobes = len(read_probe_group(ddir).probes)
    data_list = [{} for _ in range(nprobes)]
    # load dictionary from npz file
    with np.load(Path(ddir,'lfp_bp.npz'), allow_pickle=True) as npz:
        keys = list(npz.keys())
        for k,v in npz.items():
            if key in keys and k != key:    # skip non-matching keys
                continue
            for i in range(nprobes):
                if k==key:
                    data_list[i] = v[i]     # arrays at key $k
                else:
                    data_list[i][k] = v[i]  # dict with all keys
    if 0 <= iprb < len(data_list):
        return data_list[iprb]
    else:
        return data_list
    
def load_lfp(ddir, key='', iprb=-1):
    """ Load LFP signals, timestamps, and sampling rate from NPY/NPZ files """
    DATA = load_bp(ddir, key=key, iprb=iprb)
    lfp_time = np.load(Path(ddir, 'lfp_time.npy'))
    lfp_fs = int(np.load(Path(ddir, 'lfp_fs.npy')))
    return DATA, lfp_time, lfp_fs

def load_noise_channels(ddir, iprb=-1):
    """ Load or create channel noise annotations (0=clean, 1=noisy) from NPY file """
    if os.path.exists(Path(ddir, 'noise_channels.npy')):
        # load noise trains, listed by probe
        noise_list = list(np.load(Path(ddir, 'noise_channels.npy')))
    else:
        # initialize noise trains with zeroes (channels are "clean" by default)
        probes = read_probe_group(ddir).probes
        noise_list = [np.zeros(prb.get_contact_count(), dtype='int') for prb in probes]
    if 0 <= iprb < len(noise_list):
        return noise_list[iprb]
    else:
        return noise_list

def csv2list(ddir, f=''):
    """ Return list of dataframes from keyed .csv file """
    ddf = pd.read_csv(Path(ddir, f))
    llist = [x.droplevel(0) for _,x in ddf.groupby(level=0)]
    return llist

def load_csv_event_dfs(ddir, event, iprb=-1):
    """ Load event dataframes (ripples or DS) for 1+ probes from keyed CSVs """
    DFS = list(zip(csv2list(ddir, f'ALL_{event.upper()}'), # event dfs
                   csv2list(ddir, 'channel_bp_std')))  # ch bandpass power
    probes = read_probe_group(ddir).probes
    LLIST = []
    for i,(DF_ALL, STD) in enumerate(DFS):
        DF_ALL = clean_event_df(DF_ALL, STD, probes[i])
        DF_MEAN = get_mean_event_df(DF_ALL, STD)
        LLIST.append([DF_ALL, DF_MEAN])
    if 0 <= iprb < len(LLIST):
        return LLIST[iprb]
    else:
        return LLIST
    
def save_csv_event_dfs(ddir, event, DF, iprb):
    """ Update saved event dataframe """
    tmp = [x[0] for x in load_csv_event_dfs(ddir, event)]
    tmp[iprb] = DF
    ALL_DF = pd.concat(tmp, keys=range(len(tmp)), ignore_index=False)
    ALL_DF = ALL_DF.drop('ch', axis=1)
    ALL_DF.to_csv(Path(ddir, f'ALL_{event.upper()}'), index_label=False)

###   HDF5 FILES   ###

def get_h5_key(name, iprb=None, ishank=None):
    """ Get HDF5 dataset key """
    key = f'/{name}'
    if ishank is not None: key = f'/{ishank}' + key
    if iprb is not None:   key = f'/{iprb}' + key
    return key

def load_h5_lfp(ff, key='', iprb=0, in_memory=False):
    """ Load bandpass-filtered LFP data from HDF5 file """
    ddict = ff[f'{iprb}']['LFP']
    if key in ddict.keys():
        if in_memory: return ddict[key][:]
        return ddict[key]
    if in_memory:
        ddict = {k:v[:] for k,v in ddict.items()}
    return ddict
    
def load_h5_array(ff, name, iprb=None, ishank=None, in_memory=False):
    """ Load data arrays (e.g. lfp_time) from HDF5 file """
    key = get_h5_key(name, iprb, ishank)
    arr = ff[key]
    if in_memory: return arr[:]
    return arr

def load_h5_thresholds(ff, iprb=0):
    """ Load ripple and DS detection thresholds from HDF5 file """
    thresholds = {}
    for event in ['SWR','DS']:
        ddict = load_h5_df(ff, f'{event}_THRES', iprb).T.to_dict()
        thresholds[event] = {k:pd.Series(v) for k,v in ddict.items()}
    return thresholds
    
def load_h5_df(ff, name, iprb=None, ishank=None):
    """ Load dataframes (e.g. ripple and DS event data) from HDF5 file """
    key = get_h5_key(name, iprb, ishank)
    DF = pd.read_hdf(ff.filename, key=key)
    return DF

def save_h5_df(DF, ff, name, iprb=None, ishank=None):
    """ Save dataframes to HDF5 format"""
    key = get_h5_key(name, iprb, ishank)
    DF.to_hdf(ff.filename, key=key)

def load_h5_event_df(ff, event, iprb=0):
    """ Load event dataframes for 1+ probes from HDF5 file """
    ddir = os.path.dirname(ff.filename)
    probe = read_probe_group(ddir).probes[iprb]
    DF_ALL = load_h5_df(ff, name=f'ALL_{event.upper()}', iprb=iprb)
    STD = load_h5_df(ff, name='STD', iprb=iprb)
    DF_ALL = clean_event_df(DF_ALL, STD, probe)
    DF_MEAN = get_mean_event_df(DF_ALL, STD)
    return DF_ALL, DF_MEAN

###   PROCESS EVENT DATAFRAMES   ###

def clean_event_df(DF_ALL, STD, probe):
    """ Return single-event dataframe with channel, shanks, event statuses, 
    and frequency band amplitudes for each event instance """
    DF_ALL.insert(0, 'ch', np.array(DF_ALL.index.values))
    if 'shank' not in DF_ALL.columns:
        DF_ALL['shank'] = pd.Series(probe.shank_ids.astype('int'))
    if 'status' not in DF_ALL.columns: # track original/added/removed events
        DF_ALL['status'] = 1
    if 'is_valid' not in DF_ALL.columns:
        DF_ALL['is_valid'] = np.array(DF_ALL['status'] > 0, dtype='int')
    # add freq band amplitudes (one value per channel)
    DF_ALL[STD.columns] = STD
    DF_ALL['n_valid'] = DF_ALL.groupby('ch')['is_valid'].agg('sum')
    return DF_ALL

def get_mean_event_df(DF_ALL, STD):
    """ Return dataframe of mean event values for each channel """
    channels = np.arange(len(STD))
    # average values by channel (NaNs for channels with no valid events)
    DF_VALID = pd.DataFrame(DF_ALL[DF_ALL['is_valid'] == 1])
    DF_MEAN = DF_VALID.groupby('ch').agg('mean')
    DF_MEAN = replace_missing_channels(DF_MEAN, channels).astype({'n_valid':int})
    DF_MEAN[STD.columns] = np.array(STD)
    DF_MEAN.insert(0, 'ch', DF_MEAN.index.values)
    return DF_MEAN

def replace_missing_channels(DF, channels):
    """ Replace any missing channels in mean event dataframe with NaNs"""
    if len(DF) == len(channels):
        return DF
    # fill in rows for any channels with no detected events
    missing_ch = np.setdiff1d(channels, DF.index.values)
    missing_df = pd.DataFrame(0.0, index=missing_ch, columns=DF.columns)
    DF = pd.concat([DF, missing_df], axis=0, ignore_index=False).sort_index()
    # set missing values to NaN (except for event counts, which are zero)
    DF.loc[missing_ch, [c for c in DF.columns if c!='n_valid']] = np.nan
    return DF

def load_aux(ddir):
    """ Load AUX files """
    aux_files = sorted([f for f in os.listdir(ddir) if re.match('AUX\d+.npy', f)])
    aux_array = np.array([np.load(Path(ddir, f)) for f in aux_files])
    return aux_array
    
def load_iis(ddir, iprb):
    """ Load interictal spikes """
    if os.path.exists(Path(ddir, f'iis_{iprb}.npy')):
        ddict = np.load(Path(ddir, f'iis_{iprb}.npy'), allow_pickle=True).item()
        seqs, thres, thr = list(ddict.values())
    else:
        seqs, thres, thr = [], None, None
    return seqs, thres, thr

def load_seizures(ddir):
    """ Load seizure events """
    if os.path.exists(Path(ddir, 'seizures.npy')):
        ddict = np.load(Path(ddir, 'seizures.npy'), allow_pickle=True).item()
        seqs, thres, thr = list(ddict.values())
    else:
        seqs, thres, thr = [], None, None
    return seqs, thres, thr

###   HANDLE EVENT CHANNELS   ###

def init_event_channels(ddir, probes=None, psave=True):
    """ Return event channel dictionary with placeholder [None,None,None] 
    for each probe and shank in the recording """
    if probes is None: probes = read_probe_group(ddir).probes
    event_channels = {}
    for iprb,prb in enumerate(probes):
        event_channels[iprb] = {ishank:[None,None,None] for ishank in range(prb.get_shank_count())}
    if psave:
        np.save(Path(ddir, 'theta_ripple_hil_chan.npy'), event_channels, allow_pickle=True)
    return event_channels

def load_event_channels(ddir, iprb=None, ishank=None):
    """ Load theta/ripple/hilus channels for given probe(s) and shank(s) """
    fpath = Path(ddir, 'theta_ripple_hil_chan.npy')
    if not os.path.isfile(fpath):
        _ = init_event_channels(ddir)
    event_channel_dict = np.load(Path(ddir, 'theta_ripple_hil_chan.npy'), 
                                 allow_pickle=True).item()
    if iprb is None: return event_channel_dict
    deprec_path = Path(ddir, f'theta_ripple_hil_chan_{iprb}.npy')
    if os.path.isfile(deprec_path):
        deprec_event_channels = [*map(list, np.load(deprec_path, allow_pickle=True))]
        for ii,ll in enumerate(deprec_event_channels):
            if len(ll)==3: event_channel_dict[iprb][ii] = ll
        np.save(fpath, event_channel_dict, allow_pickle=True)
        os.remove(deprec_path)
    ddict = event_channel_dict[iprb]
    if ishank is None: return list(ddict.values())
    return ddict[ishank]
        
def save_event_channels(ddir, iprb, ishank, new_channels):
    """ Save new event channels for given probe and shank """
    event_channel_dict = load_event_channels(ddir)
    event_channel_dict[iprb][ishank] = list(new_channels)
    np.save(Path(ddir, 'theta_ripple_hil_chan.npy'), event_channel_dict, allow_pickle=True)

def load_event_channels_deprec(ddir, iprb, ishank=-1):
    """ Deprecated function for loading event channels """
    probes = read_probe_group(ddir).probes
    probe = probes[iprb]
    nshanks = probe.get_shank_count()
    # try loading channels
    ppath1 = Path(ddir, f'theta_ripple_hil_chan_{iprb}.npy')
    ppath2 = Path(ddir, 'theta_ripple_hil_chan.npy')
    if os.path.isfile(ppath1): # event channels saved for the given probe
        event_channels = list(np.load(ppath1, allow_pickle=True))
        # for single-shank probe, convert [a,b,c] to [[a,b,c]] 
        if nshanks==1 and len(event_channels)==3 and all([np.ndim(x)==0 for x in event_channels]):
            event_channels = [event_channels]
            np.save(ppath1, event_channels)
        event_channels = [list(x) for x in event_channels]
    elif os.path.isfile(ppath2) and len(probes)==1 and nshanks==1:
        # for recording with one single-shank probe, can use generic channels
        event_channels = [list(np.load(ppath2))]
    else:
        event_channels = [[] for _ in range(nshanks)]
    if 0 <= ishank < len(event_channels):
        return event_channels[ishank]
    else:
        return event_channels

def load_ds_dataset(ddir, iprb, ishank=-1, fill_shank=True):
    """ Load DS event dataframe for given probe and shank(s) """
    try:
        DS_DF = pd.read_csv(Path(ddir, f'DS_DF_{iprb}')).reset_index(drop=True)
    except:
        return None
    if 'shank' not in DS_DF.columns:
        probe = read_probe_group(ddir).probes[iprb]
        DS_DF['shank'] = pd.Series(probe.shank_ids.astype('int'))
    if ishank == -1 or ishank is None:
        return DS_DF
    elif ishank in DS_DF.shank:
        ddf = pd.DataFrame(DS_DF[DS_DF.shank==ishank]).reset_index(drop=True)
        return ddf
    else:
        return pd.DataFrame(columns=DS_DF.columns)


##############################################################################
##############################################################################
################                                              ################
################               DATA MANIPULATION              ################
################                                              ################
##############################################################################
##############################################################################


def get_asym(ipk, istart, istop):
    """ Compute asymmetry of waveform spanning $istart-$iend with peak $ipk """
    i0,i1 = ipk-istart, istop-ipk
    asym = (i1-i0) / min(i0,i1) * 100
    return asym

def getwaves(LFP, iev, iwin, center=False):
    """ Get LFP waveforms surrounding the given event indices $iev """
    if center:
        iwin += 1
    arr = np.full((len(iev), iwin*2), np.nan)
    iev = np.atleast_1d(iev).astype('int')
    for i,idx in enumerate(iev):
        arr[i,:] = pad_lfp(LFP, idx, iwin)
    if center:
        arr = arr[:, 1:]
    return arr

def getavg(LFP, iev, iwin, center=False):
    """ Get event-averaged LFP waveform """
    return np.nanmean(getwaves(LFP, iev, iwin, center), axis=0)

def getyerrs(arr, mode='std'):
    """ Get mean signal and variance for 2D array $arr (instance x timepoint) """
    d = np.nanmean(arr, axis=0)
    yerr = np.nanstd(arr, axis=0)
    if mode == 'sem':
        yerr /= np.sqrt(arr.shape[0])
    return (d-yerr, d+yerr), d

def pad_lfp(LFP, idx, iwin, pad_val=np.nan):
    """ Add padding to data windows that extend past the recording boundaries """
    if idx >= iwin and idx < len(LFP)-iwin:
        return LFP[idx-iwin : idx+iwin]
    elif idx < iwin:
        pad = np.full(iwin*2 - (idx+iwin), pad_val)
        return np.concatenate([pad, LFP[0 : idx+iwin]])
    else:
        #pad = np.full(len(LFP)-idx, pad_val)
        pad = np.full(iwin*2 - (iwin+len(LFP)-idx), pad_val)
        return np.concatenate([LFP[idx-iwin :], pad])

def get_csd_obj(data, coord_electrode, ddict):
    """ Calculate CSD using the given parameters """
    # update default dictionary with new params
    lfp_data = (data * pq.mV).rescale('V') # assume data units (mV)
    
    # set general params
    method = ddict['csd_method']
    args = {'lfp'             : lfp_data,
            'coord_electrode' : coord_electrode,
            'sigma'           : ddict['cond'] * pq.S / pq.m,
            'f_type'          : ddict['f_type'],
            'f_order'         : ddict['f_order']}
    if ddict['f_type'] == 'gaussian':
        args['f_order'] = (ddict['f_order'], ddict['f_sigma'])
    
    # create CSD object
    if method == 'standard':
        args['vaknin_el'] = bool(ddict['vaknin_el'])
        csd_obj = icsd.StandardCSD(**args)
    else:
        args['sigma_top'] = ddict['cond'] * pq.S/pq.m
        args['diam']      = (ddict['src_diam'] * pq.mm).rescale(pq.m)
        if method == 'delta':
            csd_obj = icsd.DeltaiCSD(**args)
        else:
            args['tol'] = ddict['tol']
            if method == 'step':
                args['h'] = (ddict['src_diam'] * pq.mm).rescale(pq.m)
                csd_obj = icsd.StepiCSD(**args)
            elif method == 'spline':
                args['num_steps'] = int(ddict['spline_nsteps'])
                csd_obj = icsd.SplineiCSD(**args)
    return csd_obj

def csd_obj2arrs(csd_obj):
    """ Convert ICSD object to raw, filtered, and normalized CSD arrays """
    raw_csd       = csd_obj.get_csd()
    filt_csd      = csd_obj.filter_csd(raw_csd)
    norm_filt_csd = np.array([*map(pyfx.Normalize, filt_csd.T)]).T
    return (raw_csd.magnitude, filt_csd.magnitude, norm_filt_csd)

def encoder2pos(chA, chB, circumference=44.8, ppr=256):
    """ Get wheel position from rotary encoder A and B signals """
    chA_plus_minus = (chA * 2) - 1
    chB_diff = np.diff(chB, prepend=[0])
    chA_prod = chA_plus_minus * chB_diff
    chB_plus_minus = (chB * 2) - 1
    chA_diff = np.diff(chA, prepend=[0])
    chB_prod = -chB_plus_minus * chA_diff
    position = np.cumsum(chA_prod + chB_prod)
    # circumfence (cm) divided by number of pulses per revolution
    pos = position * (circumference/ppr)
    pos_dif = np.diff(pos, prepend=[0])
    return pos_dif

def pos2speed(pos, sf=500):
    """ Translate wheel position to speed with Gaussian smoothing factor $sf """
    pos_dif = np.concatenate((np.array([0]), np.diff(pos)))
    speed_smth = scipy.ndimage.gaussian_filter1d(pos_dif, sf)
    return speed_smth


##############################################################################
##############################################################################
################                                              ################
################                EVENT DETECTION               ################
################                                              ################
##############################################################################
##############################################################################


def estimate_theta_chan(STD, noise_idx=np.array([], dtype='int')):
    """ Estimate optimal theta (fissure) channel (max power in theta range) """
    theta_pwr = np.array(STD.theta.values)
    theta_pwr[noise_idx] = np.nan
    return np.nanargmax(theta_pwr)

def estimate_ripple_chan(STD, noise_idx=np.array([], dtype='int')):
    """ Estimate optimal ripple channel (max ripple power among channels with low theta power) """
    arr = STD[['theta','swr']].values.T
    arr[:, noise_idx] = np.nan
    norm_theta, norm_swr = map(pyfx.Normalize, arr)
    # eliminate channels above 60th %ile of theta power
    norm_swr[norm_theta >= np.nanpercentile(norm_theta, 60)] = np.nan
    norm_swr[norm_theta == 0] = np.nan
    return np.nanargmax(norm_swr / norm_theta)
    
def estimate_hil_chan(DS_MEAN, noise_idx=np.array([], dtype='int')):
    """ Estimate optimal DS (hilus) channel (large and frequent waveforms) """
    arr = DS_MEAN[['amp','n_valid']].values.T
    arr[:, noise_idx] = np.nan
    norm_amp, norm_n = map(pyfx.Normalize, arr)
    return np.nanargmax(norm_amp * norm_n)

def get_inst_freq(x, lfp_fs, swr_freq=[120,180]):
    """ Calculate LFP instantaneous frequency for ripple detection """
    angle  = np.angle(x)      # radian phase (-π to π) of each LFP timepoint
    iphase = np.unwrap(angle) # phase + 2π*k, where k=cycle number (0-K total cycles)
    difs   = np.diff(iphase)/(2.0*np.pi) # distance (% of 2π cycle) between consecutive points
    ifreq  = np.clip(difs*lfp_fs, *swr_freq) # inst. freq (Hz) at each point (bounds=SWR cutoff freqs)
    return ifreq

def get_swr_peaks(LFP, lfp_time, lfp_fs, pprint=True, **kwargs):
    """ Detect peaks in the envelope of sharp-wave ripple activity """
    # load optional keyword args
    swr_freq     = kwargs.get('swr_freq', [120,180])
    swr_min_dur  = kwargs.get('swr_min_dur',  0) / 1000  # ms -> s
    swr_freq_thr = kwargs.get('swr_freq_thr', 0)
    swr_min_dist = kwargs.get('swr_dist_thr', 0) / 1000  # ms -> s
    swr_fwin     = int(round(kwargs.get('swr_freq_win', 8)/1000 * lfp_fs))
    swr_ampwin   = int(round(kwargs.get('swr_maxamp_win', 40)/1000 * lfp_fs))
    height, distance, swr_min = None,None,None
    
    # get SWR envelope, calculate detection thresholds
    hilb = scipy.signal.hilbert(LFP)     # Hilbert transform of SWR LFP signal
    env = np.abs(hilb).astype('float32') # Hilbert absolute value (amp. of pos/neg peaks)
    std = np.std(env)                    # standard deviation of SWR envelope
    if 'swr_height_thr' in kwargs:
        height = std * kwargs['swr_height_thr']
    if 'swr_min_thr' in kwargs:
        swr_min = std * kwargs['swr_min_thr']
    if swr_min_dist > 0:
        distance = int(round(lfp_fs * swr_min_dist))
    thresholds = dict(dur=swr_min_dur,         # min. SWR duration (s)
                      inst_freq=swr_freq_thr,  # min. SWR instantaneous freq (Hz)
                      peak_height=height,      # min. SWR peak amplitude
                      edge_height=swr_min,     # min. SWR edge amplitude
                      isi=swr_min_dist)        # min. distance (s) between SWRs
    thresholds = pd.Series(thresholds)
        
    # get instantaneous frequency for each timepoint
    ifreq = get_inst_freq(hilb, lfp_fs, swr_freq)
    env_clip = np.clip(env, swr_min, max(env))
    
    # get indexes of putative SWR envelope peaks
    ippks = scipy.signal.find_peaks(env, height=height, distance=distance)[0]
    ippks = ippks[np.where((ippks > lfp_fs) & (ippks < len(LFP)-lfp_fs))[0]]
    ppk_freqs = np.array([np.mean(ifreq[i-swr_fwin:i+swr_fwin]) for i in ippks])
    
    # get width of each SWR (first point above SWR min to next point below SWR min) 
    durs, _, starts, stops = scipy.signal.peak_widths(env_clip, peaks=ippks, rel_height=1)
    
    # filter for peaks above duration/frequency thresholds
    idur = np.where(durs/lfp_fs > swr_min_dur)[0] # SWRs > min. duration
    ifreq = np.where(ppk_freqs > swr_freq_thr)[0] # SWRs > min. inst. freq
    idx = np.intersect1d(idur, ifreq)
    swr_rate = len(idx) / (lfp_time[-1]-lfp_time[0])
    
    if pprint:
        print((f'{len(idx)} sharp-wave ripples detected; '
               f'SWR rate = {swr_rate:0.3f} Hz ({swr_rate*60:0.1f} events/min)'))
    
    ipks = ippks[idx]
    istarts, istops = [x[idx].astype('int') for x in [starts, stops]]
    
    # get timepoint of largest positive cycle for each SWR
    offsets = [np.argmax(LFP[i-swr_ampwin:i+swr_ampwin]) - swr_ampwin for i in ipks]
    imax = np.array(ipks + np.array(offsets), dtype='int')
    ddict = dict(time      = lfp_time[imax],  # times (s) of largest ripple oscillations
                 amp       = env[ipks],       # SWR envelope peak amplitudes
                 dur       = durs[idx] / (lfp_fs/1000), # SWR durations (ms)
                 freq      = ppk_freqs[idx],     # SWR instantaneous freqs
                 start     = lfp_time[istarts],  # SWR start times
                 stop      = lfp_time[istops],   # SWR end times
                 idx       = imax,    # idx of largest ripple oscillations
                 idx_peak  = ipks,    # idx of max envelope amplitudes
                 idx_start = istarts, # idx of SWR starts
                 idx_stop  = istops)  # idx of SWR stops
    df = pd.DataFrame(ddict)
    return df, thresholds


def get_ds_peaks(LFP, lfp_time, lfp_fs, pprint=True, **kwargs):
    """ Detect peaks of dentate spike waveforms """
    # load optional keyword args
    ds_min_dist = kwargs.get('ds_dist_thr', 0) / 1000  # ms -> s
    height, distance, wlen, LFPraw = None,None,None,None
    if 'ds_height_thr' in kwargs:
        height = np.std(LFP) * kwargs['ds_height_thr']
    if 'ds_wlen' in kwargs:
        wlen = int(round(lfp_fs * kwargs['ds_wlen'] / 1000)) # ms -> s
    if ds_min_dist > 0:
        distance = int(round(lfp_fs * ds_min_dist))
    min_prominence = kwargs.get('ds_prom_thr', 0)
    min_amp = kwargs.get('ds_abs_thr', 0)
    thresholds = dict(peak_height=height,  # min. DS peak height
                      isi=ds_min_dist,     # min. distance (s) between DS events
                      min_amp=min_amp)     # min. DS amplitude in mV
    thresholds = pd.Series(thresholds)
    thres_mv = max(height, min_amp)
    # detect qualifying peaks
    ipks,props = scipy.signal.find_peaks(LFP, height=thres_mv, distance=distance, 
                                         prominence=min_prominence)
    ds_prom = props['prominences']
    
    # get peak size/shape
    pws = scipy.signal.peak_widths(LFP, peaks=ipks, rel_height=0.5, wlen=wlen)
    ds_half_width, ds_width_height, starts, stops = pws
    
    # calculate peak half-widths and asymmetry (peak pos. relative to bases)
    istarts, istops = [x.astype('int') for x in [starts, stops]]
    ds_half_width = (ds_half_width/lfp_fs) * 1000  # convert nsamples to ms
    ds_asym = list(map(get_asym, ipks, istarts, istops))
    
    # for each peak, get index of max raw LFP value in surrounding 20 samples
    LFPraw = kwargs.get('LFPraw')
    if type(LFPraw) in [list,tuple,np.ndarray] and len(LFPraw) == len(LFP):
        max_ds_loc = [np.argmax(LFPraw[ipk-10:ipk+10]) for ipk in ipks]
        imax   = np.array([ipk-10+max_ds_loc[i] for i,ipk in enumerate(ipks)])
    else:
        imax = np.array(ipks)
    ds_rate = len(ipks) / (lfp_time[-1]-lfp_time[0])
    if pprint:
        print((f'{len(ipks)} dentate spikes detected; '
               f'DS rate = {ds_rate:0.3f} Hz ({ds_rate*60:0.1f} spks/min)'))
    ddict = dict(time         = lfp_time[imax],     # times (s) of DS peak
                 amp          = LFP[ipks],          # DS peak amplitudes
                 half_width   = ds_half_width,      # half-widths (ms) of DS waveforms
                 width_height = ds_width_height,    # DS height at 0.5 peak prominence
                 asym         = ds_asym,  # DS asymmetry (peak pos. relative to bases)
                 prom         = ds_prom,  # DS peak prominence (relative to surround)
                 start        = lfp_time[istarts],  # DS start times
                 stop         = lfp_time[istops],   # DS end times
                 idx          = imax,    # idx of max DS amplitudes
                 idx_peak     = ipks,    # idx of DS scipy peaks
                 idx_start    = istarts, # idx of DS starts
                 idx_stop     = istops)  # idx of DS stops
    df = pd.DataFrame(ddict)
    return df, thresholds


def get_seizures(spks, lfp_time, lfp_fs, baseline=[0,5], thres=10, sep=10, 
                  pprint=True, pplot=True):
    """ Basic seizure detection using summed magnitude of all LFP channels """
    # get min separation (# samples) between individual seizure events
    isep = int(round(sep/1000 * lfp_fs))
    # calculate threshold from baseline interval
    ibase0, ibase1 = [int(round(x*lfp_fs)) for x in baseline]
    base_spks = spks[ibase0 : ibase1]
    thr = base_spks.std() * thres
    # get sequences of consecutive indices above threshold
    idx = np.where(spks >= thr)[0]
    edges = np.concatenate([[0], np.where(np.diff(idx) > isep)[0]+1, [len(idx)]])
    seqs = [idx[a:b] for a,b in zip(edges[0:-1], edges[1:])]
    if pprint:
        print(f'{len(seqs)} spikes detected (thres = {thres})')
    if pplot:
        fig,ax = plt.subplots(layout='tight')
        ax.plot(lfp_time, spks)
        ax.axhline(thr, color='red')
        for seq in seqs:
            ax.plot(lfp_time[seq], spks[seq])
        ax.set(xlabel='Time (s)', ylabel='$\Sigma$ LFP ampl. (mV)', 
               title=f'Interictal spikes (n={len(seqs)})')
        sns.despine()
    return seqs, thr


##############################################################################
##############################################################################
################                                              ################
################                 STATIC PLOTS                 ################
################                                              ################
##############################################################################
##############################################################################


def plot_num_events(DF_MEAN, ax, pal):
    """ Plot number of valid events for each channel on given axes $ax """
    _ = sns.barplot(DF_MEAN, x='ch', y='n_valid', order=range(len(DF_MEAN)), 
                    lw=1, ax=ax)
    ax.ch_bars = list(ax.patches)
    ax.cmap = pyfx.Cmap(DF_MEAN.n_valid, pal)
    ax.CM = ax.cmap  # save colormap (ch x [R,G,B])
    return ax

def plot_event_amps(DF_ALL, DF_MEAN, ax, pal):
    """ Plot event amplitudes for each channel on given axes $ax """
    _ = sns.stripplot(data=DF_ALL, x='ch', y='amp', order=range(len(DF_MEAN)), 
                      linewidth=0.5, edgecolor='lightgray', ax=ax)
    # each PathCollection must be colormapped separately, but still relative to the entire dataset
    bounds = (np.nanmin(DF_ALL.amp), np.nanmax(DF_ALL.amp))
    ax.ch_collections = list(ax.collections)
    ax.cmap = []
    for coll in ax.collections:
        ydata = coll.get_offsets().data[:,1]
        clrs = pyfx.Cmap(ydata, pal, norm_data=bounds)  # colors for each collection
        ax.cmap.append(clrs)
    ax.CM = ax.cmap  # save colormap (ch x [R,G,B])
    return ax

def plot_ds_width_height(DF_ALL, DF_MEAN, ax, pal):
    """ Plot DS waveform widths at half-prominence height on give axes $ax """
    # plot errorbars and background markers
    err_kwargs = dict(errorbar='sd', err_kws=dict(lw=3), mfc='white', ms=10, mew=0)
    _ = sns.pointplot(DF_ALL, x='ch', y='width_height', order=range(len(DF_MEAN)),
                      linestyle='none', zorder=1, ax=ax, **err_kwargs)
    ax.lines[0].set(zorder=2)
    ax.err_bars = list(ax.lines)[1:]
    # plot foreground markers; "hue" param required to color each channel marker individually
    mkr_kwargs = dict(errorbar=None, ms=10, mew=3, zorder=3)
    _ = sns.pointplot(DF_MEAN, x='ch', y='width_height', hue='ch', order=range(len(DF_MEAN)), 
                      linestyle='none', ax=ax, legend=False, **mkr_kwargs)
    # save errorbar/marker items and their corresponding colormaps
    ax.outlines = ax.lines[len(ax.err_bars)+1 : ]
    ax.cmap = pyfx.Cmap(DF_MEAN.width_height, pal)
    ax.CM = ax.cmap
    # initialize markers as a neutral color
    _ = [ol.set(mec=ax.err_bars[0].get_c(), mfc='white') for ol in ax.outlines]
    return ax

def plot_channel_events(DF_ALL, DF_MEAN, ax0, ax1, ax2, pal='default', **kwargs):
    """ Plot summary statistics for ripples or DSs on each LFP channel """
    # plot ripple or DS events
    if 'width_height' in DF_ALL.columns : EVENT = 'DS'
    elif 'freq' in DF_ALL.columns       : EVENT = 'Ripple'
    channels = np.array(DF_MEAN.ch)
    linewidth = kwargs.get('lw', 3.5)
    markersize = kwargs.get('ms', 5)
    
    swr_markersize = kwargs.get('swr_ms', 50)
    swr_markeredgewidth = kwargs.get('swr_mew', 2)
    
    # set default palette
    if pal == 'default':
        pal = sns.cubehelix_palette(dark=0.2, light=0.9, rot=0.4, as_cmap=True)
    
    # plot number of events
    if ax0 is not None:
        _ = ax0.bar(DF_MEAN.ch, DF_MEAN.n_valid, lw=1, color=pyfx.Cmap(DF_MEAN.n_valid, pal))
        ax0.set(xlabel='Channel', ylabel='# events', xmargin=0.05)
        ax0.set_title(f'{EVENT} count', fontdict=dict(fontweight='bold'))
        ax0.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    
    # plot event amplitude
    if ax1 is not None:
        _ = sns.stripplot(DF_ALL, x='ch', y='amp', hue='amp', size=markersize, 
                          palette=pal, legend=False, ax=ax1)
        ax1.set(xlabel='Channel', ylabel='Amplitude', xmargin=0.05)
        ax1.set_title(f'{EVENT} amplitude', fontdict=dict(fontweight='bold'))
        ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    
    if EVENT == 'DS' and ax2 is not None:
        # get standard error for channel half-width heights
        sem = DF_ALL[['ch','width_height']].groupby('ch').agg('sem')
        sem = replace_missing_channels(sem, channels)
        d,yerr = np.array([DF_MEAN.width_height.values, sem.width_height.values])
        clrs = pyfx.Cmap(d, pal, use_alpha=True)
        # plot summary data
        _ = ax2.vlines(DF_MEAN.ch, d-yerr, d+yerr, lw=3.5, zorder=-1, colors=clrs)
        _ = ax2.scatter(DF_MEAN.ch, d, ec=clrs, fc='white', s=75, lw=3, zorder=0)
        _ = ax2.scatter(DF_MEAN.ch, d, ec=clrs, fc=clrs*[1,1,1,0.2], s=75, lw=3, zorder=1)
        ax2.set(xlabel='Channel', ylabel='prominence / 2', xmargin=0.05)
        ax2.set_title('DS height above surround', fontdict=dict(fontweight='bold'))
        ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
       
    elif EVENT == 'Ripple' and ax2 is not None:
        tmp = d0,d1 = DF_MEAN[['norm_swr','norm_theta']].values.T
        _ = ax2.scatter(DF_MEAN.ch, d0, fc='w', ec='g', s=swr_markersize, 
                        lw=swr_markeredgewidth, label='ripple power')
        _ = ax2.scatter(DF_MEAN.ch, d1, fc='w', ec='b', s=swr_markersize, 
                        lw=swr_markeredgewidth, label='theta power')
        _ = ax2.vlines(DF_MEAN.ch, *np.sort(tmp.T).T, lw=linewidth, zorder=0, 
                       colors=pyfx.Cmap(d0-d1, pal))
        _ = ax2.legend(frameon=False)
        ax2.set(xlabel='Channel', ylabel='Norm. power', xmargin=0.05)
        ax2.set_title('Ripple/theta power', fontdict=dict(fontweight='bold'))
        ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    sns.despine()
    
    return (ax0,ax1,ax2)

def new_plot_channel_events(DF_ALL, DF_MEAN, ax0, ax1, ax2, pal='default', 
                            noise_train=None, exclude_noise=False, CHC=None):
    """ Plot summary statistics for ripples or DSs on each channel (in progress) """
    channels = np.array(DF_MEAN.ch)
    if CHC is None:
        CHC = pd.Series(pyfx.rand_hex(len(channels)))
        
    # plot ripple or DS events
    if 'width_height' in DF_ALL.columns : EVENT = 'DS'
    elif 'freq' in DF_ALL.columns       : EVENT = 'Ripple'
    
    # set default palette
    if pal == 'default':
        pal = sns.cubehelix_palette(dark=0.2, light=0.9, rot=0.4, as_cmap=True)
    if noise_train is None:
        noise_train = np.zeros(len(channels), dtype='int')
    noise_train = noise_train.astype('bool')
    noise_train_ev = np.in1d(DF_ALL.ch, np.nonzero(noise_train)[0]).astype('bool')
    
    # plot number of events per channel
    ax0 = plot_num_events(DF_MEAN, ax0, pal)
    ax0.cmapNE = pyfx.Cmap(DF_MEAN.n_valid.mask(noise_train), pal)
    if exclude_noise: ax0.CM = ax0.cmapNE
    ax0.set(xlabel='Channel', ylabel='# events', xmargin=0.05)
    ax0.set_title(f'{EVENT} count', fontdict=dict(fontweight='bold'))
    
    # plot event amplitudes
    ax1 = plot_event_amps(DF_ALL, DF_MEAN, ax1, pal)
    bounds = pyfx.MinMax(DF_ALL.amp.mask(noise_train_ev))
    fx = lambda coll: coll.get_offsets().data[:,1]
    ax1.cmapNE = [pyfx.Cmap(fx(coll), pal, bounds) for coll in ax1.collections]
    if exclude_noise: ax1.CM = ax1.cmapNE
    ax1.set(xlabel='Channel', ylabel='Amplitude', xmargin=0.05)
    ax1.set_title(f'{EVENT} amplitude', fontdict=dict(fontweight='bold'))
    
    if EVENT == 'DS':
        ax2 = plot_ds_width_height(DF_ALL, DF_MEAN, ax2, pal)
        ax2.cmapNE = pyfx.Cmap(DF_MEAN.width_height.mask(noise_train), pal)
        if exclude_noise: ax2.CM = ax2.cmapNE
        ax2.set(xlabel='Channel', ylabel='prominence / 2', xmargin=0.05)
        ax2.set_title('DS height above surround', fontdict=dict(fontweight='bold'))
        
    elif EVENT == 'Ripple':
        # plot theta and ripple power for all channels
        tmp = d0,d1 = DF_MEAN[['norm_swr','norm_theta']].values.T
        _ = ax2.scatter(DF_MEAN.ch, d0, fc='w', ec='g', s=50, lw=2, label='ripple power')
        _ = ax2.scatter(DF_MEAN.ch, d1, fc='w', ec='b', s=50, lw=2, label='theta power')
        _ = ax2.vlines(DF_MEAN.ch, *np.sort(tmp.T).T, lw=3, zorder=0, 
                       colors=pyfx.Cmap(d0-d1, pal))
        _ = ax2.legend(frameon=False)
        ax2.set(xlabel='Channel', ylabel='Norm. power', xmargin=0.05)
        ax2.set_title('Ripple/theta power', fontdict=dict(fontweight='bold'))
        
    kw = dict(lw=5, alpha=0.7, zorder=-5)
    ax2.ch_vlines = [ax2.axvline(ch,**kw) for ch in channels]
    _ = [vl.set_visible(False) for vl in ax2.ch_vlines]
    
    ax0.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    sns.despine()
    # set initial colormaps
    _ = [bar.set(color=c) for bar,c in zip(ax0.ch_bars, ax0.CM)]
    _ = [coll.set_fc(pyfx.Cmap_alpha(cm, 0.5)) for coll,cm in zip(ax1.ch_collections, ax1.CM)]
    _ = [coll.set(lw=0.1, ec='lightgray') for coll in ax1.ch_collections]
    if EVENT == 'DS':
        _ = [err.set(color=c) for err,c in zip(ax2.err_bars, ax2.CM)]
        _ = [ol.set(mec=c, mfc=pyfx.alpha_like(c)) for ol,c in zip(ax2.outlines, ax2.CM)]
    
    if exclude_noise:
        _ = [ax0.ch_bars[i] for i in np.nonzero(noise_train)[0]]
        _ = [ax1.ch_collections[i] for i in np.nonzero(noise_train)[0]]
        if EVENT == 'DS':
            _ = [ax2.err_bars[i] for i in np.nonzero(noise_train)[0]]
    
    return (ax0,ax1,ax2)


##############################################################################
##############################################################################
################                                              ################
################               INTERACTIVE PLOTS              ################
################                                              ################
##############################################################################
##############################################################################


def plot_signals(t, ddict, fs, twin=4, step_perc=0.25, **kwargs):
    """
    Show timecourse data on interactive Matplotlib plot with scrollable x-axis
    @Params
    t - time vector (x-axis)
    ddict - dictionary of labeled data vectors
    fs - sampling rate (Hz) of data signals
    twin - time window (s) to show in plot
    step_perc - size of each slider step, as a percentage of $twin
    **kwargs - t_init       : initialize slider at given timepoint (default = minimum t-value)
               hide         : list of data signal(s) to exclude from plot
               plot_nonzero : list of data signal(s) for which to plot nonzero values only
               
               color      : default color for all data signals (if $colordict not given)
               colordict  : dictionary matching data signal(s) with specific colors
               OTHER STYLE PROPERTIES: * lw, lwdict (linewidths)
                                       * ls, lsdict (linestyles)
                                       * mkr, mkrdict (marker shapes)
    @Returns
    fig, ax, slider - Matplotlib figure, data axes, and slider widget
    """
    if isinstance(ddict, np.ndarray):
        if ddict.ndim==1:
            ddict = dict(data=np.array(ddict))
        elif ddict.ndim==2:
            ddict = {chr(ia):ddict[i] for i,ia in enumerate(range(ord('A'), ord('A')+len(ddict)))}
            
    # clean keyword arguments
    t_init     = kwargs.get('t_init', None)   # initial plot timepoint
    hide       = kwargs.get('hide', [])       # hidden data items
    title      = kwargs.get('title', '')
    
    # get dictionary of visible data
    data_dict = {k:v for k,v in ddict.items() if k not in hide}
    
    # set up Matplotlib style properties, set y-axis limits
    props = pd.Series()
    for k,v in zip(['color','lw','ls','mkr'], [None,None,'-',None]):
        dflt_dict = dict.fromkeys(data_dict.keys(), kwargs.get(k,v))
        props[k] = {**dflt_dict, **kwargs.get(k + 'dict', {})}
    ylim = pyfx.Limit(np.concatenate(list(data_dict.values())), pad=0.05)
    
    # get number of samples in plot window / per slider step
    iwin = int(round(twin/2*fs))
    istep = int(round(iwin/4))
    tpad = twin*0.05/2
    # get initial slider value
    if t_init is None : val_init = iwin
    else              : val_init = pyfx.IdxClosest(t_init, t)
    
    # create Matplotlib figure and axes, create slider widget
    fig, (sax0,ax) = plt.subplots(nrows=2, height_ratios=[1,9])
    slider = matplotlib.widgets.Slider(ax=sax0, label='', valmin=iwin, valmax=len(t)-iwin-1, 
                                       valstep=istep, initcolor='none')
    slider.valtext.set_visible(False)
    
    # create data items
    line_dict = {}
    for lbl,data in ddict.items():
        if lbl not in hide:
            #line = ax.plot([0,0], [0,0], color=cdict[lbl], marker=mdict[lbl], label=lbl)[0]
            line = ax.plot([0,0], [0,0], color=props.color[lbl], marker=props.mkr[lbl], 
                           linewidth=props.lw[lbl], linestyle=props.ls[lbl], label=lbl)[0]
            line_dict[lbl] = line
    # set axis limits and legend
    ax.set_ylim(ylim)
    ax.set_title(title)
    leg = ax.legend()
    leg.set_draggable(True)
    sns.despine()
    
    def plot(i):
        """ Update each data item for current time window """
        x = t[i-iwin : i+iwin]
        for lbl,data in data_dict.items():
            line_dict[lbl].set_data(x, data[i-iwin : i+iwin])
        ax.set_xlim([x[0]-tpad, x[-1]+tpad])
        fig.canvas.draw_idle()
        
    def on_press(event):
        """ Scroll backward/forward using left/right arrow keys """
        if event.key == 'left': 
            slider.set_val(max(slider.val - istep, slider.valmin))
        elif event.key == 'right': 
            slider.set_val(min(slider.val + istep, slider.valmax))
    fig.canvas.mpl_connect("key_press_event", on_press)
    
    # connect slider to plot function, plot initial value
    slider.on_changed(plot)
    slider.set_val(val_init)
    
    return fig, ax, slider
