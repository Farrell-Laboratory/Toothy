#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 04:00:25 2024

@author: amandaschott
"""
import os
from pathlib import Path
import scipy.io as so
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pickle
import quantities as pq
from PyQt5 import QtWidgets, QtCore
from open_ephys.analysis import Session
import probeinterface as prif
import pdb
# custom modules
import pyfx
import icsd


##################################################
########          FILE MANAGEMENT         ########
##################################################


def base_dirs(return_keys=False):
    """ Return default data directories saved in default_folders.txt """
    # Mode 0 for paths only, 1 for keys only
    with open('default_folders.txt', 'r') as fid:
        keys,vals = zip(*[map(str.strip, l.split('=')) for l in fid.readlines()])
    if not return_keys:
        return list(vals)
    return list(zip(keys,vals))

def write_base_dirs(ddir_list):
    """ Save input directories to default_folders.txt """
    assert len(ddir_list) == 5
    with open('default_folders.txt', 'w') as fid:
        for k,path in zip(['RAW_DATA','PROCESSED_DATA','PROBE_FILES', 'DEFAULT_PROBE', 'DEFAULT_PARAMETERS'], ddir_list):
            fid.write(k + ' = ' + str(path) + '\n')

def lines2dict(lines):
    """ Parse list of text strings ($lines) and return parameter dictionary """
    PARAMS = {}
    for line in lines:
        d = line.split(';')[0]
        k,v = [x.strip() for x in d.split('=')]
        if v.startswith('[') and v.endswith(']'):
            val = [float(x.strip()) for x in v[1:-1].split(',')]
        else:
            try    : val = float(v)
            except : val = str(v)
        if   val == 'True'  : val = True
        elif val == 'False' : val = False
        PARAMS[k] = val
    return PARAMS

def get_original_defaults():
    PARAMS = {
             'lfp_fs' : 1000.0,
             'theta' : [6.0, 10.0],
             'slow_gamma' : [25.0, 55.0],
             'fast_gamma' : [60.0, 100.0],
             'ds_freq' : [5.0, 100.0],
             'ds_height_thr' : 4.5,
             'ds_dist_thr' : 100.0,
             'ds_prom_thr' : 0.0,
             'ds_wlen' : 125.0,
             'swr_freq' : [120.0, 180.0],
             'swr_ch_bound' : 5.0,
             'swr_height_thr' : 5.0,
             'swr_min_thr' : 3.0,
             'swr_dist_thr' : 100.0,
             'swr_min_dur' : 25.0,
             'swr_freq_thr' : 125.0,
             'swr_freq_win' : 8.0,
             'swr_maxamp_win' : 40.0,
             'csd_method' : 'standard',
             'f_type' : 'gaussian',
             'f_order' : 3.0,
             'f_sigma' : 1.0,
             'vaknin_el' : True, 
             'tol' : 1e-06,
             'spline_nsteps' : 200.0,
             'src_diam' : 0.5,
             'src_h' : 0.1,
             'cond' : 0.3,
             'cond_top' : 0.3,
             'clus_algo' : 'kmeans',
             'nclusters' : 2.0,
             'eps' : 0.2,
             'min_clus_samples' : 3.0,
             'el_w' : 15.0,
             'el_h' : 15.0,
             'el_shape' : 'circle'
             }
    return PARAMS
    

def validate_params(ddict):
    """ Determine whether input $ddict is a valid parameter dictionary
    @Returns
    is_valid - boolean value (True if all critical parameters are valid, False if not)
    valid_ddict - dictionary with validation result for each critical parameter
    """
    
    def is_number(key): # numerical parameter value
        return bool(key in ddict and type(ddict[key]) in [float, int])
    
    def is_range(key):  # list of two parameter values
        return bool(key in ddict and isinstance(ddict[key], list) and len(ddict[key])==2)
    
    def is_category(key, options):  # categorical (string) parameter value
        return bool(key in ddict and ddict[key] in options)
        
    # make sure each parameter 1) is present in dictionary and 2) has a valid value
    valid_ddict = {
                  'lfp_fs' : is_number('lfp_fs'),
                  'theta' : is_range('theta'),
                  'slow_gamma' : is_range('slow_gamma'),
                  'fast_gamma' : is_range('fast_gamma'),
                  'ds_freq' : is_range('ds_freq'),
                  'swr_freq' : is_range('swr_freq'),
                  'ds_height_thr' : is_number('ds_height_thr'),
                  'ds_dist_thr' : is_number('ds_dist_thr'),
                  'ds_prom_thr' : is_number('ds_prom_thr'),
                  'ds_wlen' : is_number('ds_wlen'),
                  'swr_ch_bound' : is_number('swr_ch_bound'),
                  'swr_height_thr' : is_number('swr_height_thr'),
                  'swr_min_thr' : is_number('swr_min_thr'),
                  'swr_dist_thr' : is_number('swr_dist_thr'),
                  'swr_min_dur' : is_number('swr_min_dur'),
                  'swr_freq_thr' : is_number('swr_freq_thr'),
                  'swr_freq_win' : is_number('swr_freq_win'),
                  'swr_maxamp_win' : is_number('swr_maxamp_win'),
                  'csd_method' : is_category('csd_method', ['standard','delta','step','spline']), 
                  'f_type' : is_category('f_type', ['gaussian','identity','boxcar','hamming','triangular']),
                  'f_order' : is_number('f_order'),
                  'f_sigma' : is_number('f_sigma'),
                  'vaknin_el' : is_category('vaknin_el', [True, False]), 
                  'tol' : is_number('tol'),
                  'spline_nsteps' : is_number('spline_nsteps'),
                  'src_diam' : is_number('src_diam'),
                  'src_h' : is_number('src_h'),
                  'cond' : is_number('cond'),
                  'cond_top' : is_number('cond_top'),
                  'clus_algo' : is_category('clus_algo', ['kmeans','dbscan']),
                  'nclusters' : is_number('nclusters'),
                  'eps' : is_number('eps'),
                  'min_clus_samples' : is_number('min_clus_samples'),
                  'el_w' : is_number('el_w'),
                  'el_h' : is_number('el_h'),
                  'el_shape' : is_category('el_shape', ['circle', 'square', 'rectangle'])
                  }
    is_valid = bool(all(valid_ddict.values()))
    return is_valid, valid_ddict
    

def read_param_file(filepath='default_params.txt', exclude_fs=False):
    """ Return dictionary of parameters loaded from .txt file """
    if not os.path.exists(filepath):
        return None, []
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    lines = [l for l in lines if not l.startswith('#') and len(l) > 0]
    # convert text lines to parameter dictionary, validate param values
    PARAMS = lines2dict(lines)
    is_valid, valid_ddict = validate_params(PARAMS)
    if is_valid:
        if exclude_fs: _ = PARAMS.pop('lfp_fs')
        return PARAMS, []  # return parameter dictionary and empty list
    else:
        invalid_params = [k for k,v in valid_ddict.items() if v==False]
        return None, invalid_params  # return None and list of invalid parameters

def write_param_file(PARAMS, filepath='default_params.txt'):
    """ Save parameter dictionary to .txt file """
    fid = open(filepath, 'w')
    fid.write('###  PARAMETERS  ###' + os.linesep)
    fid.write(os.linesep)
    for k,v in PARAMS.items():
        fid.write(f'{k} = {v};' + os.linesep)
    fid.close()

def read_notes(filepath):
    """ Load any recording notes from .txt file """
    try:
        with open(filepath, 'r') as fid:
            notes_txt = fid.read()
    except: 
        notes_txt = ''
    return notes_txt

def write_notes(filepath, txt):
    """ Write recording note to .txt file """
    with open(filepath, 'w') as fid:
        fid.write(str(txt))
    
def load_recording_info(ddir):
    """ Load info dictionary from processed recording """
    INFO = pd.Series(pickle.load(open(Path(ddir, 'info.pkl'), 'rb')))
    return INFO

def load_recording_params(ddir):
    """ Load param dictionary from processed recording """
    PARAMS = pd.Series(pickle.load(open(Path(ddir, 'params.pkl'), 'rb')))
    return PARAMS

def save_recording_info(ddir, INFO):
    """ Save recording metadata to processed data folder """
    with open(Path(ddir, 'info.pkl'), 'wb') as f:
        pickle.dump(INFO, f)

def save_recording_params(ddir, PARAMS):
    """ Save param values used to analyze processed data """
    PARAMS = dict(PARAMS)
    if 'RAW_DATA_FOLDER' in PARAMS.keys()       : del PARAMS['RAW_DATA_FOLDER']
    if 'PROCESSED_DATA_FOLDER' in PARAMS.keys() : del PARAMS['PROCESSED_DATA_FOLDER']
    with open(Path(ddir, 'params.pkl'), 'wb') as f:
        pickle.dump(PARAMS, f)
        

def get_openephys_session(ddir):
    """ Return top-level Session object of recording directory $ddir """
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
            probe = prif.io.read_probeinterface(fpath).probes[0]
        elif ext == '.prb':
            probe = prif.io.read_prb(fpath).probes[0]
        elif ext == '.mat':
            probe = mat2probe(fpath)
        # keep probe name consistent with the file name
        probe.name = os.path.splitext(os.path.basename(fpath))[0].replace('_config','')
    except:
        probe = None
        if raise_exception:
            raise Exception('Invalid probe file')
    return probe

        
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

        
def write_probe_file(probe, fpath):
    """ Write probe configuration to .json, .mat, .prb, or .csv file """
    ext = os.path.splitext(fpath)[-1]
    
    if ext == '.json':   # best for probeinterface
        prif.io.write_probeinterface(fpath, probe)
        
    elif ext == '.prb':  # loses a bunch of data, but required by some systems
        probegroup = prif.ProbeGroup()
        probegroup.add_probe(probe)
        prif.io.write_prb(fpath, probegroup)
        
    elif ext == '.mat':  # preserves data, not automatically handled by probeinterface
        _ = probe2mat(probe, fpath)
        
    elif ext == '.csv':  # straightforward, easy to view (TBD)
        probe.to_dataframe(complete=True)
        return False
    return True


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


def read_array_file(fpath, raise_exception=False):
    """ Load raw data from .npy, .mat, or .csv file """
    if not os.path.exists(fpath):
        if raise_exception:
            raise Exception('Data file does not exist')
        return
    
    def choose_mat_key(keys):
        pyfx.qapp()
        lbl = QtWidgets.QLabel('The following keys represent 2-dimensional data arrays.<br>'
                               'Which one corresponds to the raw LFP signals?')
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        qlist = QtWidgets.QListWidget()
        qlist.addItems(list(data_dict.keys()))
        qlist.setFocusPolicy(QtCore.Qt.NoFocus)
        qlist.setStyleSheet('QListWidget {'
                            'border : 4px solid lightgray;'
                            'border-style : double;'
                            'selection-color : white;}'
                            'QListWidget::item {'
                            'border : none;'
                            'border-bottom : 2px solid lightgray;'
                            'background-color : white;'
                            'padding : 4px;}'
                            'QListWidget::item:selected {'
                            'background-color : blue;}')
        go_btn = QtWidgets.QPushButton('Continue')
        go_btn.setEnabled(False)
        dlg = QtWidgets.QDialog()
        dlg.setStyleSheet('QWidget {font-size : 15pt;}')
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(lbl)
        lay.addWidget(qlist)
        lay.addWidget(go_btn)
        qlist.itemSelectionChanged.connect(lambda: go_btn.setEnabled(len(qlist.selectedItems())==1))
        go_btn.clicked.connect(dlg.accept)
        res = dlg.exec()
        if res : return qlist.selectedItems()[0].text()
        else   : return None
    
    # load data according to file extension
    ext = os.path.splitext(fpath)[-1]
    try:
        if ext == '.npy':
            data = np.load(fpath)
        elif ext == '.mat':
            matfile = so.loadmat(fpath, squeeze_me=True)
            # find the real data
            data_dict = {}
            for k,v in matfile.items():
                if k.startswith('__'): continue
                if not hasattr(v, '__iter__'): continue
                if len(v) == 0: continue
                if np.array(v).ndim != 2: continue
                data_dict[k] = v
            if len(data_dict) == 1:
                data = list(data_dict.values())[0]
            elif len(data_dict) > 1:
                key = choose_mat_key(list(data_dict.keys()))
                if key is not None:
                    data = data_dict[key] # user selected key
                else:
                    data = np.array([])   # user exited popup
            elif len(data_dict) == 0:
                data = np.array([])
                #raise Exception('No 2-dimensional arrays found in the dataset')
        # make sure data is 2-dimensional
        if data.ndim != 2:
            raise Exception('Data must be a 2-dimensional array')
    except:
        data = None
        if raise_exception:
            raise Exception('Invalid data file')
    return data


def load_lfp(ddir, key='', iprb=-1):
    """ Load LFP signals, timestamps, and sampling rate """
    DATA = load_bp(ddir, key=key, iprb=iprb)
    lfp_time = np.load(Path(ddir, 'lfp_time.npy'))
    lfp_fs = int(np.load(Path(ddir, 'lfp_fs.npy')))
    return DATA, lfp_time, lfp_fs

def load_aux(ddir):
    import re
    aux_files = [f for f in os.listdir(ddir) if re.match('AUX\d+.npy', f)]
    aux_array = np.array([np.load(Path(ddir, f)) for f in aux_files])
    return aux_array
    

def load_iis(ddir, iprb):
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


def load_ds_csd(ddir, iprb):
    """ Load or create dictionary of CSD data for probe $iprb """
    # load DS CSDs
    if os.path.exists(Path(ddir, f'ds_csd_{iprb}.npz')):
        with np.load(Path(ddir,f'ds_csd_{iprb}.npz'), allow_pickle=True) as npz:
            csd_dict = dict(npz)
    else:
        csd_dict = dict.fromkeys(['raw_csd','filt_csd','norm_filt_csd','csd_chs'])
    return csd_dict
    

def load_bp(ddir, key='', iprb=-1):
    """ Load bandpass-filtered LFP data for 1 or more probes """
    nprobes = len(prif.io.read_probeinterface(str(Path(ddir, 'probe_group'))).probes)
    #nprobes = load_recording_info(ddir).nprobes
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


def load_noise_channels(ddir, iprb=-1):
    """ Load or create channel noise annotations (0=clean, 1=noisy) for 1 or more probes """
    if os.path.exists(Path(ddir, 'noise_channels.npy')):
        # load noise trains, listed by probe
        noise_list = list(np.load(Path(ddir, 'noise_channels.npy')))
    else:
        # initialize noise trains with zeroes (channels are "clean" by default)
        probes = prif.io.read_probeinterface(str(Path(ddir, 'probe_group'))).probes
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


def load_event_dfs(ddir, event, iprb=-1, mask=False):
    """ Load event dataframes (ripples or DS) for 1 or more probes """
    fname = f'ALL_{event.upper()}'
    if mask:
        fname += '_MASK'
    DFS = list(zip(csv2list(ddir, fname), # event dfs
                   csv2list(ddir, 'channel_bp_std')))      # ch bandpass power
    LLIST = []
    for i,(DF_ALL, STD) in enumerate(DFS):
        if 'status' not in DF_ALL.columns:
            DF_ALL['status'] = 1
        if 'is_valid' not in DF_ALL.columns:
            DF_ALL['is_valid'] = np.array(DF_ALL['status'] > 0, dtype='int')
        #nch = load_recording_info(ddir).probe_nch[i]
        channels = np.arange(len(STD))
        # create 'ch' column from index, merge with bandpass power, add (valid) event counts
        try:
            DF_ALL.insert(0, 'ch', np.array(DF_ALL.index.values))#, dtype='float64'))
        except:
            pdb.set_trace()
        DF_ALL[STD.columns] = STD
        DF_ALL['n_valid'] = DF_ALL.groupby('ch')['is_valid'].agg('sum')
        
        # average values by channel (NaNs for channels with no valid events)
        DF_VALID = pd.DataFrame(DF_ALL[DF_ALL['is_valid'] == 1])
        DF_MEAN = DF_VALID.groupby('ch').agg('mean')
        DF_MEAN = replace_missing_channels(DF_MEAN, channels).astype({'n_valid':int})
        DF_MEAN[STD.columns] = np.array(STD)
        
        DF_MEAN.insert(0, 'ch', DF_MEAN.index.values)
        LLIST.append([DF_ALL, DF_MEAN])
        
    if 0 <= iprb < len(LLIST):
        return LLIST[i]
    else:
        return LLIST


def estimate_theta_chan(STD, noise_idx=np.array([], dtype='int')):
    """ Estimate optimal theta (fissure) channel (max power in theta range) """
    theta_pwr = np.array(STD.theta.values)
    theta_pwr[noise_idx] = np.nan
    return np.nanargmax(theta_pwr)


def estimate_ripple_chan(STD, noise_idx=np.array([], dtype='int')):
    """ Estimate optimal ripple channel (max ripple power among channels with low theta power) """
    arr = STD[['theta','swr']].values.T
    arr[:, noise_idx] = np.nan
    theta_pwr, ripple_pwr = arr
    # convert values above 60th %ile of theta power to NaN
    x = theta_pwr >= np.nanpercentile(theta_pwr, 60)
    ripple_pwr[x] = np.nan
    return np.nanargmax(ripple_pwr)
    

def estimate_hil_chan(SWR_DATA, noise_idx=np.array([], dtype='int')):
    """ Estimate optimal DS (hilus) channel (max positive peaks in ripple freq band) """
    ripple_percentile = np.percentile(SWR_DATA, 99.9, axis=1)
    ripple_percentile[noise_idx] = np.nan
    return np.nanargmax(ripple_percentile)
    

def load_event_channels(ddir, iprb, DATA=None, STD=None, NOISE_TRAIN=None):
    """ Load/estimate optimal theta, ripple, and hilus channels for given probe """
    if os.path.exists(Path(ddir, f'theta_ripple_hil_chan_{iprb}.npy')):
        # load event channels for given probe
        event_channels = np.load(Path(ddir, f'theta_ripple_hil_chan_{iprb}.npy'))
    elif os.path.exists(Path(ddir, 'theta_ripple_hil_chan.npy')):
        # load general event channels
        event_channels = np.load(Path(ddir, 'theta_ripple_hil_chan.npy'))
    else:
        # estimate optimal event channels
        if DATA is None : DATA = load_bp(ddir, '', iprb)
        if STD is None  : STD  = csv2list(ddir, 'channel_bp_std')[iprb]
        if NOISE_TRAIN is None: NOISE_TRAIN = load_noise_channels(ddir, iprb)
        noise_idx = np.nonzero(NOISE_TRAIN)[0]
        theta_chan  = estimate_theta_chan(STD, noise_idx)
        ripple_chan = estimate_ripple_chan(STD, noise_idx)
        hil_chan    = estimate_hil_chan(DATA['swr'], noise_idx)
        event_channels = [theta_chan, ripple_chan, hil_chan]
    return event_channels


##################################################
########         DATA MANIPULATION        ########
##################################################

def get_asym(ipk, istart, istop):
    """ Compute asymmetry of waveform spanning $istart-$iend with peak $ipk """
    i0,i1 = ipk-istart, istop-ipk
    asym = (i1-i0) / min(i0,i1) * 100
    return asym


def getwaves(LFP, iev, iwin):
    """ Get LFP waveforms surrounding the given event indices $iev """
    arr = np.full((len(iev), iwin*2), np.nan)
    for i,idx in enumerate(iev):
        arr[i,:] = pad_lfp(LFP, idx, iwin)
    return arr


def getavg(LFP, iev, iwin):
    """ Get event-averaged LFP waveform """
    return np.nanmean(getwaves(LFP, iev, iwin), axis=0)


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


def replace_missing_channels(DF, channels):
    """ Replace any missing channels in event dataframe with rows of NaNs"""
    if len(DF) == len(channels):
        return DF
    
    # # fill in rows for any channels with no detected events
    # missing_ch = np.setdiff1d(channels, DF.index.values)
    # ddf = pd.concat([DF, pd.DataFrame(dict(ch=missing_ch))], ignore_index=True)
    # ddf = ddf.sort_values('ch').reset_index(drop=True)
    # ddf.loc[missing_ch,'n_valid'] = 0
    # return ddf
    
    # fill in rows for any channels with no detected events
    missing_ch = np.setdiff1d(channels, DF.index.values)
    missing_df = pd.DataFrame(0.0, index=missing_ch, columns=DF.columns)
    DF = pd.concat([DF, missing_df], axis=0, ignore_index=False).sort_index()
    # set missing values to NaN (except for event counts, which are zero)
    DF.loc[missing_ch, [c for c in DF.columns if c!='n_valid']] = np.nan
    return DF


def get_csd_obj(data, coord_electrode, ddict):
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


def encoder2pos(ddir='', A_signal=None, B_signal=None, circumference=44.8, ppr=256):
    """ Get wheel position from rotary encoder A and B signals """
    if A_signal is None or B_signal is None:
        # load A and B signals
        A_path, B_path = Path(ddir, 'A_signal.npy'), Path(ddir, 'B_signal.npy')
        if not (A_path.exists() and B_path.exists()):
            raise Exception('ERROR: Encoder A/B signals not found')
        A_signal, B_signal = np.load(A_path), np.load(B_path)
    # binarize signals
    chA = (A_signal > 0.5) * 1
    chB = (B_signal > 0.5) * 1
    chA_plus_minus = (chA * 2) - 1
    chB_diff = np.concatenate(([0], np.diff(chB)))
    chA_prod = chA_plus_minus * chB_diff
    chB_plus_minus = (chB * 2) - 1
    chA_diff = np.concatenate(([0], np.diff(chA)))
    chB_prod = -chB_plus_minus * chA_diff
    position = np.cumsum(chA_prod + chB_prod)
    # circumfence in cm divided by no. pulses per revolution
    position_cm = position * (circumference/ppr)
    return position_cm


def pos2speed(pos, sf=500):
    """ Translate wheel position to speed with Gaussian smoothing factor $sf """
    pos_dif = np.concatenate((np.array([0]), np.diff(pos)))
    speed_smth = scipy.ndimage.gaussian_filter1d(pos_dif, sf)
    return speed_smth


##################################################
########          EVENT DETECTION         ########
##################################################


def get_inst_freq(x, lfp_fs, swr_freq=[120,180]):
    """ Calculate LFP instantaneous frequency """
    angle  = np.angle(x)       # radian phase (-π to π) of each LFP timepoint
    iphase = np.unwrap(angle)  # phase + 2π*k, where k=cycle number (0-K total cycles)
    difs   = np.diff(iphase)/(2.0*np.pi)  # distance (% of 2π cycle) between consecutive points
    ifreq  = np.clip(difs*lfp_fs, *swr_freq)  # inst. freq (Hz) at each point (bounds=SWR cutoff freqs)
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
    env = np.abs(hilb).astype('float32') # Hilbert absolute value (amplitude of pos/neg peaks)
    std = np.std(env)                        # standard deviation of SWR envelope
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
    thresholds = dict(peak_height=height,  # min. DS peak amplitude
                      isi=ds_min_dist)     # min. distance (s) between DS events
    thresholds = pd.Series(thresholds)
    
    # detect qualifying peaks
    ipks,props = scipy.signal.find_peaks(LFP, height=height, distance=distance, 
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
        ax.set(xlabel='Time (s)', ylabel='$\Sigma$ LFP ampl. (mV)', title=f'Interictal spikes (n={len(seqs)})')
        sns.despine()
    return seqs, thr


##################################################
########           STATIC PLOTS           ########
##################################################


def plot_num_events(DF_MEAN, ax, pal):
    # plot number of valid events for each channel, save colormap (ch x R,G,B)
    _ = sns.barplot(DF_MEAN, x='ch', y='n_valid', order=range(len(DF_MEAN)), 
                    lw=1, ax=ax)
    ax.ch_bars = list(ax.patches)
    ax.cmap = pyfx.Cmap(DF_MEAN.n_valid, pal)
    ax.CM = ax.cmap
    return ax

def plot_event_amps(DF_ALL, DF_MEAN, ax, pal):
    # plot event amplitudes for each channel
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
    ax.CM = ax.cmap
    return ax


def plot_ds_width_height(DF_ALL, DF_MEAN, ax, pal):
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


def new_plot_channel_events(DF_ALL, DF_MEAN, ax0, ax1, ax2, pal='default', 
                            noise_train=None, exclude_noise=False, CHC=None):
    """
    Plot summary statistics for ripples or dentate spikes on each LFP channel
    """
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
        print('did nothing')
        # plot theta and ripple power for all channels
        tmp = d0,d1 = DF_MEAN[['norm_swr','norm_theta']].values.T
        _ = ax2.scatter(DF_MEAN.ch, d0, fc='w', ec='g', s=50, lw=2, label='ripple power')
        _ = ax2.scatter(DF_MEAN.ch, d1, fc='w', ec='b', s=50, lw=2, label='theta power')
        _ = ax2.vlines(DF_MEAN.ch, *np.sort(tmp.T).T, lw=3, zorder=0, colors=pyfx.Cmap(d0-d1, pal))
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
        noise_bars = [ax0.ch_bars[i] for i in np.nonzero(noise_train)[0]]
        noise_colls = [ax1.ch_collections[i] for i in np.nonzero(noise_train)[0]]
        if EVENT == 'DS':
            noise_errs = [ax2.err_bars[i] for i in np.nonzero(noise_train)[0]]
    
    return (ax0,ax1,ax2)
    
    
def plot_channel_events(DF_ALL, DF_MEAN, ax0, ax1, ax2, pal='default', **kwargs):
    """
    Plot summary statistics for ripples or dentate spikes on each LFP channel
    """
    # plot ripple or DS events
    if 'width_height' in DF_ALL.columns : EVENT = 'DS'
    elif 'freq' in DF_ALL.columns       : EVENT = 'Ripple'
    channels = np.array(DF_MEAN.ch)
    
    # set default palette
    if pal == 'default':
        pal = sns.cubehelix_palette(dark=0.2, light=0.9, rot=0.4, as_cmap=True)
    
    # plot number of events
    _ = ax0.bar(DF_MEAN.ch, DF_MEAN.n_valid, lw=1, color=pyfx.Cmap(DF_MEAN.n_valid, pal))
    ax0.set(xlabel='Channel', ylabel='# events', xmargin=0.05)
    ax0.set_title(f'{EVENT} count', fontdict=dict(fontweight='bold'))
    
    # plot event amplitude
    _ = sns.stripplot(DF_ALL, x='ch', y='amp', hue='amp', palette=pal, legend=False, ax=ax1)
    ax1.set(xlabel='Channel', ylabel='Amplitude', xmargin=0.05)
    ax1.set_title(f'{EVENT} amplitude', fontdict=dict(fontweight='bold'))
    
    if EVENT == 'DS':
        # get standard error for channel half-width heights
        sem = DF_ALL[['ch','width_height']].groupby('ch').agg('sem')
        sem = replace_missing_channels(sem, channels)
        d,yerr = np.array([DF_MEAN.width_height.values, sem.width_height.values])
        clrs = pyfx.Cmap(d, pal, use_alpha=True)
        # plot summary data
        try:
            _ = ax2.vlines(DF_MEAN.ch, d-yerr, d+yerr, lw=3.5, zorder=-1, colors=clrs)
            _ = ax2.scatter(DF_MEAN.ch, d, ec=clrs, fc='white', s=75, lw=3, zorder=0)
            _ = ax2.scatter(DF_MEAN.ch, d, ec=clrs, fc=clrs*[1,1,1,0.2], s=75, lw=3, zorder=1)
            ax2.set(xlabel='Channel', ylabel='prominence / 2', xmargin=0.05)
            ax2.set_title('DS height above surround', fontdict=dict(fontweight='bold'))
        except:
            pdb.set_trace()
       
    elif EVENT == 'Ripple':
        # plot theta and ripple power for all channels
        tmp = d0,d1 = DF_MEAN[['norm_swr','norm_theta']].values.T
        _ = ax2.scatter(DF_MEAN.ch, d0, fc='w', ec='g', s=50, lw=2, label='ripple power')
        _ = ax2.scatter(DF_MEAN.ch, d1, fc='w', ec='b', s=50, lw=2, label='theta power')
        _ = ax2.vlines(DF_MEAN.ch, *np.sort(tmp.T).T, lw=3, zorder=0, colors=pyfx.Cmap(d0-d1, pal))
        _ = ax2.legend(frameon=False)
        ax2.set(xlabel='Channel', ylabel='Norm. power', xmargin=0.05)
        ax2.set_title('Ripple/theta power', fontdict=dict(fontweight='bold'))
    
    
         
        fig,ax2 = plt.subplots()
        
        tmp = d0,d1 = DF_MEAN[['norm_swr','norm_theta']].values.T
        _ = ax2.scatter(DF_MEAN.ch, d0, fc='w', ec='g', s=50, lw=2, label='ripple power')
        _ = ax2.scatter(DF_MEAN.ch, d1, fc='w', ec='b', s=50, lw=2, label='theta power')
        _ = ax2.vlines(DF_MEAN.ch, *np.sort(tmp.T).T, lw=3, zorder=0, colors=pyfx.Cmap(d0-d1, pal))
        _ = ax2.legend(frameon=False)
        ax2.set(xlabel='Channel', ylabel='Norm. power', xmargin=0.05)
        ax2.set_title('Ripple/theta power', fontdict=dict(fontweight='bold'))
        
    ax0.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax1.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    
    sns.despine()
    
    return (ax0,ax1,ax2)


##################################################
########          LIVE DATA VIEW          ########
##################################################


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
        ddict = dict(data=np.array(ddict))
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
        # update each data item for current time window
        x = t[i-iwin : i+iwin]
        for lbl,data in data_dict.items():
            line_dict[lbl].set_data(x, data[i-iwin : i+iwin])
        ax.set_xlim([x[0]-tpad, x[-1]+tpad])
        fig.canvas.draw_idle()
        
    # connect slider to plot function, plot initial value
    slider.on_changed(plot)
    slider.set_val(val_init)
    
    return fig, ax, slider
