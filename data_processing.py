#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:00:24 2024

@author: amandaschott
"""
import os
from pathlib import Path
import re
import json
import pickle
import numpy as np
import pandas as pd
import probeinterface as prif
import quantities as pq
import warnings
import pdb
# custom modules
import pyfx
import ephys
import gui_items as gi


def validate_raw_ddir(ddir):
    """ Check whether directory contains raw Open Ephys or NeuroNexus data """
    if not os.path.exists(ddir):
        return False
    files = os.listdir(ddir)
    a = bool('structure.oebin' in files)
    b = bool(len([f for f in files if f.endswith('.xdat.json')]) > 0)
    return bool(a or b)


def bp_filter_lfps(lfp, lfp_fs, **kwargs):
    """ Bandpass filter LFP signals within fixed frequency bands """
    # set filter cutoffs
    theta      = kwargs.get('theta',      [6,10])
    slow_gamma = kwargs.get('slow_gamma', [25,55])
    fast_gamma = kwargs.get('fast_gamma', [60,100])
    swr_freq   = kwargs.get('swr_freq',   [120,180])
    ds_freq    = kwargs.get('ds_freq',    [5,100])
    
    # collect filtered LFPs in data dictionary
    bp_dict = {'raw' : lfp}
    bp_dict['theta']      = pyfx.butter_bandpass_filter(lfp, *theta,      lfp_fs=lfp_fs, axis=1)
    bp_dict['slow_gamma'] = pyfx.butter_bandpass_filter(lfp, *slow_gamma, lfp_fs=lfp_fs, axis=1)
    bp_dict['fast_gamma'] = pyfx.butter_bandpass_filter(lfp, *fast_gamma, lfp_fs=lfp_fs, axis=1)
    bp_dict['swr']        = pyfx.butter_bandpass_filter(lfp, *swr_freq,   lfp_fs=lfp_fs, axis=1)
    bp_dict['ds']         = pyfx.butter_bandpass_filter(lfp, *ds_freq,    lfp_fs=lfp_fs, axis=1)
    
    return bp_dict


def load_openephys_data(ddir):
    """ Load raw data files from Open Ephys recording software """
    # initialize Open Ephys data objects
    session = ephys.get_openephys_session(ddir)
    OE = ephys.oeNodes(session, ddir)
    node = OE['node']            # node containing the selected recording
    recording = OE['recording']  # experimental recording object
    continuous_list = OE['continuous']  # continuous data from 1 or more processors
    #settings_file = str(Path(node.directory, 'settings.xml'))
    metadata_list = recording.info['continuous']
    
    continuous, meta = continuous_list[0], metadata_list[0]
    if len(continuous_list) > 1:
        print(f'WARNING: Data extracted from 1 of {len(continuous_list)} existing processors')
    
    # load sampling rate, timestamps, and number of channels
    fs = meta['sample_rate']
    num_channels = meta['num_channels']
    tstart, tend = pyfx.Edges(continuous.timestamps)
    
    # load channel names and bit volt conversions, find primary channels
    ch_names, bit_volts = zip(*[(d['channel_name'], d['bit_volts']) for d in meta['channels']])
    ipri = np.nonzero([*map(lambda n: bool(re.match('C\d+', n)), ch_names)])[0]
    if len(ipri) > 0:
        try:
            units = meta['channels'][ipri[0]]['units']
        except:
            pdb.set_trace()
    else:
        units = None
    iaux = np.nonzero([*map(lambda n: bool(re.match('ADC\d+', n)), ch_names)])[0]
    if ipri.size == 0 and iaux == 0:
        return
    
    # load raw signals (uV)
    first,last = pyfx.Edges(continuous.sample_numbers)
    raw_signal_array = np.array([x*bv for x,bv in zip(continuous.get_samples(0, last-first+1).T, bit_volts)])
    if ipri.size > 0:
        A,B = ipri[[0,-1]] + [0,1]
        pri_mx = raw_signal_array[A:B]
    else: pri_mx = np.array([])
    # look for aux channels
    if iaux.size > 0:
        A_AUX, B_AUX = iaux[[0,-1]] + [0,1]
        aux_mx = raw_signal_array[A_AUX : B_AUX]
    else: aux_mx = np.array([])
    
    return (pri_mx, aux_mx), fs, units


def load_neuronexus_data(ddir):
    """ Load raw data files from Allego NeuroNexus recording software """
    # get raw file names
    meta_file = [f for f in os.listdir(ddir) if f.endswith('.xdat.json')][0]
    stem = meta_file.replace('.xdat.json', '')
    data_file = os.path.join(ddir, stem + '_data.xdat')
    
    # load metadata
    with open(os.path.join(ddir, meta_file), 'rb') as f:
        metadata = json.load(f)
    fs             = metadata['status']['samp_freq']
    num_channels   = metadata['status']['signals']['pri']
    total_channels = metadata['status']['signals']['total']
    tstart, tend   = metadata['status']['t_range']
    num_samples    = int(round(tend * fs)) - int(round(tstart * fs))
    # get SI units
    udict = {'micro-volts':'uV', 'milli-volts':'mV', 'volts':'V'}
    units = metadata['sapiens_base']['sigUnits']['sig_units_pri']
    units = udict.get(units, units)
    
    # organize electrode channels by port
    ports,ddicts = map(list, zip(*metadata['sapiens_base']['sensors_by_port'].items()))
    nprobes = len(ports)
    probe_nch = [d['num_channels'] for d in ddicts]
    
    # separate primary and aux channels
    ch_names = metadata['sapiens_base']['biointerface_map']['chan_name']
    ipri = np.array([i for i,n in enumerate(ch_names) if n.startswith('pri')])
    A,B = ipri[[0,-1]] + [0,1]
    
    # load raw probe data
    with open(data_file, 'rb') as fid:
        fid.seek(0, os.SEEK_SET)
        raw_signals = np.fromfile(fid, dtype=np.float32, count=num_samples*total_channels)
    raw_signal_array = np.reshape(raw_signals, (num_samples, total_channels)).T #[a:b+1]#[ipri]#[0:num_channels]
    pri_mx = raw_signal_array[A:B]
    # look for aux channels
    iaux = np.array([i for i,n in enumerate(ch_names) if n.startswith('aux')])
    if iaux.size > 0:
        A_AUX, B_AUX = iaux[[0,-1]] + [0,1]
        aux_mx = raw_signal_array[A_AUX : B_AUX]
    else: aux_mx = np.array([])
    
    return (pri_mx, aux_mx), fs, units#, info


def load_ncs_file(file_path):
    # make sure .ncs file exists
    assert os.path.isfile(file_path) and file_path.endswith('.ncs')
    
    HEADER_LENGTH = 16 * 1024  # 16 kilobytes of header
    NCS_SAMPLES_PER_RECORD = 512
    NCS_RECORD = np.dtype([('TimeStamp',       np.uint64),       # Cheetah timestamp for this record. This corresponds to
                                                                 # the sample time for the first data point in the Samples
                                                                 # array. This value is in microseconds.
                           ('ChannelNumber',   np.uint32),       # The channel number for this record. This is NOT the A/D
                                                                 # channel number
                           ('SampleFreq',      np.uint32),       # The sampling frequency (Hz) for the data stored in the
                                                                 # Samples Field in this record
                           ('NumValidSamples', np.uint32),       # Number of values in Samples containing valid data
                           ('Samples',         np.int16, NCS_SAMPLES_PER_RECORD)])  # Data points for this record. Cheetah
                                                                                    # currently supports 512 data points per
                                                                                    # record. At this time, the Samples
                                                                                    # array is a [512] array.
    
    def parse_header(raw_header):
        """ Parse Neuralynx file header """
        # decode header as iso-8859-1 (the spec says ASCII, but there is at least one case of 0xB5 in some headers)
        raw_hdr = raw_header.decode('iso-8859-1')
        hdr_lines = [line.strip() for line in raw_hdr.split('\r\n') if line != '']
        # look for line identifiying Neuralynx file
        if hdr_lines[0] != '######## Neuralynx Data File Header':
            warnings.warn('Unexpected start to header: ' + hdr_lines[0])
        # return header information as dictionary
        tmp = [l.split() for l in hdr_lines[1:]]
        tmp = [x + [''] if len(x)==1 else x for x in tmp]
        header = {x[0].replace('-','') : ' '.join(x[1:]) for x in tmp}
        return header
    
    # read in .ncs file
    with open(file_path, 'rb') as fid:
        # Read the raw header data (16 kb) from the file object fid. Restores the position in the file object after reading.
        pos = fid.tell()
        fid.seek(0)
        raw_header = fid.read(HEADER_LENGTH).strip(b'\0')
        records = np.fromfile(fid, NCS_RECORD, count=-1)
        fid.seek(pos)
    header = parse_header(raw_header)
    fs = records['SampleFreq'][0]                   # get sampling rate
    bit_volts = float(header['ADBitVolts']) * 1000  # convert ADC counts to mV

    # load data
    D = np.array(records['Samples'].reshape(-1) * bit_volts, dtype=np.float32)
    ts = np.linspace(0, len(D) / fs, len(D))
    return D, ts, fs


def load_neuralynx_data(ddir, pprint=True, use_array=True, save_array=True):
    """ Load raw data files from Neuralynx recording software """
    # identify and sort all .ncs files in data directory
    flist = np.array([f for f in os.listdir(ddir) if f.endswith('.ncs')])
    fnums = [int(f.strip('CSC').strip('.ncs')) for f in flist]
    fpaths = [str(Path(ddir, f)) for f in flist[np.argsort(fnums)]]
    # get number of total channels, timestamps, and sampling rate
    nch = len(fpaths)
    _, ts, fs = load_ncs_file(fpaths[0])
    if pprint: 
        print(os.linesep + '###   LOADING NEURALYNX DATA   ###' + os.linesep)
        
    data_path = str(Path(ddir, 'DATA_ARRAY.npy'))
    if use_array and os.path.isfile(data_path):
        # load existing array
        if pprint: print('Loading existing DATA_ARRAY.npy file ...')
        pri_mx = np.load(data_path)
    else:
        print_progress = np.round(np.linspace(0, nch-1, 10)).astype('int')
        # initialize data array (channels x timepoints)
        pri_mx = np.empty((nch, len(ts)), dtype=np.float32)
        for i,f in enumerate(fpaths):
            if pprint and (i in print_progress):
                print(f'Loading NCS file {i+1}/{nch} ...')
            pri_mx[i,:] = load_ncs_file(f)[0]
        if save_array:
            print('Saving data array ...')
            np.save(data_path, pri_mx)
    if pprint: print('Done!' + os.linesep)
    aux_mx = np.array([])
    return (pri_mx, aux_mx), fs, 'mV'


def load_raw_data(ddir, pprint=True):
    """ Load raw data files from Open Ephys, NeuroNexus, or Neuralynx software """
    try:
        files = os.listdir(ddir)
    except:
        raise Exception(f'Directory "{ddir}" does not exist')
    xdat_files = [f for f in files if f.endswith('.xdat.json')]
    # load Open Ephys data
    if 'structure.oebin' in files:
        if pprint: print('Loading Open Ephys raw data ...')
        res = load_openephys_data(ddir) # removed info, added fs
        if not res:
            msgbox = gi.MsgboxError.run('Unable to load channels from Open Ephys data')
            return
        (pri_array, aux_array), fs, units = res
    # load NeuroNexus data
    elif len(xdat_files) > 0:
        if pprint: print('Loading NeuroNexus raw data ...')
        (pri_array, aux_array), fs, units = load_neuronexus_data(ddir)
    # load Neuralynx data
    elif len([f for f in files if f.endswith('.ncs')]) > 0:
        (pri_array, aux_array), fs, units = load_neuralynx_data(ddir, pprint=pprint)
    # no valid raw data found
    else:
        raise Exception(f'No raw Open Ephys (.oebin), NeuroNexus (.xdat.json), or Neuralynx (.ncs) files found in directory "{ddir}"')
    return (pri_array, aux_array), fs, units
    

def get_idx_by_probe(probe):
    """ Clean $probe input, return list of channel maps """
    if probe.__class__ == prif.Probe:
        idx_by_probe = [probe.device_channel_indices]
    elif probe.__class__ == prif.ProbeGroup:
        idx_by_probe = [prb.device_channel_indices for prb in probe.probes]
    elif type(probe) in [list, np.ndarray]:
        if type(probe) == list:
            probe = np.array(probe)
        if type(probe) == np.ndarray:
            if probe.ndim == 1:
                idx_by_probe = [probe]
            elif probe.ndim == 2:
                idx_by_probe = [x for x in probe]
    return idx_by_probe
        

def extract_data(raw_signal_array, idx, fs=30000, lfp_fs=1000, units='uV', lfp_units='mV'):
    """ Extract, scale, and downsample each raw signal in depth order down the probe """
    ds_factor = int(fs / lfp_fs)  # calculate downsampling factor
    cf = pq.Quantity(1, units).rescale(lfp_units).magnitude  # mV conversion factor
    lfp = np.array([pyfx.Downsample(raw_signal_array[i], ds_factor)*cf for i in idx])
    return lfp
    
def extract_data_by_probe(raw_signal_array, chMap, fs=30000, lfp_fs=1000, units='uV', lfp_units='mV'):
    """ Get LFP array for each probe represented in $chMap """
    idx_by_probe = get_idx_by_probe(chMap)
    ds_factor = int(fs / lfp_fs)  # calculate downsampling factor
    cf = pq.Quantity(1, units).rescale(lfp_units).magnitude  # uV -> mV conversion factor
    lfp_list = []
    for idx in idx_by_probe:
        lfp = np.array([pyfx.Downsample(raw_signal_array[i], ds_factor)*cf for i in idx])
        lfp_list.append(lfp)
    return lfp_list


def process_probe_data(_lfp, lfp_time, lfp_fs, PARAMS, pprint=True):
    """ Filter LFPs, run ripple and DS detection on each channel """
    
    # bandpass filter LFPs within different frequency bands
    if pprint: print('Bandpass filtering signals ...')    
    bp_dict = bp_filter_lfps(_lfp, lfp_fs, **PARAMS)
    # get standard deviation (raw and normalized) for each filtered signal
    std_dict = {k : np.std(v, axis=1) for k,v in bp_dict.items()}
    std_dict.update({f'norm_{k}' : pyfx.Normalize(v) for k,v in std_dict.items()})
    STD = pd.DataFrame(std_dict)
    
    # run ripple detection on all channels
    SWR_DF = pd.DataFrame()
    SWR_THRES = {}
    if pprint: print('Detecting ripples on each channel ...')
    for ch in range(_lfp.shape[0]):
        # sharp-wave ripples
        swr_df, swr_thres = ephys.get_swr_peaks(bp_dict['swr'][ch], lfp_time, lfp_fs, 
                                                pprint=False, **PARAMS)
        swr_df.set_index(np.repeat(ch, len(swr_df)), inplace=True)
        SWR_DF = pd.concat([SWR_DF, swr_df], ignore_index=False)
        SWR_THRES[ch] = swr_thres
    
    # run DS detection on all channels
    DS_DF = pd.DataFrame()
    DS_THRES = {}
    if pprint: print('Detecting dentate spikes on each channel ...')
    for ch in range(_lfp.shape[0]):
        # dentate spikes
        ds_df, ds_thres = ephys.get_ds_peaks(bp_dict['ds'][ch], lfp_time, lfp_fs, 
                                             pprint=False, **PARAMS)
        ds_df.set_index(np.repeat(ch, len(ds_df)), inplace=True)
        DS_DF = pd.concat([DS_DF, ds_df], ignore_index=False)
        DS_THRES[ch] = ds_thres
    THRESHOLDS = dict(SWR=SWR_THRES, DS=DS_THRES)
    
    return bp_dict, STD, SWR_DF, DS_DF, THRESHOLDS


def process_all_probes(lfp_list, lfp_time, lfp_fs, PARAMS, save_ddir, pprint=True):
    """
    Process LFPs for each probe in dataset, save to new data folder
    """
    if type(lfp_list) == np.ndarray:
        lfp_list = [lfp_list]
    bp_dicts = {'raw':[], 'theta':[], 'slow_gamma':[], 'fast_gamma':[], 'swr':[], 'ds':[]}
    std_dfs, swr_dfs, ds_dfs, thresholds, noise_trains = [], [], [], [], []
    
    for i,_lfp in enumerate(lfp_list):
        if pprint: print(f'\n#####   PROBE {i+1} / {len(lfp_list)}   #####\n')
        bp_dict, STD, SWR_DF, DS_DF, THRESHOLDS = process_probe_data(_lfp, lfp_time, lfp_fs, 
                                                                     PARAMS, pprint=pprint)
        for k,l in bp_dicts.items(): l.append(bp_dict[k])
        std_dfs.append(STD)
        swr_dfs.append(SWR_DF)
        ds_dfs.append(DS_DF)
        thresholds.append(THRESHOLDS)
        noise_trains.append(np.zeros(len(_lfp), dtype='int'))
    ALL_STD = pd.concat(std_dfs, keys=range(len(std_dfs)), ignore_index=False)
    ALL_SWR = pd.concat(swr_dfs, keys=range(len(swr_dfs)), ignore_index=False)
    ALL_DS = pd.concat(ds_dfs, keys=range(len(ds_dfs)), ignore_index=False)
    
    # save downsampled data
    if pprint: print('Saving files ...')
    if not os.path.isdir(save_ddir):
        os.mkdir(save_ddir)
    np.save(Path(save_ddir, 'lfp_time.npy'), lfp_time)
    np.save(Path(save_ddir, 'lfp_fs.npy'), lfp_fs)
    np.savez(Path(save_ddir, 'lfp_bp.npz'), **bp_dicts)
    
    # save bandpass-filtered power in each channel (index)
    ALL_STD.to_csv(Path(save_ddir, 'channel_bp_std'), index_label=False)
    
    # save event quantifications and thresholds
    ALL_SWR.to_csv(Path(save_ddir, 'ALL_SWR'), index_label=False)
    ALL_DS.to_csv(Path(save_ddir, 'ALL_DS'), index_label=False)
    np.save(Path(save_ddir, 'THRESHOLDS.npy'), thresholds)
    # initialize noise channels
    np.save(Path(save_ddir, 'noise_channels.npy'), noise_trains)
    
    # save params and info file
    with open(Path(save_ddir, 'params.pkl'), 'wb') as f:
        pickle.dump(PARAMS, f)
    
    if pprint: print('Done!' + os.linesep)


def process_aux(aux_mx, fs, lfp_fs, save_ddir, pprint=True):
    ds_factor = int(fs / lfp_fs)
    for i,aux in enumerate(aux_mx):
        if pprint: print(f'Saving AUX {i+1} / {len(aux_mx)} ...')
        aux_dn = pyfx.Downsample(aux, ds_factor)
        np.save(Path(save_ddir, f'AUX{i}.npy'), aux_dn)
        