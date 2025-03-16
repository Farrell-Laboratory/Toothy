#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 08:58:32 2025

@author: amandaschott
"""
import sys
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

###########################################################
########            PARAMETER MANAGEMENT           ########
###########################################################

def get_original_defaults():
    
    PARAMS = {
    'lfp_fs' : 1000.0,
    'trange' : [0.0, -1.0],
    'theta' : [6.0, 10.0],
    'slow_gamma' : [25.0, 55.0],
    'fast_gamma' : [60.0, 100.0],
    'ds_freq' : [5.0, 100.0],
    'swr_freq' : [120.0, 180.0],
    'ds_height_thr' : 4.5,
    'ds_dist_thr' : 100.0,
    'ds_prom_thr' : 0.0,
    'ds_wlen' : 125.0,
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


def get_param_info():

    PARAM_INFO = {
        
    'lfp_fs'     : ['Downsampled FS', 'Target sampling rate (Hz) for downsampled LFP data.'],
    'trange'     : ['Recording cutoffs', 'Start and end recording timepoints (s) to process for analysis.'],
    'theta'      : ['Theta range', 'Bandpass filter cutoff frequencies (Hz) in the theta frequency range.'],
    'slow_gamma' : ['Slow gamma range', 'Bandpass filter cutoff frequencies (Hz) in the slow gamma frequency range.'],
    'fast_gamma' : ['Fast gamma range', 'Bandpass filter cutoff frequencies (Hz) in the fast gamma frequency range.'],
    'ds_freq'    : ['DS bandpass filter', 'Bandpass filter cutoff frequencies (Hz) for detecting dentate spikes.'],
    'swr_freq'   : ['Ripple bandpass filter', 'Bandpass filter cutoff frequencies (Hz) for detecting sharp-wave ripples.'],
                  
    'ds_height_thr' : ['DS height', 'Minimum peak height (standard deviations) of a dentate spike.'],
    'ds_dist_thr'   : ['DS separation', 'Minimum distance (ms) between neighboring dentate spikes.'],
    'ds_prom_thr'   : ['DS prominence', 'Minimum dentate spike prominence (relative to the surrounding signal).'],
    'ds_wlen'       : ['DS window length', 'Window size (ms) for evaluating dentate spikes.'],
                 
    'swr_ch_bound'   : ['Ripple window', 'Number of channels on either side of the ripple LFP to include in ripple profile.'],
    'swr_height_thr' : ['Ripple height', 'Minimum height (standard deviations) at the peak of a ripple envelope.'],
    'swr_min_thr'    : ['Ripple min. height', 'Minimum height (standard deviations) at the edges of a ripple envelope.'],
    'swr_dist_thr'   : ['Ripple separation', 'Minimum distance (ms) between neighboring ripples.'],
    'swr_min_dur'    : ['Ripple min. duration', 'Minimum duration (ms) of a ripple.'],
    'swr_freq_thr'   : ['Ripple frequency', 'Minimum instantaneous frequency (Hz) of a ripple.'],
    'swr_freq_win'   : ['Ripple freq window', 'Window size (ms) for calculating ripple instantaneous frequency.'],
    'swr_maxamp_win' : ['Ripple peak window', 'Window size (ms) to look for maximum ripple LFP amplitude.'],
                        
    'csd_method'     : ['CSD mode', 'Current source density (CSD) estimation method.'],
    'f_type'         : ['CSD filter', 'Spatial filter for estimated CSD.'],
    'f_order'        : ['Filter order', 'CSD spatial filter settings (passed to scipy.signal method).'],
    'f_sigma'        : ['Filter sigma (\u03C3)', 'Sigma (or standard deviation) parameter; applies to Gaussian filter only.'],
    'vaknin_el'      : ['Vaknin electrode', "Calculate CSD with or without Vaknin's method of duplicating endpoint electrodes."],
    'tol'            : ['Tolerance', 'Tolerance of numerical integration in CSD estimation; applies to step and spline methods only.'],
    'spline_nsteps'  : ['Spline steps', 'Number of upsampled data points in CSD estimation; applies to spline method only.'],
    'src_diam'       : ['Source diameter', 'Diameter (mm) of the assumed circular current sources.'],
    'src_h'          : ['Source thickness', 'Thickness (mm) of the assumed cylindrical current sources.'],
    'cond'           : ['Conductivity', 'Conductivity (Siemens per meter) through brain tissue.'],
    'cond_top'       : ['Conductivity (top)', 'Conductivity (Siemens per meter) on top of brain tissue.'],
                  
    'clus_algo'        : ['Clustering method', 'Clustering algorithm used to classify dentate spikes.'],
    'nclusters'        : ['# clusters', 'Number of target clusters; applies to K-means algorithm only.'],
    'eps'              : ['Epsilon (\u03B5)', 'Maximum distance between points in the same cluster; applies to DBSCAN algorithm only.'],
    'min_clus_samples' : ['Min. cluster samples', 'Minimum number of samples per cluster; applies to DBSCAN algorithm only.'],
                          
    'el_w'     : ['Contact width', 'Default width (\u00B5m) of probe electrode contacts.'],
    'el_h'     : ['Contact height', 'Default height (\u00B5m) of probe electrode contacts.'],
    'el_shape' : ['Contact shape', 'Default shape of probe electrode contacts.']
    }
    
    return PARAM_INFO


def read_lines(filepath):
    """ Read in parameter file lines, return dictionary of key:value pairs """
    # strip whitespace and ignore lines beginning with comment (#)
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    PARAMS = parse_lines(lines)
    return PARAMS


def parse_lines(lines):
    PARAMS = {}
    lines = [l for l in lines if not l.startswith('#') and len(l) > 0]
    for line in lines:
        d = line.split(';')[0]  # ";" is the end of the line
        k,v = [x.strip() for x in d.split('=')]   # "=" splits keys and values
        if v.startswith('[') and v.endswith(']'): # values in brackets are lists
            val = []
            for x in v[1:-1].split(','):
                try: x2 = float(x.strip())
                except: x2 = str(x.strip())
                val.append(x2)
            #val = [float(x.strip()) for x in v[1:-1].split(',')]
        else:
            try    : val = float(v)  # if value is numerical, store as float
            except : val = str(v)    # otherwise, store as string
        if   val == 'True'  : val = True  # convert "True"/"False" strings to boolean
        elif val == 'False' : val = False
        PARAMS[k] = val
    return PARAMS


def validate_params(ddict):
    """ Determine whether input $ddict is a valid parameter dictionary
    @Returns
    is_valid - boolean value (True if all critical parameters are valid, False if not)
    valid_ddict - dictionary with validation result for each critical parameter
    """
    
    def is_number(key): # numerical parameter value
        try: val = ddict[key]
        except: return False
        return bool(type(val) in [float, int])
    
    def is_range(key):  # list of two parameter values
        try: val = ddict[key]
        except: return False
        return bool(isinstance(val,list) and len(val)==2 and all(map(pyfx.IsNum, val)))
    
    def is_category(key, options):  # categorical (string) parameter value
        try: val = ddict[key]
        except: return False
        return bool(val in options)
        
    # make sure each parameter 1) is present in dictionary and 2) has a valid value
    valid_ddict = {
                  'lfp_fs' : is_number('lfp_fs'),
                  'trange' : is_range('trange'),
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


def fix_params(ddict, default_dict=None):
    """ Fill in missing/invalid parameters with default values """
    # set "default" values to use in place of missing/invalid parameters
    if default_dict is None:
        default_dict = get_original_defaults()
    assert validate_params(default_dict)[0] == True
    # populate new dictionary with hard-coded parameter keys
    keys = list(get_original_defaults().keys())
    valid_ddict = validate_params(dict(ddict))[1]
    new_dict = {}
    for k in keys:
        if valid_ddict[k] == True:
            new_dict[k] = ddict[k]
        else:
            new_dict[k] = default_dict[k]
    return new_dict
    