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
    'el_shape' : 'circle',
    'el_area' : 225.0,
    'el_h' : 12.5
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
    
    'el_shape' : ['Contact shape', 'Default shape of probe electrode contacts.'],
    'el_area'  : ['Contact area', 'Default area (\u00B5m\u00B2) of probe electrode contacts.'],
    'el_h'     : ['Contact height', 'Default height (\u00B5m) of probe electrode contacts.']
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
                  'el_shape' : is_category('el_shape', ['circle', 'square', 'rectangle']),
                  'el_area' : is_number('el_area'),
                  'el_h' : is_number('el_h')
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



###########################################################
########             PARAMETER WIDGETS             ########
###########################################################


class Spinbox(QtWidgets.QDoubleSpinBox):
    param_changed_signal = QtCore.pyqtSignal()
    
    def __init__(self, key, isvalid=True, parent=None):
        super().__init__(parent)
        self.key = key
        self.setObjectName(key)
        self.isvalid = isvalid
        self.wheelEvent = lambda event: None
        self.ss = 'QAbstractSpinBox {background-color : %s; color : %s;}'
        self.valueChanged.connect(self.param_changed_signal.emit)
    
    def set_widget_valid(self, x):
        self.isvalid = bool(x)
        ss = self.ss % [('red','transparent'),('white','black')][int(x)]
        self.setStyleSheet(ss)
    
    def get_param_value(self):
        return self.value()
        
    def update_param(self, val):
        self.blockSignals(True)
        self.setValue(val)
        self.blockSignals(False)


class Combobox(QtWidgets.QComboBox):
    param_changed_signal = QtCore.pyqtSignal()
    
    def __init__(self, key, isvalid=True, parent=None):
        super().__init__(parent)
        self.key = key
        self.setObjectName(key)
        self.isvalid = isvalid
        self.wheelEvent = lambda event: None
        
        if key in ['csd_method', 'f_type', 'el_shape']:
            self.get_valtxt = lambda val: val.capitalize()
            self.export_valtxt = lambda txt: txt.lower()
        elif key == 'vaknin_el':
            self.get_valtxt = lambda val: str(bool(val))
            self.export_valtxt = lambda txt: bool(txt)
        elif key == 'clus_algo':
            self.get_valtxt = lambda val: dict(kmeans='K-means', dbscan='DBSCAN')[val]
            self.export_valtxt = lambda txt: txt.replace('-','').lower()
        
        self.ss = ('QComboBox {color : black;}'
                   'QComboBox:open {background-color : %s; color : %s;}')
        self.currentIndexChanged.connect(self.param_changed_signal.emit)
        self.activated.connect(lambda i: self.set_widget_valid(True))
    
    def set_widget_valid(self, x):
        self.isvalid = bool(x)
        ss = self.ss % [('red','transparent'),('white','black')][int(x)]
        self.setStyleSheet(ss)
        
    def get_param_value(self):
        res = self.export_valtxt(self.currentText())
        return res
    
    def update_param(self, val):
        txt = self.get_valtxt(val)
        self.blockSignals(True)
        self.setCurrentText(txt)
        self.blockSignals(False)

class SpinboxRange(QtWidgets.QWidget):
    param_changed_signal = QtCore.pyqtSignal()
    
    def __init__(self, key, isvalid=True, double=False, alignment=QtCore.Qt.AlignLeft, parent=None, **kwargs):
        super().__init__(parent)
        self.key = key
        self.setObjectName(key)
        self.isvalid = isvalid
        if double:
            self.box0 = QtWidgets.QDoubleSpinBox()
            self.box1 = QtWidgets.QDoubleSpinBox()
        else:
            self.box0 = QtWidgets.QSpinBox()
            self.box1 = QtWidgets.QSpinBox()
        self.boxes = [self.box0, self.box1]
        for box in self.boxes:
            box.wheelEvent = lambda event: None
            box.setAlignment(alignment)
            if 'suffix' in kwargs: box.setSuffix(kwargs['suffix'])
            if 'minimum' in kwargs: box.setMinimum(kwargs['minimum'])
            if 'maximum' in kwargs: box.setMaximum(kwargs['maximum'])
            if 'decimals' in kwargs: box.setDecimals(kwargs['decimals'])
            if 'step' in kwargs: box.setSingleStep(kwargs['step'])
            box.valueChanged.connect(lambda: self.param_changed_signal.emit())
            
        self.dash = QtWidgets.QLabel(' — ')
        self.dash.setAlignment(QtCore.Qt.AlignCenter)
        
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.box0, stretch=2)
        self.layout.addWidget(self.dash, stretch=0)
        self.layout.addWidget(self.box1, stretch=2)
        self.ss = ('QAbstractSpinBox {background-color : %s; color : %s;}')
    
    def set_widget_valid(self, x):
        self.isvalid = bool(x)
        ss = self.ss % [('red','transparent'),('white','black')][int(x)]
        self.setStyleSheet(ss)
    
    def get_param_value(self):
        return [self.box0.value(), self.box1.value()]
    
    def update_param(self, val):
        for i,val in enumerate(val):
            self.boxes[i].blockSignals(True)
            self.boxes[i].setValue(val)
            self.boxes[i].blockSignals(False)


class ParamObject(QtWidgets.QWidget):
    update_signal = QtCore.pyqtSignal(dict, list)
    
    def __init__(self, params={}, mode='all', parent=None):
        """
        mode: set visible parameters for different analysis modes
            'signal_processing': LFP downsampling and filtering params
            'ds_detection': thresholds for detecting DS events
            'swr_detection': thresholds for detecting SWR events
            'data_processing': includes all the above params
            'ds_classification': CSD calculation and PCA clustering params
            'all': all parameters
        """
        super().__init__(parent)
        # set "default values" based on input param dictionary
        dflt = get_original_defaults()
        _,valid_ddict = validate_params(params)
        for k,x in valid_ddict.items():
            if x: dflt[k] = params[k]
        self.DEFAULTS = dict(dflt)
        
        self.gen_layout()
        self.init_widgets()
        self.set_mode(mode)
        
        self.update_gui_from_ddict(params)
        self.connect_signals()
        
        # self.setStyleSheet('QWidget {'
        #                    'font-size : 15pt;'
        #                    '}'
        #                    'QToolTip {'
        #                    'background-color : lightyellow;'
        #                    'border : 2px solid black;'
        #                    'font-size : 15pt;'
        #                    'padding : 4px;'
        #                    '}')
        
    def gen_layout(self):
        self.setContentsMargins(0,0,15,0)
        
        ####################
        ####   WIDGETS   ###
        ####################
        
        ### signal_processing
        self.lfp_fs = Spinbox(key='lfp_fs')
        self.lfp_fs.setDecimals(1)
        self.lfp_fs.setSingleStep(0.5)
        self.lfp_fs.setSuffix(' Hz')
        self.trange = SpinboxRange(key='trange', double=True, decimals=0, maximum=999999, suffix=' s')
        self.trange.box1.setMinimum(-1)
        kwargs = dict(double=True, decimals=1, step=0.5, maximum=999999, suffix=' Hz')
        self.theta = SpinboxRange(key='theta', **kwargs)
        self.slow_gamma = SpinboxRange(key='slow_gamma', **kwargs)
        self.fast_gamma = SpinboxRange(key='fast_gamma', **kwargs)
        self.ds_freq = SpinboxRange(key='ds_freq', **kwargs)
        self.swr_freq = SpinboxRange(key='swr_freq', **kwargs)
        ### ds_detection
        self.ds_height_thr = Spinbox(key='ds_height_thr')
        self.ds_height_thr.setSuffix(' S.D.')
        self.ds_dist_thr = Spinbox(key='ds_dist_thr')
        self.ds_dist_thr.setSuffix(' ms')
        self.ds_dist_thr.setDecimals(0)
        self.ds_prom_thr = Spinbox(key='ds_prom_thr')
        self.ds_prom_thr.setSuffix(' a.u.')
        self.ds_wlen = Spinbox(key='ds_wlen')
        self.ds_wlen.setSuffix(' ms')
        self.ds_wlen.setDecimals(0)
        ### swr_detection
        self.swr_ch_bound = Spinbox(key='swr_ch_bound')
        self.swr_ch_bound.setSuffix(' channels')
        self.swr_ch_bound.setDecimals(0)
        self.swr_height_thr = Spinbox(key='swr_height_thr')
        self.swr_height_thr.setSuffix(' S.D.')
        self.swr_min_thr = Spinbox(key='swr_min_thr')
        self.swr_min_thr.setSuffix(' S.D.')
        self.swr_dist_thr = Spinbox(key='swr_dist_thr')
        self.swr_dist_thr.setSuffix(' ms')
        self.swr_dist_thr.setDecimals(0)
        self.swr_min_dur = Spinbox(key='swr_min_dur')
        self.swr_min_dur.setSuffix(' ms')
        self.swr_min_dur.setDecimals(0)
        self.swr_freq_thr = Spinbox(key='swr_freq_thr')
        self.swr_freq_thr.setDecimals(1)
        self.swr_freq_thr.setSuffix(' Hz')
        self.swr_freq_win = Spinbox(key='swr_freq_win')
        self.swr_freq_win.setSuffix(' ms')
        self.swr_freq_win.setDecimals(0)
        self.swr_maxamp_win = Spinbox(key='swr_maxamp_win')
        self.swr_maxamp_win.setSuffix(' ms')
        self.swr_maxamp_win.setDecimals(0)
        ### csd_calculation
        self.csd_method = Combobox(key='csd_method')
        self.csd_method.addItems(['Standard','Delta','Step','Spline'])
        self.f_type = Combobox(key='f_type')
        self.f_type.addItems(['Gaussian','Identity','Boxcar','Hamming','Triangular'])
        self.f_order = Spinbox(key='f_order')
        self.f_order.setMinimum(1)
        self.f_order.setDecimals(1)
        self.f_sigma = Spinbox(key='f_sigma')
        self.f_sigma.setDecimals(1)
        self.f_sigma.setSingleStep(0.1)
        self.vaknin_el = Combobox(key='vaknin_el')
        self.vaknin_el.addItems(['True','False'])
        self.tol = Spinbox(key='tol')
        self.tol.setDecimals(7)
        self.tol.setSingleStep(0.0000001)
        self.spline_nsteps = Spinbox(key='spline_nsteps')
        self.spline_nsteps.setMaximum(2500)
        self.spline_nsteps.setDecimals(0)
        self.src_diam = Spinbox(key='src_diam')
        self.src_diam.setDecimals(3)
        self.src_diam.setSingleStep(0.01)
        self.src_diam.setSuffix(' mm')
        self.src_h = Spinbox(key='src_h')
        self.src_h.setDecimals(3)
        self.src_h.setSingleStep(0.01)
        self.src_h.setSuffix(' mm')
        self.cond = Spinbox(key='cond')
        self.cond.setDecimals(3)
        self.cond.setSingleStep(0.01)
        self.cond.setSuffix(' S/m')
        self.cond_top = Spinbox(key='cond_top')
        self.cond_top.setDecimals(3)
        self.cond_top.setSingleStep(0.01)
        self.cond_top.setSuffix(' S/m')
        ### csd_clustering
        self.clus_algo = Combobox(key='clus_algo')
        self.clus_algo.addItems(['K-means','DBSCAN'])
        self.nclusters = Spinbox(key='nclusters')
        self.nclusters.setMinimum(1)
        self.nclusters.setSuffix(' clusters')
        self.nclusters.setDecimals(0)
        self.eps = Spinbox(key='eps')
        self.eps.setDecimals(2)
        self.eps.setSingleStep(0.1)
        self.eps.setSuffix(' a.u.')
        self.min_clus_samples = Spinbox(key='min_clus_samples')
        self.min_clus_samples.setMinimum(1)
        self.min_clus_samples.setDecimals(0)
        ### probe_geometry
        self.el_shape = Combobox(key='el_shape')
        self.el_shape.addItems(['Circle', 'Square', 'Rectangle'])
        self.el_area = Spinbox(key='el_area')
        self.el_area.setDecimals(1)
        self.el_area.setSuffix(' \u00B5m\u00B2')
        self.el_h = Spinbox(key='el_h')
        self.el_h.setDecimals(1)
        self.el_h.setSuffix(' \u00B5m')
        
        for sbox in [self.lfp_fs, self.ds_height_thr, self.ds_dist_thr, self.ds_prom_thr, self.ds_wlen, 
                     self.swr_ch_bound, self.swr_height_thr, self.swr_min_thr, self.swr_dist_thr, 
                     self.swr_min_dur, self.swr_freq_thr, self.swr_freq_win, self.swr_maxamp_win,
                     self.f_order, self.f_sigma, self.tol, self.src_diam, self.src_h, self.cond,
                     self.cond_top, self.nclusters, self.eps, self.min_clus_samples, self.el_area, self.el_h]:
            sbox.setMaximum(999999)
            
        tups = [(self.lfp_fs, 'signal_processing'),
                (self.trange, 'signal_processing'),
                (self.theta, 'signal_processing'),
                (self.slow_gamma, 'signal_processing'),
                (self.fast_gamma, 'signal_processing'),
                (self.ds_freq, 'signal_processing'),
                (self.swr_freq, 'signal_processing'),
                
                (self.ds_height_thr, 'ds_detection'),
                (self.ds_dist_thr, 'ds_detection'),
                (self.ds_prom_thr, 'ds_detection'),
                (self.ds_wlen, 'ds_detection'),
                
                (self.swr_ch_bound, 'swr_detection'),
                (self.swr_height_thr, 'swr_detection'),
                (self.swr_min_thr, 'swr_detection'),
                (self.swr_dist_thr, 'swr_detection'),
                (self.swr_min_dur, 'swr_detection'),
                (self.swr_freq_thr, 'swr_detection'),
                (self.swr_freq_win, 'swr_detection'),
                (self.swr_maxamp_win, 'swr_detection'),
                
                (self.csd_method, 'csd_calculation'),
                (self.f_type, 'csd_calculation'),
                (self.f_order, 'csd_calculation'),
                (self.f_sigma, 'csd_calculation'),
                (self.vaknin_el, 'csd_calculation'),
                (self.tol, 'csd_calculation'),
                (self.spline_nsteps, 'csd_calculation'),
                (self.src_diam, 'csd_calculation'),
                (self.src_h, 'csd_calculation'),
                (self.cond, 'csd_calculation'),
                (self.cond_top, 'csd_calculation'),
                
                (self.clus_algo, 'csd_clustering'),
                (self.nclusters, 'csd_clustering'),
                (self.eps, 'csd_clustering'),
                (self.min_clus_samples, 'csd_clustering'),
                
                (self.el_shape, 'probe_geometry'),
                (self.el_area, 'probe_geometry'),
                (self.el_h, 'probe_geometry')]
        
        # get widget items and corresponding modes, labels, and info text
        widgets,modes = map(list, zip(*tups))
        info_dict = get_param_info()
        
        ###   ORGANIZE PARAMETERS BY MODE   ###
        
        # map analysis modes to parameter variable names
        tmp = {m:[] for i,m in enumerate(modes) if modes.index(m)==i}
        mode2key = {**tmp, 'data_processing':[], 'ds_classification':[], 'all':[]}
        for key,m in zip(info_dict, modes):
            mode2key['all'].append(key)
            mode2key[m].append(key)
            if m in ['signal_processing','ds_detection','swr_detection']:
                mode2key['data_processing'].append(key)
            if m in ['csd_calculation','csd_clustering','probe_geometry']:
                mode2key['ds_classification'].append(key)
        self.mode2key = mode2key
            
        ###   LAYOUT   ###
        
        self.vlay = QtWidgets.QVBoxLayout(self)
        self.ROWS, self.WIDGETS = {},{}
        for (key,(lbl,info)),widget in zip(info_dict.items(),widgets):
            # make QLabel with hover info
            qlabel = QtWidgets.QLabel(lbl)
            qlabel.setToolTip(info)
            # create row widget, add to layout
            row = QtWidgets.QWidget()
            hbox = QtWidgets.QHBoxLayout(row)
            hbox.setContentsMargins(0,0,0,0)
            hbox.addWidget(qlabel, stretch=0)
            hbox.addWidget(widget, stretch=1)
            self.ROWS[key] = row
            self.WIDGETS[key] = widget
            self.vlay.addWidget(row)
    
    def connect_signals(self):
        for widget in self.WIDGETS.values():
            widget.param_changed_signal.connect(self.emit_signal)
            
    def ddict_from_gui(self):
        """ Return GUI widget values as parameter dictionary """
        ddict = {}
        invalid = []
        for key in self.KEYS:
            item = self.WIDGETS[key]
            if item.isvalid == True:
                ddict[key] = item.get_param_value()
            else:  # # if item remains invalid, export original param value
                if key in self.DDICT_ORIG: 
                    ddict[key] = self.DDICT_ORIG[key]
                invalid.append(key)
        return ddict, invalid
    
    def emit_signal(self, *args):
        
        # A) user toggles widget -> that param is no longer invalid
        # B) user uploads a new set of params -> validity depends on the content
        # * try to prevent emit_signal during B
        self.sender().set_widget_valid(True)
        PARAMS, invalid = self.ddict_from_gui()
        
        
        self.update_signal.emit(PARAMS, invalid)
    
    def set_mode(self, mode='all'):
        if mode not in self.mode2key: mode = 'all'
        self.KEYS = list(self.mode2key[mode])
        
        # show parameter rows for the given mode, hide the rest
        for k,row in self.ROWS.items():
            row.setVisible(k in self.KEYS)
        QtCore.QTimer.singleShot(50, self.adjust_labels)
        
    def adjust_labels(self):
        # set text label widths to the longest visible QLabel
        qlabels = [self.ROWS[mk].findChild(QtWidgets.QLabel) for mk in self.KEYS]
        mw = max([ql.width() for ql in qlabels])
        for ql in qlabels:
            ql.setAlignment(QtCore.Qt.AlignCenter)
            ql.setFixedWidth(mw)
    
    def init_widgets(self):
        func = lambda x: (lambda k,w: (w, self.DEFAULTS[k]))(*x)
        for item,val in map(func, self.WIDGETS.items()):
            item.update_param(val)
            item.set_widget_valid(True)

    def update_gui_from_ddict(self, ddict):#, block=False):
        self.DDICT_ORIG = dict(ddict)
        # check for valid $ddict values for each current param (e.g. lfp_fs:False, trange=True)
        valid_ddict = {k:b for k,b in validate_params(ddict)[1].items() if k in self.KEYS}
        param_dict = {k:ddict[k] if x else self.DEFAULTS[k] for k,x in valid_ddict.items()}
        for key,val in param_dict.items():
            item = self.WIDGETS[key]
            #if block: item.blockSignals(True)
            item.update_param(val)
            item.set_widget_valid(valid_ddict[key])
            #if block: item.blockSignals(False)
    
    def debug(self):
        pdb.set_trace()
    