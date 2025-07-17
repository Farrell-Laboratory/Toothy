#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw data processing pipeline

@author: amandaschott
"""
import sys
import os
import shutil
from pathlib import Path
import scipy.io as so
import h5py
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
import time
import pickle
import probeinterface as prif
import pdb
# custom modules
import QSS
import pyfx
import qparam
import ephys
import gui_items as gi
import data_processing as dp
from probe_handler import ProbeObjectPopup
import resources_rc


##############################################################################
##############################################################################
################                                              ################
################                 WORKER OBJECTS               ################
################                                              ################
##############################################################################
##############################################################################


class ArrayReader(QtCore.QObject):
    """ Reads data arrays from NPY and MAT files """
    progress_signal = QtCore.pyqtSignal(str)
    data_signal = QtCore.pyqtSignal(str, dict, dict)
    error_signal = QtCore.pyqtSignal(str, str)
    finished = QtCore.pyqtSignal()
    
    def __init__(self, fpath):
        """ Initialize path to raw data file """
        super().__init__()
        self.fpath = fpath
    
    def run(self):
        """ Load raw data """
        fpath = self.fpath
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fpath)[-1]
        self.progress_signal.emit(f'Loading {ext[1:].upper()} file ...')
        error_msg = ''
        try:
            if ext == '.npy':   # load NPY file
                file_data = np.load(fpath, allow_pickle=True)
            elif ext == '.mat': # load MAT file
                file_data = so.loadmat(fpath, squeeze_me=True)
            if isinstance(file_data, np.ndarray): # data saved as ndarray
                meta = {'fs':None, 'units':None}
                data_dict = {}
                if file_data.ndim == 2:
                    data_dict['data'] = np.array(file_data)
                else:  # data array not 2-dimensional
                    error_msg = 'Error: Data must be a 2-dimensional array.'
            elif isinstance(file_data, dict):     # data saved as dictionary
                self.progress_signal.emit('Parsing data dictionary ...')
                try:  # get data array (required) and SR/unit metadata (optional)
                    data_dict, meta = dp.read_data_from_dict(file_data)
                    if len(data_dict) == 0:
                        error_msg = f'Error: No 2-dimensional data arrays found in "{fname}".'
                except: error_msg = f'Error: Could not parse data from "{fname}".'
            else: error_msg = 'Error: File must contain data array or dictionary.'
        except: error_msg = f'Error: Unable to load "{fname}".'
        # return data or error message
        if error_msg == '':
            self.data_signal.emit(str(fpath), data_dict, meta)
            self.progress_signal.emit('Done!')
        else:
            self.error_signal.emit(str(fpath), error_msg)
        self.finished.emit()
        
        
class ExtractorWorker(QtCore.QObject):
    """ Returns spikeinterface extractor objects for raw recordings """
    progress_signal = QtCore.pyqtSignal(str)
    data_signal = QtCore.pyqtSignal(object)
    error_signal = QtCore.pyqtSignal(str, str)
    finished = QtCore.pyqtSignal()
    
    def __init__(self, fpath, **kwargs):
        """ Initialize recording filepath and raw data (if previously loaded) """
        super().__init__()
        self.fpath = fpath
        self.kwargs = {'data_array' : kwargs.get('data_array'),
                       'metadata' : kwargs.get('metadata', {}),
                       'electrical_series_path' : kwargs.get('electrical_series_path')}
    
    def run(self):
        """ Get recording extractor """
        data_type = dp.get_data_format(self.fpath)
        self.progress_signal.emit(f'Getting {data_type} extractor ...')
        try:  # get spikeinterface extractor for selected data type
            recording = dp.get_extractor(self.fpath, data_type, **self.kwargs)
        except Exception as e: # failed to load extractor object
            self.error_signal.emit(str(self.fpath), f'Error: {str(e)}')
        else:   # emit valid extractor
            self.data_signal.emit(recording)
            self.progress_signal.emit('Done!')
        finally:
            self.finished.emit()
    
    
class DataWorker(QtCore.QObject):
    """ Handles data imports, preprocessing, and exports in processing pipeline """
    progress_signal = QtCore.pyqtSignal(str)
    data_signal = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    
    RECORDING = None    # spikeinterface recording extractor
    PROBE_GROUP = None  # probeinterface ProbeGroup object
    PARAMS = None       # parameter dictionary
    SAVE_DDIR = None    # location of processed data files
    
    def init_source(self, RECORDING, PROBE_GROUP, PARAMS, SAVE_DDIR=None):
        """ Initialize data objects and parameter values """
        self.RECORDING = RECORDING
        self.PROBE_GROUP = PROBE_GROUP
        self.PARAMS = PARAMS
        if SAVE_DDIR is None:  # auto-generate target directory
            raw_ddir = os.path.dirname(self.RECORDING.get_annotation('ppath'))
            SAVE_DDIR = str(Path(raw_ddir, pyfx.unique_fname(raw_ddir, 'processed_data')))
        self.SAVE_DDIR = SAVE_DDIR
        
    def quit_thread(self, error_msg, ff=None, KW={}):
        """ Terminate pipeline upon data processing error """
        if ff is not None:
            ff.close()  # close HDF5 datasets
        if 'fid' in KW and KW['fid'] is not None:
            KW['fid'].close()  # close raw data file
        self.error_signal.emit(error_msg)
        self.finished.emit()
    
    def run(self, **kwargs):
        """ Raw data processing pipeline """
        # get core data objects
        recording   = kwargs.get('recording', self.RECORDING)
        probe_group = kwargs.get('probe_group', self.PROBE_GROUP)
        PARAMS      = kwargs.get('PARAMS', self.PARAMS)
        save_ddir   = kwargs.get('save_ddir', self.SAVE_DDIR)
        if not os.path.isdir(save_ddir):
            os.mkdir(save_ddir)
            
        ################################
        #####     PARSE PARAMS     #####
        ################################
        
        load_win    = kwargs.get('load_win', 600)
        recording.set_probegroup(probe_group, in_place=True)
        META = dp.get_meta_from_recording(recording)
        # get raw data directory, create processed data folder
        ppath, data_format = [recording.get_annotation(x) for x in ['ppath','data_format']]
        
        # parse metadata and analysis parameters
        FS, NSAMPLES, TOTAL_CH = META['fs'], META['nsamples'], META['total_ch']
        lfp_fs = PARAMS['lfp_fs']
        tstart, tend = PARAMS['trange']  # extract data between timepoints
        iistart, iiend = dp.get_rec_bounds(NSAMPLES, FS, tstart, tend)
        NSAMPLES_TRUNC = int(iiend-iistart)
        # calculate size of downsampled dataset
        ds_factor = FS/lfp_fs
        NSAMPLES_DN_TRUNC = int(NSAMPLES_TRUNC / ds_factor)
        
        global_dev_idx = probe_group.get_global_device_channel_indices()['device_channel_indices']
        nprobes = len(probe_group.probes)
        
        ################################
        #####      I/O OBJECTS     #####
        ################################
        
        try:  # get IO object (fid, cont, or recording)
            KW = {**dp.get_raw_source_kwargs(recording), 'total_ch':TOTAL_CH}
        except Exception as e:
            self.quit_thread(f'Data Source Error: {str(e)}')
            return
        if KW['fid'] is not None and iistart > 0:
            KW['fid'].seek(int(iistart * TOTAL_CH * 4))
        
        # create HDF5 datasets for LFP data from each probe
        ff = h5py.File(Path(save_ddir, 'DATA.hdf5'), 'w', track_order=True)
        ff.attrs['fs'] = FS
        ff.attrs['lfp_fs'] = lfp_fs
        lfp_time = ff.create_dataset('lfp_time', data=np.linspace(0, NSAMPLES_DN_TRUNC/lfp_fs, 
                                                      NSAMPLES_DN_TRUNC, dtype='float32'))
        ichannels, datasets = [], []
        for iprb in range(nprobes):  # get channel mapping and dataset for each probe
            PROBE_DSET = ff.create_group(str(iprb), track_order=True)
            PROBE_LFPS = PROBE_DSET.create_group('LFP', track_order=True)
            idx = np.where(recording.get_channel_groups()==iprb)[0]
            ichan = global_dev_idx[idx]
            dset = PROBE_LFPS.create_dataset('raw', (len(ichan), NSAMPLES_DN_TRUNC), 
                                              dtype='float32')
            ichannels.append(ichan); datasets.append(dset)
        
        ##################################
        #####     DATA EXTRACTION    #####
        ##################################
        
        # extract and downsample LFPs in ~10 min chunks
        chunkfunc, (iichunk,ichunk) = dp.get_chunkfunc(load_win, FS, NSAMPLES, lfp_fs=lfp_fs,
                                                       tstart=tstart, tend=tend)
        count = 0
        while True:
            (ii,jj),(aa,bb),txt = chunkfunc(count)
            self.progress_signal.emit(txt)
            for ichan,dset in zip(ichannels, datasets):
                try:  # read in dataset chunk (scaled to mV)
                    snip = dp.load_chunk(data_format, ii=ii, jj=jj, ichan=ichan, **KW)
                except Exception as e:
                    self.quit_thread(f'I/O Error: {str(e)}', ff=ff, KW=KW)
                    return
                
                try:  # downsample LFP signals
                    snip_dn = dp.resample_chunk(snip, bb-aa)
                    p = count*ichunk
                    dset[:, p:p+(bb-aa)] = snip_dn
                except Exception as e:
                    self.quit_thread(f'Downsampling Error: {str(e)}', ff=ff, KW=KW)
                    return
            if jj >= iiend:
                break
            count += 1
        if KW['fid'] is not None:
            KW['fid'].close()
        
        ####################################
        #####    BANDPASS FILTERING    #####
        ####################################
        
        for iprb,probe in enumerate(probe_group.probes):
            hdr = f'ANALYZING PROBE {iprb+1} / {nprobes}<br>'
            LFP_dict = ff[str(iprb)]['LFP']
            LFP_raw = LFP_dict['raw']; NCH = len(LFP_raw)
            shank_ids = np.array(probe.shank_ids, dtype='int')
            # bandpass filter signals and get mean channel amplitudes
            for k in ['theta', 'slow_gamma', 'fast_gamma', 'swr', 'ds']:
                #del LFP_dict[k]
                LFP_dict.create_dataset(k, (NCH, NSAMPLES_DN_TRUNC), dtype='float32')
                
            self.progress_signal.emit(hdr + '<br>Bandpass filtering signals ...')
            KW2 = {'lfp_fs':lfp_fs, 'axis':1}
            
            try:
                arr_list = np.array_split(LFP_raw, int(np.ceil(NSAMPLES_DN_TRUNC / ichunk)), axis=1)
                p = 0
                for i,yarr in enumerate(arr_list):
                    q = p + yarr.shape[1]
                    LFP_dict['theta'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['theta'], **KW2)
                    LFP_dict['slow_gamma'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['slow_gamma'], **KW2)
                    LFP_dict['fast_gamma'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['fast_gamma'], **KW2)
                    LFP_dict['swr'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['swr_freq'], **KW2)
                    LFP_dict['ds'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['ds_freq'], **KW2)
                    p += yarr.shape[1]
                std_dict = {}
                for k,arr in LFP_dict.items():
                    std_dict[k] = np.std(arr, axis=1)
                    std_dict[f'norm_{k}'] = pyfx.Normalize(std_dict[k])
            except Exception as e:
                self.quit_thread(f'Filtering Error: {str(e)}', ff=ff)
                return
            STD = pd.DataFrame(std_dict)
            STD.to_hdf(ff.filename, key=f'/{iprb}/STD')
            
            
            ####################################
            #####      EVENT DETECTION     #####
            ####################################
            
            
            # initial ripple and DS detection for each channel
            progress_txt = hdr + f'<br>Detecting DSs and ripples on channel<br>%s / {NCH} ...'
            SWR_DF, DS_DF, SWR_THRES, DS_THRES = None, None, None, None
            for i,(swr_d, ds_d) in enumerate(zip(LFP_dict['swr'], LFP_dict['ds'])):
                self.progress_signal.emit(progress_txt % (i+1))
                try:  # detect sharp-wave ripples
                    SWR_DF, SWR_THRES = dp.detect_channel('swr', i, swr_d, lfp_time[:],
                                                       DF=SWR_DF, THRES=SWR_THRES, 
                                                       pprint=False, **PARAMS)
                except Exception as e:
                    self.quit_thread(f'Ripple Detection Error: {str(e)}', ff=ff)
                    return
                try:  # detect DSs
                    DS_DF, DS_THRES = dp.detect_channel('ds', i, ds_d, lfp_time[:], 
                                                        DF=DS_DF, THRES=DS_THRES, 
                                                        pprint=False, **PARAMS)
                except Exception as e:
                    self.quit_thread(f'DS Detection Error: {str(e)}', ff=ff)
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
            SWR_DF.to_hdf(ff.filename, key=f'/{iprb}/ALL_SWR')
            DS_DF.to_hdf(ff.filename, key=f'/{iprb}/ALL_DS')
            for ek,THRES in [('swr',SWR_THRES),('ds',DS_THRES)]:
                THRES_DF = pd.DataFrame(THRES).T  # save threshold magnitudes
                THRES_DF.to_hdf(ff.filename, key=f'/{iprb}/{ek.upper()}_THRES')
            # initialize noise train
            ff[str(iprb)]['NOISE'] = np.zeros(NCH, dtype='int')
            
        ##########################################
        #####    SAVE PROBE/PARAM SETTINGS   #####
        ##########################################
            
        # initialize event channel dictionary with [None,None,None]
        _ = ephys.init_event_channels(save_ddir, probes=probe_group.probes, psave=True)
        # save params and info file in recording folder
        param_path = Path(save_ddir, pyfx.unique_fname(save_ddir, 'params.pkl'))
        with open(Path(param_path), 'wb') as f:
            pickle.dump(PARAMS, f)
        # write probe group to file
        probegroup_path = Path(save_ddir, pyfx.unique_fname(save_ddir, 'probe_group'))
        prif.write_probeinterface(probegroup_path, probe_group)
        ff.close()
        self.data_signal.emit()
        self.progress_signal.emit('Done!')
        time.sleep(1)
        self.finished.emit()


##############################################################################
##############################################################################
################                                              ################
################                RAW DATA MODULES              ################
################                                              ################
##############################################################################
##############################################################################
        
    
class RawRecordingSelectionWidget(gi.FileSelectionWidget):
    """ Handles raw recording selection and validation """
    
    read_array_signal = QtCore.pyqtSignal(str)
    data_array = None
    meta = {}
    
    def select_filepath(self):
        """ Launch file dialog for raw recording selection, filter unsupported extensions """
        init_ddir = self.get_init_ddir()
        supported_extensions = [x[1] for x in dp.supported_formats.values()]
        fpath = ephys.select_raw_recording_file(supported_extensions, init_ddir, parent=self)
        if fpath:
            self.data_array, self.meta = None, {}
            self.update_filepath(fpath)
        
    def update_filepath(self, ppath, x=None):
        """ Handle selection of a new filepath """
        if x is None:
            x = self.validate_ppath(ppath)
        if x is None: # selected NPY or MAT file
            self.read_array_signal.emit(ppath)
            return
        self.le.setText(ppath)
        self.update_status(x)
        self.signal.emit(self.VALID_PPATH)
    
    def enter_array_metadata(self, ppath, data_array, meta):
        """ Prompt user to enter contextual metadata for NPY/MAT files """
        # launch popup interface
        dlg = RawArrayPopup(data_array.shape, **meta, filename=os.path.basename(ppath))
        if dlg.exec():
            self.data_array = np.array(data_array)
            self.meta = {'nch' : dlg.nch, 'nts' : dlg.nts,
                         'fs' : dlg.fs_w.value(),
                         'units' : dlg.units_w.currentText()}
            self.update_filepath(ppath, True)  # raw data source validated
                
    def validate_ppath(self, ppath):
        """ Check if raw data file can be loaded """
        if not os.path.exists(ppath) or os.path.isdir(ppath):
            return False  # must be existing file
        try    : data_format = dp.get_data_format(ppath)
        except : return False  # must be supported format
        if data_format in ['NPY','MAT']:
            return None  # NPY/MAT files require additional info
        return True
    
        
class ProbeRow(QtWidgets.QWidget):
    """ Interactive widget representation of a probe object """
    
    def __init__(self, probe, nrows, start_row, mode):
        super().__init__()
        self.probe = probe
        self.nch = probe.get_contact_count()
        self.nrows = nrows
        self.div = int(self.nrows / self.nch)
        
        self.gen_layout()
        self.get_rows(start_row, mode)
        
    def gen_layout(self):
        """ Set up layout """
        # selection button
        self.btn = QtWidgets.QPushButton()
        self.btn.setCheckable(True)
        self.btn.setChecked(True)
        self.btn.setFixedSize(20,20)
        self.btn.setFlat(True)
        self.btn.setStyleSheet('QPushButton'
                               '{border : none;'
                               'image : url(:/icons/white_circle.png);'
                               'outline : none;}'
                               
                               'QPushButton:checked'
                               '{image : url(:/icons/black_circle.png);}')
        # probe info labels
        self.glabel = QtWidgets.QLabel()
        self.glabel_fmt = '<b>{a}</b><br>channels {b}'
        labels = QtWidgets.QWidget()
        self.glabel.setStyleSheet('QLabel {'
                                  'background-color:white;'
                                  'border:1px solid gray;'
                                  'padding:5px 10px;}')
        label_lay = QtWidgets.QVBoxLayout(labels)
        self.qlabel = QtWidgets.QLabel(self.probe.name)
        self.ch_label = QtWidgets.QLabel()
        label_lay.addWidget(self.qlabel)
        label_lay.addWidget(self.ch_label)
        
        # action buttons - delete & copy implemented
        self.bbox = QtWidgets.QWidget()
        policy = self.bbox.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.bbox.setSizePolicy(policy)
        bbox = QtWidgets.QGridLayout(self.bbox)
        bbox.setContentsMargins(0,0,0,0)
        bbox.setSpacing(0)
        toolbtns = [QtWidgets.QToolButton(), QtWidgets.QToolButton(), 
                    QtWidgets.QToolButton(), QtWidgets.QToolButton()]
        self.copy_btn, self.delete_btn, self.edit_btn, self.save_btn = toolbtns
        
        self.delete_btn.setIcon(QtGui.QIcon(":/icons/trash.png"))
        self.copy_btn.setIcon(QtGui.QIcon(":/icons/copy.png"))
        self.edit_btn.setIcon(QtGui.QIcon(":/icons/edit.png"))
        self.save_btn.setIcon(QtGui.QIcon(":/icons/save.png"))
        for btn in toolbtns:
            btn.setIconSize(QtCore.QSize(20,20))
            btn.setAutoRaise(True)
        bbox.addWidget(self.copy_btn, 0, 0)
        bbox.addWidget(self.delete_btn, 0, 1)
        #bbox.addWidget(self.edit_btn, 1, 0)
        #bbox.addWidget(self.save_btn, 1, 1)
        self.btn.toggled.connect(lambda chk: self.bbox.setVisible(chk))
        
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.btn, stretch=0)
        self.layout.addWidget(self.glabel, stretch=2)
        #self.layout.addWidget(labels)
        self.layout.addWidget(self.bbox, stretch=0)
        #self.layout.addWidget(self.qlabel)
        #self.layout.addWidget(self.ch_label)
        
    def get_rows(self, start_row, mode):
        """ Map probe channels to subset of data rows """
        self.MODE = mode
        if self.MODE == 0:    # M consecutive indices from starting point
            self.ROWS = np.arange(start_row, start_row+self.nch)
            txt = f'{start_row}:{start_row+self.nch}'
        elif self.MODE == 1:  # M indices distributed evenly across M*N total rows
            self.ROWS = np.arange(0, self.nch*self.div, self.div) + start_row
            txt = f'{start_row}::{self.div}::{self.nch*self.div-self.div+start_row+1}'
        self.glabel.setText(self.glabel_fmt.format(a=self.probe.name, b=txt))


class ProbeAssignmentWidget(QtWidgets.QWidget):
    """ Loads, creates and assigns probe objects to ephys data arrays """
    check_signal = QtCore.pyqtSignal()
    MODE = 0  # probes assigned to "contiguous" (0) or "alternating" (1) rows
    
    def __init__(self, nrows):
        super().__init__()
        self.nrows = nrows
        self.remaining_rows = np.arange(self.nrows)
        
        self.gen_layout()
        self.connect_signals()
        self.probe_done_btn = QtWidgets.QPushButton('i have written new text') # PROCESS DATA
        self.probe_done_btn.setEnabled(False)
    
    def gen_layout(self):
        """ Set up layout """
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        # title and status button
        self.row0 = QtWidgets.QHBoxLayout()
        self.row0.setContentsMargins(0,0,0,0)
        self.row0.setSpacing(3)
        self.prb_icon_btn = gi.StatusIcon(init_state=0)
        probe_lbl = QtWidgets.QLabel('<b><u>Probe(s)</u></b>')
        self.row0.addWidget(self.prb_icon_btn)
        self.row0.addWidget(probe_lbl)
        self.row0.addStretch()
        # load/create buttons
        self.load_prb = QtWidgets.QPushButton('Load')
        self.create_prb = QtWidgets.QPushButton('Create')
        self.row0.addWidget(self.load_prb)
        self.row0.addWidget(self.create_prb)
        # container for probe objects/rows 
        self.data_assign_df = pd.DataFrame({'Row':np.arange(self.nrows), 'Probe(s)':''})
        self.probe_bgrp = QtWidgets.QButtonGroup()
        self.qframe = QtWidgets.QFrame()
        self.qframe.setFrameShape(QtWidgets.QFrame.Panel)
        self.qframe.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.qframe.setLineWidth(3)
        self.qframe.setMidLineWidth(3)
        qframe_layout = QtWidgets.QVBoxLayout(self.qframe)
        qframe_layout.setSpacing(10)
        self.qlayout = QtWidgets.QVBoxLayout()  # probe row container
        qframe_layout.addLayout(self.qlayout, stretch=2)
        #qframe_layout.addLayout(hbox, stretch=0)
        
        # display data dimensions vs probe geometry
        self.row00 = QtWidgets.QHBoxLayout()
        self.row00.setContentsMargins(0,0,0,0)
        self.row00.setSpacing(3)
        self.view_assignments_btn = QtWidgets.QPushButton('View')
        self.row00.addStretch()
        self.row00.addWidget(self.view_assignments_btn)
        data_panel = QtWidgets.QFrame()
        data_panel.setFrameShape(QtWidgets.QFrame.Panel)
        data_panel.setFrameShadow(QtWidgets.QFrame.Sunken)
        data_lay = QtWidgets.QVBoxLayout(data_panel)
        #self.data_lbl = QtWidgets.QLabel(f'DATA: {self.nrows} channels')
        self.data_txt0 = f'{self.nrows} channels'
        self.data_txt_fmt = (f'<code>{self.nrows} data rows<br>'
                             '<font color="%s">%s channels</font></code>')
        self.data_lbl = QtWidgets.QLabel(self.data_txt_fmt % ('red', 0))
        self.data_lbl.setStyleSheet('QLabel {'
                                    'background-color:white;'
                                    'border:1px solid gray;'
                                    'padding:10px;'
                                    '}')
        # assignment mode (blocks vs interlacing)
        assign_vlay = QtWidgets.QVBoxLayout()
        assign_vlay.setSpacing(0)
        assign_lbl = QtWidgets.QLabel('<u>Index Mode</u>')
        self.block_radio = QtWidgets.QRadioButton('Contiguous rows')
        self.block_radio.setChecked(True)
        self.inter_radio = QtWidgets.QRadioButton('Alternating rows')
        assign_vlay.addWidget(assign_lbl)
        assign_vlay.addWidget(self.block_radio)
        assign_vlay.addWidget(self.inter_radio)
        data_lay.addWidget(self.data_lbl)
        data_lay.addStretch()
        data_lay.addLayout(assign_vlay)
        #data_lay.addWidget(self.view_assignments_btn)
        self.vlay0 = QtWidgets.QVBoxLayout()
        self.vlay0.addLayout(self.row0)
        self.vlay0.addWidget(self.qframe)
        self.vlay1 = QtWidgets.QVBoxLayout()
        self.vlay1.addLayout(self.row00)
        self.vlay1.addWidget(data_panel)
        self.hlay = QtWidgets.QHBoxLayout()
        self.hlay.addLayout(self.vlay0, stretch=3)
        self.hlay.addLayout(self.vlay1, stretch=1)
        
        self.layout.addLayout(self.hlay)
        
    def connect_signals(self):
        """ Connect GUI inputs """
        self.load_prb.clicked.connect(self.load_probe_from_file)
        self.create_prb.clicked.connect(self.design_probe)
        self.view_assignments_btn.clicked.connect(self.view_data_assignments)
        self.block_radio.toggled.connect(self.switch_index_mode)
    
    def view_data_assignments(self):
        """ Show probe(s) assigned to each data signal """
        tbl = gi.TableWidget(self.data_assign_df)
        dlg = gi.Popup(widgets=[tbl], title='Data Assignments', parent=self)
        dlg.exec()
        
    def switch_index_mode(self, chk):
        """ Assign probes to contiguous blocks or distributed rows of data """
        self.MODE = int(not chk)  # if block btn is checked, mode = 0
        items = pyfx.layout_items(self.qlayout)
        self.remaining_rows = np.arange(self.nrows)
        start_row = 0
        for i,item in enumerate(items):
            item.get_rows(start_row, self.MODE)
            if self.MODE == 0:
                start_row = item.ROWS[-1] + 1
            elif self.MODE == 1:
                start_row += 1
            self.remaining_rows = np.setdiff1d(self.remaining_rows, item.ROWS)
        self.check_assignments()
        
    def add_probe_row(self, probe):
        """ Add new probe to collection """
        nch = probe.get_contact_count()
        # require enough remaining rows to assign probe channels
        try:
            assert nch <= len(self.remaining_rows)
        except AssertionError:
            msg = f'Cannot map {nch}-channel probe to {len(self.remaining_rows)} remaining data rows'
            gi.MsgboxError(msg, parent=self).exec()
            return
        
        if self.MODE == 1:
            lens = [item.nch for item in pyfx.layout_items(self.qlayout)] + [nch]
            try:
                assert len(np.unique(lens)) < 2  # alternate indexing requires all same-size probes
            except AssertionError:
                msg = 'Alternate indexing requires all probes to be the same size'
                gi.MsgboxError(msg, parent=self).exec()
                return
        # get start row for probe based on prior probe assignment
        start_row = 0
        if self.qlayout.count() > 0:
            prev_rows = self.qlayout.itemAt(self.qlayout.count()-1).widget().ROWS
            start_row = pyfx.Edges(prev_rows)[1-self.MODE] + 1
        probe_row = ProbeRow(probe, self.nrows, start_row, self.MODE)
        self.probe_bgrp.addButton(probe_row.btn)
        probe_row.copy_btn.clicked.connect(lambda: self.copy_probe_row(probe_row))
        probe_row.delete_btn.clicked.connect(lambda: self.del_probe_row(probe_row))
        # probe_row.edit_btn.clicked.connect(lambda: self.edit_probe_row(probe_row))
        # probe_row.save_btn.clicked.connect(lambda: self.save_probe_row(probe_row))
        
        self.qlayout.addWidget(probe_row)
        self.remaining_rows = np.setdiff1d(self.remaining_rows, probe_row.ROWS)
        self.check_assignments()
    
    def del_probe_row(self, probe_row):
        """ Remove assigned probe from collection """
        # position of probe object to be deleted
        idx = pyfx.layout_items(self.qlayout).index(probe_row)
        
        self.probe_bgrp.removeButton(probe_row.btn)
        self.qlayout.removeWidget(probe_row)
        probe_row.setParent(None)
        
        self.remaining_rows = np.arange(self.nrows)
        items = pyfx.layout_items(self.qlayout)
        for i,item in enumerate(items):
            if i==max(idx-1,0): item.btn.setChecked(True) # auto-check row above deleted object
            if i < idx: continue  # probes above deleted object do not change assignment
            # update rows
            if i == 0 : start_row = 0
            else      : start_row = pyfx.Edges(items[i-1].ROWS)[1-self.MODE] + 1
            item.get_rows(start_row, self.MODE)
            self.remaining_rows = np.setdiff1d(self.remaining_rows, item.ROWS)
        self.check_assignments()
    
    def copy_probe_row(self, probe_row):
        """ Duplicate an assigned probe """
        # copy probe configuration to new probe object, add as row
        orig_probe = probe_row.probe
        new_probe = orig_probe.copy()
        new_probe.annotate(**dict(orig_probe.annotations))
        new_probe.set_shank_ids(np.array(orig_probe.shank_ids))
        new_probe.set_contact_ids(np.array(orig_probe.contact_ids))
        new_probe.set_device_channel_indices(np.array(orig_probe.device_channel_indices))
        self.add_probe_row(new_probe)
    
    def load_probe_from_file(self):
        """ Load probe object from saved file, add to collection """
        probe,_ = ephys.select_load_probe_file(parent=self)
        if probe is None: return
        self.add_probe_row(probe)
    
    def design_probe(self):
        """ Launch probe designer popup"""
        probe_popup = ProbeObjectPopup()
        probe_popup.setModal(True)
        probe_popup.accept_btn.setVisible(True)
        probe_popup.accept_btn.setText('CHOOSE PROBE')
        res = probe_popup.exec()
        if res:
            probe = probe_popup.probe_widget.probe
            self.add_probe_row(probe)

    def check_assignments(self):
        """ Check for valid assignment upon probe addition/deletion/reindexing """
        # list probe(s) associated with each data row
        items = pyfx.layout_items(self.qlayout)
        
        # allow different-size probes in block mode, but disable in alternate mode
        x = len(np.unique([item.nch for item in items])) < 2
        self.inter_radio.setEnabled(x)
        
        ALL_ROWS = {}
        for k in np.arange(self.nrows):
            ALL_ROWS[k] = [i for i,item in enumerate(items) if k in item.ROWS]
            
        # probe config is valid IF each row is matched with exactly 1 probe
        matches = [len(x)==1 for x in ALL_ROWS.values()]
        nvalid = len(np.nonzero(matches)[0])
        is_valid = bool(nvalid == self.nrows)
        
        probe_strings = [', '.join(np.array(x, dtype=str)) for x in ALL_ROWS.values()]
        self.data_assign_df = pd.DataFrame({'Row':ALL_ROWS.keys(), # assignment dataframe
                                            'Probe(s)':probe_strings})
        self.probe_done_btn.setEnabled(is_valid)  # require valid config for next step
        self.data_lbl.setText(self.data_txt_fmt % (['red','green'][int(is_valid)], nvalid))
        self.check_signal.emit()
        

class RawArrayPopup(QtWidgets.QDialog):
    """ Interface for user-provided metadata for NPY/MAT recordings """
    supported_units = ['uV', 'mV', 'V', 'kV']
    
    def __init__(self, data_shape, fs=None, units=None, filename='', parent=None):
        super().__init__(parent)
        assert len(data_shape) == 2, 'Data array must be 2-dimensional.'
        
        self.gen_layout(data_shape, fs, units)
        self.connect_signals()
        self.setWindowTitle(filename)
        
    def gen_layout(self, data_shape, fs, units):
        """ Set up layout """
        nrows,ncols = data_shape
        if nrows > ncols: # rows == samples
            self.nts, self.nch = data_shape
            row_lbl, col_lbl = ['samples', 'channels']
        else: # rows == channels
            self.nch, self.nts = data_shape
            row_lbl, col_lbl = ['channels', 'samples']
        txt = f'<code>{nrows} {row_lbl} x {ncols} {col_lbl}</code>'
        qlbl = QtWidgets.QLabel(txt)
        qlbl.setStyleSheet('QLabel {background-color : white; padding : 4px;}')
        qtitle = QtWidgets.QLabel('Data dimensions:')
        self.lbl_row = pyfx.get_widget_container('h', qtitle, qlbl, stretch_factors=[2,0],
                                                 widget='widget')
        # create sampling rate and unit inputs
        self.fs_w = gi.LabeledSpinbox('Sampling rate', double=True, minimum=1, 
                                      maximum=9999999999, suffix=' Hz')
        self.dur_w = gi.LabeledSpinbox('Duration', double=True, minimum=0.0001,
                                       maximum=9999999999, suffix=' s', decimals=4)
        self.dur_w.qw.setReadOnly(True)
        if fs is not None:
            self.fs_w.setValue(fs)
            self.dur_w.setValue(self.nts / fs)
        self.units_w = gi.LabeledCombobox('Units')
        self.units_w.addItems(self.supported_units)
        if units in self.supported_units:
            self.units_w.setCurrentText(units)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        hlay = pyfx.get_widget_container('h', self.fs_w, self.dur_w, self.units_w)
        
        # create action buttons
        self.bbox = QtWidgets.QWidget()
        bbox_lay = QtWidgets.QHBoxLayout(self.bbox)
        self.confirm_meta_btn = QtWidgets.QPushButton('Continue')
        self.close_btn = QtWidgets.QPushButton('Cancel')
        bbox_lay.addWidget(self.close_btn)
        bbox_lay.addWidget(self.confirm_meta_btn)
        
        self.layout.addWidget(self.lbl_row)
        self.layout.addWidget(pyfx.DividerLine(lw=1, mlw=1))
        self.layout.addLayout(hlay)
        self.layout.addWidget(self.bbox)
    
    def connect_signals(self):
        """ Connect GUI inputs """
        self.fs_w.qw.valueChanged.connect(lambda x: self.update_fs_dur(x, 0))
        self.dur_w.qw.valueChanged.connect(lambda x: self.update_fs_dur(x, 1))
        
        self.confirm_meta_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)
        
    def update_fs_dur(self, val, mode):
        """ Update duration from sampling rate (mode=0) or vice versa (mode=1) """
        #nts = self.data.shape[self.cols_w.currentIndex()]
        if mode==0:
            # calculate recording duration from sampling rate
            pyfx.stealthy(self.dur_w.qw, self.nts / self.fs_w.value())
        elif mode==1:
            pyfx.stealthy(self.fs_w.qw, self.nts / self.dur_w.value())


##############################################################################
##############################################################################
################                                              ################
################                 MAIN INTERFACE               ################
################                                              ################
##############################################################################
##############################################################################


class RawRecordingSelectionPopup(QtWidgets.QDialog):
    """ Data processing GUI for raw recording selection and probe assignment """
    recording = None
    last_saved_ddir = None
    
    def __init__(self, init_ppath=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select a raw data source')
        
        # load parameters, get initial directories
        self.PARAMS = ephys.read_params()
        raw_base, probe_base, probe_file, _  = ephys.base_dirs()
        self.default_probe_file = probe_file
        
        if init_ppath is None or not os.path.exists(init_ppath):
            init_ppath = str(raw_base)
        
        self.gen_layout()
        self.connect_signals()
        self.fsw.update_filepath(init_ppath)
        self.probe_gbox.hide()
        self.save_gbox.hide()
        QtCore.QTimer.singleShot(10, lambda: self.ddir_gbox.setFixedHeight(self.ddir_gbox.height()))
        
    def gen_layout(self):
        """ Set up layout """
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(5)
        gbox_ss = 'QGroupBox {background-color : rgba(230,230,230,255);}'
        #self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        
        ### raw recording selection
        self.ddir_container = QtWidgets.QSplitter()
        self.ddir_gbox = QtWidgets.QGroupBox()
        self.ddir_gbox.setStyleSheet(gbox_ss)
        ddir_vbox = pyfx.InterWidgets(self.ddir_gbox, 'v')[2]
        # basic directory selection widget
        self.fsw = RawRecordingSelectionWidget(title='<b><u>Raw data source</u></b>')
        ddir_vbox.addWidget(self.fsw)
        def i2rc(i, ncols=3):
            irow = int(i/ncols)
            icol = i - (irow * ncols)
            return (irow,icol)
        # buttons for supported data formats
        self.radio_btns = {}
        mygrid = QtWidgets.QGridLayout()
        for i,(k,(lbl,ext)) in enumerate(dp.supported_formats.items()):
            btn = QtWidgets.QRadioButton(f'{lbl} ({ext})')
            btn.setAutoExclusive(False)
            btn.setEnabled(False)
            btn.setStyleSheet('QRadioButton:disabled {color : gray;}'
                              'QRadioButton:disabled::checked {color : black;}')
            self.radio_btns[k] = btn
            mygrid.addWidget(btn, *i2rc(i))
        self.fsw.vlay.addLayout(mygrid)
        # raw recording metadata display
        self.meta_qform = gi.RawMeta()
        meta_qscroll = pyfx.get_widget_container('v', self.meta_qform, widget='scroll')
        self.meta_gbox = pyfx.get_widget_container('v', meta_qscroll, widget='groupbox')
        self.ddir_container.addWidget(self.ddir_gbox)
        self.ddir_container.addWidget(self.meta_gbox)
        self.meta_gbox.hide()
        
        ### results directory location
        save_lbl = QtWidgets.QLabel('<u>Results Dir.</u>')
        self.save_le = QtWidgets.QLineEdit()
        self.save_le.setReadOnly(True)
        self.save_ddir_btn = QtWidgets.QPushButton() # file dialog launch button
        self.save_ddir_btn.setIcon(QtGui.QIcon(':/icons/folder.png'))
        self.save_ddir_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.save_gbox = pyfx.get_widget_container('h', save_lbl, self.save_le, 
                                                   self.save_ddir_btn, spacing=5, 
                                                   widget='frame')
        
        ### settings widget
        self.settings_w = QtWidgets.QWidget()
        settings_vlay = QtWidgets.QVBoxLayout(self.settings_w)
        settings_vlay.setContentsMargins(0,0,0,0)
        # initialize main parameter input widget, embed in scroll area
        self.params_widget = qparam.ParamObject(params=dict(self.PARAMS), 
                                                mode='data_processing', parent=self)
        self.qscroll = QtWidgets.QScrollArea()
        self.qscroll.horizontalScrollBar().hide()
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setWidget(self.params_widget)
        qh = pyfx.ScreenRect(perc_height=0.25, keep_aspect=False).height()
        self.qscroll.setMaximumHeight(qh)
        self.qscroll.hide()
        # create settings button to show/hide param widgets
        self.params_bar = QtWidgets.QPushButton('Settings')
        self.params_bar.setCheckable(True)
        self.params_bar.setFocusPolicy(QtCore.Qt.NoFocus)
        self.params_bar.setStyleSheet(pyfx.dict2ss(QSS.EXPAND_PARAMS_BTN))
        # create icon to indicate warnings about 1 or more parameter inputs
        self.params_warning_btn = QtWidgets.QPushButton()
        self.params_warning_btn.setObjectName('params_warning')
        self.params_warning_btn.setCheckable(True)
        self.params_warning_btn.setFlat(True)
        self.params_warning_btn.setFixedSize(25,25)
        self.params_warning_btn.setEnabled(False)
        self.params_warning_btn.setStyleSheet(pyfx.dict2ss(QSS.ICON_BTN))
        self.params_warning_btn.setFlat(True)
        self.params_warning_btn.hide()
        # create button to access base folder popup
        self.base_folder_btn = QtWidgets.QPushButton('Set base folders ... ')
        self.base_folder_btn.setAutoDefault(False)
        self.base_folder_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        # create button to display raw recording metadata
        self.meta_btn = QtWidgets.QPushButton()
        self.meta_btn.setCheckable(True)
        self.meta_btn.setStyleSheet(pyfx.dict2ss(QSS.META_TOGGLE_BTN))
        self.meta_btn.setFixedWidth(self.meta_btn.sizeHint().height())
        self.meta_btn.hide()
        bbar = QtWidgets.QHBoxLayout()
        bbar.addWidget(self.params_warning_btn, stretch=0)
        bbar.addWidget(self.params_bar, stretch=2)
        bbar.addWidget(self.base_folder_btn, stretch=0)
        bbar.addWidget(self.meta_btn, stretch=0)
        settings_vlay.addLayout(bbar)
        
        #self.settings_body = QtWidgets.QWidget()
        self.settings_container = QtWidgets.QSplitter()
        self.settings_container.addWidget(self.qscroll)
        settings_vlay.addWidget(self.settings_container)
        
        ### probe assignment
        self.probe_gbox = QtWidgets.QGroupBox()
        self.probe_gbox.setStyleSheet('QGroupBox {border-width : 0px;'
                                      'font-weight : bold; text-decoration : underline;}')
        self.probe_vbox = pyfx.InterWidgets(self.probe_gbox, 'v')[2]
        self.paw = None
        
        ### action buttons
        bbox = QtWidgets.QHBoxLayout()
        self.back_btn = QtWidgets.QPushButton('Back')
        self.back_btn.setVisible(False)
        self.probe_map_btn = QtWidgets.QPushButton('Map to probe(s)')
        #self.probe_map_btn.setStyleSheet(blue_btn_ss)
        self.probe_map_btn.setEnabled(False)
        self.pipeline_btn = QtWidgets.QPushButton('Process data!')
        #self.pipeline_btn.setStyleSheet(blue_btn_ss)
        self.pipeline_btn.setVisible(False)
        self.pipeline_btn.setEnabled(False)
        bbox.addWidget(self.back_btn)
        bbox.addWidget(self.probe_map_btn)
        bbox.addWidget(self.pipeline_btn)
        # set central layout
        #self.layout.addWidget(self.ddir_gbox)
        self.layout.addWidget(self.ddir_container)
        self.layout.addWidget(self.settings_w)
        self.layout.addWidget(self.probe_gbox)
        self.layout.addWidget(self.save_gbox)
        line0 = pyfx.DividerLine()
        self.layout.addWidget(line0)
        self.layout.addLayout(bbox)
        self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        
        # "loading" spinner animation
        self.spinner_window = gi.SpinnerWindow(self)
        self.spinner_window.spinner.setInnerRadius(25)
        self.spinner_window.spinner.setNumberOfLines(10)
        #self.spinner_window.layout.setContentsMargins(5,5,5,5)
        self.spinner_window.layout.setSpacing(0)
        #self.spinner_window.adjust_labelSize(lw=2.5, lh=0.65, ww=3)
        
    def connect_signals(self):
        """ Connect GUI inputs """
        self.fsw.ppath_btn.clicked.connect(self.fsw.select_filepath)
        self.fsw.signal.connect(self.ppath_updated)
        self.fsw.read_array_signal.connect(self.load_array_worker)
        self.params_widget.warning_update_signal.connect(self.update_param_warnings)
        self.probe_map_btn.clicked.connect(self.start_probe_assignment)
        self.pipeline_btn.clicked.connect(self.pipeline_worker)
        self.back_btn.clicked.connect(self.back_to_selection)
        self.save_ddir_btn.clicked.connect(self.set_save_ddir)
        self.params_bar.toggled.connect(lambda x: self.qscroll.setVisible(x))
        self.base_folder_btn.clicked.connect(self.base_folder_popup)
        self.meta_btn.toggled.connect(lambda x: self.meta_gbox.setVisible(x))
    
    def create_workers(self):
        """ Parallel thread for long-running processing steps """
        self.worker_thread = QtCore.QThread()
        self.worker_object.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker_object.run)
        self.worker_object.progress_signal.connect(self.spinner_window.report_progress_string)
        self.worker_object.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_object.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.finished_slot)
    
    def start_qthread(self):
        """ Start worker thread """
        self.create_workers()
        self.spinner_window.start_spinner()
        self.worker_thread.start()
    
    @QtCore.pyqtSlot()
    def finished_slot(self):
        """ Worker thread stopped """
        self.spinner_window.stop_spinner() # stop "loading" spinner icon
        self.worker_object = None
        self.worker_thread = None
        
    def ppath_updated(self, x):
        """ Check if raw data source is valid """
        if x:
            ppath = self.fsw.le.text()
            data_type = dp.get_data_format(ppath)
        else:
            data_type = None
        for k,btn in self.radio_btns.items():
            btn.setChecked(k==data_type)
        self.probe_map_btn.setEnabled(x)
        self.meta_btn.setVisible(x)
        if x:
            self.extractor_worker(ppath, data_type) # load new recording
        else:
            self.recording = None # delete previous recording (if any)
            self.params_widget.trange.set_duration(np.inf)
            self.params_widget.lfp_fs.set_fs(None)
            self.meta_qform.update_recording(None)
            self.meta_btn.setChecked(False)
        
    ### array reader
        
    def load_array_worker(self, ppath):
        """ Read raw data array into memory """
        self.worker_object = ArrayReader(ppath)
        self.worker_object.data_signal.connect(self.load_array_finished_slot)
        self.worker_object.error_signal.connect(self.load_array_error_slot)
        self.start_qthread()
        
    @QtCore.pyqtSlot(str, dict, dict)
    def load_array_finished_slot(self, fpath, data_dict, meta):
        """ Pass valid array and initial metadata to the user """
        if len(data_dict) > 1:  # file contains multiple 2-dimensional datasets
            lbl = '<u>Select LFP dataset</u>'
            dlg = gi.ButtonPopup(*data_dict.keys(), label=lbl, parent=self)
            for b in dlg.btns:
                nrows, ncols = data_dict[b.text()].shape
                b.setToolTip(f'{nrows} rows x {ncols} columns')
            if dlg.exec(): # user selected LFP dataset key
                data_array = data_dict[dlg.result]
            else: # user closed dataset selection window
                gi.MsgboxError('Aborted LFP dataset selection.', parent=self).exec()
                return
        else: # file contains only one 2-dimensional dataset
            data_array = list(data_dict.values())[0]
        # manually confirm sampling rate, units, etc
        self.fsw.enter_array_metadata(fpath, data_array, meta)
        
    @QtCore.pyqtSlot(str, str)
    def load_array_error_slot(self, ppath, error_msg):
        """ Handle invalid NPY/MAT files """
        gi.MsgboxError(error_msg, parent=self).exec()
        self.fsw.update_filepath(ppath, False)
        
    def get_nwb_dataset(self, ppath):
        """ Identify target ephys dataset in NWB file """
        try:  # try getting names of available ElectricalSeries in NWB file
            eseries = dp.get_nwb_eseries(ppath)
        except Exception as e:  # file missing or corrupted
            gi.MsgboxError(f'Error: {str(e)}', parent=self).exec()
            return False
        if len(eseries) == 0:   # no analyzable data
            gi.MsgboxError('NWB file contains no ElectricalSeries '
                           'with valid electrodes.', parent=self).exec()
            return False
        elif len(eseries) == 1: # return the only available dataset
            return eseries[0]
        elif len(eseries) > 1:  # ask user to select a target dataset
            lbl = '<u>Select an ElectricalSeries dataset</u>'
            dlg = gi.ButtonPopup(*eseries, label=lbl, parent=self)
            if dlg.exec():
                return str(dlg.result)
            else:
                gi.MsgboxError('Aborted NWB dataset selection.', parent=self).exec()
                return False
            
    def extractor_worker(self, ppath, data_type):
        """ Instantiate spikeinterface Extractor for raw recording """
        data_array, metadata = None, {}
        if data_type in ['NPY','MAT']:  # create recording from loaded data/metadata
            data_array, metadata = self.fsw.data_array, self.fsw.meta
            if data_array.shape[0] == metadata['nch']:
                data_array = data_array.T  # rows=samples, columns=channels
        electrical_series_path = None
        if data_type == 'NWB':
            es_name = self.get_nwb_dataset(ppath)
            if es_name == False:
                self.fsw.update_filepath(ppath, False)
                return
            electrical_series_path = f'acquisition/{es_name}'
        KW = dict(data_array=data_array, metadata=metadata, 
                  electrical_series_path=electrical_series_path)
        # create extractor worker
        self.worker_object = ExtractorWorker(ppath, **KW)
        self.worker_object.data_signal.connect(self.extractor_finished_slot)
        self.worker_object.error_signal.connect(self.extractor_error_slot)
        self.start_qthread()
    
    @QtCore.pyqtSlot(object)
    def extractor_finished_slot(self, recording):
        """ Load recording data into pipeline """
        self.recording = recording
        META = dp.get_meta_from_recording(self.recording)
        # update parameter widgets with loaded recording
        self.params_widget.lfp_fs.set_fs(float(META['fs']))
        self.params_widget.trange.set_duration(self.recording.get_duration())
        self.meta_qform.update_recording(self.recording)
    
    @QtCore.pyqtSlot(str, str)
    def extractor_error_slot(self, ppath, error_msg):
        """ Handle data extraction errors """
        gi.MsgboxError(error_msg, parent=self).exec()
        self.fsw.update_filepath(ppath, False)
        
    def start_probe_assignment(self):
        """ Initiate probe mapping phase """
        NCH = self.recording.get_num_channels()
        # initialize probe box
        self.paw = ProbeAssignmentWidget(NCH)
        self.paw.check_signal.connect(self.update_probe_config)
        self.probe_vbox.addWidget(self.paw)
        self.probe_gbox.setVisible(True)
        self.save_gbox.setVisible(True)
        self.back_btn.setVisible(True)
        self.probe_map_btn.setVisible(False)
        self.pipeline_btn.setVisible(True)
        
        # initialize save box
        ppath = self.recording.get_annotation('ppath')
        raw_ddir = os.path.dirname(ppath)
        init_save_ddir = str(Path(raw_ddir, pyfx.unique_fname(raw_ddir, 'processed_data')))
        self.save_le.setText(init_save_ddir)
        # try loading and adding default probe if it meets the criteria
        dflt_probe = ephys.read_probe_file(self.default_probe_file)
        if (dflt_probe is not None) and (dflt_probe.get_contact_count() <= NCH):
            self.paw.add_probe_row(dflt_probe)
        # disable data loading
        self.ddir_gbox.setEnabled(False)
        self.setWindowTitle('Map to probe(s)')
        pyfx.center_window(self)
        
    def assemble_probe_group(self):
        """ Return ProbeGroup object from row items in probe assignment widget """
        PROBE_GROUP = prif.ProbeGroup()
        items = pyfx.layout_items(self.paw.qlayout)
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
        
    def update_probe_config(self):
        """ Check for valid probe assignment """
        x = bool(self.paw.probe_done_btn.isEnabled())
        self.paw.prb_icon_btn.new_status(x)
        self.enable_pipeline_btn()
    
    def update_param_warnings(self, n, ddict):
        """ Check for any input parameter warnings """
        x = n > 0
        self.params_warning_btn.setChecked(x)
        self.params_warning_btn.setVisible(x)
        self.enable_pipeline_btn()
    
    def enable_pipeline_btn(self):
        """ Enable processing pipeline """
        a = self.paw is not None and self.paw.probe_done_btn.isEnabled()
        b = not self.params_warning_btn.isChecked()
        self.pipeline_btn.setEnabled(a and b)
    
    ### processing pipeline
    
    def pipeline_worker(self):
        """ Run processing pipeline """
        META = dp.get_meta_from_recording(self.recording)
        # create probe group with all probe objects used in the recording
        self.PROBE_GROUP = self.assemble_probe_group()
        # get updated analysis parameters
        PARAMS = self.params_widget.DEFAULTS  # validated input params
        param_dict = self.params_widget.ddict_from_gui()[0]  # current data processing params
        PARAMS.update(param_dict)
        # create empty folder for processed data
        save_ddir = self.save_le.text()
        if os.path.isdir(save_ddir):
            shutil.rmtree(save_ddir)  # delete existing directory
        if os.path.isfile(save_ddir):
            os.remove(save_ddir)  # delete existing file
        os.makedirs(save_ddir)
        # create data processor worker
        self.worker_object = DataWorker()
        self.worker_object.init_source(self.recording, self.PROBE_GROUP, 
                                       PARAMS=PARAMS, SAVE_DDIR=save_ddir)
        self.worker_object.data_signal.connect(self.pipeline_finished_slot)
        self.worker_object.error_signal.connect(self.pipeline_error_slot)
        self.start_qthread()
    
    @QtCore.pyqtSlot()
    def pipeline_finished_slot(self):
        """ Worker successfully completed the processing pipeline """
        self.last_saved_ddir = str(self.save_le.text())
        msg = 'Data processing complete!<br><br>Load another recording?'
        res = gi.MsgboxSave(msg, parent=self).exec()
        if res == QtWidgets.QMessageBox.Yes:
            self.back_btn.click()  # select another recording for processing
        else:  # close window
            self.accept()
    
    @QtCore.pyqtSlot(str)
    def pipeline_error_slot(self, error_msg):
        """ Worker encountered an error in the processing pipeline """
        gi.MsgboxError(error_msg, parent=self).exec()
        # delete incomplete recording folder
        save_ddir = self.save_le.text()
        if os.path.isdir(save_ddir):
            folder_name = os.path.basename(save_ddir)
            filestring = ', '.join(map(lambda f: f'"{f}"', os.listdir(save_ddir)))
            print(f'"{folder_name}" contains the following file(s): {filestring}' + os.linesep)
            res = ''
            while res.lower() not in ['y','n']:
                res = input('Delete folder? (y/n) --> ')
            print('')
            if res == 'y':
                shutil.rmtree(save_ddir) # delete incomplete recording folder
                print(f'"{folder_name}" folder deleted!')
    
    def set_save_ddir(self):
        """ Select location of processed data folder """
        init_ddir = os.path.dirname(self.save_le.text())
        init_dirname = os.path.basename(self.save_le.text())
        save_ddir = ephys.select_save_directory(init_ddir=init_ddir, init_dirname=init_dirname,
                                                title='Set processed data directory', parent=self)
        if save_ddir:
            self.save_le.setText(save_ddir)
    
    def back_to_selection(self):
        """ Return to the raw data selection step """
        self.setWindowTitle('Select a raw data source')
        self.back_btn.setVisible(False)
        self.probe_map_btn.setVisible(True)
        self.pipeline_btn.setVisible(False)
        self.pipeline_btn.setEnabled(False)
        self.ddir_gbox.setEnabled(True)
        self.probe_gbox.setVisible(False)
        self.save_gbox.setVisible(False)
        # delete probe assignment widget
        self.probe_vbox.removeWidget(self.paw)
        self.paw.deleteLater()
        self.paw = None
    
    def base_folder_popup(self):
        """ Allow user to set base folders from pipeline GUI """
        raw_ddir, probe_ddir, probe_file, _ = ephys.base_dirs()
        self.basedirs_popup = gi.BaseFolderPopup()
        self.basedirs_popup.widget.param_w.hide() # cannot change param file
        if self.basedirs_popup.exec():
            raw_ddir2, probe_ddir2, probe_file2, _ = ephys.base_dirs()
            # if current filepath is the raw data folder, update it
            if (self.fsw.le.text()==raw_ddir) and raw_ddir != raw_ddir2:
                self.fsw.update_filepath(raw_ddir2)
            # update default probe configuration file
            self.default_probe_file = str(probe_file2)
        
if __name__ == '__main__':
    app = pyfx.qapp()
    qfd = QtWidgets.QFileDialog()
    init_ddir = str(qfd.directory().path())
    if init_ddir == os.getcwd():
        init_ddir=None
    w = RawRecordingSelectionPopup()
    w.show()
    w.raise_()
    sys.exit(app.exec())