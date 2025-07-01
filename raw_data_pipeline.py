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
    data_signal = QtCore.pyqtSignal(str, np.ndarray, dict)
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
                data_array = np.array(file_data)
                meta = {'fs':None, 'units':None}
            elif isinstance(file_data, dict):     # data saved as dictionary
                self.progress_signal.emit('Parsing data dictionary ...')
                try:  # get data array (required) and SR/unit metadata (optional)
                    data_array, meta = dp.read_data_from_dict(file_data)
                except: error_msg = f'Error: Could not parse data from "{fname}".'
            else: error_msg = 'Error: File must contain data array or dictionary.'
        except: error_msg = f'Error: Unable to load "{fname}".'
        if error_msg == '' and data_array.ndim != 2: # data array not 2-dimensional
            error_msg = 'Error: Data must be a 2-dimensional array.'
        # return data or error message
        if error_msg == '':
            self.data_signal.emit(str(fpath), data_array, meta)
            self.progress_signal.emit('Done!')
        else:
            self.error_signal.emit(str(fpath), error_msg)
        self.finished.emit()
        
        
class ExtractorWorker(QtCore.QObject):
    """ Returns spikeinterface extractor objects for raw recordings """
    progress_signal = QtCore.pyqtSignal(str)
    data_signal = QtCore.pyqtSignal(object)
    error_signal = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    
    def __init__(self, fpath, data_array=None, metadata={}):
        """ Initialize recording filepath and raw data (if previously loaded) """
        super().__init__()
        self.fpath = fpath
        self.data_array = data_array
        self.metadata = metadata
    
    def run(self):
        """ Get recording extractor """
        fpath, data_array, metadata = self.fpath, self.data_array, self.metadata
        data_type = dp.get_data_format(fpath)
        self.progress_signal.emit(f'Getting {data_type} extractor ...')
        try:  # get spikeinterface extractor for selected data type
            recording = dp.get_extractor(fpath, data_type, data_array=data_array,
                                         metadata=metadata)
        except: # failed to load extractor object
            self.error_signal.emit(f'Error: Unable to load {data_type} extractor.')
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
        load_win    = kwargs.get('load_win', 600)
        recording.set_probegroup(probe_group, in_place=True)
        META = dp.get_meta_from_recording(recording)
        # get raw data directory, create processed data folder
        ppath, data_format = [recording.get_annotation(x) for x in ['ppath','data_format']]
        
        # parse metadata and analysis parameters
        FS, NSAMPLES, TOTAL_CH = META['fs'], META['nsamples'], META['total_ch']
        lfp_fs = PARAMS['lfp_fs']; ds_factor = int(FS/lfp_fs) # downsampling factor
        tstart, tend = PARAMS['trange']  # extract data between timepoints
        iistart, iiend = dp.get_rec_bounds(NSAMPLES, FS, tstart, tend)
        NSAMPLES_DN_TRUNC = int((iiend-iistart) / ds_factor)
        global_dev_idx = probe_group.get_global_device_channel_indices()['device_channel_indices']
        nprobes = len(probe_group.probes)
        
        # get IO object (fid, cont, or recording)
        try:
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
            
        # extract and downsample LFPs in ~10 min chunks
        chunkfunc, (iichunk,_) = dp.get_chunkfunc(load_win, FS, NSAMPLES, lfp_fs=lfp_fs,
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
                    snip_dn = dp.downsample_chunk(snip, ds_factor)
                    dset[:, aa:bb] = snip_dn
                except Exception as e:
                    self.quit_thread(f'Downsampling Error: {str(e)}', ff=ff, KW=KW)
                    return
            if jj >= iiend:
                break
            count += 1
        if KW['fid'] is not None:
            KW['fid'].close()
        
        for iprb,probe in enumerate(probe_group.probes):
            hdr = f'ANALYZING PROBE {iprb+1} / {nprobes}<br>'
            LFP_dict = ff[str(iprb)]['LFP']
            LFP_raw = LFP_dict['raw']; NCH = len(LFP_raw)
            shank_ids = np.array(probe.shank_ids, dtype='int')
            # bandpass filter signals and get mean channel amplitudes
            self.progress_signal.emit(hdr + '<br>Bandpass filtering signals ...')
            std_dict = {}
            for fb in ['theta', 'slow_gamma', 'fast_gamma', 'swr_freq', 'ds_freq']:
                fbk = fb.replace('_freq','')
                try:
                    farr = LFP_dict.create_dataset(fbk, dtype='float32', 
                                    data=pyfx.butter_bandpass_filter(LFP_raw, *PARAMS[fb], 
                                                                     lfp_fs=lfp_fs, axis=1))
                    std_dict[fbk] = np.std(farr, axis=1)
                    std_dict[f'norm_{fbk}'] = pyfx.Normalize(std_dict[fbk])
                except Exception as e:
                    self.quit_thread(f'Filtering Error: {str(e)}', ff=ff)
                    return
            STD = pd.DataFrame(std_dict)
            STD.to_hdf(ff.filename, key=f'/{iprb}/STD')
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
            # save event dataframes and thresholds
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
        
    def read_data_array(self, ppath):
        """ Load NPY/MAT data locally (not implemented) """
        try:
            data_array, meta = dp.read_array_file(ppath, raise_exception=True)
            if data_array.ndim != 2: # data array not 2-dimensional
                gi.MsgboxError('Error: Data must be a 2-dimensional array.').exec()
                self.update_filepath(ppath, False)
                return
        except Exception as e: # failed to load data file
            gi.MsgboxError(f'Error: {str(e)}').exec()
            self.update_filepath(ppath, False)
            return
        # pass valid data array and initial metadata to pop-up window
        self.enter_array_metadata(ppath, data_array, meta)
    
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
        self.pipeline_btn = QtWidgets.QPushButton('PROCESS DATA')
        self.pipeline_btn.setEnabled(False)
    
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
            gi.MsgboxError(msg).exec()
            return
        
        if self.MODE == 1:
            lens = [item.nch for item in pyfx.layout_items(self.qlayout)] + [nch]
            try:
                assert len(np.unique(lens)) < 2  # alternate indexing requires all same-size probes
            except AssertionError:
                msg = 'Alternate indexing requires all probes to be the same size'
                gi.MsgboxError(msg).exec()
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
        self.pipeline_btn.setEnabled(is_valid)  # require valid config for next step
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
        _ = f'{nrows} {row_lbl} x {ncols} {col_lbl}'
        # create sampling rate and unit inputs
        self.fs_w = gi.LabeledSpinbox('Sampling rate', double=True, minimum=1, 
                                      maximum=9999999999, suffix=' Hz')
        if fs is not None:
            self.fs_w.setValue(fs)
        self.dur_w = gi.LabeledSpinbox('Duration', double=True, minimum=0.0001,
                                       maximum=9999999999, suffix=' s', decimals=4)
        self.dur_w.qw.setReadOnly(True)
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
        
    def gen_layout(self):
        """ Set up layout """
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(5)
        gbox_ss = 'QGroupBox {background-color : rgba(230,230,230,255);}'
        #self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        
        ### raw recording selection
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
        self.params_widget = qparam.ParamObject(params=dict(self.PARAMS), mode='data_processing', parent=self)
        self.qscroll = QtWidgets.QScrollArea()
        self.qscroll.horizontalScrollBar().hide()
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setWidget(self.params_widget)
        qh = pyfx.ScreenRect(perc_height=0.25, keep_aspect=False).height()
        self.qscroll.setMaximumHeight(qh)
        self.qscroll.hide()
        # create settings button to show/hide param widgets
        bbar = QtWidgets.QHBoxLayout()
        self.settings_btn0 = QtWidgets.QPushButton('Settings')
        self.settings_btn0.setStyleSheet('QPushButton {'
                                         'border : 1px solid rgba(0,0,0,50);'
                                         'background-color : rgba(0,0,0,10);'
                                         'color : black;'
                                         'icon : url(:/icons/settings.png);'
                                         'padding : 2px;'
                                         '}')
        self.settings_btn = QtWidgets.QPushButton()
        self.settings_btn.setStyleSheet('QPushButton {'
                                        'icon : url(:/icons/double_chevron_down.png);'
                                        'padding : 2px;'
                                        'outline : none;}'
                                        'QPushButton:checked {'
                                        'icon : url(:/icons/double_chevron_up.png);}')
        self.settings_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.settings_btn.setFixedWidth(self.settings_btn.sizeHint().height())
        self.settings_btn.setCheckable(True)
        #self.settings_btn.setChecked(True)
        self.settings_btn.toggled.connect(lambda x: self.qscroll.setVisible(x))
        bbar.addWidget(self.settings_btn0)
        bbar.addWidget(self.settings_btn)
        settings_vlay.addLayout(bbar)
        settings_vlay.addWidget(self.qscroll)
        
        ### probe assignment
        self.probe_gbox = QtWidgets.QGroupBox()
        self.probe_gbox.setStyleSheet('QGroupBox {border-width : 0px;'
                                      'font-weight : bold; text-decoration : underline;}')
        self.probe_vbox = pyfx.InterWidgets(self.probe_gbox, 'v')[2]
        
        ### action buttons
        bbox = QtWidgets.QHBoxLayout()
        self.back_btn = QtWidgets.QPushButton('Back')
        self.back_btn.setVisible(False)
        self.extract_btn = QtWidgets.QPushButton('Map to probe(s)')
        #self.extract_btn.setStyleSheet(blue_btn_ss)
        self.extract_btn.setEnabled(False)
        self.pipeline_btn = QtWidgets.QPushButton('Process data!')
        #self.pipeline_btn.setStyleSheet(blue_btn_ss)
        self.pipeline_btn.setVisible(False)
        self.pipeline_btn.setEnabled(False)
        bbox.addWidget(self.back_btn)
        bbox.addWidget(self.extract_btn)
        bbox.addWidget(self.pipeline_btn)
        # set central layout
        self.layout.addWidget(self.ddir_gbox)
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
        self.extract_btn.clicked.connect(self.extractor_worker)
        self.pipeline_btn.clicked.connect(self.pipeline_worker)
        self.back_btn.clicked.connect(self.back_to_selection)
        self.save_ddir_btn.clicked.connect(self.set_save_ddir)
        
    def ppath_updated(self, x):
        """ If raw data source is valid, enable probe assignment step """
        if x:
            ppath = self.fsw.le.text()
            data_type = dp.get_data_format(ppath)
        else:
            data_type = None
        for k,btn in self.radio_btns.items():
            btn.setChecked(k==data_type)
        self.extract_btn.setEnabled(x)
        
    def update_probe_config(self):
        """ If probe assignment is valid, enable processing step """
        x = bool(self.paw.pipeline_btn.isEnabled())
        self.paw.prb_icon_btn.new_status(x)
        self.pipeline_btn.setEnabled(bool(x and self.PARAMS is not None))
        
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
        
    ### array reader
        
    def load_array_worker(self, ppath):
        """ Worker loads raw data array into memory """
        self.worker_object = ArrayReader(ppath)
        self.worker_object.data_signal.connect(self.load_array_finished_slot)
        self.worker_object.error_signal.connect(self.load_array_error_slot)
        self.start_qthread()
        
    @QtCore.pyqtSlot(str, np.ndarray, dict)
    def load_array_finished_slot(self, fpath, data_array, meta):
        """ Worker returned valid data array and initial metadata """
        # pass LFP array and metadata dictionary to widget
        self.fsw.enter_array_metadata(fpath, data_array, meta)
        
    @QtCore.pyqtSlot(str, str)
    def load_array_error_slot(self, ppath, error_msg):
        """ Worker encountered an error while loading array """
        gi.MsgboxError(error_msg).exec()
        self.fsw.update_filepath(ppath, False)
    
    ### recording extractor
        
    def extractor_worker(self):
        """ Worker instantiates a spikeinterface Extractor for a raw recording """
        ppath = self.fsw.le.text()
        data_type = dp.get_data_format(ppath)
        data_array, metadata = None, {}
        if data_type in ['NPY','MAT']:  # create recording from loaded data/metadata
            data_array, metadata = self.fsw.data_array, self.fsw.meta
            if data_array.shape[0] == metadata['nch']:
                data_array = data_array.T  # rows=samples, columns=channels
        # create extractor worker
        self.worker_object = ExtractorWorker(ppath, data_array, metadata)
        self.worker_object.data_signal.connect(self.extractor_finished_slot)
        self.worker_object.error_signal.connect(self.extractor_error_slot)
        self.start_qthread()
    
    @QtCore.pyqtSlot(object)
    def extractor_finished_slot(self, recording):
        """ Worker returned valid extractor object """
        self.recording = recording
        NCH = self.recording.get_num_channels()
        # initialize probe box
        self.paw = ProbeAssignmentWidget(NCH)
        self.paw.check_signal.connect(self.update_probe_config)
        self.probe_vbox.addWidget(self.paw)
        self.probe_gbox.setVisible(True)
        self.save_gbox.setVisible(True)
        self.back_btn.setVisible(True)
        self.extract_btn.setVisible(False)
        self.pipeline_btn.setVisible(True)
        self.settings_btn.setChecked(False)
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
        
    @QtCore.pyqtSlot(str)
    def extractor_error_slot(self, error_msg):
        """ Worker encountered an error while extracting recording """
        gi.MsgboxError(error_msg).exec()
        
    ### processing pipeline
    
    def pipeline_worker(self):
        """ Worker starts the processing pipeline """
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
        res = gi.MsgboxSave(msg).exec()
        if res == QtWidgets.QMessageBox.Yes:
            self.back_btn.click()  # select another recording for processing
        else:  # close window
            self.accept()
    
    @QtCore.pyqtSlot(str)
    def pipeline_error_slot(self, error_msg):
        """ Worker encountered an error in the processing pipeline """
        gi.MsgboxError(error_msg).exec()
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
        self.extract_btn.setVisible(True)
        self.pipeline_btn.setVisible(False)
        self.ddir_gbox.setEnabled(True)
        self.probe_gbox.setVisible(False)
        self.save_gbox.setVisible(False)
        # delete probe assignment widget
        self.probe_vbox.removeWidget(self.paw)
        self.paw.deleteLater()
        self.paw = None
        
        
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