#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:46:44 2025

@author: amandaschott
"""
import sys
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import time
import pdb
# custom modules
import QSS
import pyfx
import ephys
import data_processing as dp
import gui_items as gi
from channel_selection_gui import ChannelSelectionWindow
from ds_classification_gui import DS_CSDWindow


##############################################################################
##############################################################################
################                                              ################
################                 WORKER OBJECTS               ################
################                                              ################
##############################################################################
##############################################################################


class ConverterBot(QtCore.QObject):
    """ Converts recording folder from previous (NPZ/NPY) to current (HDF5) format """
    progress_signal = QtCore.pyqtSignal(str)
    data_signal = QtCore.pyqtSignal(list)
    finished = QtCore.pyqtSignal()
    
    ddir = None
    transferred = []  # store names of converted files
    
    def run(self):
        """ Organize recording data in HDF5 files """
        ddir = self.ddir
        ff = h5py.File(Path(ddir, 'DATA.hdf5'), 'w', track_order=True)
        probes = ephys.read_probe_group(ddir).probes
        # convert LFP timestamps and sampling rate
        if 'lfp_time.npy' in os.listdir(ddir):
            self.report_progress('lfp_time.npy')
            ff.create_dataset('lfp_time', data=np.load(Path(ddir, 'lfp_time.npy')))
        if  'lfp_fs.npy' in os.listdir(ddir):
            self.report_progress('lfp_fs.npy')
            ff.attrs['lfp_fs'] = int(np.load(Path(ddir, 'lfp_fs.npy')))
        # convert bandpass-filtered LFP signals and summary stats table
        if 'lfp_bp.npz' in os.listdir(ddir):
            self.report_progress('lfp_bp.npz')
            npz = np.load(Path(ddir, 'lfp_bp.npz'), allow_pickle=True, mmap_mode='r')
            for iprb in range(len(probes)):
                PROBE_DSET = ff.create_group(str(iprb), track_order=True)
                PROBE_LFPS = PROBE_DSET.create_group('LFP', track_order=True)
                for k in npz.keys():
                    PROBE_LFPS.create_dataset(k, data=npz[k][iprb])
            npz.close()
        if 'channel_bp_std' in os.listdir(ddir):
            self.report_progress('channel_bp_std')
            STD_list = ephys.csv2list(ddir, 'channel_bp_std')
            for iprb,STD in enumerate(STD_list):
                STD.to_hdf(ff.filename, key=f'/{iprb}/STD')
        # convert ripple and DS event dataframes
        for fname in ['ALL_SWR','ALL_DS']:
            if fname in os.listdir(ddir):
                self.report_progress(fname)
                DF_list = ephys.csv2list(ddir, fname)
                for iprb,DF in enumerate(DF_list):
                    if not 'shank' in DF.columns:
                        DF['shank'] = pd.Series(probes[iprb].shank_ids.astype('int'))
                    DF.to_hdf(ff.filename, key=f'/{iprb}/{fname}')
        if 'THRESHOLDS.npy' in os.listdir(ddir):  # convert detection thresholds
            self.report_progress('THRESHOLDS.npy')
            threshold_list = list(np.load(Path(ddir, 'THRESHOLDS.npy'), allow_pickle=True))
            for iprb, thres_dict in enumerate(threshold_list):
                for k,v in thres_dict.items():
                    thres_df = pd.DataFrame(v).T
                    thres_df.to_hdf(ff.filename, key=f'/{iprb}/{k}_THRES')
        if 'noise_channels.npy' in os.listdir(ddir):  # convert noise channels
            self.report_progress('noise_channels.npy')
            noise_list = list(np.load(Path(ddir, 'noise_channels.npy')))
            for iprb,noise in enumerate(noise_list):
                ff[f'{iprb}']['NOISE'] = np.array(noise, dtype='int')
        # convert event channels, organize by probe -> shank -> event type
        event_channel_dict = ephys.init_event_channels(ddir, probes=probes, psave=False)
        for iprb,pdict in event_channel_dict.items():
            epath = Path(ddir, f'theta_ripple_hil_chan_{iprb}.npy')
            if os.path.isfile(epath):
                self.report_progress(f'theta_ripple_hil_chan_{iprb}.npy')
                event_channels = list(np.load(epath, allow_pickle=True))
                for ii,ll in enumerate(event_channels):
                    if len(ll)==3: pdict[ii] = list(ll)
            event_channel_dict[iprb] = pdict
        np.save(Path(ddir, 'theta_ripple_hil_chan.npy'), event_channel_dict, allow_pickle=True)
        # convert saved CSDs and DS classifications
        gg = h5py.File(Path(ddir, 'CSDs.hdf5'), 'w', track_order=True)
        for iprb,probe in enumerate(probes):
            if f'ds_csd_{iprb}.npz' in os.listdir(ddir):
                self.report_progress(f'ds_csd_{iprb}.npz')
                #CSD_dict = gg.create_group(str(iprb), track_order=True)
                csd_npz = np.load(Path(ddir, f'ds_csd_{iprb}.npz'), allow_pickle=True, mmap_mode='r')
                for ishank,shank in enumerate(probe.get_shanks()):
                    if str(ishank) in csd_npz:
                        K = f'/{iprb}/{ishank}'
                        for k,v in csd_npz[f'{ishank}'].item().items():
                            gg[f'{K}/{k}'] = v
                        ddf = ephys.load_ds_dataset(ddir, iprb=iprb, ishank=ishank)
                        if ddf is not None and ddf.size > 0 and 'type' in ddf.columns:
                            ddf.to_hdf(gg.filename, key=f'{K}/DS_DF')
                        if f'{ishank}_params' in csd_npz: # store params as attributes
                            paramdict = dict(csd_npz[f'{ishank}_params'].item())
                            gg[K].attrs.update(paramdict)
                csd_npz.close()
        ff.close()
        gg.close()
        self.data_signal.emit(list(self.transferred))
        self.progress_signal.emit('Done!')
        time.sleep(0.5)
        self.finished.emit()
    
    def report_progress(self, txt=None):
        """ Collect successfully converted data files """
        #hdr = 'Converting the following data files:'
        if txt is not None:
            self.transferred.append(txt)
        fstring = ', '.join(self.transferred) + ' ... '
        if txt is None: fstring += 'Done!'
        #msg = f'<center>{hdr}<br><br>{fstring}</center>'
        #self.progress_signal.emit(msg)
        self.progress_signal.emit('Converting files ...')


def get_analysis_btn(label, color, enabled=True):
    """ Return labeled analysis button """
    cbase = pyfx.hue(color, 0.7, 1)  # base color (lighter)
    cpress = pyfx.hue(color, 0.4, 0) # pressed color (darker)
    # create button and label
    btn = QtWidgets.QPushButton()
    ss_dict = QSS.ANALYSIS_BTN
    ss_dict['QPushButton']['background-color'] = f'rgba{cbase}'
    ss_dict['QPushButton:pressed']['background-color'] = f'rgba{cpress}'
    btn.setStyleSheet(pyfx.dict2ss(ss_dict))
    lbl = QtWidgets.QLabel(label)
    # create parent widget
    w = pyfx.get_widget_container('h', btn, lbl, spacing=8, widget='widget')
    w.btn = btn
    w.lbl = lbl
    w.setEnabled(enabled)
    return w

class ProcessedRecordingSelectionWidget(gi.FileSelectionWidget):
    """ Custom FileSelectionWidget for selecting processed datasets """
    
    def __init__(self, title='', parent=None):
        super().__init__(title=title, parent=parent)
        self.gen_layout()
    
    def gen_layout(self):
        """ Create dropdowns for probe and shank selection """
        self.le.setStyleSheet(pyfx.dict2ss(self.le_styledict) % 'gray')
        self.icon_btn.hide()
        # probe/shank selection widgets
        self.probe_dropdown = QtWidgets.QComboBox()
        self.shank_dropdown = QtWidgets.QComboBox()
        self.probe_box = pyfx.get_widget_container('h', self.probe_dropdown, 
                                                   self.shank_dropdown, spacing=5, widget='widget')
        self.vlay.addWidget(self.probe_box)
    
    def select_filepath(self):
        """ Launch file dialog for directory selection """
        ddir = ephys.select_directory(init_ddir=self.get_init_ddir(), 
                                      title='Select processed recording', parent=self)
        if ddir:
            self.update_filepath(ddir)
            
    def update_filepath(self, ppath):
        """ Update QLineEdit with selected directory path """
        self.le.setText(ppath)
        self.signal.emit(True)
        

class ProcessedRecordingSelectionPopup(QtWidgets.QDialog):
    """ Hub for analyzing processed data """
    
    def __init__(self, init_ppath=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Analyze recording')
        self.setMinimumWidth(300)
        
        # initialize recording folder
        if init_ppath is None : self.ddir = ephys.base_dirs()[0]
        else                  : self.ddir = init_ppath
        self.gen_layout()
        self.connect_signals()
        
        self.fsw.update_filepath(self.ddir)
    
    def gen_layout(self):
        """ Set up layout """
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(10)
        
        ### processed data selection
        self.ddir_gbox = QtWidgets.QGroupBox()
        ddir_vbox = pyfx.InterWidgets(self.ddir_gbox, 'v')[2]
        self.fsw = ProcessedRecordingSelectionWidget(title='<b><u>Processed data source</u></b>')
        self.probe_dropdown = self.fsw.probe_dropdown
        self.shank_dropdown = self.fsw.shank_dropdown
        ddir_vbox.addWidget(self.fsw)
        self.layout.addWidget(self.ddir_gbox)
        
        ### analysis buttons
        self.channel_selection_btn = get_analysis_btn('Select event channels', 
                                                      'green', enabled=False)
        self.ds_classification_btn = get_analysis_btn('Classify dentate spikes', 
                                                      'blue', enabled=False)
        self.ab_btns = [self.channel_selection_btn, self.ds_classification_btn]
        self.layout.addWidget(pyfx.DividerLine())
        bbox = pyfx.get_widget_container('v', *self.ab_btns, widget='widget')
        self.layout.addWidget(bbox)
        
        ### "waiting" spinner icon
        self.spinner_window = gi.SpinnerWindow(self)
        self.spinner_window.spinner.setInnerRadius(25)
        self.spinner_window.spinner.setNumberOfLines(10)
        #self.spinner_window.layout.setContentsMargins(5,5,5,5)
        self.spinner_window.layout.setSpacing(0)
        self.spinner_window.adjust_labelSize(lw=2.5, lh=0.65, ww=3)
        #self.spinner_window.adjust_labelSize(lw=5, lh=1.5, ww=3)
    
    def connect_signals(self):
        """ Connect GUI inputs """
        self.fsw.ppath_btn.clicked.connect(self.fsw.select_filepath)
        self.probe_dropdown.currentTextChanged.connect(self.probe_updated)
        self.shank_dropdown.currentTextChanged.connect(self.shank_updated)
        self.fsw.signal.connect(self.ddir_updated)
        self.channel_selection_btn.btn.clicked.connect(self.run_channel_selection_gui)
        self.ds_classification_btn.btn.clicked.connect(self.run_classification_gui)
        
    def create_workers(self):
        """ Parallel thread for migrating processed data """
        self.worker_thread = QtCore.QThread()
        self.worker_object.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker_object.run)
        self.worker_object.progress_signal.connect(self.spinner_window.report_progress_string)
        self.worker_object.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_object.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.finished_conversion)
    
    def start_conversion(self):
        """ Worker thread starts the data conversion pipeline """
        self.worker_object = ConverterBot()
        self.worker_object.ddir = self.ddir
        self.worker_object.data_signal.connect(self.conversion_slot)
        self.create_workers()
        self.spinner_window.start_spinner()
        self.worker_thread.start()
        
    def finished_conversion(self):
        """ Worker thread completes the data conversion pipeline """
        self.spinner_window.stop_spinner()
        self.worker_object = None
        self.worker_thread = None
    
    @QtCore.pyqtSlot(list)
    def conversion_slot(self, transferred):
        # prompt user to delete original data files after migration
        msg = '<center>Conversion successful!<br>Delete remaining NPZ files?</center>'
        res = gi.MsgboxSave(msg, parent=self).exec()
        if res == QtWidgets.QMessageBox.Yes:
            for fname in transferred:
                os.remove(Path(self.ddir, fname))
            print(f'{len(transferred)} NPZ files removed from directory!')
        self.ddir_updated()
        
    def clear_dropdown(self, dropdown):
        """ Clear probes/shanks from dropdowns """
        # reset probe/shank dropdowns
        dropdown.blockSignals(True)
        for i in reversed(range(dropdown.count())):
            dropdown.removeItem(i)
        dropdown.blockSignals(False)
    
    def update_probe_dropdown(self, probes):
        """ Set new probe items when data directory is changed """
        self.clear_dropdown(self.probe_dropdown)
        probe_items = [f'probe {iprb}' for iprb in range(len(probes))]
        pyfx.stealthy(self.probe_dropdown, probe_items)
        self.update_shank_dropdown(probes[0])
    
    def update_shank_dropdown(self, probe):
        """ Set new shank items when probe is changed """
        self.clear_dropdown(self.shank_dropdown)
        shank_items = [f'shank {i}' for i in range(probe.get_shank_count())]
        pyfx.stealthy(self.shank_dropdown, shank_items)
        
    def ddir_updated(self):
        """ Assess the properties and analysis options for a given recording """
        self.ddir = str(self.fsw.le.text())
        # reset probe widgets when directory is changed
        self.clear_dropdown(self.probe_dropdown)
        self.clear_dropdown(self.shank_dropdown)
        self.channel_selection_btn.setEnabled(False)
        self.ds_classification_btn.setEnabled(False)
        
        # check if directory contains required LFP/probe files
        opt1 = dp.validate_processed_ddir(self.ddir)
        if opt1 == 0:
            return
        # for valid recording, update probe/shank dropdowns
        probes = ephys.read_probe_group(self.ddir).probes
        self.update_probe_dropdown(probes)
        
        if opt1 == 2:  # prompt user to convert data to new HDF5 format
            res = gi.MsgboxQuestion('Convert NPZ to HDF5?', parent=self).exec()
            if res == QtWidgets.QMessageBox.Yes:
                self.start_conversion()
                return
        # enable channel selection GUI and/or classification GUI
        self.channel_selection_btn.setEnabled(True)
        self.enable_disable_classification()
        
    def enable_disable_classification(self):
        """ Enable DS classification if the optimal hilus channel has been chosen """
        iprb = self.probe_dropdown.currentIndex()
        ishank = self.shank_dropdown.currentIndex()
        opt2 = dp.validate_classification_ddir(self.ddir, iprb, ishank)
        self.ds_classification_btn.setEnabled(opt2)
    
    def probe_updated(self):
        """ User selected a new probe """
        iprb = self.probe_dropdown.currentIndex()
        probe = ephys.read_probe_group(self.ddir).probes[iprb]
        self.update_shank_dropdown(probe)
        self.enable_disable_classification()
    
    def shank_updated(self):
        """ User selected a new shank """
        self.enable_disable_classification()
    
    def run_channel_selection_gui(self):
        """ Launch main analysis GUI, initialize with selected probe and shank """
        iprb = self.probe_dropdown.currentIndex()
        ishank = self.shank_dropdown.currentIndex()
        self.ch_selection_dlg = ChannelSelectionWindow(self.ddir, iprb=iprb, ishank=ishank)
        self.ch_selection_dlg.exec()
        # update probe/shank selection from main analysis GUI
        self.probe_dropdown.setCurrentIndex(int(self.ch_selection_dlg.iprb))
        self.shank_dropdown.setCurrentIndex(int(self.ch_selection_dlg.ishank))
        self.enable_disable_classification()
    
    def run_classification_gui(self):
        """ Launch DS classification GUI for events on selected probe and shank """
        iprb = self.probe_dropdown.currentIndex()
        ishank = self.shank_dropdown.currentIndex()
        # load DS dataframe
        DS_DF = ephys.load_ds_dataset(self.ddir, iprb=iprb, ishank=ishank)
        if len(DS_DF) < 2:
            pref = ['No dentate spikes','Only 1 dentate spike'][len(DS_DF)]
            gi.MsgboxError(f'{pref} detected on the hilus channel.', parent=self).exec()
            return
        self.ds_classification_dlg = DS_CSDWindow(self.ddir, iprb=iprb, ishank=ishank)
        self.ds_classification_dlg.exec()


if __name__ == '__main__':
    app = pyfx.qapp()
    qfd = QtWidgets.QFileDialog()
    init_ddir = str(qfd.directory().path())
    if init_ddir == os.getcwd():
        init_ddir=None
    w = ProcessedRecordingSelectionPopup()
    w.show()
    w.raise_()
    sys.exit(app.exec())