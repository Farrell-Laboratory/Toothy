#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 13:46:44 2025

@author: amandaschott
"""

# import h5py
# import numpy as np
# import pandas as pd
# import time
# import pdb

import sys
import os
from pathlib import Path

from PyQt5 import QtWidgets, QtCore

# custom modules
from . import QSS
from . import pyfx
from . import helpers_io as h_io
from . import data_processing as dp
from . import gui_items as gi
from .gui_channel_selection import ChannelSelectionWindow
from .ds_classification_gui import DS_CSDWindow


##############################################################################
##############################################################################
################                                              ################
################                 WORKER OBJECTS               ################
################                                              ################
##############################################################################
##############################################################################

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
        ddir = h_io.select_directory(init_dir=self.get_init_ddir(), 
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
        if init_ppath is None : self.ddir = h_io.get_toothy_paths()[0]
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
        print("Here??")
        self.fsw.signal.connect(self.ddir_updated)
        self.channel_selection_btn.btn.clicked.connect(self.run_channel_selection_gui)
        self.ds_classification_btn.btn.clicked.connect(self.run_classification_gui)
        
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

        # The list of "Probe" objects is only referenced here to use the *length* of the list
        probe_items = [f'probe {iprb}' for iprb in range(len(probes))]
        pyfx.stealthy(self.probe_dropdown, probe_items)
        self.update_shank_dropdown(probes[0])
    
    def update_shank_dropdown(self, probe):
        """ Set new shank items when probe is changed """
        self.clear_dropdown(self.shank_dropdown)

        # the "Probe" object is actually used here to get a shank count.
        shank_items = [f'shank {i}' for i in range(probe.get_shank_count())]
        pyfx.stealthy(self.shank_dropdown, shank_items)
        
    def ddir_updated(self):
        """ Assess the properties and analysis options for a given recording """
        self.ddir = str(self.fsw.le.text())
        print(self.ddir)
        # reset probe widgets when directory is changed
        self.clear_dropdown(self.probe_dropdown)
        self.clear_dropdown(self.shank_dropdown)
        self.channel_selection_btn.setEnabled(False)
        self.ds_classification_btn.setEnabled(False)
        
        # check if directory contains required LFP/probe files
        opt1 = dp.validate_processed_ddir(self.ddir)
        if opt1 == 0:
            print("HERE?!!!!!")
            return
        

        # for valid recording, update probe/shank dropdowns

        # Read in a "ProbeGroup" object and get the list of "Probe" objects
        probes = h_io.read_probe_group(self.ddir).probes
        self.update_probe_dropdown(probes)
        
        # if opt1 == 2:  # prompt user to convert data to new HDF5 format
        #     res = gi.MsgboxQuestion('Convert NPZ to HDF5?', parent=self).exec()
        #     if res == QtWidgets.QMessageBox.Yes:
        #         self.start_conversion()
        #         return
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
        probe = h_io.read_probe_group(self.ddir).probes[iprb] # Once again the "Probe" object is used just to get the shanks
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
        DS_DF = h_io.load_ds_dataset(self.ddir, iprb=iprb, ishank=ishank)
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