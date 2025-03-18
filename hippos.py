#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:25:32 2024

@author: amandaschott
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
import probeinterface as prif
import pdb
# set app folder as working directory
app_ddir = Path(__file__).parent
os.chdir(app_ddir)
# import custom modules
import pyfx
import qparam
import ephys
import selection_popups as sp
import gui_items as gi
from probe_handler import probegod
from channel_selection_gui import ChannelSelectionWindow
from ds_classification_gui import DS_CSDWindow
    
class hippos(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # load base data directories
        if not os.path.exists('default_folders.txt'):
            ephys.init_default_folders()
        ephys.clean_base_dirs()
        data_path, probe_path, probe_file, param_file = ephys.base_dirs()
        self.init_raw_ddir = str(data_path)
        self.init_processed_ddir = str(data_path)
        
        self.gen_layout()
        
        self.basedirs_popup   = None
        self.parameters_popup = None
        self.probe_popup      = None
        self.rawdata_popup    = None
        self.analysis_popup   = None
        self.ch_selection_dlg = None
        self.classify_ds_dlg  = None
        
        self.show()
        self.center_window()
        
    
    def gen_layout(self):
        """ Set up layout """
        self.setWindowTitle('Hippos')
        self.setContentsMargins(25,25,25,25)
        
        self.centralWidget = QtWidgets.QWidget()
        self.centralLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.centralLayout.setSpacing(20)
        
        mode_btn_ss = ('QPushButton {'
                       'background-color : gainsboro;'
                       'border : 3px outset gray;'
                       'border-radius : 2px;'
                       'color : black;'
                       'padding : 4px;'
                       'font-weight : bold;'
                       '}'
                       
                       'QPushButton:pressed {'
                       'background-color : dimgray;'
                       'border : 3px inset gray;'
                       'color : white;'
                       '}'
                       
                       'QPushButton:checked {'
                       'background-color : darkgray;'
                       'border : 3px inset gray;'
                       'color : black;'
                       '}'
                       
                       'QPushButton:disabled {'
                       'background-color : gainsboro;'
                       'border : 3px outset darkgray;'
                       'color : gray;'
                       '}'
                       
                       'QPushButton:disabled:checked {'
                       'background-color : darkgray;'
                       'border : 3px inset darkgray;'
                       'color : dimgray;'
                       '}'
                       )
        
        # create main buttons
        self.base_folder_btn = QtWidgets.QPushButton('Base folders')
        self.base_folder_btn.setStyleSheet(mode_btn_ss)
        self.view_params_btn = QtWidgets.QPushButton('View parameters')
        self.view_params_btn.setStyleSheet(mode_btn_ss)
        self.probe_btn =  QtWidgets.QPushButton('Create probe')
        self.probe_btn.setStyleSheet(mode_btn_ss)
        self.process_btn = QtWidgets.QPushButton('Process raw data')
        self.process_btn.setStyleSheet(mode_btn_ss)
        self.analyze_btn = QtWidgets.QPushButton('Analyze data')
        self.analyze_btn.setStyleSheet(mode_btn_ss)
        
        # connect to functions
        self.base_folder_btn.clicked.connect(self.base_folder_popup)
        self.view_params_btn.clicked.connect(self.view_param_popup)
        self.probe_btn.clicked.connect(self.probe_popup)
        self.process_btn.clicked.connect(self.raw_data_popup)
        self.analyze_btn.clicked.connect(self.processed_data_popup)
        
        self.centralLayout.addWidget(self.base_folder_btn)
        self.centralLayout.addWidget(self.view_params_btn)
        self.centralLayout.addWidget(self.probe_btn)
        self.centralLayout.addWidget(self.process_btn)
        self.centralLayout.addWidget(self.analyze_btn)
        
        self.setCentralWidget(self.centralWidget)
    
    def base_folder_popup(self):
        """ View or change base data directories """
        self.basedirs_popup = sp.BaseFolderPopup()
        self.basedirs_popup.setModal(True)
        self.basedirs_popup.show()
    
    def view_param_popup(self):
        """ View/edit default parameters """
        self.parameters_popup = gi.ParamSettings()
        self.parameters_popup.setModal(True)
        res = self.parameters_popup.exec()
        if res and (self.parameters_popup.SAVE_LOCATION != ephys.base_dirs()[3]):
            save_path = str(self.parameters_popup.SAVE_LOCATION)
            res2 = QtWidgets.QMessageBox.question(None, '', (f'Use {os.path.basename(save_path)} '
                                                             'as default parameter file?'))
            if res2:
                ephys.write_base_dirs(ephys.base_dirs()[0:3] + [save_path])
        
    def probe_popup(self, *args, init_probe=None):
        """ Build probe objects """
        self.probeobj_popup = sp.ProbeObjectPopup(probe=init_probe)
        self.probeobj_popup.setModal(True)
        self.probeobj_popup.show()
        
    def raw_data_popup(self, *args, mode=2, init_raw_ddir=''):
        """ Select raw data for processing """
        self.rawdata_popup = sp.RawDirectorySelectionPopup(raw_ddir=init_raw_ddir)
        self.rawdata_popup.setModal(True)
        self.rawdata_popup.show()
        
    def processed_data_popup(self, *args, init_ddir=''):
        """ Show processed data options """
        # create popup window for processed data
        self.analysis_popup = sp.ProcessedDirectorySelectionPopup(init_ddir=init_ddir)
        #self.analysis_popup.setModal(True)
        self.analysis_popup.ab.option1_btn.clicked.connect(self.ch_selection_popup)
        self.analysis_popup.ab.option2_btn.clicked.connect(self.classify_ds_popup)
        self.analysis_popup.show()
    
    def ch_selection_popup(self):
        # beta
        ddir = self.analysis_popup.ddir
        probe_list = self.analysis_popup.probe_group.probes
        iprb = self.analysis_popup.probe_idx
        ishank = self.analysis_popup.shank_idx
        self.ch_selection_dlg = ChannelSelectionWindow(ddir, probe_list=probe_list, 
                                                       iprb=iprb, ishank=ishank)
        self.ch_selection_dlg.setModal(True)
        self.ch_selection_dlg.show()
        self.ch_selection_dlg.raise_()
        _ = self.ch_selection_dlg.exec()
        # check for updated files, enable/disable analysis options
        iprb = int(self.ch_selection_dlg.iprb)
        ishank = int(self.ch_selection_dlg.ishank)
        self.analysis_popup.ab.ddir_toggled(ddir, probe_idx=iprb, shank_idx=ishank)
    
    def classify_ds_popup(self):
        """ Launch DS analysis window """
        ddir = self.analysis_popup.ddir
        iprb = self.analysis_popup.probe_idx
        ishank = self.analysis_popup.shank_idx
        # load DS dataframe
        DS_DF = ephys.load_ds_dataset(ddir, iprb, ishank=ishank)
        if DS_DF.empty:
            QtWidgets.QMessageBox.critical(self.analysis_popup, '', 'No dentate spikes detected on the hilus channel.')
            return
        self.classify_ds_dlg = DS_CSDWindow(ddir, iprb=iprb, ishank=ishank)
        self.classify_ds_dlg.setModal(True)
        self.classify_ds_dlg.show()
        self.classify_ds_dlg.raise_()
        _ = self.classify_ds_dlg.exec()
        # check for updated files, enable/disable analysis options
        self.analysis_popup.ab.ddir_toggled(ddir, probe_idx=iprb, shank_idx=ishank)
        
    def center_window(self):
        """ Move GUI to center of screen """
        qrect = self.frameGeometry()  # proxy rectangle for window with frame
        screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
        qrect.moveCenter(screen_rect.center())  # move center of qr to center of screen
        self.move(qrect.topLeft())

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setQuitOnLastWindowClosed(True)
    
    w = hippos()
    w.raise_()
    sys.exit(app.exec())