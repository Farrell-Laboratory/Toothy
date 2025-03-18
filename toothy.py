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
    
class toothy(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # load base data directories
        if not os.path.exists('default_folders.txt'):
            ephys.init_default_folders()
        ephys.clean_base_dirs()
        
        self.gen_layout()
        self.show()
        self.center_window()
        
    
    def gen_layout(self):
        """ Set up layout """
        self.setWindowTitle('Toothy')
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
        
        # create popup window for processed data
        self.analysis_popup = sp.ProcessedDirectorySelectionPopup(go_to_last=False, parent=self)
        self.analysis_popup.ab.option1_btn.clicked.connect(self.ch_selection_popup)
        self.analysis_popup.ab.option2_btn.clicked.connect(self.classify_ds_popup)
        
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
        self.process_btn.clicked.connect(self.raw_data_popup)
        self.analyze_btn.clicked.connect(self.processed_data_popup)
        self.probe_btn.clicked.connect(self.probe_popup)
        self.view_params_btn.clicked.connect(self.view_param_popup)
        self.base_folder_btn.clicked.connect(self.base_folder_popup)
        
        self.centralLayout.addWidget(self.base_folder_btn)
        self.centralLayout.addWidget(self.view_params_btn)
        self.centralLayout.addWidget(self.probe_btn)
        self.centralLayout.addWidget(self.process_btn)
        self.centralLayout.addWidget(self.analyze_btn)
        
        self.setCentralWidget(self.centralWidget)
    
    def base_folder_popup(self):
        """ View or change base data directories """
        sp.BaseFolderPopup.run(parent=self)
    
    def view_param_popup(self):
        """ View/edit default parameters """
        PARAMS = ephys.read_params()
        self.param_dlg = gi.ParamSettings(PARAMS, parent=self)
        self.param_dlg.show()
        self.param_dlg.raise_()
        res = self.param_dlg.exec()
        if res:  # existing param file updated by user
            a, b, c, param_path = ephys.base_dirs()
            save_path = str(self.param_dlg.SAVE_LOCATION)
            if param_path != save_path:
                fname = os.path.basename(save_path)
                res2 = QtWidgets.QMessageBox.question(self, '', f'Use {fname} as default parameter file?')
                if res2 == QtWidgets.QMessageBox.Yes:
                    llist = [a,b,c,save_path]
                    ephys.write_base_dirs(llist)
        
    def probe_popup(self):
        """ Build probe objects """
        _ = probegod.run_probe_window(accept_visible=False, title='Create probe', parent=self)
        
    def raw_data_popup(self, mode=2, init_raw_ddir=''):
        """ Select raw data for processing """
        popup = sp.RawDirectorySelectionPopup(mode, init_raw_ddir, parent=self)
        res = popup.exec()
        if not res:
            return
        
    def processed_data_popup(self, _, init_ddir=None, go_to_last=True):
        """ Show processed data options """
        self.analysis_popup.show()
        self.analysis_popup.raise_()
        
    def ch_selection_popup(self):
        """ Launch event channel selection window """
        ddir = self.analysis_popup.ddir
        probe_list = self.analysis_popup.probe_group.probes
        iprb = self.analysis_popup.probe_idx
        self.ch_selection_dlg = ChannelSelectionWindow(ddir, probe_list=probe_list, 
                                                       iprb=iprb, parent=self.analysis_popup)
        self.ch_selection_dlg.show()
        self.ch_selection_dlg.raise_()
        _ = self.ch_selection_dlg.exec()
        # check for updated files, enable/disable analysis options
        iprb = int(self.ch_selection_dlg.iprb)
        self.analysis_popup.ab.ddir_toggled(ddir, iprb)
    
    def classify_ds_popup(self):
        """ Launch DS analysis window """
        ddir = self.analysis_popup.ddir
        iprb = self.analysis_popup.probe_idx
        DS_DF = pd.read_csv(Path(ddir, f'DS_DF_{iprb}'))
        if DS_DF.empty:
            QtWidgets.QMessageBox.critical(self.analysis_popup, '', 'No dentate spikes detected on the hilus channel.')
            return
        PARAMS = ephys.load_recording_params(ddir)
        self.classify_ds_dlg = DS_CSDWindow(ddir, iprb, PARAMS, parent=self.analysis_popup)
        self.classify_ds_dlg.show()
        self.classify_ds_dlg.raise_()
        _ = self.classify_ds_dlg.exec()
        # check for updated files, enable/disable analysis options
        self.analysis_popup.ab.ddir_toggled(ddir)
        
    def center_window(self):
        """ Move GUI to center of screen """
        qrect = self.frameGeometry()  # proxy rectangle for window with frame
        screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
        qrect.moveCenter(screen_rect.center())  # move center of qr to center of screen
        self.move(qrect.topLeft())

if __name__ == '__main__':
    app = pyfx.qapp()
    w = toothy()
    w.show()
    w.raise_()
    sys.exit(app.exec())
