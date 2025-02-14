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
from PyQt5 import QtWidgets
import probeinterface as prif
import pdb
# set app folder as working directory
app_ddir = Path(__file__).parent
os.chdir(app_ddir)
# import custom modules
import pyfx
import ephys
import selection_popups as sp
import gui_items as gi
from probe_handler import probegod
from channel_selection_gui import ChannelSelectionWindow
from ds_classification_gui import DS_CSDWindow
    
class hippos(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        if not os.path.exists('default_folders.txt'):
            default_probe_folder = str(Path(os.getcwd(), 'probe_configs'))
            default_probe_file = str(Path(default_probe_folder, 'demo_probe_config.json'))
            default_param_file = str(Path(os.getcwd(), 'default_params.txt'))
            if not os.path.exists(default_probe_folder):
                os.makedirs(default_probe_folder)  # initialize probe folder
            if not os.path.exists(default_probe_file):
                # create demo probe
                shank = prif.generate_linear_probe(num_elec=8, ypitch=50)
                pos_adj = shank.contact_positions[::-1] + np.array([0, 50])
                shank.set_contacts(pos_adj, shapes='circle', 
                                        shape_params=dict(radius=7.5))
                shank.create_auto_shape('tip', margin=20)
                shank.set_shank_ids(np.ones(8, dtype='int'))
                demo_probe = prif.combine_probes([shank])
                demo_probe.set_contact_ids(np.arange(8))
                demo_probe.set_device_channel_indices(np.array([5,2,3,0,6,7,4,1]))
                demo_probe.annotate(**{'name':'demo_probe'})
                _ = ephys.write_probe_file(demo_probe, default_probe_file)
            if not os.path.exists(default_param_file):
                param_dict = ephys.get_original_defaults()
                _ = ephys.write_param_file(param_dict, default_param_file)
            # save base folders
            llist = [str(os.getcwd()),      # raw data folder
                     str(os.getcwd()),      # processed data folder
                     default_probe_folder,  # probe configuration folder
                     '',                    # probe configuration file (optional)
                     default_param_file]    # param configuration file
            ephys.write_base_dirs(llist)
        
        # load base data directories
        self.BASE_FOLDERS = ephys.base_dirs()
        if not os.path.isdir(self.BASE_FOLDERS[0]):  # raw data
            self.BASE_FOLDERS[0] = str(os.getcwd())
        if not os.path.isdir(self.BASE_FOLDERS[1]):  # processed data
            self.BASE_FOLDERS[1] = str(os.getcwd())
        if not os.path.isdir(self.BASE_FOLDERS[2]):  # probe files
            self.BASE_FOLDERS[2] = str(os.getcwd())
        if not os.path.isfile(self.BASE_FOLDERS[3]): # default probe
            self.BASE_FOLDERS[3] = ''
        if not os.path.isfile(self.BASE_FOLDERS[4]): # default settings
            self.BASE_FOLDERS[4] = ''
        ephys.write_base_dirs(list(self.BASE_FOLDERS))
        
        # initial parameter dictionary (or None)
        self.PARAMS, _ = ephys.read_param_file(filepath=self.BASE_FOLDERS[4])
        if self.PARAMS is None:
            self.PARAMS = ephys.get_original_defaults()  # use hard-coded defaults
        
        self.gen_layout()
        
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
        
    def probe_popup(self):
        _ = probegod.run_probe_window(accept_visible=False, title='Create probe', parent=self)
        
        
    def raw_data_popup(self, mode=2, init_raw_ddir='', init_save_ddir=''):
        """ Select raw data for processing """
        popup = sp.RawDirectorySelectionPopup(mode, init_raw_ddir, init_save_ddir, 
                                              PARAMS=self.PARAMS, parent=self)
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
        self.classify_ds_dlg = DS_CSDWindow(ddir, iprb, self.PARAMS, parent=self.analysis_popup)
        self.classify_ds_dlg.show()
        self.classify_ds_dlg.raise_()
        _ = self.classify_ds_dlg.exec()
        # check for updated files, enable/disable analysis options
        self.analysis_popup.ab.ddir_toggled(ddir)
        
        
    def view_param_popup(self):
        """ View/edit default parameters """
        if self.PARAMS is None:
            self.param_msgbox = gi.MsgboxParams(ephys.base_dirs()[4], parent=self)
            self.param_msgbox.show()
            self.param_msgbox.raise_()
            res = self.param_msgbox.exec()
            #pdb.set_trace()
            if res:  # new valid param file loaded or created by user
                self.PARAMS,_ = ephys.read_param_file(self.param_msgbox.PARAM_FILE)
                self.BASE_FOLDERS[4] = self.param_msgbox.PARAM_FILE
                ephys.write_base_dirs(list(self.BASE_FOLDERS))
        else:
            self.param_dlg = gi.ParamSettings(self.PARAMS, parent=self)
            self.param_dlg.show()
            self.param_dlg.raise_()
            res = self.param_dlg.exec()
            if res:  # existing param file updated by user
                self.PARAMS,_ = ephys.read_param_file(self.param_dlg.SAVE_LOCATION)
                self.BASE_FOLDERS[4] = self.param_dlg.SAVE_LOCATION
                ephys.write_base_dirs(list(self.BASE_FOLDERS))
        
        
    def base_folder_popup(self):
        """ View or change base data directories """
        dlg = sp.BaseFolderPopup(parent=self)
        dlg.show()
        dlg.raise_()
        dlg.exec()
        self.BASE_FOLDERS = ephys.base_dirs()
        self.PARAMS, _ = ephys.read_param_file(filepath=self.BASE_FOLDERS[4])
        if self.PARAMS is None:
            self.PARAMS = ephys.get_original_defaults()  # use hard-coded defaults
        
    
    def center_window(self):
        """ Move GUI to center of screen """
        qrect = self.frameGeometry()  # proxy rectangle for window with frame
        screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
        qrect.moveCenter(screen_rect.center())  # move center of qr to center of screen
        self.move(qrect.topLeft())
        
    
if __name__ == '__main__':
    app = pyfx.qapp()
    w = hippos()
    w.show()
    w.raise_()
    sys.exit(app.exec())