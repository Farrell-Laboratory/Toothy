#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toothy home interface

@author: amandaschott
"""
import sys
import os
from pathlib import Path
from PyQt5 import QtWidgets, QtCore
import pdb
# set app folder as working directory
app_ddir = Path(__file__).parent
os.chdir(app_ddir)
# import custom modules
import QSS
import pyfx
import ephys
import gui_items as gi
from gui_items import BaseFolderPopup, ParameterPopup
from probe_handler import ProbeObjectPopup
from raw_data_pipeline import RawRecordingSelectionPopup
from processed_data_hub import ProcessedRecordingSelectionPopup


class toothy(QtWidgets.QMainWindow):
    resize_signal = QtCore.pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        # load base data directories
        if not os.path.exists('default_folders.txt'):
            ephys.init_default_folders()
        ephys.clean_base_dirs()
        
        self.init_raw_ddir = None
        self.init_processed_ddir = None
        
        self.gen_layout()
        self.connect_signals()
        
        self.basedirs_popup   = None
        self.parameters_popup = None
        self.probe_popup      = None
        self.rawdata_popup    = None
        self.analysis_popup   = None
        
        self.show()
        self.center_window()
        
    def gen_layout(self):
        """ Set up layout """
        self.setWindowTitle('Toothy')
        self.setContentsMargins(25,25,25,25)
        self.centralWidget = QtWidgets.QWidget()
        self.centralLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.centralLayout.setSpacing(20)
        
        # create main buttons
        self.base_folder_btn = QtWidgets.QPushButton('Base folders')
        self.view_params_btn = QtWidgets.QPushButton('View parameters')
        self.probe_btn = QtWidgets.QPushButton('Create probe')
        self.process_btn = QtWidgets.QPushButton('Process raw data')
        self.analyze_btn = QtWidgets.QPushButton('Analyze data')
        self.home_btns = [self.base_folder_btn, # update data/probe/param files
                          self.view_params_btn, # edit param values
                          self.probe_btn,       # create probe object
                          self.process_btn,     # process raw recording
                          self.analyze_btn]     # launch analysis window
        for btn in self.home_btns:
            btn.setStyleSheet(pyfx.dict2ss(QSS.INSET_BTN))
            self.centralLayout.addWidget(btn)
        self.setCentralWidget(self.centralWidget)
    
    def connect_signals(self):
        """ Connect GUI buttons """
        self.base_folder_btn.clicked.connect(self.base_folder_popup)
        self.view_params_btn.clicked.connect(self.view_param_popup)
        self.probe_btn.clicked.connect(self.probe_popup)
        self.process_btn.clicked.connect(self.raw_data_popup)
        self.analyze_btn.clicked.connect(self.processed_data_popup)
        self.resize_signal.connect(lambda win: QtCore.QTimer.singleShot(50, win.adjustSize))
    
    def base_folder_popup(self):
        """ View or change base data directories """
        self.basedirs_popup = BaseFolderPopup()
        self.basedirs_popup.path_updated_signal.connect(lambda i: self.resize_signal.emit(self.basedirs_popup))
        self.basedirs_popup.exec()
    
    def view_param_popup(self):
        """ View/edit default parameters """
        self.parameters_popup = ParameterPopup(ddict=ephys.read_params())
        if self.parameters_popup.exec():
            save_path = str(self.parameters_popup.SAVE_LOCATION)
            if save_path == ephys.base_dirs()[3]: return
            msg = f'Use "{os.path.basename(save_path)}" as default parameter file?'
            res = gi.MsgboxQuestion(msg).exec()
            if res == QtWidgets.QMessageBox.Yes:
                ephys.write_base_dirs(ephys.base_dirs()[0:3] + [save_path])
        
    def probe_popup(self, *args, init_probe=None):
        """ Build probe objects """
        self.probeobj_popup = ProbeObjectPopup(probe=init_probe)
        self.probeobj_popup.setModal(True)
        self.probeobj_popup.show()
        self.resize_signal.emit(self.probeobj_popup)
        
    def raw_data_popup(self, *args):
        """ Raw data processing pipeline """
        self.rawdata_popup = RawRecordingSelectionPopup(init_ppath=self.init_raw_ddir)
        self.rawdata_popup.exec()
        # set analysis hub to the most recently processed recording folder
        if self.rawdata_popup.last_saved_ddir is not None:
            self.init_processed_ddir = str(self.rawdata_popup.last_saved_ddir)
        
    def processed_data_popup(self, *args):
        """ Show processed data options """
        # create popup window for processed data
        self.analysis_popup = ProcessedRecordingSelectionPopup(init_ppath=self.init_processed_ddir, parent=self)
        self.analysis_popup.exec()
        self.init_processed_ddir = str(self.analysis_popup.ddir)
        
    def center_window(self):
        """ Move GUI to center of screen """
        qrect = self.frameGeometry()  # proxy rectangle for window with frame
        screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
        qrect.moveCenter(screen_rect.center())  # move center of qr to center of screen
        self.move(qrect.topLeft())
    

if __name__ == '__main__':
    app = pyfx.qapp()
    
    ToothyWindow = toothy()
    ToothyWindow.raise_()
    
    sys.exit(app.exec())
    