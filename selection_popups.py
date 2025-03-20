#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:28:55 2024

@author: amandaschott
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
import time
import pickle
import quantities as pq
import probeinterface as prif
import pdb
# custom modules
import pyfx
import qparam
import ephys
import gui_items as gi
import data_processing as dp
from probe_handler import probegod, IFigProbe
import resources_rc


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

btn_ss = ('QPushButton {'
          'background-color : rgba%s;'  # light
          'border : 4px outset rgb(128,128,128);'
          'border-radius : 11px;'
          'min-width : 15px;'
          'max-width : 15px;'
          'min-height : 15px;'
          'max-height : 15px;'
          '}'
          
          'QPushButton:disabled {'
          'background-color : rgb(220,220,220);'#'rgba%s;'
          #'border : 4px outset rgb(128,128,128);'
          '}'
          
          'QPushButton:pressed {'
          'background-color : rgba%s;'  # dark
          '}'
          )

blue_btn_ss = ('QPushButton {'
               'background-color : royalblue;'
               'color : white;'
               'font-weight : bold;'
               '}'
               
               'QPushButton:pressed {'
               'background-color : navy;}'
               
               'QPushButton:disabled {'
               'background-color : lightgray;'
               'color : gray;'
               'font-weight : normal;'
               '}')

def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return suffix

def clean(mode, base, last, init_ddir=''):
    if os.path.exists(init_ddir) and os.path.isdir(init_ddir):
        return init_ddir
    if mode==0: return base
    if mode==1: return last
    if mode==2:
        res = last if (Path(base) in Path(last).parents) else base
        return res
    
    
def validate_raw_ddir(ddir):
    if not os.path.isdir(ddir):
        return False
    files = os.listdir(ddir)
    a = bool('structure.oebin' in files)
    b = bool(len([f for f in files if f.endswith('.xdat.json')]) > 0)
    return bool(a or b)


def validate_processed_ddir(ddir):
    if not os.path.isdir(ddir):
        return False
    files = os.listdir(ddir)
    flist = ['lfp_bp.npz', 'lfp_time.npy', 'lfp_fs.npy']
    x = all([bool(f in files) for f in flist])
    return x


class DirectorySelectionWidget(QtWidgets.QWidget):
    def __init__(self, title='', simple=False, qedit_simple=False, parent=None):
        super().__init__(parent)
        self.simple_mode = simple
        
        # row 1 - QLabel and status icon (formerly info button)
        self.ddir_lbl_hbox = QtWidgets.QHBoxLayout()
        self.ddir_lbl_hbox.setContentsMargins(0,0,0,0)
        self.ddir_lbl_hbox.setSpacing(3)
        self.ddir_lbl = QtWidgets.QLabel()
        self.ddir_lbl.setText(title)
        
        # yes/no icons
        self.ddir_icon_btn = gi.StatusIcon(init_state=0) # folder status icon
        self.ddir_lbl_hbox.addWidget(self.ddir_icon_btn)
        self.ddir_lbl_hbox.addWidget(self.ddir_lbl)
        
        # row 2 - QLineEdit, and folder button
        self.qedit_hbox = gi.QEdit_HBox(simple=qedit_simple)  # 3 x QLineEdit items
        self.ddir_btn = QtWidgets.QPushButton()  # file dlg launch button
        self.ddir_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.ddir_btn.setMinimumSize(30,30)
        self.ddir_btn.setIconSize(QtCore.QSize(20,20))
        
        # assemble layout
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setHorizontalSpacing(5)
        self.grid.setVerticalSpacing(8)
        self.grid.addLayout(self.ddir_lbl_hbox,      0, 1)
        #self.grid.addWidget(self.ddir_icon_btn, 1, 0)
        self.grid.addLayout(self.qedit_hbox,    1, 1, 1, 4)
        self.grid.addWidget(self.ddir_btn,      1, 5)
        self.grid.setColumnStretch(0, 0)
        self.grid.setColumnStretch(1, 2)
        self.grid.setColumnStretch(2, 2)
        self.grid.setColumnStretch(3, 2)
        self.grid.setColumnStretch(4, 2)
        self.grid.setColumnStretch(5, 0)
        
        if self.simple_mode:
            #self.ddir_icon_btn.hide()
            self.grid.setColumnStretch(0,0)
    
    def update_status(self, ddir, x=False):
        self.qedit_hbox.update_qedit(ddir, x)  # line edit
        self.ddir_icon_btn.new_status(x)  # status icon
        

class ProcessedDirectorySelectionPopup(QtWidgets.QDialog):
    def __init__(self, init_ddir='', go_to_last=False, parent=None):
        super().__init__(parent)
        self.probe_group = None
        self.probe = None
        self.probe_idx = -1
        self.shank_idx = 0
        
        if init_ddir == '':
            self.ddir = ephys.base_dirs()[0]
        else:
            self.ddir = init_ddir
        self.gen_layout()
        
        if os.path.isdir(self.ddir):
            self.update_ddir(self.ddir)
    
    def gen_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(20)
        
        ###   SELECT PROCESSED DATA FOLDER
        
        self.ddir_gbox = QtWidgets.QGroupBox()
        #ddir_gbox.setStyleSheet(gbox_ss)
        ddir_vbox = pyfx.InterWidgets(self.ddir_gbox, 'v')[2]
        # basic directory selection widget
        self.ddw = DirectorySelectionWidget(title='<b><u>Processed Data Folder</u></b>')
        self.qedit_hbox = self.ddw.qedit_hbox
        # dropdown lists of probes and shanks
        self.probe_dropdown = QtWidgets.QComboBox()
        self.probe_dropdown.hide()
        self.shank_dropdown = QtWidgets.QComboBox()
        self.shank_dropdown.hide()
        self.info_view_btn = QtWidgets.QPushButton('More Info')
        self.info_view_btn.setEnabled(False)
        self.info_view_btn.hide()
        self.ddw.grid.addWidget(self.probe_dropdown, 2, 0, 1, 3)
        self.ddw.grid.addWidget(self.shank_dropdown, 2, 3, 1, 3)
        #self.ddw.grid.addWidget(self.info_view_btn, 2, 3, 1, 3)
        ddir_vbox.addWidget(self.ddw)
        
        ###   ACTION BUTTONS
        
        self.ab = gi.AnalysisBtns()
        intraline = pyfx.DividerLine()
        self.ddw.grid.addWidget(intraline, 3, 0, 1, 6)
        self.ddw.grid.addWidget(self.ab.option1_widget, 4, 1)
        self.ddw.grid.addWidget(self.ab.option2_widget, 5, 1)
        self.ddw.grid.setRowMinimumHeight(2, 20)

        # assemble layout
        self.layout.addWidget(self.ddir_gbox)
        
        # connect buttons
        self.ddw.ddir_btn.clicked.connect(self.select_ddir)
        self.probe_dropdown.currentTextChanged.connect(self.update_probe)
        self.shank_dropdown.currentTextChanged.connect(self.update_shank)
        self.info_view_btn.clicked.connect(self.show_info_popup)
    
    def show_info_popup(self):
        fig = IFigProbe(self.probe)
        dlg = gi.MatplotlibPopup(fig, toolbar_pos='left')
        dlg.setMinimumHeight(600)
        dlg.exec()
        
    def select_ddir(self):
        # open file popup to select processed data folder
        dlg = gi.FileDialog(init_ddir=self.ddir, parent=self)
        res = dlg.exec()
        if res:
            self.update_ddir(str(dlg.directory().path()))
    
    def update_ddir(self, ddir):
        self.ddir = ddir
        x = validate_processed_ddir(self.ddir)  # valid data folder?
        self.ddw.update_status(self.ddir, x)    # update folder path/icon style
        
        # reset probe and shank dropdowns, populate with probes in $info
        self.probe_dropdown.blockSignals(True)
        for i in reversed(range(self.probe_dropdown.count())):
            self.probe_dropdown.removeItem(i)
        self.probe_dropdown.setVisible(x)
        self.shank_dropdown.setVisible(x)
        #self.info_view_btn.setVisible(x)
        if x:
            # read in recording info, clear and update probe dropdown menu
            self.probe_group = prif.read_probeinterface(Path(self.ddir, 'probe_group'))
            items = [f'probe {i}' for i in range(len(self.probe_group.probes))]
            self.probe_dropdown.addItems(items)
            self.probe_dropdown.setEnabled(len(items) > 1)
            self.update_probe()
        else:
            self.probe_group, self.probe, self.probe_idx = None, None, -1
            self.ab.ddir_toggled(self.ddir)
            
        # probe index (e.g. 0,1,...) if directory is valid, otherwise -1
        self.probe_dropdown.blockSignals(False)
    
    def update_probe(self):
        if self.probe_group is None: return
        self.probe_idx = self.probe_dropdown.currentIndex()
        self.probe = self.probe_group.probes[self.probe_idx]
        # set shank dropdown items from current probe
        self.shank_dropdown.blockSignals(True)
        for j in reversed(range(self.shank_dropdown.count())):
            self.shank_dropdown.removeItem(j)
        items = [f'shank {i}' for i in range(self.probe.get_shank_count())]
        self.shank_dropdown.addItems(items)
        self.shank_dropdown.setEnabled(len(items) > 1)
        self.shank_dropdown.blockSignals(False)
        self.shank_idx = self.shank_dropdown.currentIndex()
        self.ab.ddir_toggled(self.ddir, probe_idx=self.probe_idx, shank_idx=self.shank_idx)
    
    def update_shank(self):
        self.shank_idx = self.shank_dropdown.currentIndex() # update action widgets
        self.ab.ddir_toggled(self.ddir, probe_idx=self.probe_idx, shank_idx=self.shank_idx)


class ProbeObjectPopup(QtWidgets.QDialog):
    def __init__(self, probe=None, auto_plot=True, parent=None):
        super().__init__(parent)
        
        # initialize central widget
        self.probegod = probegod(probe=probe, auto_plot=auto_plot)
        self.accept_btn = self.probegod.accept_btn
        self.accept_btn.setVisible(False)
        self.accept_btn.clicked.connect(self.accept)
        self.probegod.widget_updated_signal.connect(self.adjustSize)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.probegod)
        #self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.layout.addWidget(self.accept_btn)
        self.setWindowTitle('Create probe')
    

class ParamFilePopup(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        txt = QtWidgets.QLabel('Parameter file not found')
        bbox = QtWidgets.QHBoxLayout()
        select_btn = QtWidgets.QPushButton('Select existing file')
        create_btn = QtWidgets.QPushButton('Create new file')
        bbox.addWidget(select_btn)
        bbox.addWidget(create_btn)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(txt, stretch=2)
        lay.addLayout(bbox, stretch=0)
        
        select_btn.clicked.connect(self.select_param_file)
        create_btn.clicked.connect(self.create_param_file)
    

class BaseFolderPopup(QtWidgets.QDialog):
    path_updated_signal = QtCore.pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Base Folders')
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        self.BASE_FOLDERS = [str(x) for x in ephys.base_dirs()]
        qlabel_ss = ('QLabel {background-color:white;'
                             'border:1px solid gray;'
                             'border-radius:4px;'
                             'padding:5px;}')
        fmt = '<code>{}</code>'
        
        def makebtn(icon):
            btn = QtWidgets.QPushButton()
            btn.setMinimumSize(30,30)
            btn.setIcon(icon)
            btn.setIconSize(QtCore.QSize(20,20))
            return btn
        
        ###   RAW BASE FOLDER
        self.raw_w = QtWidgets.QWidget()
        raw_vlay, raw_row0, raw_row1 = self.create_hbox_rows()
        self.raw_w.setLayout(raw_vlay)
        raw_header = QtWidgets.QLabel('<b>RAW DATA</b>')
        raw_row0.addWidget(raw_header)
        raw_row0.addStretch()
        self.raw_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[0]))
        self.raw_qlabel.setStyleSheet(qlabel_ss)
        self.raw_btn = makebtn(icon=QtGui.QIcon(':/icons/folder.png'))
        raw_row1.addWidget(self.raw_qlabel, stretch=2)
        raw_row1.addWidget(self.raw_btn, stretch=0)
        layout.addWidget(self.raw_w)
        
        ###   PROBE BASE FOLDER
        self.probe_w = QtWidgets.QWidget()
        probe_vlay, probe_row0, probe_row1 = self.create_hbox_rows()
        self.probe_w.setLayout(probe_vlay)
        probe_header = QtWidgets.QLabel('<b>PROBE FILES</b>')
        probe_row0.addWidget(probe_header)
        probe_row0.addStretch()
        self.probe_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[1]))
        self.probe_qlabel.setStyleSheet(qlabel_ss)
        self.probe_btn = makebtn(icon=QtGui.QIcon(':/icons/folder.png'))
        probe_row1.addWidget(self.probe_qlabel, stretch=2)
        probe_row1.addWidget(self.probe_btn, stretch=0)
        layout.addWidget(self.probe_w)
        
        ###   DEFAULT PROBE FILE
        self.probefile_w = QtWidgets.QWidget()
        probefile_vlay, probefile_row0, probefile_row1 = self.create_hbox_rows()
        self.probefile_w.setLayout(probefile_vlay)
        probefile_header = QtWidgets.QLabel('<b>DEFAULT PROBE</b>')
        probefile_row0.addWidget(probefile_header)
        probefile_row0.addStretch()
        self.probefile_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[2]))
        self.probefile_qlabel.setStyleSheet(qlabel_ss)
        self.probefile_btn = makebtn(icon=QtGui.QIcon(':/icons/load.png'))
        self.probefile_clear = makebtn(icon=QtGui.QIcon(':/icons/trash.png'))
        probefile_row1.addWidget(self.probefile_qlabel, stretch=2)
        probefile_row1.addWidget(self.probefile_btn, stretch=0)
        probefile_row1.addWidget(self.probefile_clear, stretch=0)
        layout.addWidget(self.probefile_w)
        
        ###   DEFAULT PARAM FILE
        self.paramfile_w = QtWidgets.QWidget()
        paramfile_vlay, paramfile_row0, paramfile_row1 = self.create_hbox_rows()
        self.paramfile_w.setLayout(paramfile_vlay)
        paramfile_header = QtWidgets.QLabel('<b>DEFAULT PARAMETERS</b>')
        paramfile_row0.addWidget(paramfile_header)
        paramfile_row0.addStretch()
        self.paramfile_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[3]))
        self.paramfile_qlabel.setStyleSheet(qlabel_ss)
        self.paramfile_btn = makebtn(icon=QtGui.QIcon(':/icons/load.png'))
        self.paramfile_auto = makebtn(icon=QtGui.QIcon(':/icons/load_txt.png'))
        self.paramfile_auto.setToolTip('Auto-generate file from default params')
        paramfile_row1.addWidget(self.paramfile_qlabel, stretch=2)
        paramfile_row1.addWidget(self.paramfile_btn, stretch=0)
        paramfile_row1.addWidget(self.paramfile_auto, stretch=0)
        layout.addWidget(self.paramfile_w)
        
        bbox = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton('Save')
        self.save_btn.setStyleSheet(blue_btn_ss)
        self.save_btn.setEnabled(False)
        bbox.addWidget(self.save_btn)
        layout.addLayout(bbox)
        
        self.raw_btn.clicked.connect(lambda: self.choose_base_ddir(0))
        self.probe_btn.clicked.connect(lambda: self.choose_base_ddir(1))
        self.probefile_btn.clicked.connect(self.choose_probe_file)
        self.paramfile_btn.clicked.connect(self.choose_param_file)
        self.probefile_clear.clicked.connect(self.clear_probe_file)
        self.paramfile_auto.clicked.connect(self.auto_param_file)
        self.path_updated_signal.connect(lambda i: self.update_base_ddir(i))
        self.save_btn.clicked.connect(self.save_base_folders)
    
    def choose_base_ddir(self, i):
        init_ddir = str(self.BASE_FOLDERS[i])
        fmt = 'Base folder for %s'
        titles = [fmt % x for x in ['raw data', 'probe files']]
        # when activated, initialize at ddir and save new base folder at index i
        dlg = gi.FileDialog(init_ddir=init_ddir)
        dlg.setWindowTitle(titles[i])
        res = dlg.exec()
        if res:
            self.BASE_FOLDERS[i] = str(dlg.directory().path())
            #self.save_btn.setEnabled(True)
            self.path_updated_signal.emit(i)
            #self.update_base_ddir(i)
    
    def choose_probe_file(self):
        probe, fpath = gi.FileDialog.load_file(filetype='probe',
                                               init_ddir=str(self.BASE_FOLDERS[1]))
        if probe is not None:
            self.BASE_FOLDERS[2] = str(fpath)
            #self.save_btn.setEnabled(True)
            self.path_updated_signal.emit(2)
            #self.update_base_ddir(2)
        
    def clear_probe_file(self):
        self.BASE_FOLDERS[2] = ''
        #self.save_btn.setEnabled(True)
        self.path_updated_signal.emit(2)
        #self.update_base_ddir(2)
            
    def choose_param_file(self):
        param_dict, fpath = gi.FileDialog.load_file(filetype='param')
        if param_dict is not None:
            self.BASE_FOLDERS[3] = str(fpath)
            #self.save_btn.setEnabled(True)
            self.path_updated_signal.emit(3)
            #self.update_base_ddir(3)
    
    def auto_param_file(self):
        fpath = gi.FileDialog.save_file(qparam.get_original_defaults(), filetype='param',
                                        init_ddir=os.getcwd(), init_fname='default_params.txt')
        if fpath:
            self.BASE_FOLDERS[3] = str(fpath)
            #self.save_btn.setEnabled(True)
            self.path_updated_signal.emit(3)
            #self.update_base_ddir(3)
            
    def update_base_ddir(self, i):
        fmt = '<code>{}</code>'
        if i==0:
            self.raw_qlabel.setText(fmt.format(self.BASE_FOLDERS[0]))
        elif i==1:
            self.probe_qlabel.setText(fmt.format(self.BASE_FOLDERS[1]))
        elif i==2:
            self.probefile_qlabel.setText(fmt.format(self.BASE_FOLDERS[2]))
        elif i==3:
            self.paramfile_qlabel.setText(fmt.format(self.BASE_FOLDERS[3]))
        self.save_btn.setEnabled(True)
    
    def save_base_folders(self):
        ddir_list = list(self.BASE_FOLDERS)
        ephys.write_base_dirs(ddir_list)
        self.accept()
    
    def create_hbox_rows(self):
        vlay = QtWidgets.QVBoxLayout()
        vlay.setSpacing(2)
        row0 = QtWidgets.QHBoxLayout()
        row0.setContentsMargins(0,0,0,0)
        row1 = QtWidgets.QHBoxLayout()
        row1.setContentsMargins(0,0,0,0)
        row1.setSpacing(5)
        vlay.addLayout(row0)
        vlay.addLayout(row1)
        return vlay, row0, row1
    
    @classmethod
    def run(cls, qrect=None, parent=None):
        pyfx.qapp()
        dlg = BaseFolderPopup(parent)
        if isinstance(qrect, QtCore.QRect):
            dlg.setGeometry(qrect)
        dlg.show()
        dlg.raise_()
        res = dlg.exec()
        return res


class RawArrayPopup(QtWidgets.QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        
        self.gen_layout()
        self.connect_signals()
        
    def gen_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        gbox_ss = 'QGroupBox {background-color : rgba(230,230,230,255); font-weight:bold;}'
        self.dims_gbox = QtWidgets.QGroupBox('Data Array')
        self.dims_gbox.setStyleSheet(gbox_ss)
        dims_hlay = pyfx.InterWidgets(self.dims_gbox, 'h')[2]
        #dims_hlay = QtWidgets.QHBoxLayout(self.dims_gbox)
        
        ###   DATA STRUCTURE
        
        nrows, ncols = self.data.shape
        self.rows_w = gi.LabeledCombobox(f'Rows (<code>n={nrows}</code>)')
        self.rows_w.addItems(['Channels', 'Timepoints'])
        self.rows_w.setCurrentIndex(int(nrows > ncols))
        self.cols_w = gi.LabeledCombobox(f'Columns (<code>n={ncols:,}</code>)')
        self.cols_w.addItems(['Channels', 'Timepoints'])
        self.cols_w.setCurrentIndex(int(ncols > nrows))
        self.order_w = gi.LabeledCombobox('Channel order')
        self.order_w.addItems(['Shallow \u2192 Deep', 'Deep \u2192 Shallow'])
        dims_hlay.addWidget(self.rows_w)
        dims_hlay.addWidget(self.cols_w)
        dims_hlay.addWidget(self.order_w)
        
        ###   RECORDING PARAMS
        
        self.rec_gbox = QtWidgets.QGroupBox('Recording')
        self.rec_gbox.setStyleSheet(gbox_ss)
        rec_lay = pyfx.InterWidgets(self.rec_gbox, 'v')[2]
        # sampling rate and recording duration
        rec_hlay1 = QtWidgets.QHBoxLayout()
        self.fs_w = gi.LabeledSpinbox('Sampling rate', double=True, minimum=1, 
                                      maximum=9999999999, suffix=' Hz')
        self.dur_w = gi.LabeledSpinbox('Duration', double=True, minimum=0.0001,
                                       maximum=9999999999, suffix=' s')
        self.dur_w.qw.setDecimals(4)
        self.dur_w.qw.setReadOnly(True)
        rec_hlay1.addWidget(self.fs_w)
        rec_hlay1.addWidget(self.dur_w)
        # units
        self.units_w = gi.LabeledCombobox('Units')
        self.units_w.addItems(['uV', 'mV', 'V', 'kV'])
        rec_hlay1.addWidget(self.units_w)
        rec_lay.addLayout(rec_hlay1)

        ###   ACTION BUTTONS
        
        self.bbox = QtWidgets.QWidget()
        bbox_lay = QtWidgets.QHBoxLayout(self.bbox)
        self.continue_btn = QtWidgets.QPushButton('Continue')
        self.close_btn = QtWidgets.QPushButton('Cancel')
        bbox_lay.addWidget(self.close_btn)
        bbox_lay.addWidget(self.continue_btn)
        
        self.layout.addWidget(self.dims_gbox)
        self.layout.addWidget(self.rec_gbox)
        self.layout.addWidget(self.bbox)
    
    def connect_signals(self):
        # set row and column dropdowns to be mutually exclusive
        self.rows_w.qw.currentIndexChanged.connect(self.label_dims)
        self.cols_w.qw.currentIndexChanged.connect(self.label_dims)
        # update duration from sampling rate (or vice versa)
        self.fs_w.qw.valueChanged.connect(lambda x: self.update_fs_dur(x, 0))
        self.dur_w.qw.valueChanged.connect(lambda x: self.update_fs_dur(x, 1))
        
        self.continue_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)
    
    def label_dims(self, idx):
        """ Label data rows/columns as channels/timepoints """
        if self.sender()   == self.rows_w.qw: uw = self.cols_w
        elif self.sender() == self.cols_w.qw: uw = self.rows_w
        uw.qw.blockSignals(True)
        uw.setCurrentIndex(int(-(idx-1)))
        uw.qw.blockSignals(False)
        self.update_fs_dur(None, mode=0)
        
    
    def update_fs_dur(self, val, mode):
        """ Update duration from sampling rate (mode=0) or vice versa (mode=1) """
        nts = self.data.shape[self.cols_w.currentIndex()]
        if mode==0:
            # calculate recording duration from sampling rate
            self.dur_w.qw.blockSignals(True)
            self.dur_w.setValue(nts / self.fs_w.value())
            self.dur_w.qw.blockSignals(False)
        elif mode==1:
            self.fs_w.qw.blockSignals(True)
            self.fs_w.setValue(nts / self.dur_w.value())
            self.fs_w.qw.blockSignals(False)
    
    def accept(self):
        if self.rows_w.currentIndex() == 1:
            self.data = self.data.T
        if self.order_w.currentIndex() == 1:
            self.data = self.data[::-1]
        super().accept()
    
    @classmethod
    def run(cls, data, default_fs=1000, parent=None):
        # launch popup with loaded data
        dlg = cls(data, parent=parent)
        dlg.fs_w.setValue(default_fs)
        dlg.show()
        dlg.raise_()
        res = dlg.exec()
        if res:
            # return correctly oriented array, sampling rate/duration, and units
            ddict = dict(data=dlg.data, fs = dlg.fs_w.value(),
                         dur = dlg.dur_w.value(), units = dlg.units_w.currentText())
            return ddict


class DataWorker(QtCore.QObject):
    progress_signal = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(bool)
    
    DATA = None
    AUX_DATA = None
    FS = None
    PROBE_GROUP = None
    PARAMS = None
    
    units = 'uV'
    lfp_units = 'mV'
    
    def set_filepaths(self, raw_ddir, save_ddir=None):
        """ Paths to raw data source file and processed data target folder """
        assert os.path.isdir(raw_ddir)
        if save_ddir is None:
            save_ddir = str(Path(raw_ddir, gi.unique_fname(raw_ddir, 'processed_data')))
        self.raw_ddir = raw_ddir
        self.save_ddir = save_ddir
    
    def load_raw_data(self, raw_ddir):
        """ Load ephys data from supported recording system """
        (self.DATA, self.AUX_DATA), self.FS, self.units = dp.load_raw_data(raw_ddir)
    
    def run(self):
        # analyze recording between tstart and tend
        nts = self.DATA.shape[1]  # original number of data points
        dur = nts / self.FS
        tstart, tend = self.PARAMS['trange']
        tstart = min(tstart, dur-1)  # tstart must be at least 1s before last timepoint
        if (tend != -1) and (tend <= tstart): # tend must be greater than tstart OR -1
            tend = -1
        self.PARAMS['trange'] = [tstart, tend]
        if [tstart, tend] != [0,-1]:
            istart = int(tstart*self.FS)
            iend = nts if tend==-1 else min(int(tend*self.FS), nts)
            self.DATA = self.DATA[:, istart:iend]
        
        self.lfp_fs = self.PARAMS.pop('lfp_fs')
        self.dur = self.DATA.shape[1] / self.FS
        idx_by_probe = [prb.device_channel_indices for prb in self.PROBE_GROUP.probes]
        ds_factor = int(self.FS / self.lfp_fs)  # calculate downsampling factor
        cf = pq.Quantity(1, self.units).rescale(self.lfp_units).magnitude  # uV -> mV conversion factor
        
        self.progress_signal.emit('Extracting LFPs by probe ...')
        
        self.lfp_list = []
        for idx in idx_by_probe:
            lfp = np.array([pyfx.Downsample(self.DATA[i], ds_factor)*cf for i in idx])
            self.lfp_list.append(lfp)
        self.lfp_time = np.linspace(0, self.dur, lfp.shape[1])
        
        
        bp_dicts = {'raw':[], 'theta':[], 'slow_gamma':[], 'fast_gamma':[], 'swr':[], 'ds':[]}
        std_dfs, swr_dfs, ds_dfs, thresholds, noise_trains = [], [], [], [], []
        
        for i,_lfp in enumerate(self.lfp_list):
            hdr = f'ANALYZING PROBE {i+1} / {len(self.lfp_list)}'
            self.progress_signal.emit(f'analyzing probe {i+1} / {len(self.lfp_list)}')
            channels = np.arange(_lfp.shape[0])
            # bandpass filter LFPs within different frequency bands
            self.progress_signal.emit(hdr + '<br>Bandpass filtering signals ...')
            bp_dict = dp.bp_filter_lfps(_lfp, self.lfp_fs, **self.PARAMS)
            # get standard deviation (raw and normalized) for each filtered signal
            std_dict = {k : np.std(v, axis=1) for k,v in bp_dict.items()}
            std_dict.update({f'norm_{k}' : pyfx.Normalize(v) for k,v in std_dict.items()})
            STD = pd.DataFrame(std_dict)
            
            # run ripple detection on all channels
            SWR_DF = pd.DataFrame()
            SWR_THRES = {}
            self.progress_signal.emit(hdr + '<br>Detecting ripples ...')
            for ch in range(_lfp.shape[0]):
                # sharp-wave ripples
                swr_df, swr_thres = ephys.get_swr_peaks(bp_dict['swr'][ch], self.lfp_time, 
                                                        self.lfp_fs, pprint=False, **self.PARAMS)
                swr_df.set_index(np.repeat(ch, len(swr_df)), inplace=True)
                SWR_DF = pd.concat([SWR_DF, swr_df], ignore_index=False)
                SWR_THRES[ch] = swr_thres
            if SWR_DF.size == 0: 
                SWR_DF.loc[0] = np.nan
            # run DS detection on all channels
            DS_DF = pd.DataFrame()
            DS_THRES = {}
            self.progress_signal.emit(hdr + '<br>Detecting DS events ...')
            for ch in range(_lfp.shape[0]):
                # dentate spikes
                ds_df, ds_thres = ephys.get_ds_peaks(bp_dict['ds'][ch], self.lfp_time, 
                                                     self.lfp_fs, pprint=False, **self.PARAMS)
                ds_df.set_index(np.repeat(ch, len(ds_df)), inplace=True)
                DS_DF = pd.concat([DS_DF, ds_df], ignore_index=False)
                DS_THRES[ch] = ds_thres
            if DS_DF.size == 0: 
                DS_DF.loc[0] = np.nan
            THRESHOLDS = dict(SWR=SWR_THRES, DS=DS_THRES)
            
            for k,l in bp_dicts.items(): l.append(bp_dict[k])
            std_dfs.append(STD)
            swr_dfs.append(SWR_DF)
            ds_dfs.append(DS_DF)
            thresholds.append(THRESHOLDS)
            noise_trains.append(np.zeros(len(channels), dtype='int'))
        
        self.progress_signal.emit('Saving data ...')
        ALL_STD = pd.concat(std_dfs, keys=range(len(std_dfs)), ignore_index=False)
        ALL_SWR = pd.concat(swr_dfs, keys=range(len(swr_dfs)), ignore_index=False)
        ALL_SWR['status'] = 1   # 1=auto-detected; 2=added by user; -1=removed by user
        ALL_SWR['is_valid'] = 1 # valid events are either auto-detected and not user-removed OR user-added
        ALL_DS = pd.concat(ds_dfs, keys=range(len(ds_dfs)), ignore_index=False)
        ALL_DS['status'] = 1
        ALL_DS['is_valid'] = 1
        
        if not os.path.isdir(self.save_ddir):
            os.mkdir(self.save_ddir)
        
        fx = lambda fname: gi.unique_fname(self.save_ddir, fname)
        np.save(Path(self.save_ddir, fx('lfp_time.npy')), self.lfp_time)
        np.save(Path(self.save_ddir, fx('lfp_fs.npy')), self.lfp_fs)
        np.savez(Path(self.save_ddir, fx('lfp_bp.npz')), **bp_dicts)
        
        # save bandpass-filtered power in each channel (index)
        ALL_STD.to_csv(Path(self.save_ddir, fx('channel_bp_std')), index_label=False)
        
        # save event quantifications and thresholds
        ALL_SWR.to_csv(Path(self.save_ddir, fx('ALL_SWR')), index_label=False)
        ALL_DS.to_csv(Path(self.save_ddir, fx('ALL_DS')), index_label=False)
        np.save(Path(self.save_ddir, fx('THRESHOLDS.npy')), thresholds)
        # initialize noise channels
        np.save(Path(self.save_ddir, fx('noise_channels.npy')), noise_trains)
        
        # save params and info file
        with open(Path(self.save_ddir, fx('params.pkl')), 'wb') as f:
            pickle.dump(self.PARAMS, f)
            
        # write probe group to file
        prif.write_probeinterface(Path(self.save_ddir, fx('probe_group')), self.PROBE_GROUP)
        
        # save AUX data
        dp.process_aux(self.AUX_DATA, self.FS, self.lfp_fs, self.save_ddir, pprint=False)
        
        self.progress_signal.emit('Done!')
        time.sleep(1)
        self.finished.emit(True)


class RawDirectorySelectionPopup(QtWidgets.QDialog):
    RAW_DDIR_VALID = False
    RAW_ARRAY = None
    
    def __init__(self, mode=2, raw_ddir='', parent=None):
        # 0==base, 1=last visited, 2=last visited IF it's within base
        super().__init__(parent)
        self.setWindowTitle('Select a raw data source')
        
        # load parameters, get initial directories
        self.PARAMS = ephys.read_params()
        raw_base, probe_base, probe_file, _  = ephys.base_dirs()
        
        #qfd = QtWidgets.QFileDialog()
        #last_ddir = str(qfd.directory().path()) # most recently entered directory
        #self.raw_ddir = clean(mode, raw_base, last_ddir, str(raw_ddir))
        if not (os.path.exists(raw_ddir) and os.path.isdir(raw_ddir)):
            raw_ddir = str(raw_base)
        self.raw_ddir = raw_ddir
        self.default_probe_file = probe_file
        
        self.gen_layout()
        self.probe_gbox.hide()
        self.update_raw_ddir()
        
        # create worker thread, connect functions
        self.worker_object = DataWorker()
        self.create_worker_thread()
        
    
    def gen_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(5)
        gbox_ss = 'QGroupBox {background-color : rgba(230,230,230,255);}'
        #self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        
        ###   SELECT RAW DATA FOLDER
        
        self.ddir_gbox = QtWidgets.QGroupBox()
        self.ddir_gbox.setStyleSheet(gbox_ss)
        ddir_vbox = pyfx.InterWidgets(self.ddir_gbox, 'v')[2]
        # basic directory selection widget
        self.ddw = DirectorySelectionWidget(title='<b><u>Raw data directory</u></b>')
        self.qedit_hbox = self.ddw.qedit_hbox
        # types of recording files
        self.oe_radio = QtWidgets.QRadioButton('Open Ephys (.oebin)')
        self.nn_radio = QtWidgets.QRadioButton('NeuroNexus (.xdat.json)')
        self.nl_radio = QtWidgets.QRadioButton('Neuralynx (.ncs)')
        self.manual_radio = QtWidgets.QRadioButton('Upload custom file')
        for btn in [self.oe_radio, self.nn_radio, self.nl_radio, self.manual_radio]:
            btn.setAutoExclusive(False)
            #btn.setCheckable(False)
            btn.setEnabled(False)
            btn.setStyleSheet('QRadioButton:disabled {color : black;}')
        self.ddw.grid.addWidget(self.oe_radio, 2, 0, 1, 2)
        self.ddw.grid.addWidget(self.nn_radio, 2, 2, 1, 2)
        self.ddw.grid.addWidget(self.nl_radio, 2, 4, 1, 2)
        
        ddir_vbox.addWidget(self.ddw)
        
        # manual file upload with custom parameters 
        custom_bar = QtWidgets.QFrame()
        frame_hlay = QtWidgets.QHBoxLayout(custom_bar)
        frame_hlay.setContentsMargins(0,0,0,0)
        self.manual_upload_btn = QtWidgets.QPushButton()
        self.manual_upload_btn.setIcon(QtGui.QIcon(":/icons/load.png"))
        self.manual_upload_flabel = QtWidgets.QLineEdit()
        self.manual_upload_flabel.setReadOnly(True)
        frame_hlay.addWidget(self.manual_radio)
        frame_hlay.addWidget(self.manual_upload_flabel)
        frame_hlay.addWidget(self.manual_upload_btn)
        ddir_vbox.addWidget(custom_bar)
        
        ###   SETTINGS WIDGET
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
        #self.settings_btn0.setIcon(QtGui.QIcon(":/icons/settings.png"))
        #self.settings_btn0.setIconSize(QtCore.QSize(20,20))
        #self.settings_btn0.setEnabled(False)
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
        
        ###   ASSIGN PROBE(S)
        self.probe_gbox = QtWidgets.QGroupBox()
        self.probe_gbox.setStyleSheet('QGroupBox {border-width : 0px;'
                                      'font-weight : bold; text-decoration : underline;}')
        self.probe_vbox = pyfx.InterWidgets(self.probe_gbox, 'v')[2]
        
        ###   ACTION BUTTONS
        
        # continue button
        bbox = QtWidgets.QHBoxLayout()
        
        self.continue_btn = QtWidgets.QPushButton('Map to probe(s)')
        self.continue_btn.setStyleSheet(blue_btn_ss)
        self.continue_btn.setEnabled(False)
        self.tort_btn = QtWidgets.QPushButton('Process data!')
        self.tort_btn.setStyleSheet(blue_btn_ss)
        self.tort_btn.setVisible(False)
        self.tort_btn.setEnabled(False)
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        bbox.addWidget(self.continue_btn)
        bbox.addWidget(self.tort_btn)
        
        self.layout.addWidget(self.ddir_gbox)
        self.layout.addWidget(self.settings_w)
        self.layout.addWidget(self.probe_gbox)
        line0 = pyfx.DividerLine()
        self.layout.addWidget(line0)
        self.layout.addLayout(bbox)
        self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        
        # connect buttons
        self.ddw.ddir_btn.clicked.connect(self.select_ddir)
        self.manual_upload_btn.clicked.connect(self.load_array)
        
        self.continue_btn.clicked.connect(self.load_data)
        self.cancel_btn.clicked.connect(self.reject)
        self.tort_btn.clicked.connect(self.PROCESS_THE_DATA)
        
        self.spinner_window = gi.SpinnerWindow(self)
        self.spinner_window.spinner.setInnerRadius(25)
        self.spinner_window.spinner.setNumberOfLines(10)
        self.spinner_window.layout.setContentsMargins(5,5,5,5)
        self.spinner_window.layout.setSpacing(0)
        self.spinner_window.adjust_labelSize(lw=2.5, lh=0.65, ww=3)
    
    
    def create_worker_thread(self):
        self.worker_thread = QtCore.QThread()
        self.worker_object.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker_object.run)
        self.worker_object.progress_signal.connect(self.spinner_window.report_progress_string)
        self.worker_object.finished.connect(self.finished_work)
        self.worker_object.finished.connect(self.worker_thread.quit)
    
    
    def start_work(self):
        self.spinner_window.start_spinner()
        self.worker_thread.start()
    
    def finished_work(self, bool):
        self.spinner_window.stop_spinner()
        self.worker_thread.quit()
        
        self.worker_object.deleteLater()
        self.worker_thread.deleteLater()
        time.sleep(1)
        
        msgbox = gi.MsgboxSave('Data processing complete!\nExit window?', parent=self)
        res = msgbox.exec()
        if res == QtWidgets.QMessageBox.Yes:
            self.accept()
    
    def load_array(self):
        # open file, load data array
        data, fpath = gi.FileDialog.load_file(filetype='array', parent=self)
        if data is None: return
        # launch popup to ask for context
        ddict = RawArrayPopup.run(data, default_fs=1000, parent=self)
        if ddict:
            self.RAW_ARRAY = ddict['data']  # load oriented data and FS from popup
            self.RAW_ARRAY_FS = ddict['fs'] # confirm filepath
            self.RAW_ARRAY_UNITS = ddict['units']
            self.raw_ddir = os.path.dirname(fpath)
            self.manual_upload_flabel.setText(os.path.basename(fpath))
            self.update_raw_ddir()
    
    def load_data(self):
        self.worker_object.set_filepaths(self.raw_ddir)
        # load raw data from recording software or annotated .npy file
        if self.RAW_ARRAY is not None:
            self.worker_object.DATA = self.RAW_ARRAY
            self.worker_object.AUX_DATA = np.array([])
            self.worker_object.FS = self.RAW_ARRAY_FS
            self.worker_object.units = self.RAW_ARRAY_UNITS
        else:
            self.worker_object.load_raw_data(self.raw_ddir)
            
        # initialize probe box
        self.tort = tort(self.worker_object.DATA.shape[0])
        self.tort.check_signal.connect(self.update_probe_config)
        self.probe_vbox.addWidget(self.tort)
        self.probe_gbox.setVisible(True)
        self.tort_btn.setVisible(True)
        self.continue_btn.setVisible(False)
        self.settings_btn.setChecked(False)
        
        # try loading and adding default probe if it meets the criteria
        dflt_probe = ephys.read_probe_file(self.default_probe_file)
        nrows = self.worker_object.DATA.shape[0]
        if (dflt_probe is not None) and (dflt_probe.get_contact_count() <= nrows):
            self.tort.add_probe_row(dflt_probe)
            
        # disable data loading
        self.ddir_gbox.setEnabled(False)
        for btn in [self.oe_radio, self.nn_radio, self.nl_radio, self.manual_radio]:
            btn.setStyleSheet('QRadioButton:disabled {color : gray;}')
        
        
    def update_probe_config(self):
        x = bool(self.tort.tort_btn.isEnabled())
        self.tort.prb_icon_btn.new_status(x)
        self.tort_btn.setEnabled(bool(x and self.PARAMS is not None))
        
        
    def select_ddir(self):
        # open file popup to select raw data folder
        dlg = gi.FileDialog(init_ddir=self.raw_ddir, parent=self)
        res = dlg.exec()
        if res:
            self.raw_ddir = str(dlg.directory().path())
            self.RAW_ARRAY = None
            self.manual_upload_flabel.setText('')
            self.update_raw_ddir()
            
            
    def update_raw_ddir(self):
        # check if raw data files are present
        files = os.listdir(self.raw_ddir)
        xdat_files = [f for f in files if f.endswith('.xdat.json') or f.endswith('_data.xdat')]
        a = bool('structure.oebin' in files)
        b = bool(len(xdat_files) >= 2)
        c = bool(len([f for f in files if f.endswith('.ncs')]) > 0)
        d = bool(self.manual_upload_flabel.text() in files and self.RAW_ARRAY is not None)
        x = bool(a or b or c or d)
        
        # update widgets
        self.ddw.update_status(self.raw_ddir, x)
        self.oe_radio.setChecked(a)  # recording system buttons
        self.nn_radio.setChecked(b)
        self.nl_radio.setChecked(c)
        self.manual_radio.setChecked(d)
        
        self.RAW_DDIR_VALID = bool(x)
        self.adjustSize()
        self.continue_btn.setEnabled(bool(self.RAW_DDIR_VALID)) #delprobe and self.PROBE_CONFIG_VALID))
        
    
    def PROCESS_THE_DATA(self):
        
        self.PROBE_GROUP = prif.ProbeGroup()
        items = pyfx.layout_items(self.tort.qlayout)
        
        for item in items:
            prb = item.probe
            rows = item.ROWS  # group of rows belonging to this probe
            # reorder assigned rows by device indices
            sorted_rows = [rows[dvi] for dvi in prb.device_channel_indices]
            prb.set_contact_ids(rows)
            # device_indices * nprobes + start_row = sorted_rows
            prb.set_device_channel_indices(sorted_rows)
            self.PROBE_GROUP.add_probe(item.probe)
        
        self.worker_object.PROBE_GROUP = self.PROBE_GROUP
        param_dict = self.params_widget.DEFAULTS        # validated input params
        ddict = self.params_widget.ddict_from_gui()[0]  # current data processing params
        param_dict.update(ddict)
        self.worker_object.PARAMS = param_dict
        self.start_work()
        
    # def accept(self):
    #     print(f'Raw data folder: {self.raw_ddir}')
    #     super().accept()
        
        
class tort_row(QtWidgets.QWidget):
    
    def __init__(self, probe, nrows, start_row, mode):
        super().__init__()
        self.probe = probe
        self.nch = probe.get_contact_count()
        self.nrows = nrows
        self.div = int(self.nrows / self.nch)
        
        self.gen_layout()
        self.get_rows(start_row, mode)
        
        
    def gen_layout(self):
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
        
        # action buttons
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
        self.MODE = mode
        if self.MODE == 0:    # M consecutive indices from starting point
            self.ROWS = np.arange(start_row, start_row+self.nch)
            txt = f'{start_row}:{start_row+self.nch}'
        elif self.MODE == 1:  # M indices distributed evenly across M*N total rows
            self.ROWS = np.arange(0, self.nch*self.div, self.div) + start_row
            txt = f'{start_row}::{self.div}::{self.nch*self.div-self.div+start_row+1}'
        self.glabel.setText(self.glabel_fmt.format(a=self.probe.name, b=txt))
    
    
class tort(QtWidgets.QWidget):
    check_signal = QtCore.pyqtSignal()
    MODE = 0
    def __init__(self, nrows):
        super().__init__()
        self.nrows = nrows
        self.remaining_rows = np.arange(self.nrows)
        
        self.gen_layout()
        self.connect_signals()
        self.tort_btn = QtWidgets.QPushButton('PROCESS DATA')
        self.tort_btn.setEnabled(False)
    
    def gen_layout(self):
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
        
        self.data_assign_df = pd.DataFrame({'Row':np.arange(self.nrows), 'Probe(s)':''})
        self.probe_bgrp = QtWidgets.QButtonGroup()
        self.qframe = QtWidgets.QFrame()
        self.qframe.setFrameShape(QtWidgets.QFrame.Panel)
        self.qframe.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.qframe.setLineWidth(3)
        self.qframe.setMidLineWidth(3)
        qframe_layout = QtWidgets.QVBoxLayout(self.qframe)
        qframe_layout.setSpacing(10)
        #self.qframe.setStyleSheet('QFrame {background-color:red;}')
        self.qlayout = QtWidgets.QVBoxLayout()  # probe row container
        qframe_layout.addLayout(self.qlayout, stretch=2)
        #qframe_layout.addLayout(hbox, stretch=0)
        
        # data facts
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
        
        # assignment mode
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
        
        # interactive probe widgets 
        self.vlay0 = QtWidgets.QVBoxLayout()
        self.vlay0.addLayout(self.row0)
        self.vlay0.addWidget(self.qframe)
        
        # data info widgets
        self.vlay1 = QtWidgets.QVBoxLayout()
        self.vlay1.addLayout(self.row00)
        self.vlay1.addWidget(data_panel)
        #self.vlay1.addLayout(assign_vlay)
        
        self.hlay = QtWidgets.QHBoxLayout()
        self.hlay.addLayout(self.vlay0, stretch=3)
        self.hlay.addLayout(self.vlay1, stretch=1)
        
        self.layout.addLayout(self.hlay)
        
    
    def connect_signals(self):
        """ Connect widgets to functions """
        self.load_prb.clicked.connect(self.load_probe_from_file)
        self.create_prb.clicked.connect(self.design_probe) # ffindme
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
            return gi.MsgboxError.run(msg, parent=self)
        
        if self.MODE == 1:
            lens = [item.nch for item in pyfx.layout_items(self.qlayout)] + [nch]
            try:
                assert len(np.unique(lens)) < 2  # alternate indexing requires all same-size probes
            except AssertionError:
                msgbox = gi.MsgboxError('Alternate indexing requires all probes to be the same size', parent=self)
                msgbox.exec()
                return
        
        # get start row for probe based on prior probe assignment
        start_row = 0
        if self.qlayout.count() > 0:
            prev_rows = self.qlayout.itemAt(self.qlayout.count()-1).widget().ROWS
            start_row = pyfx.Edges(prev_rows)[1-self.MODE] + 1
        probe_row = tort_row(probe, self.nrows, start_row, self.MODE) #findme
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
        probe, _ = gi.FileDialog.load_file(filetype='probe', parent=self)
        if probe is None: return
        self.add_probe_row(probe)
    
    
    def design_probe(self):
        probe_popup = ProbeObjectPopup()
        probe_popup.setModal(True)
        probe_popup.accept_btn.setVisible(True)
        probe_popup.accept_btn.setText('CHOOSE PROBE')
        res = probe_popup.exec()
        if res:
            probe = probe_popup.probegod.probe
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
        self.tort_btn.setEnabled(is_valid)  # require valid config for next step
        self.data_lbl.setText(self.data_txt_fmt % (['red','green'][int(is_valid)], nvalid))
        self.check_signal.emit()
        
        
        
        