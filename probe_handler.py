#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 03:47:14 2024

@author: amandaschott
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtCore, QtGui
import probeinterface as prif
from probeinterface.plotting import plot_probe
import pdb
# custom modules
import pyfx
import ephys
import gui_items as gi
import resources_rc
    
def get_tetrode_func(site_spacing, tet_shape):
    """ Create function to accept y-position and return 4 tetrode coordinates
    $site_spacing: distance between adjacent electrodes in a recording site
    $tet_shape: "square" or "diamond" arrangements """
    # square configuration; p = (X ± dx/2, Y ± dx/2)
    if tet_shape == 'square': 
        dist = round(site_spacing/2, 1)
        def fx(pos):
            
            return [(0 - dist, pos + dist), # top left
                    (0 - dist, pos - dist), # bottom left
                    (0 + dist, pos + dist), # top right
                    (0 + dist, pos - dist)] # bottom right
        
    elif tet_shape == 'diamond':
        dist = round(site_spacing / np.sqrt(2), 1)  # 45-45-90 triangle
        def fx(pos):
            return [(0 - dist, pos),        # left
                    (0,        pos + dist), # top
                    (0,        pos - dist), # bottom
                    (0 + dist, pos)]        # right
                    
                    
    return dist, fx


def return_valid_probe(arg, use_dummy=False, skip_loading=False):
    """ Return valid probe from input or using dummy """
    probe = None
    try:  # arg is not a probeinterface object
        probe = arg
    except AttributeError:
        try:  # arg is not a valid file path
            probe = ephys.read_probe_file(arg, raise_exception=True)
        except:
            if skip_loading==False:
                # open file dialog
                dlg = gi.FileDialog(init_ddir=ephys.base_dirs()[2], is_probe=True)
                res = dlg.exec()
                if res:
                    fpath = dlg.selectedFiles()[0]
                    tmp = ephys.read_probe_file(fpath)
                    if tmp is not None:
                        probe = tmp
    if probe is None and use_dummy==True:
        print('WARNING: No valid probe configuration found - using dummy probe instead.')
        probe = prif.generate_dummy_probe()
        probe.name = 'DUMMY_PROBE'
    return probe


class probesimple(QtWidgets.QWidget):
    check_signal = QtCore.pyqtSignal()
    generate_signal = QtCore.pyqtSignal()
    
    def __init__(self, mainWin=None, show_generate_btn=True, parent=None, **kwargs):
        super().__init__(parent)
        self.mainWin = mainWin
        self.kwargs = kwargs
        self.SHOW_GENERATE_BTN = bool(show_generate_btn)
        
        self.gen_layout()
        self.connect_signals()
        
        self.generate_btn = QtWidgets.QPushButton('Construct probe')
        self.generate_btn.clicked.connect(self.construct_probe)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setVisible(self.SHOW_GENERATE_BTN)
        self.layout.addWidget(self.generate_btn)
        
    def gen_layout(self):
        """ Layout for popup window """
        qrect = pyfx.ScreenRect()
        #self.setMinimumWidth(250)
        self.layout = QtWidgets.QVBoxLayout(self)
        #self.layout.setSpacing(20)
        
        self.main_widget = QtWidgets.QWidget()
        txtbox_height = int(qrect.height() / 10)
        
        # probe name
        self.name_gbox = QtWidgets.QGroupBox()
        self.name_gbox.setContentsMargins(0,0,0,0)
        self.name_lay = QtWidgets.QHBoxLayout(self.name_gbox)
        #row0 = QtWidgets.QHBoxLayout()
        self.name_w = gi.LabeledWidget(QtWidgets.QLineEdit, 'Name')
        self.name_w.qw.setText('Probe_0')
        self.name_lay.addWidget(self.name_w)
        #name_lay.addWidget(self.toggle_btn)
        
        # x and y-coordinates
        self.xcoor_input = gi.LabeledWidget(QtWidgets.QTextEdit,
                                            'Enter X-coordinates for each electrode')
        self.ycoor_input = gi.LabeledWidget(QtWidgets.QTextEdit,
                                            'Enter Y-coordinates for each electrode')
        self.shk_input = gi.LabeledWidget(QtWidgets.QTextEdit,
                                            'Electrode shank IDs (multi-shank probes only)')
        self.chMap_input = gi.LabeledWidget(QtWidgets.QTextEdit,
                                            'Channel map (probe to data rows)')
        for item in [self.xcoor_input, self.ycoor_input, self.shk_input, self.chMap_input]:
            item.qw.setFixedHeight(txtbox_height)
            
        # contact dimensions
        contact_w = QtWidgets.QFrame()
        contact_hlay = QtWidgets.QHBoxLayout(contact_w)
        contact_hlay.setContentsMargins(0,0,0,0)
        contact_hlay.setSpacing(15)
        self.elh_w = gi.LabeledSpinbox('Contact height', double=True, maximum=99999, decimals=1, suffix=' \u00B5m') # unicode (um)
        self.elw_w = gi.LabeledSpinbox('Contact width', double=True, maximum=99999, decimals=1, suffix=' \u00B5m')
        self.elw_w.setEnabled(False)  # enabled for rectangle shape only
        self.elshape_w = gi.LabeledCombobox('Shape')
        self.elshape_w.addItems(['Circle', 'Square', 'Rectangle'])
        # set default values if given
        self.elh_w.qw.setValue(self.kwargs.get('el_h', self.elh_w.qw.value()))
        self.elw_w.qw.setValue(self.kwargs.get('el_w', self.elw_w.qw.value()))
        if 'el_shape' in self.kwargs and self.kwargs['el_shape'] in ['circle','square','rect']: 
            self.elshape_w.setCurrentIndex(['circle','square','rect'].index(self.kwargs['el_shape']))
        contact_hlay.addWidget(self.elh_w)
        contact_hlay.addWidget(self.elw_w)
        contact_hlay.addWidget(self.elshape_w)
        
        self.layout.addWidget(self.xcoor_input)
        self.layout.addWidget(self.ycoor_input)
        self.layout.addWidget(self.shk_input)
        self.layout.addWidget(self.chMap_input)
        self.layout.addWidget(contact_w)
    
    def connect_signals(self):
        self.name_w.qw.textChanged.connect(self.enable_ccm_btn)
        self.xcoor_input.qw.textChanged.connect(self.enable_ccm_btn) # changed x-coordinates
        self.ycoor_input.qw.textChanged.connect(self.enable_ccm_btn) # changed y-coordinates
        self.shk_input.qw.textChanged.connect(self.enable_ccm_btn)   # changed shank mapping
        self.chMap_input.qw.textChanged.connect(self.enable_ccm_btn) # changed channel mapping
        self.elw_w.qw.valueChanged.connect(self.enable_ccm_btn) # changed contact width
        self.elh_w.qw.valueChanged.connect(self.enable_ccm_btn) # changed contact height
        self.elshape_w.qw.currentTextChanged.connect(self.enable_ccm_btn) # changed contact shape
        self.setStyleSheet('QTextEdit { border : 2px solid gray; }')
    
    def ddict_from_gui(self):
        probe_name = self.name_w.qw.text().strip()
        xdata      = ''.join(self.xcoor_input.qw.toPlainText().split())
        ydata      = ''.join(self.ycoor_input.qw.toPlainText().split())
        shk_data   = ''.join(self.shk_input.qw.toPlainText().split())
        cmap_data  = ''.join(self.chMap_input.qw.toPlainText().split())
        
        try: 
            xc = np.array(eval(xdata), dtype='float')   # x-coordinates
            yc = np.array(eval(ydata), dtype='float')   # y-coordinates
            if shk_data  == '' : shk = np.ones_like(xc, dtype='int')    # shank IDs
            else               : shk = np.array(eval(shk_data), dtype='int')
            if cmap_data == '' : cmap = np.arange(xc.size, dtype='int') # channel map
            else               : cmap = np.atleast_1d(np.array(eval(cmap_data), dtype='int'))
        except:
            xc, yc, shk, cmap = [np.array([]), np.array([]), np.array([]), np.array([])]
        
        ddict = dict(probe_name = probe_name,
                     xc = xc,
                     yc = yc,
                     shk = shk,
                     cmap = cmap,
                     el_w = self.elw_w.value(),
                     el_h = self.elh_w.value(),
                     el_shape = self.elshape_w.currentText().replace('angle','').lower())
        return ddict
        
    
    def enable_ccm_btn(self):
        # symmetrical contacts (i.e. circles/squares) use electrode height only
        shape = self.elshape_w.currentText().replace('angle','').lower()
        is_sym = bool(shape in ['circle', 'square'])
        self.elw_w.setEnabled(not is_sym)
        if is_sym:
            pyfx.stealthy(self.elw_w.qw, self.elh_w.value())
            
        # check if current settings describe a valid probe
        ddict = self.ddict_from_gui()
        self.check_probe(ddict)
        self.adjustSize()
        
    
    def check_probe(self, ddict):
        PP = pd.Series(ddict)
        
        nelems = [PP.xc.size, PP.yc.size, PP.shk.size, PP.cmap.size]
        a = bool(PP.probe_name != '' and                  # probe name given
                 len(PP.probe_name.split()) == 1)         # no spaces in probe name
        b = bool(nelems[0] > 0 and nelems[1] > 0)         # x and y-coordinates given
        c = bool(len(np.unique(nelems)) == 1)             # equal length arrays
        d = bool(all(sorted(PP.cmap) == np.arange(PP.cmap.size)))  # valid channel map
        #d = bool(PP.cmap.size == np.unique(PP.cmap).size) # no duplicates in channel map
        
        x = bool(a and b and c and d)
        self.generate_btn.setEnabled(x)
        self.check_signal.emit()
        
    
    def construct_probe(self):#, arg=None, pplot=True):
        PP = pd.Series(self.ddict_from_gui())
        # create dataframe
        pdf = pd.DataFrame(dict(chanMap=PP.cmap, xc=PP.xc, yc=PP.yc, shank=PP.shk))
        
        # deduce electrode config
        ishank = np.where(PP.shk==np.unique(PP.shk)[0])[0]
        shank_x, shank_y = PP.xc[ishank], PP.yc[ishank]
        ncols = len(np.unique(shank_x))
        icols = [np.where(shank_x == xc)[0] for xc in np.unique(shank_x)]
        ydifs_per_col = [np.diff(sorted(shank_y[icol])) for icol in icols]
        mono_cols = [len(set(ydifs))==1 for ydifs in ydifs_per_col]
        tet_shape = None
        if ncols in [2,3] and mono_cols[1] == False: # potential tetrode
            # square: 2 columns, both irregularly spaced
            if ncols == 2 and mono_cols == [False,False]: tet_shape = 'square'
            # diamond: 3 columns, only center irregularly spaced
            elif ncols == 3 and mono_cols == [True,False,True]: tet_shape = 'diamond'
        if tet_shape is None:
            # linear/polytrode config: index column-wise (shank > x-coor > y-coor)
            df = pdf.sort_values(['shank','xc','yc'], ascending=[True,True,False]).reset_index(drop=True)
        else:
            # tetrode config: index row-wise (shank > y-coor > x-coor), then rearrange by group
            ddf = pdf.sort_values(['shank','yc','xc'], ascending=[True,False,True]).reset_index(drop=True)
            nsites = int(len(ddf) / 4)
            if tet_shape == 'square':    # top L > bottom L > top R > bottom R
                adj = [0,1,-1,0]
            elif tet_shape == 'diamond': # L > top M > bottom M > R
                adj = [1,-1,1,-1]
            df = ddf.set_index(ddf.index.values + np.tile(adj, nsites)).sort_index()
        # map default device indices to config-specific contact indices
        if all(PP.cmap == np.arange(PP.xc.size)):
            df['chanMap'] = sorted(df['chanMap'].values)
        # get contact shape/size
        if   PP.el_shape == 'circle': shape_kw = dict(radius=PP.el_h/2)
        elif PP.el_shape == 'square': shape_kw = dict(width=PP.el_w)
        elif PP.el_shape == 'rect'  : shape_kw = dict(width=PP.el_w, height=PP.el_h)
        
        # initialize probe data object findme
        probe = prif.Probe(ndim=2, name=PP.probe_name)
        probe.set_contacts(np.array(df[['xc','yc']]), shapes=PP.el_shape, 
                           shape_params=shape_kw, shank_ids=df.shank)
        probe.create_auto_shape('tip', margin=20)
        self.probe = prif.combine_probes([probe])
        self.probe.set_contact_ids(df.index.values)
        self.probe.set_device_channel_indices(df.chanMap)
        self.probe.annotate(**{'name':PP.probe_name})
        self.generate_signal.emit()
    
    
    def update_gui_from_probe(self, probe, show_msg=True):
        # update coordinate positions and channel/shank mapping
        pyfx.stealthy(self.name_w.qw, probe.name)
        xpos, ypos = probe.contact_positions.T
        pyfx.stealthy(self.xcoor_input.qw, str(list(xpos)))
        pyfx.stealthy(self.ycoor_input.qw, str(list(ypos)))
        pyfx.stealthy(self.shk_input.qw, str(list(probe.shank_ids.astype('int'))))
        pyfx.stealthy(self.chMap_input.qw,  str(list(probe.device_channel_indices)))
        
        ### electrode contacts
        shank = probe.get_shanks()[0]
        shape = shank.contact_shapes[0]
        if shape=='circle':
            elh, elw = [shank.contact_shape_params[0]['radius'] * 2] * 2
        elif shape=='square':
            elh, elw = [shank.contact_shape_params[0]['width']] * 2
        elif shape=='rect':
            elw, elh = [shank.contact_shape_params[0][k] for k in ['width','height']]
            shape='rectangle'
        pyfx.stealthy(self.elh_w.qw, elh)
        pyfx.stealthy(self.elw_w.qw, elw)
        pyfx.stealthy(self.elshape_w, shape.capitalize())
        self.enable_ccm_btn()
        

class probething(QtWidgets.QWidget):
    """ GUI for creating and saving probe files """
    check_signal = QtCore.pyqtSignal()
    generate_signal = QtCore.pyqtSignal()
    
    def __init__(self, mainWin=None, show_generate_btn=True, parent=None, **kwargs):
        """ Initialize popup window """
        super().__init__(parent)
        self.mainWin = mainWin
        self.kwargs = kwargs
        self.SHOW_GENERATE_BTN = bool(show_generate_btn)
        ephys.read_param_file(ephys.base_dirs()[4])
        #findme
        
        self.gen_layout()
        self.connect_signals()
        
        self.generate_btn = QtWidgets.QPushButton('Generate probe')
        self.generate_btn.clicked.connect(self.generate_probe)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setVisible(self.SHOW_GENERATE_BTN)
        #self.layout.addWidget(self.generate_btn)
        
        QtCore.QTimer.singleShot(50, self.enable_ccm_btn)
    
        
    def gen_layout(self):
        """ Layout for popup window """
        
        ###   PROBE PARAMS
        
        ### probe name
        self.name_gbox = QtWidgets.QGroupBox()
        self.name_gbox.setContentsMargins(0,0,0,0)
        
        self.name_lay = QtWidgets.QHBoxLayout(self.name_gbox)
        self.name_w = gi.LabeledWidget(QtWidgets.QLineEdit, 'Name')
        self.name_w.qw.setText('Probe_0')
        self.name_lay.addWidget(self.name_w)
        
        ###   PROBE SHANKS
        
        shank_gbox = QtWidgets.QGroupBox()
        shank_gbox.setContentsMargins(0,0,0,0)
        shank_lay = QtWidgets.QVBoxLayout(shank_gbox)
        shank_lay.setSpacing(10)
        
        ###   channel/shank counts
        
        ch_w = QtWidgets.QFrame()
        ch_hlay = QtWidgets.QHBoxLayout(ch_w)
        ch_hlay.setSpacing(20)
        # total number of probe channels
        nch_lay = QtWidgets.QVBoxLayout()
        nch_lay.setSpacing(1)
        self.nch_lbl = QtWidgets.QLabel('# total channels')
        self.nch_sbox = QtWidgets.QSpinBox()
        self.nch_sbox.setMaximum(99999)
        nch_lay.addWidget(self.nch_lbl)
        nch_lay.addWidget(self.nch_sbox)
        ch_hlay.addLayout(nch_lay)
        # number of shanks
        nshk_lay = QtWidgets.QVBoxLayout()
        nshk_lay.setSpacing(1)
        self.nshk_lbl = QtWidgets.QLabel('# shanks')
        self.nshk_sbox = QtWidgets.QSpinBox()
        self.nshk_sbox.setMinimum(1)
        nshk_lay.addWidget(self.nshk_lbl)
        nshk_lay.addWidget(self.nshk_sbox)
        ch_hlay.addLayout(nshk_lay)
        shank_lay.addWidget(ch_w)
        
        ###   channel distribution among shanks
        
        self.shk_w = QtWidgets.QFrame()
        self.shk_w.setFrameShape(QtWidgets.QFrame.Panel)
        self.shk_w.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.shk_w.setLineWidth(3)
        self.shk_w.setMidLineWidth(3)
        shk_grid = QtWidgets.QGridLayout(self.shk_w)
        # number of electrodes on each shank
        self.shkch_lbl = QtWidgets.QLabel('# channels')
        self.shkch_lbl.setAlignment(QtCore.Qt.AlignCenter)
        #self.shkch_lbl.setAlignment(QtCore.Qt.AlignCenter)
        shkch_hbox = QtWidgets.QHBoxLayout()
        # inter-shank distance
        self.shkd_lbl = QtWidgets.QLabel('Shank spacing')
        self.shkd_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.shkd_sbox = QtWidgets.QDoubleSpinBox()
        self.shkd_sbox.setMaximum(99999)
        self.shkd_sbox.setDecimals(0)
        self.shkd_sbox.setSuffix(' \u00B5m')
        
        # create widgets for up to 16 shanks
        self.shkch_list = []
        for i in range(16):
            # no. channels
            chbox = gi.LabeledSpinbox(f'<small>SHANK {i+1}</small>', maximum=99999)
            chbox.setVisible(i==0)
            self.shkch_list.append(chbox)
            shkch_hbox.addWidget(chbox)
        self.shk_w.hide()
        # for multi-shank probes, show option to match values
        match_icon = self.style().standardIcon(QtWidgets.QStyle.SP_ToolBarHorizontalExtensionButton)
        sz = QtWidgets.QSpinBox().sizeHint().height()
        isz = int(sz * 0.8)
        match_btns = [self.shkchmatch_btn, self.tmatch_btn] = [QtWidgets.QPushButton(), 
                                                               QtWidgets.QPushButton()]
        for match_btn in match_btns:
            match_btn.setStyleSheet('QPushButton {border:none; padding:0px;}')
            match_btn.setFixedSize(sz, sz)
            match_btn.setIcon(match_icon)
            match_btn.setIconSize(QtCore.QSize(isz, isz))
            match_btn.hide()
        self.shkch_lbl.setFixedHeight(sz)
        # add to layout
        shk_grid.addWidget(self.shkch_lbl, 0, 0, alignment=QtCore.Qt.AlignBottom)
        shk_grid.addLayout(shkch_hbox, 0, 1)
        shk_grid.addWidget(self.shkchmatch_btn, 0, 2, alignment=QtCore.Qt.AlignBottom)
        shk_grid.addWidget(self.shkd_lbl, 1, 0)
        shk_grid.addWidget(self.shkd_sbox, 1, 1)
        shk_grid.setColumnStretch(0, 0)
        shk_grid.setColumnStretch(1, 2)
        shk_grid.setColumnStretch(2, 0)
        shank_lay.addWidget(self.shk_w)
        
        ###   ELECTRODE CONFIGURATION
        
        geom_gbox = QtWidgets.QGroupBox()
        geom_gbox.setContentsMargins(0,0,0,0)
        geom_lay = QtWidgets.QVBoxLayout(geom_gbox)
        geom_lay.setSpacing(10)
        
        ###  geometry row
        
        config_w = QtWidgets.QFrame()
        config_hlay = QtWidgets.QHBoxLayout(config_w)
        config_hlay.setContentsMargins(0,0,0,0)
        config_hlay.setSpacing(20)
        # probe configuration
        config_lay = QtWidgets.QVBoxLayout()
        config_lay.setSpacing(1)
        self.config_lbl = QtWidgets.QLabel('Probe configuration')
        self.config_cbox = QtWidgets.QComboBox()
        self.config_cbox.addItems(['Linear/Edge','Polytrode','Tetrode'])
        config_lay.addWidget(self.config_lbl)
        config_lay.addWidget(self.config_cbox)
        config_hlay.addLayout(config_lay)
        # number of electrode columns
        ncol_lay = QtWidgets.QVBoxLayout()
        ncol_lay.setSpacing(1)
        self.ncol_lbl = QtWidgets.QLabel('# electrode columns')
        self.ncol_sbox = QtWidgets.QSpinBox()
        self.ncol_sbox.setRange(1,16)
        self.ncol_sbox.setEnabled(False)
        ncol_lay.addWidget(self.ncol_lbl)
        ncol_lay.addWidget(self.ncol_sbox)
        config_hlay.addLayout(ncol_lay)
        geom_lay.addWidget(config_w)
        
        ### electrode spacing row
        
        eld_w = QtWidgets.QFrame()
        eld_w.setFrameShape(QtWidgets.QFrame.Panel)
        eld_w.setFrameShadow(QtWidgets.QFrame.Sunken)
        eld_w.setLineWidth(3)
        eld_w.setMidLineWidth(3)
        eld_vlay = QtWidgets.QVBoxLayout(eld_w)
        eld_vlay.setSpacing(10)
        # electrode spacing along shank (y-axis) and across shank (x-axis)
        sbox_kw = dict(double=True, maximum=99999, decimals=0, suffix=' \u00B5m')
        self.eldy_w = gi.LabeledSpinbox('Inter-electrode spacing', **sbox_kw)
        self.eldx_widget = QtWidgets.QWidget()
        eldx_vlay = QtWidgets.QVBoxLayout(self.eldx_widget)
        eldx_vlay.setContentsMargins(0,0,0,0)
        eldx_vlay.setSpacing(1)
        self.eldx_lbl = QtWidgets.QLabel('Intra-electrode spacing')
        eldx_hlay = QtWidgets.QHBoxLayout()
        eldx_hlay.setSpacing(10)
        self.eldx_w = gi.LabeledSpinbox('X:', orientation='h', **sbox_kw)
        self.eldx2_w = gi.LabeledSpinbox('Y:', orientation='h', **sbox_kw)
        eldx_hlay.addWidget(self.eldx_w)
        eldx_hlay.addWidget(self.eldx2_w)
        eldx_vlay.addWidget(self.eldx_lbl)
        eldx_vlay.addLayout(eldx_hlay)
        el_hbox = QtWidgets.QHBoxLayout()
        el_hbox.setSpacing(15)
        el_hbox.addWidget(self.eldy_w)
        el_hbox.addWidget(self.eldx_widget)
        # electrode offset from tip (y-axis)
        self.tip_lbl = QtWidgets.QLabel('Tip offset')
        self.tip_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.tip_lbl.setFixedHeight(sz)
        tip_hbox = QtWidgets.QHBoxLayout()
        tip_hbox.addWidget(self.tip_lbl, stretch=0, alignment=QtCore.Qt.AlignBottom)
        self.tip_list  = []
        for i in range(16):
            tbox = gi.LabeledSpinbox(f'<small>COL {i+1}</small>', **sbox_kw)
            tbox.setVisible(i==0)
            self.tip_list.append(tbox)
            tip_hbox.addWidget(tbox, stretch=2)
        tip_hbox.addWidget(self.tmatch_btn, stretch=0, alignment=QtCore.Qt.AlignBottom)
        el_line = pyfx.DividerLine(lw=2, mlw=2)
        eld_vlay.addLayout(el_hbox)
        eld_vlay.addWidget(el_line)
        eld_vlay.addLayout(tip_hbox)
        geom_lay.addWidget(eld_w)
        
        ### contact dimensions row
        
        contact_w = QtWidgets.QFrame()
        contact_hlay = QtWidgets.QHBoxLayout(contact_w)
        contact_hlay.setContentsMargins(0,0,0,0)
        contact_hlay.setSpacing(15)
        self.elh_w = gi.LabeledSpinbox('Contact height', double=True, maximum=99999, decimals=1, suffix=' \u00B5m')
        self.elw_w = gi.LabeledSpinbox('Contact width', double=True, maximum=99999, decimals=1, suffix=' \u00B5m')
        self.elw_w.setEnabled(False)  # enabled for rectangle shape only
        self.elshape_w = gi.LabeledCombobox('Shape')
        self.elshape_w.addItems(['Circle', 'Square', 'Rectangle'])
        # set default values if given
        self.elh_w.qw.setValue(self.kwargs.get('el_h', self.elh_w.qw.value()))
        self.elw_w.qw.setValue(self.kwargs.get('el_w', self.elw_w.qw.value()))
        if 'el_shape' in self.kwargs and self.kwargs['el_shape'] in ['circle','square','rect']: 
            self.elshape_w.setCurrentIndex(['circle','square','rect'].index(self.kwargs['el_shape']))
        contact_hlay.addWidget(self.elh_w)
        contact_hlay.addWidget(self.elw_w)
        contact_hlay.addWidget(self.elshape_w)
        geom_lay.addWidget(contact_w)
        
        ###   CHANNEL MAPPING   ###
        
        self.chmap_gbox = QtWidgets.QGroupBox('Channel Mapping')
        self.chmap_gbox.setCheckable(True)
        self.chmap_gbox.setChecked(False)
        chmap_gbox_lay = QtWidgets.QVBoxLayout(self.chmap_gbox)
        
        df = pd.DataFrame(columns=['id','index'])
        self.tbl = gi.TableWidget(df)
        self.tbl.setFixedHeight(self.tbl.minimumSizeHint().height())
        chmap_gbox_lay.addWidget(self.tbl)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        #self.layout.addWidget(self.name_gbox)
        self.layout.addWidget(shank_gbox)
        self.layout.addWidget(geom_gbox)
        self.layout.addSpacing(10)
        #self.layout.addWidget(self.contact_gbox)
        self.layout.addWidget(self.chmap_gbox)
        #self.contact_gbox.hide()
        
        
    def connect_signals(self):
        """ Connect widgets to functions """
        self.name_w.qw.textChanged.connect(self.enable_ccm_btn)   # changed probe name
        self.nch_sbox.valueChanged.connect(self.enable_ccm_btn)   # changed number of channels
        self.nshk_sbox.valueChanged.connect(self.enable_ccm_btn)  # changed number of shanks
        _ = [chbox.qw.valueChanged.connect(self.enable_ccm_btn) for chbox in self.shkch_list] # changed shank channels
        self.shkd_sbox.valueChanged.connect(self.enable_ccm_btn)  # changed shank spacing
        self.config_cbox.currentTextChanged.connect(self.enable_ccm_btn) # changed configuration
        self.ncol_sbox.valueChanged.connect(self.enable_ccm_btn)  # changed number of electrode columns
        _ = [tbox.qw.valueChanged.connect(self.enable_ccm_btn) for tbox in self.tip_list] # changed tip offset
        self.eldy_w.qw.valueChanged.connect(self.enable_ccm_btn)  # changed y-spacing
        self.eldx_w.qw.valueChanged.connect(self.enable_ccm_btn)  # changed x-spacing between columns/across site
        self.eldx2_w.qw.valueChanged.connect(self.enable_ccm_btn) # changed y-spacing across site
        self.shkchmatch_btn.clicked.connect(self.match_values)
        self.tmatch_btn.clicked.connect(self.match_values)
        self.elw_w.qw.valueChanged.connect(self.enable_ccm_btn) # changed contact width
        self.elh_w.qw.valueChanged.connect(self.enable_ccm_btn) # changed contact height
        self.elshape_w.qw.currentTextChanged.connect(self.enable_ccm_btn) # changed contact shape
        
        # re-check probe when user toggles or edits device indexes
        self.chmap_gbox.toggled.connect(self.enable_ccm_btn)
        self.tbl.itemChanged.connect(lambda x: self.check_probe(self.ddict_from_gui()))
        
        
    def enable_ccm_btn(self):
        """ Enable channel map creation if probe params are set """
        # make sure default channel map matches current number of channels
        ids = np.arange(self.nch_sbox.value())
        df = pd.DataFrame(self.tbl.df)
        if len(ids) > len(df):    # add more rows to channel map
            addons = np.setdiff1d(ids, np.array(df.index))
            df_new = pd.concat([df, pd.DataFrame({'id':addons, 'index':addons}, index=addons)])
        elif len(ids) < len(df):  # remove extra rows from channel map
            df_new = pd.DataFrame(df.loc[ids, :])
        if len(ids) != len(df):
            self.tbl.blockSignals(True)
            self.tbl.load_df(df_new)
            self.tbl.blockSignals(False)
        h = [self.tbl.minimumSizeHint().height(), 
             self.tbl.sizeHint().height()][int(self.chmap_gbox.isChecked())]
        self.tbl.setFixedHeight(h)
        
        nshanks = int(self.nshk_sbox.value())
        # show spinboxes for each shank on multi-shank probe (hide for single shank)
        _ = [chbox.setVisible(i < nshanks) for i,chbox in enumerate(self.shkch_list)]
        self.shk_w.setVisible(nshanks > 1)
        # for multiple shanks, show option to match values
        self.shkchmatch_btn.setVisible(nshanks > 1)
        
        # set widget properties according to probe configuration
        config = self.config_cbox.currentText()
        is_poly = bool(config == 'Polytrode')  # polytrode config
        is_tet = bool(config == 'Tetrode')     # tetrode config
        self.eldy_w.setText(f'Inter-{"site" if is_tet else "electrode"} spacing')
        self.eldx_lbl.setText(f'Intra-{"site" if is_tet else "electrode"} spacing')
        #self.eldx_w.setText(f'Intra-{"site" if is_tet else "electrode"} spacing')
        ncol_min = int(is_poly or is_tet) + 1      # min. cols (linear=1, poly/tet=2)
        ncol_max = [1,16,3][self.config_cbox.currentIndex()] # max cols (linear=1, poly=16, tet=3)
        self.ncol_sbox.blockSignals(True)
        self.ncol_sbox.setRange(ncol_min, ncol_max)
        self.ncol_sbox.blockSignals(False)
        self.ncol_sbox.setEnabled(is_poly or is_tet)
        ncols = self.ncol_sbox.value()
        
        # for multiple electrode columns, show option to match spacing values
        self.tmatch_btn.setVisible(is_poly)
        
        # show/hide electrode spacing spinboxes
        if is_poly:
            # for polytrode: distance between shank tip and electrode for each column
            _ = [tbox.setVisible(i < ncols) for i,tbox in enumerate(self.tip_list)]
        else:
            # for linear/tetrode: one tip offset value
            _ = [tbox.setVisible(i < 1) for i,tbox in enumerate(self.tip_list)]
        self.tip_list[0].label.setVisible(is_poly)     # hide "Column 1" label
        self.eldx_widget.setVisible(is_poly or is_tet) # hide X spacing for linear probe
        self.eldx2_w.setVisible(is_tet)                # hide intra-site Y spacing for non-tetrodes
        self.eldx_w.label.setVisible(is_tet)
        #self.eldx_w.setVisible(is_poly or is_tet)   
        
        # symmetrical contacts (i.e. circles/squares) use electrode height only
        shape = self.elshape_w.currentText().replace('angle','').lower()
        is_sym = bool(shape in ['circle', 'square'])
        self.elw_w.setEnabled(not is_sym)
        if is_sym:
            pyfx.stealthy(self.elw_w.qw, self.elh_w.value())
        
        # check if current settings describe a valid probe
        ddict = self.ddict_from_gui()
        self.check_probe(ddict)
        self.adjustSize()
    
    
    def check_probe(self, ddict):
        PP = pd.Series(ddict)
        
        a = bool(PP.probe_name != '' and           # probe name given
                 len(PP.probe_name.split()) == 1)  # no spaces in probe name
        b = bool(PP.nch > 0 and sum(PP.ch_per_shank) == PP.nch) # shank channels add up to total
        c = bool(PP.dy > 0)                        # electrode y-spacing set
        d = bool(all(np.array(PP.tip_offset) > 0)) # electrode tip offset set for each col
        e = bool(all(sorted(PP.dev_idx) == np.arange(PP.nch)))  # valid channel map
        x = bool(a and b and c and d and e)
        if PP.config in ['Polytrode', 'Tetrode']:
            x = bool(x and PP.dx > 0)    # require electrode x/site spacing(s)
            if PP.config == 'Polytrode': # require symmetric columns
                f = bool(not any([x is None for x in PP.ch_per_col]))
            elif PP.config == 'Tetrode': # require even groups of 4 electrodes and intra-site X/Y spacing
                f = bool(all(np.array([NCH % 4 for NCH in PP.ch_per_shank]) == 0) and PP.diy > 0)
            x = bool(x and f)
            
        if PP.nshanks > 1:
            x = bool(x and PP.shank_spacing > 0) # require shank spacing
        
        #d = bool(PP.el_w > 0 and PP.el_h)  # optional: electrode contact dimensions
        
        self.generate_btn.setEnabled(x)
        self.check_signal.emit()
    
    
    def generate_probe(self):
        PP = pd.Series(self.ddict_from_gui())
        # get config
        is_poly, is_tet = [PP.config == pc for pc in ['Polytrode','Tetrode']]
        is_lin = bool(not (is_poly or is_tet))
        # get contact shape/size
        if   PP.el_shape == 'circle': shape_kw = dict(radius=PP.el_h/2)
        elif PP.el_shape == 'square': shape_kw = dict(width=PP.el_w)
        elif PP.el_shape == 'rect'  : shape_kw = dict(width=PP.el_w, height=PP.el_h)
        
        shank_list = []
        for i in range(PP.nshanks):
            nch_shank = PP.ch_per_shank[i]  # no. channels on shank (int)
            nch_cols = PP.ch_per_col[i]     # no. channels for each column (list)
            
            ### linear probe shank
            if is_lin:
                # initialize probe, adjust y-values by tip offset
                prb = prif.generate_linear_probe(num_elec=nch_shank, ypitch=PP.dy)
                pos_adj = prb.contact_positions[::-1] + np.array([0, PP.tip_offset[0]])
                prb.set_contacts(pos_adj, shapes=PP.el_shape, shape_params=shape_kw)
            ### polytrode probe shank
            elif is_poly:
                prb = prif.generate_multi_columns_probe(num_columns=PP.ncols,
                                                        num_contact_per_column=nch_cols,
                                                        xpitch=PP.dx, ypitch=PP.dy,
                                                        y_shift_per_column=PP.tip_offset)#
                xv,yv = np.array(prb.contact_positions.T)
                yv2 = np.concatenate([yv[np.where(xv==val)[0]][::-1] for val in np.unique(xv)])
                prb.set_contacts(np.array([xv, yv2]).T, shapes=PP.el_shape, shape_params=shape_kw)
            ### tetrode probe shank
            elif is_tet:
                dix, diy = PP.dx/2, PP.diy/2
                ngrps = int(nch_shank / 4)
                yctr = np.arange(0, ngrps*PP.dy, PP.dy)[::-1] + PP.tip_offset[0]
                
                # function returns 4 tetrode coordinates relative to y-position
                if PP.ncols == 2:  # square: top left, bottom left, top right, bottom right
                    fx = lambda pos: [(-dix, pos+diy), (-dix, pos-diy), (dix, pos+diy), (dix, pos-diy)]
                elif PP.ncols == 3: # diamond: left, top middle, bottom middle, right
                    fx = lambda pos: [(-dix, pos), (0, pos+diy), (0, pos-diy), (dix, pos)]
                tet_coors = np.concatenate(list(map(fx, yctr)))
                
                # create probe
                prb = prif.Probe(ndim=2)
                prb.set_contacts(tet_coors, shapes=PP.el_shape, shape_params=shape_kw)
                
            # set shank IDs, contact IDs
            prb.set_shank_ids(np.ones(nch_shank, dtype='int') + i)
            prb.set_contact_ids(np.arange(nch_shank) + sum(PP.ch_per_shank[0:i]))
            prb.set_device_channel_indices(np.array(PP.dev_idx) + sum(PP.ch_per_shank[0:i]))
            prb.create_auto_shape('tip', margin=20)
            prb.move([PP.shank_spacing * i, 0])
            shank_list.append(prb)
        
        self.probe = prif.combine_probes(shank_list)  # multi-shank probe
        contact_ids = np.concatenate([prb.contact_ids for prb in shank_list])
        dev_indexes = np.concatenate([prb.device_channel_indices for prb in shank_list])
        self.probe.set_contact_ids(contact_ids)
        self.probe.set_device_channel_indices(dev_indexes)
        shank_spacing = PP.shank_spacing if PP.nshanks > 1 else 0
        
        self.probe.annotate(**{'name'          : PP.probe_name,
                               'shank_spacing' : shank_spacing,
                               'config' : PP.config,
                               'site_spacing' : PP.dy,
                               'intra_site_spacing' : PP.dx})
        self.generate_signal.emit()

        
    def ddict_from_gui(self):
        probe_name = self.name_w.qw.text().strip()
        # get channels, shanks, and configuration type
        nch = int(self.nch_sbox.value())
        nshanks = int(self.nshk_sbox.value())
        ch_per_shank = [chbox.value() for chbox in self.shkch_list[0:nshanks]]
        if nshanks == 1:
            ch_per_shank = [int(nch)]
        shank_spacing = self.shkd_sbox.value()
        config = self.config_cbox.currentText()
        ncols = self.ncol_sbox.value()
        ch_per_col = [self.nch_by_col(nchan, ncols) for nchan in ch_per_shank]
        # get electrode spacing/tip offset
        yval = self.eldy_w.value()
        xval = self.eldx_w.value()
        iyval = self.eldx2_w.value()
        if config == 'Polytrode':  # may be multiple tip offsets
            tip_vals = [tbox.value() for tbox in self.tip_list[0:ncols]]
        else:
            tip_vals = [self.tip_list[0].value()]  # one tip offset value
        # get electrode contact dimensions
        el_w = self.elw_w.value()
        el_h = self.elh_w.value()
        el_shape = self.elshape_w.currentText().replace('angle','').lower()
        # get channel map (default=depth order)
        dev_idx = np.arange(nch)
        if self.chmap_gbox.isChecked():
            dev_idx = np.array(self.tbl.df['index'])
        
        ddict = dict(probe_name = probe_name,
                     nch = nch,
                     nshanks = nshanks,
                     ch_per_shank = ch_per_shank,  # no. channels on each shank
                     ch_per_col = ch_per_col,      # for each shank, no. channels per column
                     shank_spacing = shank_spacing,
                     config = config,
                     ncols = ncols,
                     dy = yval,
                     dx = xval,
                     diy = iyval,
                     tip_offset = tip_vals,
                     el_w = el_w,
                     el_h = el_h,
                     el_shape = el_shape,
                     dev_idx = dev_idx)
        return ddict
    
    def nch_by_col(self, nchan, ncols):
        """ Arrange $nchan probe channels in $ncols columns """
        # more columns than channels
        if ncols > nchan:
            return None
        ch_by_col = [int(nchan / ncols) for _ in range(ncols)]
        rmd = nchan % ncols  # number of remaining electrodes
        if rmd == 0:
            # channels equally distributed!
            return ch_by_col
        if rmd >= 2:
            # add channel to left/right column
            ch_by_col[0]  += 1
            ch_by_col[-1] += 1
        if (ncols % 2 > 0) and (rmd % 2 > 0):
            # odd number of columns with odd remainder: add to center column 
            ch_by_col[int(ncols/2)] += 1
        # return if longer channel(s) in center and/or edges
        if (ncols % 2 > 0) and (rmd in [1,3]):
            return ch_by_col
        elif (ncols % 2 == 0) and (rmd == 2):
            return ch_by_col
        else:
            return None
    
    def debug(self):
        pdb.set_trace()
        
    
    def update_gui_from_probe(self, probe, show_msg=True):
        shanks = probe.get_shanks()
        nshanks = probe.get_shank_count()
        # shank spacing: get distance between median x-coor value on adjacent shanks
        if nshanks > 1:
            xctr = [np.median(shk.contact_positions[:,0]) for shk in shanks]
            shank_spacing = xctr[1] - xctr[0]
        else: shank_spacing = 0
        if nshanks > 16:
            if show_msg:
                msgbox = gi.MsgboxError(msg='Probe has too many shanks!<br>Please load using "Paste" method.')
                msgbox.exec()
            return
            
        # ncols: get number of unique x-values on first shank
        shank = shanks[0]
        shank_x, shank_y = shank.contact_positions.T
        ncols = len(np.unique(shank_x))
        if ncols > 16:
            if show_msg:
                msgbox = gi.MsgboxError(msg='Probe has too many electrode columns!<br>Please load using "Paste" method.')
                msgbox.exec()
            return
        
        ### electrode geometry (group by xcoor, use y-spacing to determine config)
        xvals = np.unique(shank_x); xdifs = np.diff(xvals)  # get unique x-values (columns)
        if len(np.unique(xdifs)) > 1:
            if show_msg:
                msgbox = gi.MsgboxError(msg='Probe has unevenly spaced electrode columns!<br>Please load using "Paste" method.')
                msgbox.exec()
            return
        icols = [np.where(shank_x == xc)[0] for xc in xvals]
        ydifs_per_col = [np.diff(sorted(shank_y[icol])) for icol in icols]
        mono_cols = [len(set(ydifs))==1 for ydifs in ydifs_per_col]
        #is_mono = all([len(set(ydifs))==1 for ydifs in ydifs_per_col])
        if ncols == 1:
            config='Linear/Edge'  # one column, monotonic y-spacing
            eldy, eldx, eldiy = ydifs_per_col[0][0], 0, 0
            tip_offsets = [min(shank_y)]
        elif (ncols in [2,3]) and (not all(mono_cols)):
            config='Tetrode'  # 2-3 columns, uneven y-spacings (intra vs inter-site)
            # intra=site top to site bottom; inter=top of site n to bottom of site n+1
            intra, inter = np.unique(ydifs_per_col[1])  # col 1 works for squares and diamonds
            eldy = inter + (intra/2*2)  # center-to-center y-distance
            eldx = np.ptp(xvals)        # x-distance across recording site
            eldiy = intra               # y-distance across recording site
            tip_offsets = [min(shank_y) + intra/2]
        else:
            config='Polytrode' # 2+ columns, monotonic y-spacings
            eldy = ydifs_per_col[0][0]
            eldx = xdifs[0]    # dist between first two columns
            eldiy = 0
            tip_offsets = [min(shank_y[icol]) for icol in icols]
        
        ### electrode contacts
        shape = shank.contact_shapes[0]
        if shape=='circle': 
            elh, elw = [shank.contact_shape_params[0]['radius'] * 2] * 2
        elif shape=='square':
            elh, elw = [shank.contact_shape_params[0]['width']] * 2
        elif shape=='rect':
            elw, elh = [shank.contact_shape_params[0][k] for k in ['width','height']]
            shape='rectangle'
        
        ### channel map
        nch = probe.get_contact_count()
        dev_idx = np.array(probe.device_channel_indices)
        chk = bool(not all(dev_idx == np.arange(nch)))
        
        ### update widget values
        pyfx.stealthy(self.name_w.qw, probe.name)
        pyfx.stealthy(self.nch_sbox, probe.get_contact_count())
        pyfx.stealthy(self.nshk_sbox, probe.get_shank_count())
        _ = [pyfx.stealthy(box.qw, shk.get_contact_count()) for box,shk in zip(self.shkch_list, shanks)]
        pyfx.stealthy(self.shkd_sbox, shank_spacing)
        pyfx.stealthy(self.ncol_sbox, (1,16))
        pyfx.stealthy(self.ncol_sbox, ncols)
        pyfx.stealthy(self.config_cbox, config)
        pyfx.stealthy(self.eldy_w.qw, eldy)
        pyfx.stealthy(self.eldx_w.qw, eldx)
        pyfx.stealthy(self.eldx2_w.qw, eldiy)
        _ = [pyfx.stealthy(box.qw, off) for box,off in zip(self.tip_list, tip_offsets)]
        pyfx.stealthy(self.elh_w.qw, elh)
        pyfx.stealthy(self.elw_w.qw, elw)
        pyfx.stealthy(self.elshape_w, shape.capitalize())
        pyfx.stealthy(self.chmap_gbox, chk)
        self.tbl.blockSignals(True)
        self.tbl.load_df(pd.DataFrame({'id':np.arange(nch), 'index':dev_idx}))
        self.tbl.blockSignals(False)
        
        # new state of the union!
        self.enable_ccm_btn()  # updates widgets, validates probe settings
    
    def match_values(self):
        # identify sender
        if   self.sender() == self.shkchmatch_btn: llist = self.shkch_list
        elif self.sender() == self.tmatch_btn    : llist = self.tip_list
        # set spacing for all columns equal to the value for column 0 
        val = llist[0].value()
        boxes = [box.qw if hasattr(box, 'qw') else box for box in llist[1:]]
        for box in boxes:
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)
        self.enable_ccm_btn()
    

class IFigProbe(QtWidgets.QWidget):
    show_contact_ids = False
    show_device_ids  = False
    show_on_click = False
    contact_fontsize = 10
    ax_fontsize = 10
    
    def __init__(self, probe=None, plot_dummy=False):
        super().__init__()
        # create axis, initialize default style kwargs
        #self.ax = self.add_subplot()
        self.create_subplots()
        self.fig_w.set_tight_layout(True)
        self.fig.set_tight_layout(True)
        
        self.canvas_w = FigureCanvas(self.fig_w)
        self.canvas_w.setFixedWidth(150)
        #self.canvas_w.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setOrientation(QtCore.Qt.Vertical)
        self.toolbar.setMaximumWidth(30)
        
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.toolbar)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.canvas_w)
        
        self.probe_shape_kwargs = dict(ec='black',lw=3)
        self.contacts_kargs=dict(ec='gray', lw=1)
        if probe is None and plot_dummy:
            probe = prif.generate_dummy_probe()
            probe.name = 'DUMMY_PROBE'
        if probe is not None:
            self.new_probe(probe)
    
    
    def create_subplots(self):
        ### BUTTON AXES
        self.fig_w = matplotlib.figure.Figure()
        self.bax = self.fig_w.add_subplot()
        self.bax.set_axis_off()
        # create viewing buttons
        self.radio_btns = matplotlib.widgets.RadioButtons(ax=self.bax,
                          labels=['None', 'Contact IDs', 'Device IDs'], active=0, activecolor='black')
        _ = self.bax.collections[0].set(sizes=[125,125,125])
        _ = [lbl.set(fontsize=12) for lbl in self.radio_btns.labels]
        
        def callback2(label):
            self.show_contact_ids = bool(label=='Contact IDs')
            self.show_device_ids  = bool(label=='Device IDs')
            self.plot_probe_config()
            self.canvas_w.draw_idle()
        self.radio_btns.on_clicked(callback2)
        
        ### PROBE AXES
        self.fig = matplotlib.figure.Figure()
        self.ax = self.fig.add_subplot()
        sns.despine(self.fig)
        
    def new_probe(self, probe):
        self.probe = probe
        if self.probe is not None:
            self.ax.contact_text = [''] * len(self.probe.contact_positions)
        self.plot_probe_config()
        
        
    def plot_probe_config(self):
        #self.ax.clear()
        for item in self.ax.collections + self.ax.texts:
            item.remove()
        
        if self.probe is None: return
        
        kwargs = dict(probe_shape_kwargs = self.probe_shape_kwargs,
                      contacts_kargs = self.contacts_kargs, title=False)
        # plot probe contacts and outline
        contacts, outline = plot_probe(self.probe, ax=self.ax, **kwargs)
        
        
        # if not in click-to-view mode, show all eligible contact IDs
        if self.show_on_click == False:
            tups = self.create_axtexts()
            for (x,y,txt) in tups:
                _ = self.ax.text(x, y, txt, color='black', ha='center', va='center',
                                 fontsize=self.contact_fontsize)
        self.ax.set_title(self.probe.name)
        self.set_ax_font()
        self.canvas.draw_idle()
    
    
    def on_click(self, event):
        xyarr = np.array([[event.xdata, event.ydata]])
        if None in xyarr: return
        # calculate distance from clicked point to each electrode; find the closest
        sq_dist = (self.probe.contact_positions - xyarr)**2
        idx = np.argmin(np.sum(sq_dist, axis=1))
        # check whether clicked point is within contact vertices 
        vertice = self.probe.get_contact_vertices()[idx]
        is_in = matplotlib.path.Path(vertice).contains_points(xyarr)[0]
        if is_in:
            if type(self.ax.contact_text[idx]) == matplotlib.text.Text:
                print('already text!')
                axt = self.ax.contact_text[idx]
                axt.remove()                   # remove from Matplotlib figure
                del self.ax.contact_text[idx]  # erase from application
                self.ax.contact_text.insert(idx, '')  # replace with empty string
            else:
                print('new text!')
                llist = []
                if self.show_contact_ids: 
                    llist.append(str(self.probe.contact_ids[idx]))
                if self.show_device_ids:
                    llist.append(str(self.probe.device_channel_indices[idx]))
                if len(llist) > 0:
                    xys = [*self.probe.contact_positions[idx], os.linesep.join(llist)]
                    axt = self.ax.text(*xys, color='black', ha='center', va='center', fontsize=self.contact_fontsize)
                    self.ax.contact_text[idx] = axt
            event.canvas.draw()
    
    
    def create_axtexts(self):
        """ Return list of (x,y,txt) tuples for each contact """
        llist = []
        if self.show_contact_ids and self.probe.contact_ids is not None:
        #if wci and self.probe.contact_ids is not None:
            llist.append(np.array(self.probe.contact_ids, dtype='str'))
        if self.show_device_ids and self.probe.device_channel_indices is not None:
        #if wdi and self.probe.device_channel_indices is not None:
            llist.append(np.array(self.probe.device_channel_indices, dtype=str))
        # no qualifying IDs
        if len(llist) == 0: return []
        if len(llist) == 1:   # one set of IDs
            strings = llist[0]
        elif len(llist) > 1:  # 2+ stacked IDs shown for each contact 
            strings = list(map(lambda x: os.linesep.join(x), zip(*llist)))
        tups = list(zip(*[*self.probe.contact_positions.T, np.array(strings)]))
        return tups
    
    def set_ax_font(self):
        # set fonts for x-axis, y-axis, and title
        self.ax.xaxis.label.set_fontsize(self.ax_fontsize)
        self.ax.yaxis.label.set_fontsize(self.ax_fontsize)
        _ = [xtxt.set_fontsize(self.ax_fontsize) for xtxt in self.ax.xaxis.properties()['ticklabels']]
        _ = [ytxt.set_fontsize(self.ax_fontsize) for ytxt in self.ax.yaxis.properties()['ticklabels']]
        self.ax.title.set_fontsize(self.ax_fontsize)
    
    @classmethod
    def run_popup(cls, probe=None, plot_dummy=True, parent=None):
        pyfx.qapp()
        #fig = cls(probe, plot_dummy)
        fig_widget = cls(probe, plot_dummy)
        dlg = gi.Popup([fig_widget], parent=parent)
        dlg.setMinimumHeight(600)
        dlg.show()
        dlg.raise_()
        dlg.exec()
        return fig_widget


class probegod(QtWidgets.QWidget):
    generate_signal = QtCore.pyqtSignal()
    SOURCE = 0
    
    def __init__(self, probe=None, auto_plot=True, **kwargs):
        super().__init__()
        self.gen_layout()
        self.connect_signals()
        if auto_plot:
            self.generate_signal.connect(self.draw_probe)
        
        if probe is not None:
            self.process_probe_object(probe, show_msg=False)
    
    def gen_layout(self):
        
        ###   PROBE DESIGN WIDGETS   ###
        
        self.probething = probething(show_generate_btn=False)
        self.probesimple = probesimple(show_generate_btn=False)
        # set size policy (limits expansion)
        self.probesimple.setMinimumSize(self.probesimple.minimumSizeHint())
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                        QtWidgets.QSizePolicy.Ignored)
        self.probesimple.setSizePolicy(policy)
        self.probesimple.hide()
        
        ###   TOGGLE BUTTONS   ###
        
        # toggle between design/coordinate widgets 
        self.toggle_bgrp = QtWidgets.QButtonGroup()
        self.toggle0, self.toggle1 = [QtWidgets.QToolButton(), 
                                      QtWidgets.QToolButton()]
        
        tups = [('Build', ':/icons/shapes.png', self.toggle0),
                ('Paste', ':/icons/excel.png', self.toggle1)]
        for i,(txt,icon,btn) in enumerate(tups):
            btn.setCheckable(True)
            btn.setChecked(i==0)
            btn.setText(txt)
            btn.setIcon(QtGui.QIcon(icon))
            btn.setIconSize(QtCore.QSize(30,30))
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            btn.setAutoRaise(True)
            self.toggle_bgrp.addButton(btn, i)
        # top row contains name widget and toggle buttons
        self.top_row = QtWidgets.QWidget()
        self.top_lay = QtWidgets.QHBoxLayout(self.top_row)
        self.top_lay.setContentsMargins(0,0,0,0)
        name_gbox = QtWidgets.QGroupBox()
        name_lay = QtWidgets.QHBoxLayout(name_gbox)
        self.name_w = gi.LabeledWidget(QtWidgets.QLineEdit, 'Name')
        self.name_w.qw.setText('Probe_0')
        name_lay.addWidget(self.name_w)
        # set probe widget name inputs to common QLineEdit, reconnect signals
        self.probething.name_w = self.name_w
        self.probething.name_w.qw.textChanged.connect(self.probething.enable_ccm_btn)
        self.probesimple.name_w = self.name_w
        self.probesimple.name_w.qw.textChanged.connect(self.probesimple.enable_ccm_btn)
        self.top_lay.addWidget(name_gbox)
        self.top_lay.addSpacing(10)
        self.top_lay.addWidget(self.toggle0)
        self.top_lay.addWidget(self.toggle1)
        
        ###   ACTION BUTTONS   ###
        
        self.bbox = QtWidgets.QWidget()
        self.bbox_lay = QtWidgets.QHBoxLayout(self.bbox)
        self.bbox_lay.setContentsMargins(0,0,0,0)
        # common probe generator button (triggers generate function of current probe widget)
        self.generate_btn = QtWidgets.QPushButton('Generate')
        self.plot_btn = QtWidgets.QPushButton('Plot')
        self.load_btn = QtWidgets.QPushButton('Load')
        self.save_btn = QtWidgets.QPushButton('Save')
        self.accept_btn = QtWidgets.QPushButton()  # placeholder
        #self.accept_btn.setVisible(False)
        self.bbox_lay.addWidget(self.generate_btn)
        self.bbox_lay.addWidget(self.load_btn)
        self.bbox_lay.addWidget(self.plot_btn)
        self.bbox_lay.addWidget(self.save_btn)
        #self.bbox_lay.addWidget(self.accept_btn)
        
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.top_row, stretch=0)
        self.layout.addWidget(self.probething, stretch=2)
        self.layout.addWidget(self.probesimple, stretch=2)
        #self.layout.addWidget(self.probemap, stretch=0)
        self.layout.addWidget(self.bbox, stretch=0)
        
        self.setLayout(self.layout)
    
    
    def connect_signals(self):
        # connect toggle buttons to widget visibility
        self.toggle0.toggled.connect(lambda x: self.probething.setVisible(x))
        self.toggle1.toggled.connect(lambda x: self.probesimple.setVisible(x))
        self.toggle_bgrp.buttonToggled.connect(self.toggle_source)
        
        # use validation signals from widgets to enable/disable generate button
        self.probething.check_signal.connect(self.enable_generate_btn)
        self.probesimple.check_signal.connect(self.enable_generate_btn)
        
        # action buttons
        self.generate_btn.clicked.connect(self.generate_probe)  # .probe, emit signal
        self.plot_btn.clicked.connect(self.draw_probe)
        self.load_btn.clicked.connect(self.load_probe_from_file)
        self.save_btn.clicked.connect(self.save_probe_to_file)
    
    
    def save_probe_to_file(self):
        res = gi.FileDialog.save_file(data_object=self.probe, filetype='probe', parent=self)
        if res:
            self.save_btn.setEnabled(False)
        
    def load_probe_from_file(self):
        probe, _ = gi.FileDialog.load_file(filetype='probe', parent=self)
        if probe is None: return
        self.process_probe_object(probe, show_msg=True)
        
    def process_probe_object(self, probe, show_msg=True):
        if self.SOURCE == 0:
            self.probething.update_gui_from_probe(probe, show_msg=show_msg)
        elif self.SOURCE == 1:
            self.probesimple.update_gui_from_probe(probe, show_msg=show_msg)
        self.generate_probe()
        
    def toggle_source(self, btn, chk):
        """ Switch between probe designer and coordinate input widgets """
        self.SOURCE = int(self.toggle_bgrp.checkedId()) #1 - int(self.SOURCE)
        self.enable_generate_btn()
    
    def enable_generate_btn(self):
        """ Enable/disable generator button based on params of current probe widget """
        if   self.SOURCE == 0 : x = bool(self.probething.generate_btn.isEnabled())
        elif self.SOURCE == 1 : x = bool(self.probesimple.generate_btn.isEnabled())
        self.generate_btn.setEnabled(x)
        self.plot_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.accept_btn.setEnabled(False)
    
    def generate_probe(self):
        """ Generate new probe using current probe widget """
        if self.SOURCE == 0:
            self.probething.generate_probe()
            self.probe = self.probething.probe
        elif self.SOURCE == 1:
            self.probesimple.construct_probe()
            self.probe = self.probesimple.probe
        self.plot_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.accept_btn.setEnabled(True)
        self.generate_signal.emit()
        
    
    def draw_probe(self):
        fig_widget = IFigProbe(self.probe)
        #fig.set_tight_layout(True)
        #dlg = gi.MatplotlibPopup(fig, toolbar_pos='left')
        dlg = gi.Popup([fig_widget])
        dlg.setMinimumHeight(600)
        dlg.exec()
    
    
    @classmethod
    def run_probe_window(cls, probe=None, accept_txt='Accept probe', accept_visible=True, 
                         title='', PARAMS=None, parent=None):
        pyfx.qapp()
        my_probegod = cls(probe=probe, auto_plot=True, PARAMS=PARAMS)
        popup = gi.Popup(widgets=[my_probegod], parent=parent)
        popup.setWindowTitle(title)
        popup.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        my_probegod.accept_btn.setText(accept_txt)
        my_probegod.accept_btn.setVisible(accept_visible)
        my_probegod.accept_btn.clicked.connect(popup.accept)
        popup.layout.addWidget(my_probegod.accept_btn)
        popup.show()
        popup.raise_()
        res = popup.exec()
        if res:
            return my_probegod.probe
        return
