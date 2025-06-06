#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 03:47:14 2024

@author: amandaschott
"""
import sys
import os
import time
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
                dlg = gi.FileDialog(init_ddir=ephys.base_dirs()[1], is_probe=True)
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


class ElectrodeWidget(QtWidgets.QFrame):
    el_updated_signal = QtCore.pyqtSignal()
    
    def __init__(self, **kwargs):
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        # contact shape
        self.elshape_w = gi.LabeledCombobox('Shape')
        self.elshape_w.addItems(['Circle', 'Square', 'Rectangle'])
        # contact area/radius/width/height
        kw = dict(double=True, maximum=99999, decimals=2, suffix=' \u00B5m')
        self.el_area_w = gi.LabeledSpinbox('Contact area', **kw) # circles0/squares0
        self.el_area_w.qw.setSuffix(' \u00B5m\u00B2') # unicode um^2
        self.el_radius_w = gi.LabeledSpinbox('Contact radius', **kw) # circles1
        self.el_height_w = gi.LabeledSpinbox('Contact height', **kw) # squares1/rect0
        self.el_width_w = gi.LabeledSpinbox('Contact width', **kw)   # rect1
        
        self.init_layout()
        
        # initialize contact params
        el_shape = kwargs.get('el_shape', 'circle')
        el_area = np.round(kwargs.get('el_area', 0.0), 2)
        el_h = np.round(kwargs.get('el_h', 0.0), 2)
        self.elshape_w.setCurrentText(el_shape.capitalize())
        self.el_area_w.setValue(el_area)
        self.el_height_w.setValue(el_h)
        self.new_shape(el_shape)
        
        self.connect_signals()
    
    def connect_signals(self):
        self.elshape_w.qw.currentTextChanged.connect(self.new_shape)
        self.el_area_w.qw.valueChanged.connect(self.new_area)
        self.el_radius_w.qw.valueChanged.connect(self.new_radius)
        self.el_height_w.qw.valueChanged.connect(self.new_height)
        self.el_width_w.qw.valueChanged.connect(self.new_width)
        
    def new_shape(self, txt, block_signal=False):
        el_shape = txt.lower()
        # show/hide widgets
        is_sym = bool(el_shape in ['circle','square'])
        self.el_area_w.setVisible(is_sym)
        self.el_width_w.setVisible(not is_sym)
        A = self.el_area_w.value()
        if is_sym:
            x = bool(el_shape=='circle')
            self.el_radius_w.setVisible(x)
            self.el_height_w.setVisible(not x)
            # update radius/width from given area
            if x: pyfx.stealthy(self.el_radius_w.qw, np.sqrt(A/np.pi))
            else: pyfx.stealthy(self.el_height_w.qw, np.sqrt(A))
        else: # update rectangle width from given area and height
            self.el_radius_w.setVisible(False)
            self.el_height_w.setVisible(True)
            if self.el_height_w.value() == 0: pyfx.stealthy(self.el_width_w.qw, 0)
            else: pyfx.stealthy(self.el_width_w.qw, A/self.el_height_w.value())
        if not block_signal:
            self.el_updated_signal.emit()
    
    def new_area(self, A):
        # user changed area (circle/square)
        pyfx.stealthy(self.el_radius_w.qw, np.sqrt(A/np.pi))
        pyfx.stealthy(self.el_height_w.qw, np.sqrt(A))
        self.el_updated_signal.emit()
    
    def new_radius(self, r):
        A = np.pi * r**2
        pyfx.stealthy(self.el_area_w.qw, A)
        self.el_updated_signal.emit()
    
    def new_height(self, h):
        if self.elshape_w.currentText() == 'Square'     : A = h**2
        elif self.elshape_w.currentText() == 'Rectangle': A = h * self.el_width_w.value()
        pyfx.stealthy(self.el_area_w.qw, A)
        self.el_updated_signal.emit()
    
    def new_width(self, w):
        A = w * self.el_height_w.value()
        pyfx.stealthy(self.el_area_w.qw, A)
        self.el_updated_signal.emit()
    
    def init_layout(self):
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.elshape_w)
        self.layout.addWidget(self.el_area_w)
        self.layout.addWidget(self.el_radius_w)
        self.layout.addWidget(self.el_width_w)
        self.layout.addWidget(self.el_height_w)
    
    def get_contact_shape_kwargs(self):
        el_shape = self.elshape_w.currentText().lower()
        if el_shape == 'circle':
            shape_kw = {'radius' : self.el_radius_w.value()}
        elif el_shape == 'square':
            shape_kw = {'width' : self.el_height_w.value()}
        elif el_shape == 'rectangle':
            el_shape = 'rect'
            shape_kw = {'width' : self.el_width_w.value(),
                        'height' : self.el_height_w.value()}
        return el_shape, shape_kw
    
    def clear_widgets(self):
        """ Reset electrode contact widgets """
        pyfx.stealthy(self.elshape_w.qw, 'Circle')
        pyfx.stealthy(self.el_area_w.qw, 0)
        pyfx.stealthy(self.el_radius_w.qw, 0)
        pyfx.stealthy(self.el_width_w.qw, 0)
        pyfx.stealthy(self.el_height_w.qw, 0)
        self.new_shape('Circle', block_signal=True)
    
    def set_params_from_probe(self, probe):
        """ Set electrode contact widget values from probe """
        # load electrode shape and size
        shank = probe.get_shanks()[0]
        el_shape = shank.contact_shapes[0]
        shape_kw = shank.contact_shape_params[0]
        if el_shape == 'circle':
            r = shape_kw['radius']
            pyfx.stealthy(self.el_radius_w.qw, r)
            A = np.pi * r**2  # calculate area from radius
        else:
            w = shape_kw['width']
            if el_shape == 'square':
                pyfx.stealthy(self.el_height_w.qw, w)
                A = w**2  # calculate area as squared width/height
            elif el_shape == 'rect':
                h = shape_kw['height']
                pyfx.stealthy(self.el_height_w.qw, h)
                pyfx.stealthy(self.el_width_w.qw, w)
                A = w * h # calculate area as length * width
                el_shape = 'rectangle'
        pyfx.stealthy(self.el_area_w.qw, A)
        pyfx.stealthy(self.elshape_w.qw, el_shape.capitalize())
        self.new_shape(el_shape, block_signal=True)


class ChannelMapWidget(QtWidgets.QWidget):
    chmap_updated_signal = QtCore.pyqtSignal()
    
    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.MODE = mode # 0=builder, 1=paster
        
        self.nch = 0
        self.xpos = np.array([])
        self.ypos = np.array([])
        self.xy = {'x':self.xpos, 'y':self.ypos}
        
        self.gen_layout()
        self.connect_signals()
    
    def gen_layout(self):
        self.gbox = QtWidgets.QGroupBox('Channel Mapping')
        self.gbox.setCheckable(True)
        self.gbox.setChecked(False)
        gbox_lay = QtWidgets.QHBoxLayout(self.gbox)
        gbox_lay.setContentsMargins(0,0,0,0)
        # create table
        df1 = pd.DataFrame(columns=['id','device_index'])
        df2 = pd.DataFrame(columns=['xpos','ypos','device_index'])
        df = df1 if self.MODE==0 else df2
        self.tbl = gi.TableWidget(df, static_columns=['id','xpos','ypos'])
        # create text field
        self.chMap_input = QtWidgets.QTextEdit()
        gbox_lay.addWidget(self.tbl)
        gbox_lay.addWidget(self.chMap_input)
        
        self.bbox = QtWidgets.QVBoxLayout()
        self.tbl_btn = QtWidgets.QRadioButton('Table')
        self.tbl_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.txt_btn = QtWidgets.QRadioButton('Text field')
        self.txt_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.bbox.addWidget(self.tbl_btn)
        self.bbox.addWidget(self.txt_btn)
        # used for coordinate-based probe building only
        self.sortByPos_btn = QtWidgets.QPushButton('Use coordinates')
        self.sortByPos_btn.setAutoDefault(False)
        self.sortByPos_btn.setEnabled(False)
        self.bbox.addWidget(self.sortByPos_btn)
        
        self.tbl.setVisible(False)
        self.chMap_input.setVisible(True)
        self.tbl_btn.setChecked(False)
        self.txt_btn.setChecked(True)
        self.sortByPos_btn.setVisible(self.MODE==1)
        self.sortByPos_btn.hide()
            
        self.hlay = QtWidgets.QHBoxLayout(self)
        self.hlay.setContentsMargins(0,0,0,0)
        self.hlay.addWidget(self.gbox)
        self.hlay.addLayout(self.bbox)
    
    def stealthy_df(self, df):
        self.tbl.blockSignals(True)
        self.tbl.load_df(df)
        self.tbl.blockSignals(False)
    
    def connect_signals(self):
        self.gbox.toggled.connect(self.toggle_active)
        self.txt_btn.toggled.connect(self.toggle_view_mode0)
        
    def toggle_active(self, chk, block_signal=False):
        """ Enable or disable device indices """
        if not chk: # disable
            # reset device indices to match contact indices
            if self.MODE == 0:
                idx = np.arange(self.nch)
                ddf = pd.DataFrame({'id':idx, 'device_index':idx})
            elif self.MODE == 1:
                self.nch = max(len(self.xpos), len(self.ypos))
                idx = np.arange(self.nch)
                ddf = pd.DataFrame(columns=['xpos','ypos','device_index'], index=idx)
                for col,d in zip(ddf.columns, [self.xpos,self.ypos,idx]):
                    if len(d)>0:
                        ddf[col].loc[0:len(d)-1] = d
            self.stealthy_df(ddf)
            pyfx.stealthy(self.chMap_input, self.iter2str(idx))
        if not block_signal:
            self.chmap_updated_signal.emit()
    
    def set_dev_idx_by_pos(self):
        """ Set device indices as the vector that sorts x and y-coordinates """
        # get indices that sort x and y-coordinates in order
        tmpdf = pd.DataFrame({'xpos':self.xpos, 'ypos':self.ypos})
        tmpdf.sort_values(['xpos','ypos'], ascending=[True,False], inplace=True)
        dev_idx = np.array(tmpdf.index.values)
        self.xpos = np.array(tmpdf.xpos.values)
        self.ypos = np.array(tmpdf.ypos.values)
        self.xy = {'x':self.xpos, 'y':self.ypos}
        # update dataframe and text field
        ddf = tmpdf.reset_index(drop=True)
        ddf['device_index'] = dev_idx
        self.stealthy_df(ddf)
        pyfx.stealthy(self.chMap_input, self.iter2str(dev_idx))
        
    def toggle_sortByPos_btn(self):
        """ Enable/disable option to set device indices by sorted x and y-coordinates """
        a = self.gbox.isChecked()
        b = len(self.xpos) > 0
        c = len(self.xpos) == len(self.ypos)
        self.sortByPos_btn.setEnabled(bool(a and b and c))
    
    def toggle_view_mode0(self, txt_chk, block_signal=False):
        """ Switch view between table and text field """
        if self.txt_btn.isChecked():   # table -> text field
            data = self.iter2str(self.tbl.df['device_index'])
            pyfx.stealthy(self.chMap_input, data)
            
        elif self.tbl_btn.isChecked(): # text field -> table
            data, is_blank = self.str2iter(self.chMap_input)
            
            if self.MODE==0:
                if is_blank:
                    print('blank text field')
                elif len(data)==0:
                    print('could not parse indices from text field')
                elif len(data) != self.nch:
                    print(f'number of indices (n={len(data)}) does not match number of channels (nch={self.nch})')
                elif not all(data==np.array(data, dtype='int')):
                    print('all indices must be integers')
                else:
                    print('passing text indices to table')
                    if self.MODE == 0:
                        ddf = pd.DataFrame({'id':np.arange(self.nch),
                                            'device_index':np.array(data, dtype='int')})
                    self.stealthy_df(ddf)
                    
            elif self.MODE==1:
                self.nch = max(len(self.xpos), len(self.ypos), len(data))
                ddf = pd.DataFrame(columns=['xpos','ypos','device_index'], index=np.arange(self.nch))
                
                for col,d in zip(ddf.columns, [self.xpos,self.ypos,data]):
                    if len(d)>0:
                        ddf[col].loc[0:len(d)-1] = d
                self.stealthy_df(ddf)
                
                
        if not block_signal:
            self.chmap_updated_signal.emit()
            
        self.tbl.setVisible(not txt_chk)
        self.chMap_input.setVisible(txt_chk)
    
    def get_dev_idx(self):
        if self.tbl_btn.isChecked():
            return self.tbl.df['device_index']
        elif self.txt_btn.isChecked():
            return self.str2iter(self.chMap_input)[0]
            
    def set_nch(self, nch):
        if nch == self.nch: return
        self.nch = nch
        df = pd.DataFrame(self.tbl.df)
        if nch > len(df):    # add more rows to channel map
            addons = np.setdiff1d(np.arange(nch), np.array(df.index))
            df_new = pd.concat([df, pd.DataFrame({'id':addons, 'device_index':addons}, index=addons)])
        elif nch < len(df):  # remove extra rows from channel map
            df_new = pd.DataFrame(df.loc[np.arange(nch), :])
        self.stealthy_df(df_new)
        if not self.gbox.isChecked():
            pyfx.stealthy(self.chMap_input, self.iter2str(np.arange(nch)))
        
    def set_pos(self, key, coor_input, block_signal=False):
        assert key in ['x','y']
        pos, _ = self.str2iter(coor_input)
        if len(pos)==len(self.xy[key]) and all(pos == self.xy[key]):
            return
        
        # # update coordinate vector
        self.xy[key] = pos
        self.xpos = self.xy['x']
        self.ypos = self.xy['y']
        if not self.gbox.isChecked():
            dev_idx = np.arange(max(len(self.xpos), len(self.ypos)))
            pyfx.stealthy(self.chMap_input, self.iter2str(dev_idx))
        else:
            dev_idx, is_blank = self.str2iter(self.chMap_input)
            
        self.nch = max(len(self.xpos), len(self.ypos), len(dev_idx))
        ddf = pd.DataFrame(columns=['xpos','ypos','device_index'], index=np.arange(self.nch))
        
        for col,d in zip(ddf.columns, [self.xpos,self.ypos,dev_idx]):
            if len(d)>0: 
                ddf[col].loc[0:len(d)-1] = d
        self.stealthy_df(ddf)
        
        if not block_signal:
            self.chmap_updated_signal.emit()
        
    def set_params_from_probe(self, probe):
        """ Set device indices and coordinates from probe """
        # set number of channels, x-coordinates, and y-coordinates
        self.nch = int(probe.get_contact_count())
        self.xpos, self.ypos = np.array(probe.contact_positions.T)
        self.xy = {'x':self.xpos, 'y':self.ypos}
        # set device indices and widgets
        idx = np.arange(self.nch)
        dev_idx = np.array(probe.device_channel_indices, dtype='int')
        chk = bool(not all(dev_idx == idx))
        pyfx.stealthy(self.gbox, chk) # check groupbox if device indices differ from contact indices
        if self.MODE == 0:
            ddf = pd.DataFrame({'id':idx, 'device_index':dev_idx})
        elif self.MODE == 1:
            ddf = pd.DataFrame({'xpos':self.xpos, 'ypos':self.ypos, 'device_index':dev_idx})
        self.stealthy_df(ddf)
        pyfx.stealthy(self.chMap_input, self.iter2str(dev_idx))
    
    def clear_params(self):
        """ Reset device indices and coordinates """
        self.nch = 0
        self.xpos = np.array([])
        self.ypos = np.array([])
        self.xy = {'x':np.array([]), 'y':np.array([])}
        ddf = pd.DataFrame(columns=list(self.tbl.df.columns))
        self.stealthy_df(ddf)
        pyfx.stealthy(self.chMap_input, '')
        pyfx.stealthy(self.gbox, False)
    
    def iter2str(self, arr):
        arr = [int(x) if x==int(x) else x for x in arr]
        return str(arr).replace('[','').replace(']','')
    
    def str2iter(self, text_input):
        data = ''.join(text_input.toPlainText().split())
        is_blank = bool(data == '')
        try: 
            coor = np.atleast_1d(np.array(eval(data), dtype='float')) # x-coordinates
        except:
            coor = np.array([])
        return coor, is_blank
    
    
class GeomBox(QtWidgets.QGroupBox):
    geom_updated_signal = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.gen_layout()
        self.connect_signals()
    
    def gen_layout(self):
        
        ###  geometry row
        
        #config_w = QtWidgets.QFrame()
        config_hlay = QtWidgets.QHBoxLayout()
        config_hlay.setContentsMargins(0,0,0,0)
        config_hlay.setSpacing(20)
        # probe configuration
        self.config_cbox = gi.LabeledCombobox('Probe configuration')
        self.config_cbox.addItems(['Linear/Edge','Polytrode','Tetrode'])
        config_hlay.addWidget(self.config_cbox)
        # number of electrode columns
        self.ncol_sbox = gi.LabeledSpinbox('# electrode columns', range=[1,16])
        self.ncol_sbox.setEnabled(False)
        config_hlay.addWidget(self.ncol_sbox)
        
        ### electrode spacing row
        
        el_hlay = QtWidgets.QHBoxLayout()
        el_hlay.setContentsMargins(0,0,0,0)
        el_hlay.setSpacing(20)
        sbox_kw = dict(double=True, maximum=99999, decimals=0, suffix=' \u00B5m')
        # electrode spacing along shank (y-axis) and across shank (x-axis)
        self.eldy_w = gi.LabeledSpinbox('Inter-electrode spacing', **sbox_kw)
        self.eldx_w = gi.LabeledSpinbox('Intra-electrode spacing', **sbox_kw)
        # tetrode site spacing
        self.intersite_w = gi.LabeledSpinbox('Inter-site spacing', **sbox_kw)
        self.intrasite_w = gi.LabeledWidget(txt='Intra-site spacing')
        intra_hbox = QtWidgets.QHBoxLayout(self.intrasite_w.qw)
        intra_hbox.setContentsMargins(0,0,0,0)
        _x, _y = [gi.LabeledSpinbox(x, orientation='h', **sbox_kw) for x in ['X:','Y:']]
        intra_hbox.addWidget(_x)
        intra_hbox.addWidget(_y)
        self.intrasite_w.qw_x = _x.qw
        self.intrasite_w.qw_y = _y.qw
        el_hlay.addWidget(self.eldy_w)
        el_hlay.addWidget(self.eldx_w)
        el_hlay.addWidget(self.intersite_w)
        el_hlay.addWidget(self.intrasite_w)
        # initial configuration is linear (hide x-spacing and site spacing)
        self.eldx_w.hide()
        self.intersite_w.hide()
        self.intrasite_w.hide()
        
        ### tip offset row
        
        tip_hlay = QtWidgets.QHBoxLayout()
        self.tip_lbl = gi.LabeledWidget(QtWidgets.QLabel)
        self.tip_lbl.qw.setText('Tip offset')
        tip_hlay.addWidget(self.tip_lbl)
        self.tip_list = []
        for i in range(16):
            tbox = gi.LabeledSpinbox(f'<small>COL {i+1}</small>', **sbox_kw)
            tbox.setVisible(i==0)
            self.tip_list.append(tbox)
            tip_hlay.addWidget(tbox, stretch=2)
        # set all offsets equal to each other
        self.tmatch_btn = gi.LabeledPushbutton(icon=QtGui.QIcon(':/icons/double_chevron_right.png'),
                                               orientation='v', label_pos=0, spacing=1)
        tip_hlay.addWidget(self.tmatch_btn)
        self.tmatch_btn.hide()
        for lw in [self.tip_lbl, self.tip_list[0], self.tmatch_btn]:
            lw.label.setVisible(False)
       
        self.vlay = QtWidgets.QVBoxLayout(self)
        self.vlay.setSpacing(10)
        self.vlay.addLayout(config_hlay)
        self.vlay.addLayout(el_hlay)
        self.vlay.addLayout(tip_hlay)
    
    def connect_signals(self):
        self.config_cbox.qw.currentTextChanged.connect(self.new_config) # geometry
        self.ncol_sbox.qw.valueChanged.connect(self.new_config) # no. electrode columns
        self.eldy_w.qw.valueChanged.connect(self.new_config) # y-spacing (along shank)
        self.eldx_w.qw.valueChanged.connect(self.new_config) # x-spacing (across shank)
        self.intersite_w.qw.valueChanged.connect(self.new_config) # spacing between sites
        self.intrasite_w.qw_x.valueChanged.connect(self.new_config) # site height
        self.intrasite_w.qw_y.valueChanged.connect(self.new_config) # site width
        _ = [x.qw.valueChanged.connect(self.new_config) for x in self.tip_list] # tip offsets
        self.tmatch_btn.qw.clicked.connect(self.match_values)
            
    def new_config(self, _, block_signal=False):
        config = self.config_cbox.currentText()
        is_poly = (config == 'Polytrode')
        is_tet  = (config == 'Tetrode')
        
        c0 = int(is_poly or is_tet) + 1  # min. cols (linear=1, poly/tet=2)
        c1 = [1,16,3][self.config_cbox.currentIndex()] # max cols (linear=1, poly=16, tet=3)
        pyfx.stealthy(self.ncol_sbox.qw, (c0,c1))
        self.ncol_sbox.setEnabled(is_poly or is_tet)
        
        if not block_signal:
            self.geom_updated_signal.emit()
        
        self.intersite_w.setVisible(is_tet) # show site spacing for tetrodes
        self.intrasite_w.setVisible(is_tet)
        self.eldy_w.setVisible(not is_tet)  # show vertical spacing for all non-tetrodes
        self.eldx_w.setVisible(is_poly)     # show horizontal spacing for polytrodes
        self.tmatch_btn.setVisible(is_poly) # allow tip offset matching for polytrodes
        
        mmax = self.ncol_sbox.value() if is_poly else 1
        # set single tip offset (linear/tetrodes) or multiple offsets (polytrode)
        _ = [tbox.setVisible(i < mmax) for i,tbox in enumerate(self.tip_list)]
        for lw in [self.tip_lbl, self.tip_list[0], self.tmatch_btn]:
            lw.label.setVisible(is_poly)  # show "Column X" label for polytrodes only
    
    def match_values(self):
        """ Set all tip offset values equal to the first column """
        val = self.tip_list[0].value()
        for tbox in self.tip_list[1:]:
            pyfx.stealthy(tbox.qw, val)
        self.geom_updated_signal.emit()
    
    def get_geom_kwargs(self):
        """ Return dictionary of electrode geometry values """
        config = self.config_cbox.currentText()
        geom_kw = {'config' : config,
                   'ncols' : self.ncol_sbox.value(),
                   'dy' : self.eldy_w.value(),
                   'dx' : self.eldx_w.value(),
                   'site_spacing' : self.intersite_w.value(),
                   'site_w' : self.intrasite_w.qw_x.value(),
                   'site_h' : self.intrasite_w.qw_y.value()}
        geom_kw['tip_offset'] = [tbox.value() for tbox in self.tip_list[0:geom_kw['ncols']]]
        if config != 'Polytrode':
            geom_kw['tip_offset'] = [geom_kw['tip_offset'][0]]
        return geom_kw
    
    def clear_widgets(self):
        """ Reset electrode geometry params """
        pyfx.stealthy(self.config_cbox.qw, 'Linear/Edge') # reset configuration
        pyfx.stealthy(self.ncol_sbox.qw, 0) # reset no. channels
        pyfx.stealthy(self.eldy_w.qw, 0) # reset y-spacing
        pyfx.stealthy(self.eldx_w.qw, 0) # reset x-spacing
        pyfx.stealthy(self.intersite_w.qw, 0) # reset intersite spacing
        pyfx.stealthy(self.intrasite_w.qw_x, 0) # reset site width
        pyfx.stealthy(self.intrasite_w.qw_y, 0) # reset site height
        _ = [pyfx.stealthy(tbox.qw, 0) for tbox in self.tip_list] # reset tip offsets
        self.new_config('Linear/Edge', block_signal=True)
    
    def set_geom_from_probe(self, **kwargs):
        """ Set electrode geometry widgets from inputs """
        config = str(kwargs['config'])
        pyfx.stealthy(self.config_cbox.qw, config)
        pyfx.stealthy(self.ncol_sbox.qw, kwargs['ncols'])
        pyfx.stealthy(self.eldy_w.qw, kwargs.get('dy', 0))
        pyfx.stealthy(self.eldx_w.qw, kwargs.get('dx', 0))
        pyfx.stealthy(self.intersite_w.qw, kwargs.get('site_spacing', 0))
        pyfx.stealthy(self.intrasite_w.qw_x, kwargs.get('site_w', 0))
        pyfx.stealthy(self.intrasite_w.qw_y, kwargs.get('site_h', 0))
        for i,tbox in enumerate(self.tip_list):
            if i < len(kwargs['tip_offset']):
                pyfx.stealthy(tbox.qw, kwargs['tip_offset'][i])
            else:
                pyfx.stealthy(tbox.qw, 0)
        self.new_config(config, block_signal=True)
        
    
class probesimple(QtWidgets.QWidget):
    check_signal = QtCore.pyqtSignal()
    generate_signal = QtCore.pyqtSignal()
    x2shank = {}
    
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
        self.name_w = gi.LabeledWidget(QtWidgets.QLineEdit, 'Name')
        self.name_lay.addWidget(self.name_w)
        
        # create QTextEdit widgets for pasting probe data
        text_inputs = [QtWidgets.QTextEdit() for _ in range(3)]
        #_ = [te.setFixedHeight(txtbox_height) for te in text_inputs]
        self.xcoor_input, self.ycoor_input, self.shk_input = text_inputs
        captions = ['Enter X-coordinates for each electrode',
                    'Enter Y-coordinates for each electrode',
                    'Electrode shank IDs']
        for i,txt in enumerate(captions):
            txt_widget = QtWidgets.QWidget()
            txt_vbox = QtWidgets.QVBoxLayout(txt_widget)
            txt_vbox.setContentsMargins(0,0,0,0)
            txt_vbox.setSpacing(1)
            caption_hlay = QtWidgets.QHBoxLayout()
            caption_hlay.setContentsMargins(0,0,0,0)
            lbl = QtWidgets.QLabel(txt)
            caption_hlay.addWidget(lbl)
            if i==2:
                self.set_shanks_btn = QtWidgets.QPushButton('Set...')
                self.set_shanks_btn.setEnabled(False)
                self.set_shanks_btn.setFocusPolicy(QtCore.Qt.NoFocus)
                caption_hlay.addStretch()
                caption_hlay.addWidget(self.set_shanks_btn)
            txt_vbox.addLayout(caption_hlay)
            txt_vbox.addWidget(text_inputs[i])
            
            policy = txt_widget.sizePolicy()
            policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
            txt_widget.setSizePolicy(policy)
            
            self.layout.addWidget(txt_widget)
            
        # contact dimensions
        self.contact_w = ElectrodeWidget()
        policy = self.contact_w.sizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        self.contact_w.setSizePolicy(policy)
        self.layout.addWidget(self.contact_w)
        
        # channel mapping
        self.chmap_gbox = ChannelMapWidget(mode=1)
        self.layout.addWidget(self.chmap_gbox)
        
    
    def connect_signals(self):
        self.name_w.qw.textChanged.connect(self.enable_ccm_btn)
        
        self.xcoor_input.textChanged.connect(lambda: self.set_xycoor('x', self.xcoor_input))
        self.ycoor_input.textChanged.connect(lambda: self.set_xycoor('y', self.ycoor_input))
        
        self.shk_input.textChanged.connect(self.enable_ccm_btn)   # changed shank mapping
        self.chmap_gbox.chmap_updated_signal.connect(self.enable_ccm_btn)
        self.chmap_gbox.sortByPos_btn.clicked.connect(self.order_pos)
        self.chmap_gbox.tbl.itemChanged.connect(lambda x: self.check_probe(self.ddict_from_gui()))
        self.chmap_gbox.chMap_input.textChanged.connect(lambda: self.check_probe(self.ddict_from_gui()))
        self.set_shanks_btn.clicked.connect(self.set_shanks)
        self.contact_w.el_updated_signal.connect(self.enable_ccm_btn)
        self.setStyleSheet('QTextEdit { border : 2px solid gray; }')
    
    
    def set_xycoor(self, key, text_input):
        """ User updates text field with new X or Y-coordinates """
        # set xpos and ypos variables in channel mapping widget
        self.chmap_gbox.set_pos(key, text_input, block_signal=True)
        if key=='x':
            _ = self.assign_shanks_by_x()
        self.enable_ccm_btn()
    
    def assign_shanks_by_x(self):
        unique_x = np.unique(self.chmap_gbox.xpos)
        if len(unique_x)==len(self.x2shank) and all(unique_x==list(self.x2shank.keys())):
            # update shank IDs based on new x-coordinates
            shank_ids = [self.x2shank[x] for x in self.chmap_gbox.xpos]
            pyfx.stealthy(self.shk_input, self.chmap_gbox.iter2str(shank_ids))
            return True
        return False
        
    def set_shanks(self):
        xpos = self.chmap_gbox.xpos
        dlg = ShankIDPopup(xpos)
        if dlg.exec():
            self.x2shank = dict(dlg.x2shank)
            _ = self.assign_shanks_by_x()
            self.enable_ccm_btn()
    
    def order_pos(self):
        """ Set channel mapping to the vector that sorts x/y-coordinates in ascending/descending order """
        self.chmap_gbox.set_dev_idx_by_pos()
        pyfx.stealthy(self.xcoor_input, self.chmap_gbox.iter2str(self.chmap_gbox.xpos))
        pyfx.stealthy(self.ycoor_input, self.chmap_gbox.iter2str(self.chmap_gbox.ypos))
        res = self.assign_shanks_by_x()
        if not res:
            shk_data = ''.join(self.shk_input.toPlainText().split())
            try:
                shk = np.atleast_1d(np.array(eval(shk_data), dtype='int'))
                if len(shk) == len(self.chmap_gbox.xpos):
                    dev_idx = self.chmap_gbox.get_dev_idx()
                    shank_ids = shk[dev_idx]
                    pyfx.stealthy(self.shk_input, self.chmap_gbox.iter2str(shank_ids))
                else:
                    pyfx.stealthy(self.shk_input, '')
            except:
                pyfx.stealthy(self.shk_input, '')
        self.enable_ccm_btn()
        
        
    def ddict_from_gui(self):
        probe_name = self.name_w.qw.text().strip()
        xc = self.chmap_gbox.xpos
        yc = self.chmap_gbox.ypos
        
        # get device indices
        dev_idx = self.chmap_gbox.get_dev_idx()
        
        shk_data   = ''.join(self.shk_input.toPlainText().split())
        # get/create shank data
        if shk_data == '':
            shk = np.zeros_like(xc, dtype='int')
        else:
            try   : shk = np.atleast_1d(np.array(eval(shk_data), dtype='int')) # shank IDs
            except: shk = np.array([])
        
        el_shape, shape_kw = self.contact_w.get_contact_shape_kwargs()
        ddict = dict(probe_name = probe_name,
                     xc = xc,
                     yc = yc,
                     shk = shk,
                     dev_idx = dev_idx,
                     el_shape = el_shape,
                     shape_kw = shape_kw)
        return ddict
        
        
    def enable_ccm_btn(self):
        self.chmap_gbox.toggle_sortByPos_btn()
        self.set_shanks_btn.setEnabled(len(self.chmap_gbox.xpos) > 0)
        
        # check if current settings describe a valid probe
        ddict = self.ddict_from_gui()
        
        self.check_probe(ddict)
        
    
    def check_probe(self, ddict):
        PP = pd.Series(ddict)
        
        nelems = [PP.xc.size, PP.yc.size, PP.shk.size]
        a = (PP.probe_name != '' and           # probe name given
             len(PP.probe_name.split()) == 1)  # no spaces in probe name
        b = (nelems[0] > 0 and nelems[1] > 0)  # x and y-coordinates given
        c = all(PP.yc >= 0)  # all y-coordinates must be positive
        d = (len(np.unique(nelems)) == 1)      # equal size arrays
        if len(PP.xc) == len(PP.yc):
            e = (len(PP.dev_idx)==len(PP.xc) and # correct number of device indices
                 all(sorted(PP.dev_idx)==np.arange(len(PP.xc)))) # valid channel map
        else: e = False # no agreed upon number of channels
        f = all(np.array(list(PP.shape_kw.values())) > 0) # valid contact size
        
        x = bool(a and b and c and d and e and f)
        self.generate_btn.setEnabled(x)
        self.check_signal.emit()
        
    
    def construct_probe(self):#, arg=None, pplot=True):
        PP = pd.Series(self.ddict_from_gui())
        pdf = pd.DataFrame(dict(xc=PP.xc, yc=PP.yc, shank=PP.shk))
        df = pd.DataFrame(pdf)
        df['chanMap'] = PP.dev_idx
        df.set_index(df['shank'].values, inplace=True)
        
        shank_ids = df['shank'].values
        ishanks, ch_per_shank = zip(*[(x,list(shank_ids).count(x)) for x in np.unique(shank_ids)])
        nshanks = len(ishanks)
        
        shank_list = []
        for i,shk in enumerate(ishanks):
            ddf = df.loc[shk]
            nch_shank = len(ddf)
            istart = sum(ch_per_shank[0:i])
            
            prb = prif.Probe(ndim=2)
            prb.set_contacts(np.array(ddf[['xc','yc']]), shapes=PP.el_shape, 
                             shape_params=PP.shape_kw)
            prb.set_shank_ids(np.ones(nch_shank, dtype='int') + i)
            prb.set_contact_ids(np.arange(nch_shank) + istart)
            prb.set_device_channel_indices(np.array(ddf['chanMap'].values))
            prb.create_auto_shape('tip', margin=20)
            shank_list.append(prb)
            
        self.probe = prif.combine_probes(shank_list)  # multi-shank probe
        contact_ids = np.concatenate([prb.contact_ids for prb in shank_list])
        dev_indexes = np.concatenate([prb.device_channel_indices for prb in shank_list])
        self.probe.set_contact_ids(contact_ids)
        self.probe.set_device_channel_indices(dev_indexes)
        self.probe.annotate(**{'name':PP.probe_name})
        self.generate_signal.emit()
    
    def clear_gui(self):
        """ Reset $probesimple widgets """
        pyfx.stealthy(self.name_w.qw, '')
        pyfx.stealthy(self.xcoor_input, '')
        pyfx.stealthy(self.ycoor_input, '')
        pyfx.stealthy(self.shk_input, '')
        self.contact_w.clear_widgets()
        self.chmap_gbox.clear_params()
        self.enable_ccm_btn()
        
    def update_gui_from_probe(self, probe, show_msg=True):
        """ Update $probesimple widgets from probe """
        pyfx.stealthy(self.name_w.qw, probe.name)
        dev_idx = np.array(probe.device_channel_indices)
        # ordered contact positions/shank IDs
        xpos, ypos = np.array(probe.contact_positions).T
        shks = probe.shank_ids.astype('int')
        
        pyfx.stealthy(self.xcoor_input, self.chmap_gbox.iter2str(xpos))
        pyfx.stealthy(self.ycoor_input, self.chmap_gbox.iter2str(ypos))
        pyfx.stealthy(self.shk_input, self.chmap_gbox.iter2str(shks))
        self.chmap_gbox.set_params_from_probe(probe)
        
        # pyfx.stealthy(self.chmap_gbox.chMap_input, self.chmap_gbox.iter2str(dev_idx))
        # ddf = pd.DataFrame({'xpos':xpos, 'ypos':ypos, 'device_index':dev_idx})
        # self.chmap_gbox.stealthy_df(ddf)
        # is_chMap = not all(dev_idx == np.arange(len(dev_idx)))
        # pyfx.stealthy(self.chmap_gbox.gbox, is_chMap)
        # #pyfx.stealthy(self.chMap_input,  str(list(dev_idx)))
        
        ### electrode contacts
        self.contact_w.set_params_from_probe(probe)
        
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
        
        self.gen_layout()
        self.connect_signals()
        
        self.generate_btn = QtWidgets.QPushButton('Construct probe')
        self.generate_btn.clicked.connect(self.construct_probe)
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
        self.name_lay.addWidget(self.name_w)
        
        ###   PROBE SHANKS
        
        shank_gbox = QtWidgets.QGroupBox()
        shank_gbox.setContentsMargins(0,0,0,0)
        shank_lay = QtWidgets.QVBoxLayout(shank_gbox)
        shank_lay.setSpacing(10)
        
        ###   channel/shank counts
        
        ch_w = QtWidgets.QFrame()
        ch_hlay = QtWidgets.QHBoxLayout(ch_w)
        ch_hlay.setContentsMargins(0,0,0,0)
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
        match_btns = [self.shkchmatch_btn, _] = [QtWidgets.QPushButton(), 
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
        
        # electrode geometry/spacing
        self.new_geom_gbox = GeomBox()
        policy = self.new_geom_gbox.sizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        self.new_geom_gbox.setSizePolicy(policy)
        
        # contact dimensions
        self.contact_w = ElectrodeWidget()
        policy = self.contact_w.sizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Maximum)
        self.contact_w.setSizePolicy(policy)
        
        # channel mapping
        self.chmap_gbox = ChannelMapWidget(mode=0)
        
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(shank_gbox)
        #self.layout.addWidget(geom_gbox)
        self.layout.addWidget(self.new_geom_gbox)
        
        self.layout.addWidget(self.contact_w)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.chmap_gbox)
        #self.layout.addSpacing(10)
        
        
        
    def connect_signals(self):
        """ Connect widgets to functions """
        self.name_w.qw.textChanged.connect(self.enable_ccm_btn)   # probe name
        self.nch_sbox.valueChanged.connect(self.enable_ccm_btn)   # no. channels
        self.nshk_sbox.valueChanged.connect(self.enable_ccm_btn)  # no. shanks/ch per shank
        _ = [chbox.qw.valueChanged.connect(self.enable_ccm_btn) for chbox in self.shkch_list]
        self.shkd_sbox.valueChanged.connect(self.enable_ccm_btn)  # shank spacing
        self.shkchmatch_btn.clicked.connect(self.match_values)
        
        # self.config_cbox.currentTextChanged.connect(self.enable_ccm_btn) # changed configuration
        # self.ncol_sbox.valueChanged.connect(self.enable_ccm_btn)  # no. electrode columns
        # _ = [tbox.qw.valueChanged.connect(self.enable_ccm_btn) for tbox in self.tip_list] # tip offset
        # self.eldy_w.qw.valueChanged.connect(self.enable_ccm_btn)  # y-spacing
        # self.eldx_w.qw.valueChanged.connect(self.enable_ccm_btn)  # x-spacing between cols/across site
        # self.eldx2_w.qw.valueChanged.connect(self.enable_ccm_btn) # y-spacing across site
        # self.tmatch_btn.clicked.connect(self.match_values)
        
        self.new_geom_gbox.geom_updated_signal.connect(self.enable_ccm_btn)
        self.chmap_gbox.chmap_updated_signal.connect(self.enable_ccm_btn)
        self.chmap_gbox.tbl.itemChanged.connect(lambda x: self.check_probe(self.ddict_from_gui()))
        self.chmap_gbox.chMap_input.textChanged.connect(lambda: self.check_probe(self.ddict_from_gui()))
        self.contact_w.el_updated_signal.connect(self.enable_ccm_btn)
        
    
    def enable_ccm_btn(self):
        """ Enable channel map creation if probe params are set """
        # make sure default channel map matches current number of channels
        self.chmap_gbox.set_nch(self.nch_sbox.value())
        
        nshanks = int(self.nshk_sbox.value())
        # show spinboxes for each shank on multi-shank probe (hide for single shank)
        _ = [chbox.setVisible(i < nshanks) for i,chbox in enumerate(self.shkch_list)]
        self.shk_w.setVisible(nshanks > 1)
        # for multiple shanks, show option to match values
        self.shkchmatch_btn.setVisible(nshanks > 1)
        
        # check if current settings describe a valid probe
        ddict = self.ddict_from_gui()
        self.check_probe(ddict)
    
    
    def check_probe(self, ddict):
        PP = pd.Series(ddict)
        
        a = (PP.probe_name != '' and           # probe name given
             len(PP.probe_name.split()) == 1)  # no spaces in probe name
        b = (PP.nch > 0 and sum(PP.ch_per_shank) == PP.nch) # shank channels add up to total
        c = all(np.array(PP.tip_offset) > 0) # electrode tip offset set for each col
        d = (len(PP.dev_idx) == PP.nch and   # correct number of device indices
            all(sorted(PP.dev_idx)==np.arange(PP.nch)))   # valid channel map
        e = all(np.array(list(PP.shape_kw.values())) > 0) # valid contact size
        x = bool(a and b and c and d and e)
        
        if PP.config == 'Tetrode': # require inter/intra-site spacing and even groups of 4 contacts
            f = (PP.site_spacing > 0 and PP.site_w > 0 and PP.site_h > 0)
            g = all(np.array([NCH % 4 for NCH in PP.ch_per_shank]) == 0)
        else:  # electrode y-spacing (along shank) is set and columns are even
            f = (PP.dy > 0) 
            if PP.config == 'Polytrode':
                f = bool(f and PP.dx > 0)
            g = not any([nch is None for nch in PP.ch_per_col])
        x = bool(x and f and g)
            
        if PP.nshanks > 1:
            x = bool(x and PP.shank_spacing > 0) # require shank spacing
        
        self.generate_btn.setEnabled(x)
        self.check_signal.emit()
    
    
    def construct_probe(self):
        """ Create probe object from current parameters (probething) """
        PP = pd.Series(self.ddict_from_gui())
        # get config
        is_poly, is_tet = [PP.config == pc for pc in ['Polytrode','Tetrode']]
        is_lin = bool(not (is_poly or is_tet))
        
        shank_list = []
        for i in range(PP.nshanks):
            nch_shank = PP.ch_per_shank[i]  # no. channels on shank (int)
            nch_cols = PP.ch_per_col[i]     # no. channels for each column (list)
            istart = sum(PP.ch_per_shank[0:i]) # no. channels on previous shanks
            
            ### linear probe shank
            if is_lin:
                # initialize probe, adjust y-values by tip offset
                prb = prif.generate_linear_probe(num_elec=nch_shank, ypitch=PP.dy)
                pos_adj = prb.contact_positions[::-1] + np.array([0, PP.tip_offset[0]])
                prb.set_contacts(pos_adj, shapes=PP.el_shape, shape_params=PP.shape_kw)
            ### polytrode probe shank
            elif is_poly:
                prb = prif.generate_multi_columns_probe(num_columns=PP.ncols,
                                                        num_contact_per_column=nch_cols,
                                                        xpitch=PP.dx, ypitch=PP.dy,
                                                        y_shift_per_column=PP.tip_offset)
                xv,yv = np.array(prb.contact_positions.T)
                yv2 = np.concatenate([yv[np.where(xv==val)[0]][::-1] for val in np.unique(xv)])
                prb.set_contacts(np.array([xv, yv2]).T, shapes=PP.el_shape, shape_params=PP.shape_kw)
            ### tetrode probe shank
            elif is_tet:
                # get y-coordinates for the center of each tetrode site
                ngrps = int(nch_shank / 4)
                yctr = np.arange(0, ngrps*PP.site_spacing, PP.site_spacing)[::-1] + PP.tip_offset[0]
                # get positions of all 4 contacts relative to the site center
                dix, diy = PP.site_w/2, PP.site_h/2
                if PP.ncols == 2:  # square: top left, bottom left, top right, bottom right
                    fx = lambda pos: [(-dix, pos+diy), (-dix, pos-diy), (dix, pos+diy), (dix, pos-diy)]
                elif PP.ncols == 3: # diamond: left, top middle, bottom middle, right
                    fx = lambda pos: [(-dix, pos), (0, pos+diy), (0, pos-diy), (dix, pos)]
                tet_coors = np.concatenate(list(map(fx, yctr)))
                # create probe
                prb = prif.Probe(ndim=2)
                prb.set_contacts(tet_coors, shapes=PP.el_shape, shape_params=PP.shape_kw)
                
            # set shank IDs, contact IDs
            prb.set_shank_ids(np.ones(nch_shank, dtype='int') + i)
            prb.set_contact_ids(np.arange(nch_shank) + istart)
            prb.set_device_channel_indices(np.array(PP.dev_idx[istart : istart+nch_shank]))
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
                               'config' : PP.config})
        self.generate_signal.emit()

        
    def ddict_from_gui(self):
        probe_name = self.name_w.qw.text().strip()
        
        # get electrode geometry/spacing/channel number
        geom_kw = self.new_geom_gbox.get_geom_kwargs()
        config, ncols = [geom_kw[k] for k in ['config','ncols']]
        
        # get channels, shanks, and configuration type
        nch = int(self.nch_sbox.value())
        nshanks = int(self.nshk_sbox.value())
        ch_per_shank = [chbox.value() for chbox in self.shkch_list[0:nshanks]]
        if nshanks == 1:
            ch_per_shank = [int(nch)]
        shank_spacing = self.shkd_sbox.value()
        ch_per_col = [self.nch_by_col(nchan, ncols) for nchan in ch_per_shank]
        
        # get device indices
        dev_idx = self.chmap_gbox.get_dev_idx()
            
        # get electrode contact dimensions
        el_shape, shape_kw = self.contact_w.get_contact_shape_kwargs()
        
        ddict = dict(probe_name = probe_name,
                     nch = nch,
                     nshanks = nshanks,
                     ch_per_shank = ch_per_shank,  # no. channels on each shank
                     ch_per_col = ch_per_col,      # for each shank, no. channels per column
                     shank_spacing = shank_spacing,
                     
                     config = config,
                     ncols = ncols,
                     dy = geom_kw['dy'],
                     dx = geom_kw['dx'],
                     site_spacing = geom_kw['site_spacing'],
                     site_w = geom_kw['site_w'],
                     site_h = geom_kw['site_h'],
                     tip_offset = geom_kw['tip_offset'],
                     
                     dev_idx = dev_idx,
                     el_shape = el_shape,
                     shape_kw = shape_kw)
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
        
    def determine_probe_config(self, probe, show_msg=True):
        kw = {}
        
        shanks = probe.get_shanks()
        nshanks = probe.get_shank_count()
        kw['nshanks'] = nshanks
        # shank spacing: get distance between median x-coor value on adjacent shanks
        if nshanks > 1:
            xctr = [np.median(shk.contact_positions[:,0]) for shk in shanks]
            kw['shank_spacing'] = xctr[1] - xctr[0]
        else: kw['shank_spacing'] = 0
        if nshanks > 16:
            if show_msg:
                msgbox = gi.MsgboxError(msg=('Probe has too many shanks!<br>'
                                             'Please load using "Paste" method.'))
                msgbox.exec()
            return
        
        # ncols: get number of unique x-values on first shank
        shank = shanks[0]
        shank_x, shank_y = shank.contact_positions.T
        ncols = len(np.unique(shank_x))
        if ncols > 16:
            if show_msg:
                msgbox = gi.MsgboxError(msg=('Probe has too many electrode columns!<br>'
                                              'Please load using "Paste" method.'))
                msgbox.exec()
            return
        kw['ncols'] = ncols
        
        ### electrode geometry (group by xcoor, use y-spacing to determine config)
        xvals = np.unique(shank_x); xdifs = np.diff(xvals)  # get unique x-values (columns)
        if len(np.unique(xdifs)) > 1:
            if show_msg:
                msgbox = gi.MsgboxError(msg='Probe has unevenly spaced electrode columns!<br>'
                                        'Please load using "Paste" method.')
                msgbox.exec()
            return
        icols = [np.where(shank_x == xc)[0] for xc in xvals]
        ydifs_per_col = [np.diff(sorted(shank_y[icol])) for icol in icols]
        mono_cols = [len(set(ydifs))==1 for ydifs in ydifs_per_col]
        #is_mono = all([len(set(ydifs))==1 for ydifs in ydifs_per_col])
        if ncols == 1:
            config = 'Linear/Edge' # one column, monotonic y-spacing
            kw['dy'] = ydifs_per_col[0][0] # distance between first two contacts
            tip_offsets = [min(shank_y)]
        elif (ncols in [2,3]) and (not all(mono_cols)):
            config = 'Tetrode' # 2-3 columns, uneven y-spacings (intra vs inter-site)
            site_h, e2e = np.unique(ydifs_per_col[1]) # site height, edge-to-edge separation 
            # bottom of site n to top of site n+1 + bottom to center of site n + top to center of site n+1
            kw['site_spacing'] = site_h + e2e
            kw['site_h'] = site_h
            kw['site_w'] = np.ptp(xvals) # distance across width of the site
            tip_offsets = [min(shank_y) + site_h/2] # lowest contact + distance to center of site
        else:
            config='Polytrode' # 2+ columns, monotonic y-spacings
            kw['dy'] = ydifs_per_col[0][0]
            kw['dx'] = xdifs[0] # dist between first two columns
            tip_offsets = [min(shank_y[icol]) for icol in icols]
        kw['config'] = config
        kw['tip_offset'] = tip_offsets
        return kw
        
    def clear_gui(self):
        """ Reset $probething widgets """
        pyfx.stealthy(self.name_w.qw, '')
        pyfx.stealthy(self.nch_sbox, 0)
        pyfx.stealthy(self.nshk_sbox, 1)
        _ = [pyfx.stealthy(box.qw, 0) for box in self.shkch_list]
        pyfx.stealthy(self.shkd_sbox, 0)
        self.new_geom_gbox.clear_widgets()
        self.contact_w.clear_widgets()
        self.chmap_gbox.clear_params()
        self.enable_ccm_btn()
        
    def update_gui_from_probe(self, probe, show_msg=True):
        """ Update $probething widgets from probe """
        kw = self.determine_probe_config(probe, show_msg=show_msg)
        if kw is None:
            raise Exception(f'ERROR: Could not determine probe {probe.name} configuration')
        
        ### update widget values
        pyfx.stealthy(self.name_w.qw, probe.name)
        pyfx.stealthy(self.nch_sbox, probe.get_contact_count())
        pyfx.stealthy(self.nshk_sbox, probe.get_shank_count())
        _ = [pyfx.stealthy(box.qw, shk.get_contact_count()) for box,shk in zip(self.shkch_list, probe.get_shanks())]
        pyfx.stealthy(self.shkd_sbox, kw['shank_spacing'])
        
        self.new_geom_gbox.set_geom_from_probe(**kw)
        self.chmap_gbox.set_params_from_probe(probe)
        self.contact_w.set_params_from_probe(probe)
        
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
    show_shank_ids   = False
    show_on_click    = False
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
                          labels=['None', 'Contact IDs', 'Device IDs', 'Shank IDs'], 
                          active=0, activecolor='black')
        _ = self.bax.collections[0].set(sizes=[125,125,125])
        _ = [lbl.set(fontsize=12) for lbl in self.radio_btns.labels]
        
        def callback2(label):
            self.show_contact_ids = bool(label=='Contact IDs')
            self.show_device_ids  = bool(label=='Device IDs')
            self.show_shank_ids   = bool(label=='Shank IDs')
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
                axt = self.ax.contact_text[idx]
                axt.remove()                   # remove from Matplotlib figure
                del self.ax.contact_text[idx]  # erase from application
                self.ax.contact_text.insert(idx, '')  # replace with empty string
            else:
                llist = []
                if self.show_contact_ids: 
                    llist.append(str(self.probe.contact_ids[idx]))
                if self.show_device_ids:
                    llist.append(str(self.probe.device_channel_indices[idx]))
                if len(llist) > 0:
                    xys = [*self.probe.contact_positions[idx], os.linesep.join(llist)]
                    axt = self.ax.text(*xys, color='black', ha='center', va='center', 
                                       fontsize=self.contact_fontsize)
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
        if self.show_shank_ids and self.probe.shank_ids is not None:
            llist.append(np.array(self.probe.shank_ids, dtype=str))
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
    widget_updated_signal = QtCore.pyqtSignal()
    generate_signal = QtCore.pyqtSignal()
    SOURCE_WIDGET = None
    
    def __init__(self, probe=None, auto_plot=True, **kwargs):
        super().__init__()
        self.gen_layout()
        self.SOURCE_WIDGET = self.probething
        self.connect_signals()
        if auto_plot:
            self.generate_signal.connect(self.draw_probe)
        
        if probe is not None:
            self.process_probe_object(probe, show_msg=False)
    
    def gen_layout(self):
        
        ###   PROBE DESIGN WIDGETS   ###
        
        self.probething = probething(show_generate_btn=False)
        self.probesimple = probesimple(show_generate_btn=False)
        self.WIDGETS = {0:self.probething, 1:self.probesimple}
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
        #self.name_w.qw.setText('Probe_0')
        name_lay.addWidget(self.name_w)
        # set probe widget name inputs to common QLineEdit, reconnect signals
        #self.probething.name_w = self.name_w
        #self.probething.name_w.qw.textChanged.connect(self.probething.enable_ccm_btn)
        #self.probesimple.name_w = self.name_w
        #self.probesimple.name_w.qw.textChanged.connect(self.probesimple.enable_ccm_btn)
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
        self.clear_btn = QtWidgets.QPushButton('Clear')
        self.accept_btn = QtWidgets.QPushButton()  # placeholder
        #self.accept_btn.setVisible(False)
        self.bbox_lay.addWidget(self.generate_btn)
        self.bbox_lay.addWidget(self.load_btn)
        self.bbox_lay.addWidget(self.plot_btn)
        self.bbox_lay.addWidget(self.save_btn)
        self.bbox_lay.addWidget(self.clear_btn)
        _ = [x.setAutoDefault(False) for x in self.bbox.children()[1:]]
        #self.bbox_lay.addWidget(self.accept_btn)
        
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.top_row, stretch=0)
        self.layout.addWidget(self.probething, stretch=2)
        self.layout.addWidget(self.probesimple, stretch=2)
        #self.layout.addWidget(self.probemap, stretch=0)
        self.layout.addWidget(self.bbox, stretch=0)
        
        self.setLayout(self.layout)
    
    
    def connect_signals(self):
        self.name_w.qw.textChanged.connect(lambda txt: self.SOURCE_WIDGET.name_w.qw.setText(txt))
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
        self.clear_btn.clicked.connect(self.clear_probe_info)
    
    def save_probe_to_file(self):
        res = gi.FileDialog.save_file(data_object=self.probe, filetype='probe', parent=self)
        if res:
            self.save_btn.setEnabled(False)
        
    def load_probe_from_file(self):
        probe, _ = gi.FileDialog.load_file(filetype='probe', parent=self)
        if probe is None: return
        self.process_probe_object(probe, show_msg=True)
        
    def process_probe_object(self, probe, show_msg=True):
        self.SOURCE_WIDGET.update_gui_from_probe(probe, show_msg=show_msg)
        pyfx.stealthy(self.name_w.qw, self.SOURCE_WIDGET.name_w.qw.text())
        self.generate_probe()
    
    def clear_probe_info(self):
        """ Clear current probe widget """
        self.SOURCE_WIDGET.clear_gui()
        pyfx.stealthy(self.name_w.qw, '')
        self.enable_generate_btn()
        
    def toggle_source(self, btn, chk):
        """ Switch between "Build" and "Paste" widgets """
        if not chk: return
        self.SOURCE_WIDGET = self.WIDGETS[int(self.toggle_bgrp.checkedId())]
        pyfx.stealthy(self.name_w.qw, self.SOURCE_WIDGET.name_w.qw.text())
        self.enable_generate_btn()
    
    def enable_generate_btn(self):
        """ Enable/disable generator button based on params of current probe widget """
        x = self.SOURCE_WIDGET.generate_btn.isEnabled()
        self.generate_btn.setEnabled(x)
        self.plot_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.accept_btn.setEnabled(False)
        self.widget_updated_signal.emit()
    
    def generate_probe(self):
        """ Generate new probe using current probe widget """
        self.SOURCE_WIDGET.construct_probe()
        self.probe = self.SOURCE_WIDGET.probe
        #pyfx.stealthy(self.name_w.qw, self.probe.name)
        self.plot_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.accept_btn.setEnabled(True)
        self.generate_signal.emit()
        
    
    def draw_probe(self):
        fig_widget = IFigProbe(self.probe)
        #fig.set_tight_layout(True)
        #dlg = gi.MatplotlibPopup(fig, toolbar_pos='left')
        dlg = gi.Popup([fig_widget])
        #dlg.setMinimumHeight(600)
        #dlg.show()
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


class ProbeObjectPopup(QtWidgets.QDialog):
    def __init__(self, probe=None, auto_plot=True, parent=None):
        super().__init__(parent)
        
        # initialize central widget
        self.probegod = probegod(probe=probe, auto_plot=auto_plot)
        self.accept_btn = self.probegod.accept_btn
        self.accept_btn.setVisible(False)
        self.accept_btn.clicked.connect(self.accept)
        self.probegod.widget_updated_signal.connect(self.adjust_the_size)
        #self.probegod.widget_updated_signal.connect(self.adjustSize)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.probegod)
        #self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.layout.addWidget(self.accept_btn)
        self.setWindowTitle('Create probe')
    
    def adjust_the_size(self):
        QtCore.QTimer.singleShot(10, lambda: self.adjustSize())
        #self.adjustSize()


class ShankIDPopup(QtWidgets.QDialog):
    def __init__(self, xpos, parent=None):
        super().__init__(parent)
        self.xpos = xpos
        self.unique_x = np.unique(self.xpos)
        self.x2shank = {}
        self.assigned_xvals = []
        self.nshanks = 1
        
        self.gen_layout()
        self.connect_signals()
        
    
    def gen_layout(self):
        self.setWindowTitle('Set shank x-coordinates')
        self.shank_grid = QtWidgets.QGridLayout()
        min_lbl = QtWidgets.QLabel('<u>Min</u>')
        max_lbl = QtWidgets.QLabel('<u>Max</u>')
        self.shank_grid.addWidget(min_lbl, 0, 1)
        self.shank_grid.addWidget(max_lbl, 0, 2)
        items = [str(x) + ' \u00B5m' for x in self.unique_x]
        self.widgets = []
        self.cboxes = []
        tmplist = [x for x in self.unique_x]
        for i in range(16):
            lbl = QtWidgets.QLabel(f'Shank {i}')
            cbox0 = QtWidgets.QComboBox()
            cbox0.setObjectName(f'{i}_0')
            cbox1 = QtWidgets.QComboBox()
            cbox1.setObjectName(f'{i}_1')
            cbox0.addItems(items)
            cbox1.addItems(items)
            minus = QtWidgets.QPushButton()
            minus.setIcon(QtGui.QIcon(':/icons/red_minus.png'))
            minus.setFocusPolicy(QtCore.Qt.NoFocus)
            minus.setEnabled(i>0)
            minus.setObjectName(str(i))
            plus = QtWidgets.QPushButton()
            plus.setIcon(QtGui.QIcon(':/icons/green_plus.png'))
            plus.setFocusPolicy(QtCore.Qt.NoFocus)
            plus.setEnabled(i<15)
            plus.setObjectName(str(i))
            self.shank_grid.addWidget(lbl, i+1, 0)
            self.shank_grid.addWidget(cbox0, i+1, 1)
            self.shank_grid.addWidget(cbox1, i+1, 2)
            self.shank_grid.addWidget(minus, i+1, 3)
            self.shank_grid.addWidget(plus, i+1, 4)
            self.widgets.append([cbox0, cbox1, lbl, minus, plus])
            self.cboxes.append([cbox0, cbox1])
        for wlist in self.widgets[1:]:
            _ = [w.hide() for w in wlist]
        
        bbox = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton('Save')
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_btn_slot)
        bbox.addWidget(self.save_btn)
            
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addLayout(self.shank_grid)
        self.layout.addLayout(bbox)
        self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
    
    def connect_signals(self):
        for (cbox0,cbox1,_,minus,plus) in self.widgets:
            cbox0.currentTextChanged.connect(self.cbox_changed)
            cbox1.currentTextChanged.connect(self.cbox_changed)
            minus.clicked.connect(self.remove_shank)
            plus.clicked.connect(self.add_shank)
    
    def add_shank(self):
        """ Store x-values for current shank, show the next shank row """
        idx = int(self.sender().objectName())
        self.widgets[idx][3].hide() # hide current - button
        self.widgets[idx][4].hide() # hide current + button
        _ = [x.setEnabled(False) for x in self.cboxes[idx]] # disable current cboxes
        _ = [x.show() for x in self.widgets[idx+1]] # show next widgets
        assigned_x = self.get_cur_x()
        self.x2shank.update({x:self.nshanks-1 for x in assigned_x}) # map x-values to shank IDs
        self.assigned_xvals += assigned_x  # add current x-values to assigned list
        self.update_cbox_items() # remove current x-values from next options
        self.nshanks += 1
        self.check_shanks()
    
    def remove_shank(self):
        """ Hide shank row, enable previous shank and forget its x-values """
        idx = int(self.sender().objectName())
        _ = [x.hide() for x in self.widgets[idx]] # hide current widgets
        self.widgets[idx-1][3].show() # show previous - button
        self.widgets[idx-1][4].show() # show previous + button
        _ = [x.setEnabled(True) for x in self.cboxes[idx-1]] # enable previous cboxes
        
        self.nshanks -= 1
        self.assigned_xvals = [x for x in self.assigned_xvals if x not in self.get_cur_x()]
        self.x2shank = {k:v for k,v in self.x2shank.items() if k in self.assigned_xvals}
        self.check_shanks()
    
    def update_cbox_items(self):
        """ Add dropdown items for unassigned x-values only """
        avail = np.setdiff1d(self.unique_x, self.assigned_xvals)
        items = [str(x) + ' \u00B5m' for x in avail]
        for cboxes in self.cboxes[self.nshanks:]:
            for cbox in cboxes:
                cbox.blockSignals(True)
                cbox.clear()
                cbox.addItems(items)
                cbox.blockSignals(False)
    
    def getval(self, cbox):
        """ Get currently selected x-value as a float """
        return float(cbox.currentText().split(' ')[0])
    
    def get_cur_x(self):
        """ Get all unique x-values that fall into the range defined by the current row """
        mmin, mmax = [self.getval(cbox) for cbox in self.cboxes[self.nshanks-1]]
        assigned_x = [x for x in self.unique_x if (mmin <= x <= mmax)]
        return assigned_x
    
    def cbox_changed(self, txt):
        """ Make sure min x-value <= max x-value for a given shank """
        cbox = self.sender()
        idx, _id = map(int, cbox.objectName().split('_'))
        other_cbox = self.widgets[idx][1-_id]
        val, other_val = [self.getval(x) for x in [cbox, other_cbox]]
        if (_id == 0 and val > other_val) or (_id == 1 and val < other_val):
            other_cbox.blockSignals(True)
            other_cbox.setCurrentIndex(cbox.currentIndex())
            other_cbox.blockSignals(False)
        self.check_shanks()
        
    def check_shanks(self):
        """ Enable/disable 'save' and 'add' buttons """
        assigned_x = self.get_cur_x()
        new_assigned = list(self.assigned_xvals + assigned_x)
        # make sure current range doesn't contain already-assigned x-values
        is_valid = len(new_assigned) == len(set(new_assigned))
        is_done = len(np.setdiff1d(self.unique_x, new_assigned)) == 0
        # check if current range + already assigned values contains all unique x-values
        self.widgets[self.nshanks-1][4].setEnabled(is_valid and not is_done)
        self.save_btn.setEnabled(is_valid and is_done)
    
    def save_btn_slot(self):
        assigned_x = self.get_cur_x()
        self.x2shank.update({x:self.nshanks-1 for x in assigned_x}) # map x-values to shank IDs
        self.assigned_xvals += assigned_x  # add current x-values to assigned list
        self.accept()
        