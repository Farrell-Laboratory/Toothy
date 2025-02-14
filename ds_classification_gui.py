#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:09:56 2024

@author: amandaschott
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import quantities as pq
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from PyQt5 import QtWidgets, QtCore
import probeinterface as prif
import pdb
# custom modules
import icsd
import pyfx
import ephys
import gui_items as gi

gbox_ss_main = ('QGroupBox {'
                'background-color : rgba(220,220,220,100);'  # gainsboro
                'border : 2px solid darkgray;'
                'border-top : 5px double black;'
                'border-radius : 6px;'
                'border-top-left-radius : 1px;'
                'border-top-right-radius : 1px;'
                'font-size : 16pt;'
                'font-weight : bold;'
                'margin-top : 10px;'
                'padding : 2px;'
                'padding-bottom : 10px;'
                '}'
               
                'QGroupBox::title {'
                'background-color : palette(button);'
                #'border-radius : 4px;'
                'subcontrol-origin : margin;'
                'subcontrol-position : top center;'
                'padding : 1px 4px;' # top, right, bottom, left
                '}')


mode_btn_ss = ('QPushButton {'
               'background-color : whitesmoke;'
               'border : 3px outset gray;'
               'border-radius : 2px;'
               'color : black;'
               'padding : 4px;'
               'font-weight : bold;'
               '}'
               
               'QPushButton:pressed {'
               'background-color : gray;'
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
    

class IFigCSD(matplotlib.figure.Figure):
    """ Interactive figure displaying channels in CSD window """
    
    # def __init__(self, LFP_arr, lfp_time, lfp_fs, DS_DF, event_channels, PARAMS, **kwargs):
    def __init__(self, init_min, init_max, nch, twin=0.2):
        super().__init__()
        
        self.axs = self.subplot_mosaic([['main','sax']], width_ratios=[20,1])#, gridspec_kw=dict(wspace=0.01))
        self.ax = self.axs['main']
        
        # create visual patch for CSD window
        self.patch = matplotlib.patches.Rectangle((-twin, init_min-0.5), twin*2, init_max-init_min+1, 
                                                  color='cyan', alpha=0.3)
        self.ax.add_patch(self.patch)
        # create slider
        self.slider = matplotlib.widgets.RangeSlider(self.axs['sax'], 'CSD', valmin=0, 
                                                     valmax=nch-1, valstep=1,
                                                     valinit=[init_min, init_max], 
                                                     orientation='vertical')
        self.slider.valtext.set_visible(False)
        self.axs['sax'].invert_yaxis()
        self.slider.on_changed(self.update_csd_window)
        
        self.ax.set(xlabel='Time (s)', ylabel='channels')
        self.ax.margins(0.02)

        
    def update_csd_window(self, bounds):
        """ Adjust patch size/position to match user inputs """
        y0,y1 = bounds
        self.patch.set_y(y0-0.5)
        self.patch.set_height(y1-y0+1)
        self.canvas.draw_idle()
        
        
class IFigPCA(matplotlib.figure.Figure):
    """ Figure displaying principal component analysis (PCA) for DS classification """
    def __init__(self, DS_DF, PARAMS):
        super().__init__()
        self.DS_DF = DS_DF
        self.PARAMS = PARAMS
        
        init_btn = ['kmeans','dbscan'].index(self.PARAMS['clus_algo'])
        self.create_subplots(init_btn=init_btn)
        self.plot_ds_pca(self.PARAMS['clus_algo'])  # initialize plot
        
        # set radio button
        ##self.btns.set_active(ibtn)
        
    
    
    def create_subplots(self, init_btn=0):
        """ Set up main PCA plot and inset button axes """
        self.ax = self.add_subplot()
        # create inset axes for radio buttons
        self.bax = self.ax.inset_axes([0, 0.9, 0.2, 0.1])
        self.bax.set_facecolor('whitesmoke')
        # create radio button widgets
        ibtn = ['kmeans', 'dbscan'].index(self.PARAMS['clus_algo'])
        self.btns = matplotlib.widgets.RadioButtons(self.bax, labels=['K-means','DBSCAN'], active=ibtn,
                                                    activecolor='black', radio_props=dict(s=100))
        self.btns.set_label_props(dict(fontsize=['x-large','x-large']))
        self.btns.on_clicked(self.plot_ds_pca)
        
        
    def plot_ds_pca(self, val):
        """ Draw scatter plot (PC1 vs PC2) and clustering results """
        
        if 'pc1' not in self.DS_DF.columns:
            return
        for item in self.ax.lines + self.ax.collections:
            item.remove()
        alg = val.lower().replace('-','')
        pal = {1:(.84,.61,.66), 2:(.3,.18,.36), 0:(.7,.7,.7)}
        (ibtn,hue_col,name) = (0,'k_type','K-means') if alg=='kmeans' else (1,'db_type','DBSCAN') if alg=='dbscan' else (None,None,None)
        hue_order = [x for x in [1,2,0] if x in self.DS_DF[hue_col].values]
        # plot PC1 vs PC2
        _ = sns.scatterplot(self.DS_DF, x='pc1', y='pc2', hue=hue_col, hue_order=hue_order,
                            s=100, palette=pal, ax=self.ax)
        handles = self.ax.legend_.legend_handles
        labels = ['undef' if h._label=='0' else f'DS {h._label}' for h in handles]
        self.ax.legend(handles=handles, labels=labels, loc='upper right', draggable=True)
        self.ax.set(xlabel='Principal Component 1', ylabel='Principal Component 2')
        self.ax.set_title(f'PCA with {name} Clustering', fontdict=dict(fontweight='bold'))
        
        sns.despine(self)
        self.canvas.draw_idle()
        
       

class DSPlotBtn(QtWidgets.QPushButton):
    """ Checkable pushbutton with "Ctrl" modifier for multiple selection """
    
    def __init__(self, text, bgrp=None, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        # button group integrates signals among plot buttons
        if bgrp is not None:
            bgrp.addButton(self)
        self.bgrp = bgrp
        self.setStyleSheet(mode_btn_ss)
    
    def mouseReleaseEvent(self, event):
        """ Button click finished """
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers != QtCore.Qt.ControlModifier:  # held down Ctrl
            if self.bgrp is not None:
                # if other checked buttons in plot bar: uncheck them
                shown_btns = [btn for btn in self.bgrp.buttons() if btn.isChecked() and btn != self]
                _ = [btn.setChecked(False) for btn in shown_btns]
                # click one of several checked buttons -> only the clicked button remains on
                if len(shown_btns) > 0 and self.isChecked():
                    return
        super().mouseReleaseEvent(event)
    

class DSPlotBar(QtWidgets.QFrame):
    """ Toolbar with plot buttons """
    def __init__(self, parent=None):
        super().__init__(parent)
        # bar with show/hide widgets for each plot
        self.layout = QtWidgets.QHBoxLayout(self)
        self.bgrp = QtWidgets.QButtonGroup()
        self.bgrp.setExclusive(False)
        
        # FIGURE 0: mean DS LFPs; adjust channels in CSD window
        self.fig0_btn = DSPlotBtn('CSD Window', self.bgrp)
        self.fig0_btn.setChecked(True)
        # FIGURE 1: plot DS CSD heatmaps for raw LFP, raw CSD, and filtered CSD
        self.fig1_btn = DSPlotBtn('CSD Heatmaps', self.bgrp)
        # FIGURE 2: scatterplot of principal components and clustering results
        self.fig27_btn = DSPlotBtn('PCA Clustering', self.bgrp)
        # FIGURE 3: mean Type 1 and 2 waveforms
        self.fig3_btn = DSPlotBtn('Mean waveforms', self.bgrp)
        
        self.layout.addWidget(self.fig0_btn)
        self.layout.addWidget(self.fig1_btn)
        self.layout.addWidget(self.fig3_btn)
        self.layout.addWidget(self.fig27_btn)
        
        
class DS_CSDWidget(QtWidgets.QFrame):
    """ Settings widget for main DS analysis GUI """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setLineWidth(3)
        self.setMidLineWidth(2)
        
        # channel selection widgets
        self.vlay = QtWidgets.QVBoxLayout()
        self.vlay.setSpacing(20)
        
        # probe params
        self.gbox0 = QtWidgets.QGroupBox('Probe Settings')
        gbox0_grid = QtWidgets.QGridLayout(self.gbox0)
        # assumed source diameter
        diam_lbl = QtWidgets.QLabel('Source\ndiameter:')
        diam_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.diam_sbox = QtWidgets.QDoubleSpinBox()
        self.diam_sbox.setDecimals(3)
        self.diam_sbox.setSingleStep(0.01)
        self.diam_sbox.setSuffix(' mm')
        # assumed source cylinder thickness
        h_lbl = QtWidgets.QLabel('Source\nthickness:')
        h_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.h_sbox = QtWidgets.QDoubleSpinBox()
        self.h_sbox.setDecimals(3)
        self.h_sbox.setSingleStep(0.01)
        self.h_sbox.setSuffix(' mm')
        # tissue conductivity
        cond_lbl = QtWidgets.QLabel('Tissue\nconductivity:')
        cond_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.cond_sbox = QtWidgets.QDoubleSpinBox()
        self.cond_sbox.setDecimals(3)
        self.cond_sbox.setSingleStep(0.01)
        self.cond_sbox.setSuffix(' S/m')
        gbox0_grid.addWidget(diam_lbl, 0, 0)
        gbox0_grid.addWidget(self.diam_sbox, 0, 1)
        gbox0_grid.addWidget(h_lbl, 1, 0)
        gbox0_grid.addWidget(self.h_sbox, 1, 1)
        gbox0_grid.addWidget(cond_lbl, 2, 0)
        gbox0_grid.addWidget(self.cond_sbox, 2, 1)
        self.vlay.addWidget(self.gbox0)
        
        # CSD mode
        self.gbox2 = QtWidgets.QGroupBox('CSD Mode')
        gbox2_grid = QtWidgets.QGridLayout(self.gbox2)
        csdmode_lbl = QtWidgets.QLabel('Method:')
        # calculation mode
        self.csd_mode = QtWidgets.QComboBox()
        modes = ['standard', 'delta', 'step', 'spline']
        self.csd_mode.addItems([m.capitalize() for m in modes])
        self.csd_mode.currentTextChanged.connect(self.update_filter_widgets)
        # tolerance
        tol_lbl = QtWidgets.QLabel('Tolerance:')
        self.tol_sbox = QtWidgets.QDoubleSpinBox()
        self.tol_sbox.setDecimals(7)
        self.tol_sbox.setSingleStep(0.0000001)
        # upsampling factor
        nstep_lbl = QtWidgets.QLabel('Upsample:')
        self.nstep_sbox = QtWidgets.QSpinBox()
        self.nstep_sbox.setMaximum(2500)
        # use Vaknin electrode?
        self.vaknin_chk = QtWidgets.QCheckBox('Use Vaknin electrode')
        gbox2_grid.addWidget(csdmode_lbl, 0, 0)
        gbox2_grid.addWidget(self.csd_mode, 0, 1)
        gbox2_grid.addWidget(tol_lbl, 1, 0)
        gbox2_grid.addWidget(self.tol_sbox, 1, 1)
        gbox2_grid.addWidget(nstep_lbl, 2, 0)
        gbox2_grid.addWidget(self.nstep_sbox, 2, 1)
        gbox2_grid.addWidget(self.vaknin_chk, 3, 0, 1, 2)
        intraline = pyfx.DividerLine()
        gbox2_grid.addWidget(intraline, 4, 0, 1, 2)
        
        # CSD filter type
        csd_filter_lbl = QtWidgets.QLabel('CSD Filter:')
        csd_filter_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.csd_filter = QtWidgets.QComboBox()
        filters = ['gaussian','identity','boxcar','hamming','triangular']
        self.csd_filter.addItems([f.capitalize() for f in filters])
        self.csd_filter.currentTextChanged.connect(self.update_filter_widgets)
        fhbox1 = QtWidgets.QHBoxLayout()
        # filter order
        csd_filter_order_lbl = QtWidgets.QLabel('M:')
        self.csd_filter_order = QtWidgets.QSpinBox()
        self.csd_filter_order.setMinimum(1)
        fhbox1.addStretch()
        fhbox1.addWidget(csd_filter_order_lbl)
        fhbox1.addWidget(self.csd_filter_order)
        fhbox1.addStretch()
        fhbox2 = QtWidgets.QHBoxLayout()
        csd_filter_sigma_lbl = QtWidgets.QLabel('\u03C3:') # unicode sigma (σ)
        # filter sigma (st. deviation)
        self.csd_filter_sigma = QtWidgets.QDoubleSpinBox()
        self.csd_filter_sigma.setDecimals(1)
        self.csd_filter_sigma.setSingleStep(0.1)
        fhbox2.addStretch()
        fhbox2.addWidget(csd_filter_sigma_lbl)
        fhbox2.addWidget(self.csd_filter_sigma)
        fhbox2.addStretch()
        gbox2_grid.addWidget(csd_filter_lbl, 5, 0)
        gbox2_grid.addWidget(self.csd_filter, 5, 1)
        gbox2_grid.addLayout(fhbox1, 6, 0)
        gbox2_grid.addLayout(fhbox2, 6, 1)
        self.vlay.addWidget(self.gbox2)
        
        # clustering algorithm
        self.gbox4 = QtWidgets.QGroupBox('Clustering Algorithm')
        gbox4_grid = QtWidgets.QGridLayout(self.gbox4)
        # use K-means or DBSCAN?
        self.kmeans_radio = QtWidgets.QRadioButton('K-means')
        self.kmeans_radio.setChecked(True)
        self.dbscan_radio = QtWidgets.QRadioButton('DBSCAN')
        self.kmeans_radio.toggled.connect(self.update_cluster_widgets)
        # K-means: no. target clusters
        nclus_lbl = QtWidgets.QLabel('# clusters')
        self.nclus_sbox = QtWidgets.QSpinBox()
        self.nclus_sbox.setMinimum(1)
        # DBSCAN: epsilon, min samples
        eps_lbl = QtWidgets.QLabel('Epsilon (\u03B5)')
        self.eps_sbox = QtWidgets.QDoubleSpinBox()
        self.eps_sbox.setDecimals(2)
        self.eps_sbox.setSingleStep(0.1)
        minN_lbl = QtWidgets.QLabel('Min. samples')
        self.minN_sbox = QtWidgets.QSpinBox()
        self.minN_sbox.setMinimum(1)
        gbox4_grid.addWidget(self.kmeans_radio, 0, 0)
        gbox4_grid.addWidget(self.dbscan_radio, 0, 1)
        gbox4_grid.addWidget(nclus_lbl, 1, 0)
        gbox4_grid.addWidget(self.nclus_sbox, 1, 1)
        gbox4_grid.addWidget(eps_lbl, 2, 0)
        gbox4_grid.addWidget(self.eps_sbox, 2, 1)
        gbox4_grid.addWidget(minN_lbl, 3, 0)
        gbox4_grid.addWidget(self.minN_sbox, 3, 1)
        self.vlay.addWidget(self.gbox4)
        
        # action buttons
        bbox = QtWidgets.QHBoxLayout()
        self.go_btn = QtWidgets.QPushButton('Calculate')
        self.save_btn = QtWidgets.QPushButton('Save')
        self.save_btn.setEnabled(False)
        bbox.addWidget(self.go_btn)   # perform CSD calculation/clustering
        bbox.addWidget(self.save_btn) # save CSD and DS classification
        self.vlay.addLayout(bbox)
        
        self.setLayout(self.vlay)
    
    
    def update_gui_from_ddict(self, ddict):
        """ Set GUI widget values from input ddict """
        # probe settings
        #self.eldist_sbox.setValue(ddict['el_dist'])
        self.diam_sbox.setValue(ddict['src_diam'])
        self.h_sbox.setValue(ddict['src_h'])
        self.cond_sbox.setValue(ddict['cond'])
        # CSD params
        self.csd_mode.setCurrentText(ddict['csd_method'].capitalize())
        self.csd_filter.setCurrentText(ddict['f_type'].capitalize())
        self.csd_filter_order.setValue(int(ddict['f_order']))
        self.csd_filter_sigma.setValue(ddict['f_sigma'])
        self.vaknin_chk.setChecked(ddict['vaknin_el'])
        self.tol_sbox.setValue(ddict['tol'])
        self.nstep_sbox.setValue(int(ddict['spline_nsteps']))
        # clustering params
        self.kmeans_radio.setChecked(ddict['clus_algo']=='kmeans')
        self.nclus_sbox.setValue(int(ddict['nclusters']))
        self.eps_sbox.setValue(ddict['eps'])
        self.minN_sbox.setValue(int(ddict['min_clus_samples']))
        
        self.update_filter_widgets()
        self.update_cluster_widgets()
    
    
    def ddict_from_gui(self):
        """ Return GUI widget values as parameter dictionary """
        ddict = dict(csd_method       = self.csd_mode.currentText().lower(),
                     f_type           = self.csd_filter.currentText().lower(),
                     f_order          = self.csd_filter_order.value(),
                     f_sigma          = self.csd_filter_sigma.value(),
                     vaknin_el        = bool(self.vaknin_chk.isChecked()),
                     tol              = self.tol_sbox.value(),
                     spline_nsteps    = self.nstep_sbox.value(),
                     #el_dist          = self.eldist_sbox.value(),
                     src_diam         = self.diam_sbox.value(),
                     src_h            = self.h_sbox.value(),
                     cond             = self.cond_sbox.value(),
                     cond_top         = self.cond_sbox.value(),
                     clus_algo        = 'kmeans' if self.kmeans_radio.isChecked() else 'dbscan',
                     nclusters        = self.nclus_sbox.value(),
                     eps              = self.eps_sbox.value(),
                     min_clus_samples = self.minN_sbox.value())
        return ddict
        
    def update_filter_widgets(self):
        """ Enable/disable widgets based on selected filter """
        mmode = self.csd_mode.currentText().lower()
        self.tol_sbox.setEnabled(mmode in ['step','spline'])
        self.nstep_sbox.setEnabled(mmode=='spline')
        self.vaknin_chk.setEnabled(mmode=='standard')
        
        ffilt = self.csd_filter.currentText().lower()
        self.csd_filter_order.setEnabled(ffilt != 'identity')
        self.csd_filter_sigma.setEnabled(ffilt == 'gaussian')
    
    def update_cluster_widgets(self):
        """ Enable/disable widgets based on selected clustering algorithm """
        self.nclus_sbox.setEnabled(self.kmeans_radio.isChecked())
        self.eps_sbox.setEnabled(self.dbscan_radio.isChecked())
        self.minN_sbox.setEnabled(self.dbscan_radio.isChecked())
        
    def update_ch_win(self, bounds):
        """ Update CSD channel range from GUI """
        ch0, ch1 = bounds
        self.csd_chs = np.arange(ch0, ch1+1)
    
        
class DS_CSDWindow(QtWidgets.QDialog):
    """ Main DS analysis GUI """
    
    cmap = plt.get_cmap('bwr')
    cmap2 = pyfx.truncate_cmap(cmap, 0.2, 0.8)
    
    def __init__(self, ddir, iprb=0, PARAMS=None, parent=None):
        super().__init__()
        qrect = pyfx.ScreenRect(perc_width=0.8, keep_aspect=False)
        self.setGeometry(qrect)
        self.ddir = ddir
        self.iprb = iprb
        if PARAMS is None: PARAMS = ephys.load_recording_params(self.ddir)
        self.PARAMS = dict(PARAMS)
        
        # required: event channels file, probe DS_DF file
        self.lfp_list, self.lfp_time, self.lfp_fs = ephys.load_lfp(ddir, 'raw', -1)
        self.noise_list = ephys.load_noise_channels(ddir, -1)
        self.probe_group = prif.read_probeinterface(Path(ddir, 'probe_group'))
        self.load_probe_data(self.iprb)
        
        self.gen_layout()
        
        if self.csd_chs is not None:
            ddict = self.widget.ddict_from_gui()
            # compute mean CSD for time window surrounding all DS peaks
            self.mean_csds = self.get_csd_surround(self.csd_chs, self.iev, ddict, twin=0.05)
            self.mean_lfp = self.mean_csds[0]
        if self.idx_ds1 is not None:
            self.mean_csds_1 = self.get_csd_surround(self.csd_chs, self.idx_ds1, ddict, twin=0.05)
            self.mean_csds_2 = self.get_csd_surround(self.csd_chs, self.idx_ds2, ddict, twin=0.05)
        
        self.plot_csd_window()  # CSD movable window
        if self.csd_chs is not None:
            tup = pyfx.Edges(self.csd_chs)
            self.fig0.slider.set_val(tup)
            
        if self.raw_csd is not None:  # DS peak CSD heatmaps
            self.plot_ds_csds(twin=0.05)
            
        if self.idx_ds1 is not None:  # DS1 vs DS1 waveforms/CSDs
            self.plot_ds_by_type(0.05)
        
        if 'pc1' in self.DS_DF.columns:  # PCA scatterplot
            self.fig27.plot_ds_pca(self.PARAMS['clus_algo'])
            init_btn = ['kmeans','dbscan'].index(self.PARAMS['clus_algo'])
            self.fig27.btns.set_active(init_btn)
    
    
    def load_probe_data(self, iprb):
        """ Set data corresponding to the given probe """
        self.iprb = iprb
        self.lfp_all = np.array(self.lfp_list[iprb])  # original LFP signals
        self.channels = np.arange(self.lfp_all.shape[0])
        self.lfp = deepcopy(self.lfp_all)             
        self.lfp_interp = deepcopy(self.lfp_all)      # noisy channels are interpolated
        self.NOISE_TRAIN = np.array(self.noise_list[iprb])
        noise_idx = np.nonzero(self.NOISE_TRAIN)[0]
        clean_idx = np.setdiff1d(np.arange(len(self.channels)), noise_idx)
        for i in noise_idx:
            self.lfp[i,:] = np.nan  # replace noisy channels in lfp with np.nan
            # replace noisy channels in lfp_interp with average of two closest (clean) signals
            if i==0: 
                self.lfp_interp[i,:] = self.lfp_all[min(clean_idx)]
            elif i > max(clean_idx):
                self.lfp_interp[i,:] = self.lfp_all[max(clean_idx)]
            else:
                sig1 = self.lfp_all[pyfx.Closest(i, clean_idx[clean_idx < i])]
                sig2 = self.lfp_all[pyfx.Closest(i, clean_idx[clean_idx > i])]
                self.lfp_interp[i,:] = np.nanmean([sig1, sig2], axis=0)
        self.DS_DF = pd.read_csv(Path(self.ddir, f'DS_DF_{iprb}'))  # load DS dataframe
        self.iev = self.DS_DF.idx.values
        self.probe = self.probe_group.probes[iprb]
        # get electrode geometry in meters
        ypos = np.array(sorted(self.probe.contact_positions[:, 1]))
        self.coord_electrode = pq.Quantity(ypos, self.probe.si_units).rescale('m')  # um -> m
        
        self.event_channels= np.load(Path(self.ddir, f'theta_ripple_hil_chan_{iprb}.npy'))
        self.theta_chan, self.ripple_chan, self.hil_chan = self.event_channels
        
        # load/create CSDs
        csd_dict = ephys.load_ds_csd(self.ddir, iprb)
        self.raw_csd        = csd_dict.get('raw_csd')
        self.filt_csd       = csd_dict.get('filt_csd')
        self.norm_filt_csd  = csd_dict.get('norm_filt_csd')
        self.csd_chs        = csd_dict.get('csd_chs')
        if self.csd_chs is not None:
            self.csd_lfp = self.lfp_interp[self.csd_chs, :][:, self.iev]
        
        if 'type' in self.DS_DF.columns:
            # get table rows and recording indexes of DS1 vs DS2
            self.irows_ds1 = np.where(self.DS_DF.type == 1)[0]
            self.irows_ds2 = np.where(self.DS_DF.type == 2)[0]
            self.idx_ds1 = self.DS_DF.idx.values[self.irows_ds1]
            self.idx_ds2 = self.DS_DF.idx.values[self.irows_ds2]
        else:
            self.irows_ds1 = None
            self.irows_ds2 = None
            self.idx_ds1   = None
            self.idx_ds2   = None
        
    def gen_layout(self):
        """ Set up layout """
        title = f'{os.path.basename(self.ddir)} (probe={self.iprb})'
        self.setWindowTitle(title)
        self.layout = QtWidgets.QHBoxLayout(self)
        
        # container for plot bar (top widget) and all shown/hidden plots (bottom layout)
        self.plot_panel = QtWidgets.QWidget()
        plot_panel_lay = QtWidgets.QVBoxLayout(self.plot_panel)
        
        #self.fig_container = QtWidgets.QHBoxLayout()
        self.fig_container = QtWidgets.QSplitter()
        self.fig_container.setChildrenCollapsible(False)
        
        # FIGURE 0: Interactive CSD window
        self.fig0 = IFigCSD(init_min=self.theta_chan, init_max=len(self.channels)-1,
                            nch=len(self.channels))
        self.canvas0 = FigureCanvas(self.fig0)
        self.canvas0.setMinimumWidth(100)
            
        # FIGURE 1: Heatmaps of raw LFP, raw CSD, and filtered CSD during DS events
        self.fig1, self.csd_axs = plt.subplots(nrows=4, ncols=2, sharey=True, width_ratios=[4,2])
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setMinimumWidth(100)
        self.canvas1.hide()
        
        
        # scatterplot of PC1 vs PC2
        self.fig27 = IFigPCA(self.DS_DF, self.PARAMS)
        self.fig27.set_tight_layout(True)
        self.canvas27 = FigureCanvas(self.fig27)
        self.canvas27.setMinimumWidth(100)
        self.canvas27.hide()
        
        # mean type 1 and 2 DS waveforms
        self.fig3, self.type_axs = plt.subplots(nrows=2, ncols=2, sharey='row')
        self.fig3.set_tight_layout(True)
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setMinimumWidth(100)
        self.canvas3.hide()
        
        self.fig_container.addWidget(self.canvas0)
        self.fig_container.addWidget(self.canvas1)
        self.fig_container.addWidget(self.canvas3)
        self.fig_container.addWidget(self.canvas27)
        
        # bar with show/hide widgets for each plot
        self.plot_bar = DSPlotBar()
        self.plot_bar.fig1_btn.setEnabled(self.raw_csd is not None)
        self.plot_bar.fig27_btn.setEnabled('pc1' in self.DS_DF.columns)
        self.plot_bar.fig3_btn.setEnabled('type' in self.DS_DF.columns)
        self.plot_bar.fig0_btn.toggled.connect(lambda x: self.canvas0.setVisible(x))
        self.plot_bar.fig1_btn.toggled.connect(lambda x: self.canvas1.setVisible(x))
        self.plot_bar.fig27_btn.toggled.connect(lambda x: self.canvas27.setVisible(x))
        self.plot_bar.fig3_btn.toggled.connect(lambda x: self.canvas3.setVisible(x))
        
        plot_panel_lay.addWidget(self.plot_bar, stretch=0)
        plot_panel_lay.addWidget(self.fig_container, stretch=2)
        
        # create settings widget
        self.widget = DS_CSDWidget()
        self.widget.setMaximumWidth(250)
        self.widget.update_ch_win(self.fig0.slider.val)
        self.widget.update_gui_from_ddict(self.PARAMS)
        
        # navigation toolbar
        # self.toolbar = NavigationToolbar(self.canvas0, self)
        # self.toolbar.setOrientation(QtCore.Qt.Vertical)
        # self.toolbar.setMaximumWidth(30)
        
        #self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.plot_panel)
        #self.layout.addWidget(self.canvas0)
        self.layout.addWidget(self.widget)
        
        # connect signals
        self.fig0.slider.on_changed(self.widget.update_ch_win)
        self.widget.go_btn.clicked.connect(self.calculate_csd)
        self.widget.save_btn.clicked.connect(self.save_csd)
        
    def get_csd(self, channels, idx, ddict):
        csd_lfp = self.lfp_interp[channels, :][:, idx]
        csd_obj = ephys.get_csd_obj(csd_lfp, self.coord_electrode, ddict)
        csds = ephys.csd_obj2arrs(csd_obj)
        return (csd_lfp, *csds)
        
    def get_csd_surround(self, channels, idx, ddict, twin):
        iwin = int(round(twin*self.lfp_fs))
        mean_lfp = np.array([ephys.getavg(self.lfp_interp[i], idx, iwin) for i in channels])
        csd_obj = ephys.get_csd_obj(mean_lfp, self.coord_electrode, ddict)
        mean_csds = ephys.csd_obj2arrs(csd_obj)
        return (mean_lfp, *mean_csds)
    
    def calculate_csd(self, btn=None, twin=0.05, twin2=0.1):
        """ Current source density (CSD) analysis """
        self.csd_chs = np.array(self.widget.csd_chs)
        ddict = self.widget.ddict_from_gui()
        
        # compute CSD of each DS peak using iCSD functions
        self.csds = self.get_csd(self.csd_chs, self.iev, ddict) # LFP value for each DS on each channel
        self.csd_lfp, self.raw_csd, self.filt_csd, self.norm_filt_csd = self.csds
        
        # compute mean CSD for time window surrounding all DS peaks
        self.mean_csds = self.get_csd_surround(self.csd_chs, self.iev, ddict, twin=twin)
        self.mean_lfp = self.mean_csds[0]

        # run clustering algorithms
        self.run_pca(ddict)
        
        # get table rows and recording indexes of DS1 vs DS2
        self.irows_ds1 = np.where(self.DS_DF.type == 1)[0]
        self.irows_ds2 = np.where(self.DS_DF.type == 2)[0]
        self.idx_ds1 = self.DS_DF.idx.values[self.irows_ds1]
        self.idx_ds2 = self.DS_DF.idx.values[self.irows_ds2]
        
        self.mean_csds_1 = self.get_csd_surround(self.csd_chs, self.idx_ds1, ddict, twin=twin)
        self.mean_csds_2 = self.get_csd_surround(self.csd_chs, self.idx_ds2, ddict, twin=twin)
        
        # update params, allow save
        self.PARAMS.update(**ddict)
        self.widget.save_btn.setEnabled(True)
        
        # plot new CSDs, hide window
        self.plot_ds_csds(twin=twin)
        self.plot_bar.fig0_btn.setChecked(False)
        self.plot_bar.fig1_btn.setEnabled(True)
        self.plot_bar.fig1_btn.setChecked(True)
        
        # plot PCA scatterplot
        self.fig27.DS_DF = self.DS_DF
        self.plot_bar.fig27_btn.setEnabled(True)
        init_btn = ['kmeans','dbscan'].index(self.PARAMS['clus_algo'])
        self.fig27.plot_ds_pca(self.PARAMS['clus_algo'])
        self.fig27.btns.set_active(init_btn)
        
        # plot mean waveforms
        self.plot_ds_by_type(twin=twin)
        self.plot_bar.fig3_btn.setEnabled(True)
    
    
    def plot_csd_window(self, twin=0.2):
        # plot signals
        iwin = int(round(twin*self.lfp_fs))
        arr = np.array([ephys.getavg(self.lfp[i], self.iev, iwin) for i in self.channels])
        xax = np.linspace(-twin, twin, arr.shape[1])
        for irow,y in enumerate(arr):
            if self.NOISE_TRAIN[irow] == 1:  # for noisy signals, plot a flat line
                _ = self.fig0.ax.plot(xax, np.repeat(irow, len(xax)), color='lightgray', lw=2)[0]
            else:
                _ = self.fig0.ax.plot(xax, -y+irow, color='black', lw=2)[0]
        self.fig0.ax.invert_yaxis()
        self.fig0.ax.lines[self.hil_chan].set(color='red', lw=3)
        self.fig0.ax.lines[self.ripple_chan].set(color='green', lw=3)
        self.fig0.ax.lines[self.theta_chan].set(color='blue', lw=3)
        self.fig0.set_tight_layout(True)
        sns.despine(self.fig0)
        self.canvas0.draw_idle()
    
    
    def plot_ds_csds(self, twin):
        """ Plot heatmaps for LFP and the raw, filtered, and normalized CSDs """
        _ = [ax.clear() for ax in self.csd_axs.flatten()]
        xax = np.arange(len(self.DS_DF))
        xax2 = np.linspace(-twin*self.lfp_fs, twin*self.lfp_fs, self.mean_lfp.shape[1])
        
        def rowplot(i, d, dsurround, title=''):
            try:
                ax, ax_mean = self.csd_axs[i]
                try:
                    im = ax.pcolorfast(xax, self.csd_chs, d, cmap=self.cmap)
                    im_mean = ax_mean.pcolorfast(xax2, self.csd_chs, dsurround, cmap=self.cmap2)
                except:
                    im = ax.pcolorfast(pyfx.Edges(xax), pyfx.Edges(self.csd_chs), d, cmap=self.cmap)
                    im_mean = ax_mean.pcolorfast(pyfx.Edges(xax2), pyfx.Edges(self.csd_chs), dsurround, cmap=self.cmap2)
                for irow,y in zip(self.csd_chs, self.mean_lfp):
                    _ = ax_mean.plot(xax2, -y+irow, color='black', lw=2)[0]
                ax.set(ylabel='Channels')
                if i==0:
                    ax.invert_yaxis()
                    ax_mean.set_title('Mean activity', fontdict=dict(fontweight='bold'))
                ax.set_title(title, fontdict=dict(fontweight='bold'))
            except:
                pdb.set_trace()
        
        rowplot(0, self.csd_lfp, self.mean_lfp, 'Raw LFP')
        rowplot(1, self.raw_csd, self.mean_csds[1], 'Raw CSD')
        rowplot(2, self.filt_csd, self.mean_csds[2], 'Filtered CSD')
        rowplot(3, self.norm_filt_csd, self.mean_csds[3], 'Norm. Filtered CSD')
        self.csd_axs[-1][-1].set_visible(False)
        self.csd_axs[-1][0].set_xlabel('# dentate spikes')
        self.csd_axs[-2][1].set_xlabel('Time (ms)')
        
        self.fig1.set_tight_layout(True)
        sns.despine(self.fig1)
        self.canvas1.draw_idle()
        
        
    def plot_ds_by_type(self, twin):
        _ = [ax.clear() for ax in self.type_axs.flatten()]
        iwin = int(round(twin*self.lfp_fs))
        xax = np.linspace(-twin, twin, self.mean_csds_1[0].shape[1])
        
        self.ds1_arr = np.array(ephys.getwaves(self.lfp[self.hil_chan], self.idx_ds1, iwin))
        self.ds2_arr = np.array(ephys.getwaves(self.lfp[self.hil_chan], self.idx_ds2, iwin))
        
        def rowplot(i, arr, csd, csd_lfp):
            ax1_w, ax1_c = self.type_axs[:,i]
            # mean waveforms
            d = np.nanmean(arr, axis=0)
            yerr = np.nanstd(arr, axis=0)
            ax1_w.plot(xax, d, color='black', lw=2)[0]
            ax1_w.fill_between(xax, d-yerr, d+yerr, color='black', alpha=0.3, zorder=-2)
            # raw CSD
            try:
                ax1_c.pcolorfast(xax, self.csd_chs, csd, cmap=self.cmap2)
            except:
                ax1_c.pcolorfast(pyfx.Edges(xax), pyfx.Edges(self.csd_chs), csd, cmap=self.cmap2)
            for irow,y in zip(self.csd_chs, csd_lfp):
                _ = ax1_c.plot(xax, -y+irow, color='black', lw=2)[0]
                
        rowplot(0, self.ds1_arr, self.mean_csds_1[2], self.mean_csds_1[0])
        rowplot(1, self.ds2_arr, self.mean_csds_2[2], self.mean_csds_2[0])
        self.type_axs[1][0].invert_yaxis()
        self.type_axs[0][0].set_title(f'DS Type 1\nN={len(self.ds1_arr)}', fontdict=dict(fontweight='bold'))
        self.type_axs[0][1].set_title(f'DS Type 2\nN={len(self.ds2_arr)}', fontdict=dict(fontweight='bold'))
        
        self.fig3.set_tight_layout(True)
        sns.despine(self.fig3)
        self.canvas1.draw_idle()
        
        
    def run_pca(self, ddict):
        # principal components analysis
        pca = PCA(n_components=2)
        pca_fit = pca.fit_transform(self.norm_filt_csd.T) # PCA
        
        # unsupervised clustering via K-means and DBSCAN algorithms
        self.kmeans = KMeans(n_clusters=int(ddict['nclusters']), n_init='auto').fit(pca_fit)
        self.dbscan = DBSCAN(eps=ddict['eps'], min_samples=int(ddict['min_clus_samples'])).fit(pca_fit)  # 0->1, 1->2
        kmeans_types = np.array([{0:2, 1:1}.get(x, 0) for x in self.kmeans.labels_])  # 0->2, 1->1, other->0
        db_types = np.array([{0:1, 1:2}.get(x, 0) for x in self.dbscan.labels_])      # 0->1, 1->2, other->0
        
        # label DS1 vs DS2 by sink position
        k1_rows = np.where(kmeans_types == 1)[0]  # initially assigned K-means class
        k2_rows = np.where(kmeans_types == 2)[0]
        # DS2 sink is lower than DS1
        k1_csd = self.get_csd(self.csd_chs, self.DS_DF.idx.values[k1_rows], ddict)[2]
        k2_csd = self.get_csd(self.csd_chs, self.DS_DF.idx.values[k2_rows], ddict)[2]
        k1_imin = np.argmin(np.nanmean(k1_csd, axis=1))
        k2_imin = np.argmin(np.nanmean(k2_csd, axis=1))
        if k1_imin > k2_imin:
            kmeans_types[k1_rows] = 2
            kmeans_types[k2_rows] = 1
        db1_rows = np.where(db_types == 1)[0]  # initially assigned DBSCAN class
        db2_rows = np.where(db_types == 2)[0]
        db1_csd = self.get_csd(self.csd_chs, self.DS_DF.idx.values[db1_rows], ddict)[2]
        db2_csd = self.get_csd(self.csd_chs, self.DS_DF.idx.values[db2_rows], ddict)[2]
        db1_imin = np.argmin(np.nanmean(db1_csd, axis=1))
        db2_imin = np.argmin(np.nanmean(db2_csd, axis=1))
        if db1_imin > db2_imin:
            db_types[db1_rows] = 2
            db_types[db2_rows] = 1
            
        # update PCA and classifications in dataframe
        self.DS_DF.loc[:, ['pc1', 'pc2']] = pca_fit
        dstypes = np.array(kmeans_types) if ddict['clus_algo']=='kmeans' else np.array(db_types)
        self.DS_DF.loc[:, ['k_type', 'db_type', 'type']] = np.array([kmeans_types, db_types, dstypes]).T
        
        
    def save_csd(self):
        """ Write CSDs to .npz file, save classifications in DS_DF  """
        # save raw, filtered, and normalized CSDs
        csd_path = Path(self.ddir, f'ds_csd_{self.iprb}.npz')
        np.savez(csd_path, raw_csd = self.raw_csd,
                           filt_csd = self.filt_csd,
                           norm_filt_csd = self.norm_filt_csd,
                           csd_chs = self.csd_chs)
        
        # save DS dataframe with PCA values/DS types
        self.DS_DF.to_csv(Path(self.ddir, f'DS_DF_{self.iprb}'), index_label=False)
        ephys.save_recording_params(self.ddir, self.PARAMS)  # save params
        
        # pop-up messagebox appears when save is complete
        msgbox = gi.MsgboxSave('CSD data saved!\nExit window?', parent=self)
        res = msgbox.exec()
        if res == QtWidgets.QMessageBox.Yes:
            self.accept()
    
    
    def closeEvent(self, event):
        plt.close()
        event.accept()


if __name__ == '__main__':
    # ddir = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/raw_data/'
    #         'JG007_2_2024-07-09_15-40-43_openephys/Record Node 103/experiment1/recording1')
    #ddir = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/JG008')
    #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/stanford/JF513_saved'
    ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/neuralynx_saved'
    pyfx.qapp()
    w = DS_CSDWindow(ddir, 0)
    w.show()
        
    #w.show()
    w.raise_()
    w.exec()
    #sys.exit(app.exec())