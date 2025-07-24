#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS classification GUI

@author: amandaschott
"""
import os
from pathlib import Path
import h5py
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
import pdb
# custom modules
import QSS
import pyfx
import ephys
import gui_items as gi
import data_processing as dp


class IFigCSD(matplotlib.figure.Figure):
    """ Interactive figure displaying channels in CSD window """
    
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
        

class IFigPCA(QtWidgets.QWidget):
    switch_alg_signal = QtCore.pyqtSignal(str)
    switch_class_signal = QtCore.pyqtSignal()
    
    """ Figure displaying principal component analysis (PCA) for DS classification """
    def __init__(self, DS_DF, INIT_CLUS_ALGO):
        super().__init__()
        init_btn = ['kmeans','dbscan'].index(INIT_CLUS_ALGO)
        self.create_subplots(init_btn=init_btn)
        self.plot_ds_pca(DS_DF)  # initialize plot
        
    def create_subplots(self, init_btn=0):
        """ Set up main PCA plot and inset button axes """
        self.fig = matplotlib.figure.Figure()
        self.ax = self.fig.add_subplot()
        # create inset axes for radio buttons
        self.bax = self.ax.inset_axes([0, 0.9, 0.2, 0.1])
        self.bax.set_facecolor('whitesmoke')
        # create radio button widgets
        self.btns = matplotlib.widgets.RadioButtons(self.bax, labels=['K-means','DBSCAN'], active=init_btn,
                                                    activecolor='black', radio_props=dict(s=100))
        self.btns.set_label_props(dict(fontsize=['x-large','x-large']))
        self.btns.on_clicked(lambda lbl: self.switch_alg_signal.emit(lbl))
        # create toggle button to switch DS1 vs DS2 designation
        self.sax = self.ax.inset_axes([0, 0.85, 0.15, 0.05])
        self.sax.axis('off')
        self.chks = matplotlib.widgets.CheckButtons(self.sax, labels=['Switch DS1 vs DS2'], actives=[False],
                                                    frame_props=dict(s=100))
        self.chks.set_check_props=(dict(facecolors=['black'], edgecolors=['black']))
        self.chks.set_label_props(dict(fontsize=['large','large']))
        self.chks.on_clicked(lambda x: self.switch_class_signal.emit())
        
    def plot_ds_pca(self, DS_DF): # fflag (val -> DS_DF)
        """ Draw scatter plot (PC1 vs PC2) and clustering results """
        
        if 'pc1' not in DS_DF.columns: return
        for item in self.ax.lines + self.ax.collections:
            item.remove()
        
        pal = {1:(.84,.61,.66), 2:(.3,.18,.36), 0:(.7,.7,.7)}
        hue_order = [x for x in [1,2,0] if x in DS_DF['type'].values]
        # plot PC1 vs PC2
        _ = sns.scatterplot(DS_DF, x='pc1', y='pc2', hue='type', hue_order=hue_order,
                            s=100, palette=pal, ax=self.ax)
        handles = self.ax.legend_.legend_handles
        labels = ['undef' if h._label=='0' else f'DS {h._label}' for h in handles]
        self.ax.legend(handles=handles, labels=labels, loc='upper right', draggable=True)
        self.ax.set(xlabel='Principal Component 1', ylabel='Principal Component 2')
        self.ax.set_title(f'PCA with {self.btns.value_selected} Clustering', 
                          fontdict=dict(fontweight='bold'))
        sns.despine(self.fig)
        self.fig.canvas.draw_idle()
        

class DSPlotBtn(QtWidgets.QPushButton):
    """ Buttons for toggling between plot windows """
    
    def __init__(self, text, bgrp=None, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        # button group integrates signals among plot buttons
        if bgrp is not None:
            bgrp.addButton(self)
        self.bgrp = bgrp
        # set stylesheet
        self.setStyleSheet(pyfx.dict2ss(QSS.BOLD_INSET_BTN))
    
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
        
        # update classification dataframe with current clustering method/class labels in PCA plot
        self.save_df_btn = QtWidgets.QPushButton('Re-save classification')
        self.save_df_btn.setEnabled(False)
        
        # action buttons
        bbox = QtWidgets.QHBoxLayout()
        self.go_btn = QtWidgets.QPushButton('Calculate')
        self.save_btn = QtWidgets.QPushButton('Save')
        self.save_btn.setEnabled(False)
        bbox.addWidget(self.go_btn)   # perform CSD calculation/clustering
        bbox.addWidget(self.save_btn) # save CSD and DS classification
        self.vlay.addLayout(bbox)
        self.vlay.addWidget(self.save_df_btn)
        self.setLayout(self.vlay)
    
    def update_gui_from_ddict(self, ddict):
        """ Initialize GUI widget values from input ddict """
        # probe settings
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
        self.dbscan_radio.setChecked(ddict['clus_algo']=='dbscan')
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
    pca_cols = ['pc1', 'pc2', 'k_type', 'db_type', 'type']
    placeholders = [np.nan, np.nan, -1, -1, -1]
    
    def __init__(self, ddir, iprb=0, ishank=0, parent=None):
        super().__init__()
        qrect = pyfx.ScreenRect(perc_width=0.8, keep_aspect=False)
        self.setGeometry(qrect)
        
        self.init_data(ddir, iprb, ishank)
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
            alg = 'kmeans' if self.DS_DF['k_type'].equals(self.DS_DF['type']) else 'dbscan'
            self.CSD_PARAMS['clus_algo'] = alg
            self.pca_widget.plot_ds_pca(self.DS_DF)
            init_btn = ['kmeans','dbscan'].index(alg)
            self.pca_widget.btns.set_active(init_btn)
            self.widget.save_df_btn.setEnabled(True)
    
    def init_data(self, ddir, iprb, ishank):
        """ Initialize all recording variables """
        self.ddir = ddir
        self.iprb = iprb
        self.ishank = ishank
        self.probe_list = ephys.read_probe_group(ddir).probes
        self.probe = self.probe_list[iprb]
        self.shank = self.probe.get_shanks()[ishank]
        # get probe geometry
        ypos = np.array(sorted(self.shank.contact_positions[:, 1]))
        self.coord_electrode = pq.Quantity(ypos, self.probe.si_units).rescale('m', dtype='float32')  # um -> m
        # get absolute and relative channels
        self.shank_channels = self.shank.get_indices()
        self.channels = np.arange(len(self.shank_channels), dtype='int')
        # each event channel index is relative to its shank
        event_channels = ephys.load_event_channels(ddir, iprb, ishank=ishank)
        rel_event_channels = [list(self.shank_channels).index(ch) for ch in event_channels]
        self.rel_theta_chan, self.rel_ripple_chan, self.rel_hil_chan = rel_event_channels
        # load LFP data
        dmode = dp.validate_processed_ddir(ddir)
        if dmode == 1: # load HDF5 files
            self.FF = h5py.File(Path(ddir, 'DATA.hdf5'), 'r+')
            self.lfp_fs = int(self.FF.attrs['lfp_fs'])
            self.lfp_time = ephys.load_h5_array(self.FF, 'lfp_time', in_memory=True)
            self.lfp_all = ephys.load_h5_lfp(self.FF, key='raw', iprb=iprb)[self.shank_channels, :]
            self.NOISE_TRAIN = ephys.load_h5_array(self.FF, 'NOISE', iprb, in_memory=True)[self.shank_channels]
        elif dmode == 2:
            self.FF = None # load NPY/NPZ files
            self.lfp_time = np.load(Path(ddir, 'lfp_time.npy'))
            self.lfp_fs = int(np.load(Path(ddir, 'lfp_fs.npy')))
            self.lfp_all = ephys.load_bp(ddir, key='raw', iprb=iprb)[self.shank_channels, :]
            self.NOISE_TRAIN = ephys.load_noise_channels(ddir, iprb=iprb)[self.shank_channels]
        self.lfp = deepcopy(self.lfp_all)         # noisy channels are replaced with np.nan
        self.lfp_interp = deepcopy(self.lfp_all)  # noisy channels are interpolated
        self.interp_data()
        
        # load DS dataframe
        self.DS_DF = ephys.load_ds_dataset(ddir, iprb, ishank=ishank)
        if 'pc1' in self.DS_DF.columns and self.DS_DF['pc1'].isnull().all():
            self.DS_DF.drop(columns=self.pca_cols, inplace=True)
        self.iev = np.atleast_1d(self.DS_DF.idx.values)
        
        # load saved CSDs and classifications
        CSDpath = str(Path(ddir, 'CSDs.hdf5'))
        self.raw_csd,self.filt_csd,self.norm_filt_csd,self.csd_chs = [None,None,None,None]
        self.CSD_PARAMS = dict(ephys.load_recording_params(ddir))
        if os.path.exists(CSDpath):  # CSD file exists for this recording
            K = f'/{iprb}/{ishank}'
            with h5py.File(CSDpath, 'r+') as gg:
                if f'{K}/DS_DF' in gg:  # CSD exists for this probe and shank
                    ddf = pd.read_hdf(gg.filename, key=f'{K}/DS_DF')
                    if len(ddf)==len(self.DS_DF) and all(ddf.idx==self.DS_DF.idx):
                        self.DS_DF = ddf
                        self.raw_csd = gg[f'{K}/raw_csd'][:]
                        self.filt_csd = gg[f'{K}/filt_csd'][:]
                        self.norm_filt_csd = gg[f'{K}/norm_filt_csd'][:]
                        self.csd_chs = gg[f'{K}/csd_chs'][:]
                        self.CSD_PARAMS = dict(gg[K].attrs)
                        self.CSD_PARAMS['vaknin_el'] = bool(self.CSD_PARAMS['vaknin_el'])
                    else:
                        print('Saved dataframe does not match the current DS dataset.')
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
        
    def interp_data(self):
        """ Replace noisy channels with interpolated values """
        noise_idx = np.nonzero(self.NOISE_TRAIN)[0]
        clean_idx = np.setdiff1d(np.arange(len(self.channels)), noise_idx)
        if len(noise_idx) > 0: print('Interpolating noisy channels...')
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
        
    def gen_layout(self):
        """ Set up layout """
        title = f'{os.path.basename(self.ddir)} (probe={self.iprb}, shank={self.ishank})'
        self.setWindowTitle(title)
        self.layout = QtWidgets.QHBoxLayout(self)
        
        # container for plot bar (top widget) and all shown/hidden plots (bottom layout)
        self.plot_panel = QtWidgets.QWidget()
        plot_panel_lay = QtWidgets.QVBoxLayout(self.plot_panel)
        
        self.fig_container = QtWidgets.QSplitter()
        self.fig_container.setChildrenCollapsible(False)
        
        # FIGURE 0: Interactive CSD window
        self.fig0 = IFigCSD(init_min=self.rel_theta_chan, init_max=len(self.channels)-1,
                            nch=len(self.channels))
        self.canvas0 = FigureCanvas(self.fig0)
        self.canvas0.setMinimumWidth(100)
            
        # FIGURE 1: Heatmaps of raw LFP, raw CSD, and filtered CSD during DS events
        self.fig1, self.csd_axs = plt.subplots(nrows=4, ncols=2, sharey=True, width_ratios=[4,2])
        self.canvas1 = FigureCanvas(self.fig1)
        self.canvas1.setMinimumWidth(100)
        self.canvas1.hide()
        
        # scatterplot of PC1 vs PC2
        self.pca_widget = IFigPCA(self.DS_DF, self.CSD_PARAMS['clus_algo'])
        self.pca_widget.switch_alg_signal.connect(self.switch_alg_slot)
        self.pca_widget.switch_class_signal.connect(self.switch_class_slot)
        self.fig27 = self.pca_widget.fig
        self.fig27.set_tight_layout(True)
        self.canvas27 = FigureCanvas(self.fig27)
        self.canvas27.setMinimumWidth(100)
        self.canvas27.hide()
        
        # mean type 1 and 2 DS waveforms
        self.fig3, self.type_axs = plt.subplots(nrows=2, ncols=2, sharey='row',
                                                constrained_layout=True)
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
        self.widget.update_gui_from_ddict(self.CSD_PARAMS)
        
        # navigation toolbar
        # self.toolbar = NavigationToolbar(self.canvas0, self)
        # self.toolbar.setOrientation(QtCore.Qt.Vertical)
        # self.toolbar.setMaximumWidth(30)
        
        #self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.plot_panel)
        self.layout.addWidget(self.widget)
        
        # connect signals
        self.fig0.slider.on_changed(self.widget.update_ch_win)
        self.widget.go_btn.clicked.connect(self.calculate_csd)
        self.widget.save_btn.clicked.connect(self.save_csd)
        self.widget.save_df_btn.clicked.connect(self.save_classification)
        
    def get_csd(self, channels, idx, ddict):
        """ Calculate and filter the CSDs for DS events """
        csd_lfp = self.lfp_interp[channels, :][:, idx]
        csd_obj = ephys.get_csd_obj(csd_lfp, self.coord_electrode[channels], ddict)
        csds = ephys.csd_obj2arrs(csd_obj)
        return (csd_lfp, *csds)
        
    def get_csd_surround(self, channels, idx, ddict, twin):
        """ Calculate the mean CSD surrounding DSs """
        iwin = int(round(twin*self.lfp_fs))
        mean_lfp = np.array([ephys.getavg(self.lfp_interp[i], idx, iwin) for i in channels])
        csd_obj = ephys.get_csd_obj(mean_lfp, self.coord_electrode[channels], ddict)
        mean_csds = ephys.csd_obj2arrs(csd_obj)
        return (mean_lfp, *mean_csds)
    
    def calculate_csd(self, btn=None, twin=0.05, twin2=0.1):
        """ Current source density (CSD) analysis """
        self.widget.save_df_btn.setEnabled(False)
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
        self.CSD_PARAMS.update(**ddict)
        self.widget.save_btn.setEnabled(True)
        
        # plot new CSDs, hide window
        self.plot_ds_csds(twin=twin)
        self.plot_bar.fig0_btn.setChecked(False)
        self.plot_bar.fig1_btn.setEnabled(True)
        self.plot_bar.fig1_btn.setChecked(False)
        
        # plot PCA scatterplot
        self.plot_bar.fig27_btn.setEnabled(True)
        self.plot_bar.fig27_btn.setChecked(False)
        alg = str(self.CSD_PARAMS['clus_algo']) # fflag
        init_btn = ['kmeans','dbscan'].index(alg)
        self.pca_widget.blockSignals(True)
        self.pca_widget.btns.set_active(init_btn) # initialize clustering algorithm
        if self.pca_widget.chks.get_status()[0]==True: # reset "switch" to False
            self.pca_widget.chks.set_active(0)
        self.pca_widget.blockSignals(False)
        self.pca_widget.plot_ds_pca(self.DS_DF)
        
        # plot mean waveforms and CSDs
        self.plot_ds_by_type(twin=twin)
        self.plot_bar.fig3_btn.setEnabled(True)
        self.plot_bar.fig3_btn.setChecked(True)
    
    def plot_csd_window(self, twin=0.2):
        """ Interactive selection of CSD channels """
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
        self.fig0.ax.lines[self.rel_hil_chan].set(color='red', lw=3)
        self.fig0.ax.lines[self.rel_ripple_chan].set(color='green', lw=3)
        self.fig0.ax.lines[self.rel_theta_chan].set(color='blue', lw=3)
        self.fig0.set_tight_layout(True)
        sns.despine(self.fig0)
        self.canvas0.draw_idle()
    
    def plot_ds_csds(self, twin):
        """ Plot heatmaps for LFP and the raw, filtered, and normalized CSDs """
        _ = [ax.clear() for ax in self.csd_axs.flatten()]
        xax = np.arange(len(self.DS_DF))
        xax2 = np.linspace(-twin*self.lfp_fs, twin*self.lfp_fs, self.mean_lfp.shape[1])
        
        def rowplot(i, d, dsurround, title=''):
            ax, ax_mean = self.csd_axs[i]
            try:  # plot CSD heatmaps
                ax.pcolorfast(xax, self.csd_chs, d, cmap=self.cmap)
                ax_mean.pcolorfast(xax2, self.csd_chs, dsurround, cmap=self.cmap2)
            except:
                ax.pcolorfast(pyfx.Edges(xax), pyfx.Edges(self.csd_chs), d, cmap=self.cmap)
                ax_mean.pcolorfast(pyfx.Edges(xax2), pyfx.Edges(self.csd_chs), dsurround, cmap=self.cmap2)
            for irow,y in zip(self.csd_chs, self.mean_lfp): # plot waveforms
                _ = ax_mean.plot(xax2, -y+irow, color='black', lw=2)[0]
            ax.set(ylabel='Channels')
            if i==0:
                ax.invert_yaxis()
                ax_mean.set_title('Mean activity', fontdict=dict(fontweight='bold'))
            ax.set_title(title, fontdict=dict(fontweight='bold'))
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
        """ Plot mean waveform and CSD heatmap for DS1 vs DS2 """
        _ = [ax.clear() for ax in self.type_axs.flatten()]
        iwin = int(round(twin*self.lfp_fs))
        xax = np.linspace(-twin, twin, self.mean_csds_1[0].shape[1])
        ds1_arr = np.array(ephys.getwaves(self.lfp[self.rel_hil_chan], self.idx_ds1, iwin))
        ds2_arr = np.array(ephys.getwaves(self.lfp[self.rel_hil_chan], self.idx_ds2, iwin))
        
        def rowplot(i, arr, csd, csd_lfp):
            ax1_w, ax1_c = self.type_axs[:,i] # waveform and CSD axes
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
            if i == 0:
                ax1_w.set_ylabel('Amplitude (mV)')
                ax1_c.set_ylabel('Channels')
            ax1_c.set_xlabel('Time (s)')
                
        rowplot(0, ds1_arr, self.mean_csds_1[2], self.mean_csds_1[0])
        rowplot(1, ds2_arr, self.mean_csds_2[2], self.mean_csds_2[0])
        self.type_axs[1][0].invert_yaxis()
        self.type_axs[0][0].set_title(f'DS Type 1\nN={len(ds1_arr)}', fontdict=dict(fontweight='bold'))
        self.type_axs[0][1].set_title(f'DS Type 2\nN={len(ds2_arr)}', fontdict=dict(fontweight='bold'))
        
        sns.despine(self.fig3)
        #self.fig3.set_tight_layout(True)
        self.canvas3.draw_idle()
        
    def run_pca(self, ddict):
        """ PCA-based clustering analysis """
        # principal components analysis
        pca = PCA(n_components=2)
        pca_fit = pca.fit_transform(self.norm_filt_csd.T) # PCA
        
        def set_ds_type(types):
            """ Label DS1 vs DS2 by sink position """
            rows1 = np.where(types == 1)[0] # initially assigned class
            rows2 = np.where(types == 2)[0]
            if len(rows1)==0 or len(rows2)==0: return types
            # get sink positions from CSD (higher index == lower sink)
            csd1 = self.get_csd(self.csd_chs, self.DS_DF.idx.values[rows1], ddict)[2]
            csd2 = self.get_csd(self.csd_chs, self.DS_DF.idx.values[rows2], ddict)[2]
            imin1 = np.argmin(np.nanmean(csd1, axis=1))
            imin2 = np.argmin(np.nanmean(csd2, axis=1))
            if imin1 > imin2: # if "DS1" sink is lower than "DS2", swap labels
                types[rows1] = 2
                types[rows2] = 1
            return types
        
        # unsupervised clustering via K-means and DBSCAN algorithms
        self.kmeans = KMeans(n_clusters=int(ddict['nclusters']), n_init='auto').fit(pca_fit)
        self.dbscan = DBSCAN(eps=ddict['eps'], min_samples=int(ddict['min_clus_samples'])).fit(pca_fit)
        kmeans_types_init = np.array([{0:2, 1:1}.get(x, 0) for x in self.kmeans.labels_])
        kmeans_types = set_ds_type(kmeans_types_init)  # 0->1, 1->2
        db_types_init = np.array([{0:1, 1:2}.get(x, 0) for x in self.dbscan.labels_])
        db_types = set_ds_type(db_types_init) # 0->1, 1->2, other->0
        
        # update PCA and classifications in dataframe
        self.DS_DF.loc[:, ['pc1', 'pc2']] = pca_fit
        dstypes = np.array(kmeans_types) if ddict['clus_algo']=='kmeans' else np.array(db_types)
        self.DS_DF.loc[:, ['k_type', 'db_type', 'type']] = np.array([kmeans_types, db_types, dstypes]).T
    
    @QtCore.pyqtSlot(str)
    def switch_alg_slot(self, lbl):
        """ Switch classification algorithm """
        alg = lbl.lower().replace('-','')
        col = dict(kmeans='k_type', dbscan='db_type')[alg]
        self.DS_DF['type'] = np.array(self.DS_DF[col])
        self.CSD_PARAMS['clus_algo'] = alg # fflag
        self.irows_ds1 = np.where(self.DS_DF.type == 1)[0]
        self.irows_ds2 = np.where(self.DS_DF.type == 2)[0]
        self.idx_ds1 = self.DS_DF.idx.values[self.irows_ds1]
        self.idx_ds2 = self.DS_DF.idx.values[self.irows_ds2]
        self.mean_csds_1 = self.get_csd_surround(self.csd_chs, self.idx_ds1, self.CSD_PARAMS, twin=0.05)
        self.mean_csds_2 = self.get_csd_surround(self.csd_chs, self.idx_ds2, self.CSD_PARAMS, twin=0.05)
        self.pca_widget.plot_ds_pca(self.DS_DF)
        self.plot_ds_by_type(0.05)
    
    @QtCore.pyqtSlot()
    def switch_class_slot(self):
        """ Swap DS1 and DS2 labels """
        for col in ['k_type','db_type','type']:
            self.DS_DF[col].replace({1:2, 2:1}, inplace=True)
        self.irows_ds1, self.irows_ds2 = np.array(self.irows_ds2), np.array(self.irows_ds1)
        self.idx_ds1, self.idx_ds2 = np.array(self.idx_ds2), np.array(self.idx_ds1)
        self.mean_csds_1, self.mean_csds_2 = tuple(self.mean_csds_2), tuple(self.mean_csds_1)
        # update scatterplot and mean waveforms/CSDs
        self.pca_widget.plot_ds_pca(self.DS_DF)
        self.plot_ds_by_type(0.05)
        
    def save_csd(self):
        """ Write CSDs and classifications to HDF5 file  """
        # load CSD file if it exists, create if not
        with h5py.File(Path(self.ddir, 'CSDs.hdf5'), 'a') as gg:
            K = f'/{self.iprb}/{self.ishank}'
            if K in gg: del gg[K]  # delete shank dictionary if it exists
            csd_dict = dict(raw_csd=self.raw_csd, filt_csd=self.filt_csd,
                            norm_filt_csd=self.norm_filt_csd, csd_chs=self.csd_chs)
            for k,d in csd_dict.items():  # save CSD datasets
                gg[f'{K}/{k}'] = d
            # save PCA values and classification with the dataset
            self.DS_DF.to_hdf(gg.filename, key=f'{K}/DS_DF')
            gg[K].attrs.update(self.CSD_PARAMS)  # save params
        self.DS_DF.to_csv(Path(self.ddir, f'DS_DF_probe{self.iprb}-shank{self.ishank}'))
        # pop-up messagebox appears when save is complete
        res = gi.MsgboxSave('CSD data saved!\nExit window?').exec()
        if res == QtWidgets.QMessageBox.Yes:
            if self.FF is not None:
                self.FF.close()
            self.accept()
        self.widget.save_df_btn.setEnabled(True)
    
    def save_classification(self):
        """ Update classification to the currently displayed settings  """
        K = f'/{self.iprb}/{self.ishank}'
        # save current clustering algorithm in HDF5 file
        with h5py.File(Path(self.ddir, 'CSDs.hdf5'), 'r+') as gg:
            self.DS_DF.to_hdf(gg.filename, key=f'{K}/DS_DF')
            gg[K].attrs['clus_algo'] = str(self.CSD_PARAMS['clus_algo'])
        self.DS_DF.to_csv(Path(self.ddir, f'DS_DF_probe{self.iprb}-shank{self.ishank}'))
        msgbox = gi.MsgboxSave('Current DS classification saved to file!')
        msgbox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgbox.exec()
    
    def cleanup(self):
        """ Close plots and HDF5 datasets """
        plt.close()
        if self.FF is not None:
            self.FF.close()
        
    def accept(self):
        self.cleanup()
        super().accept()
    
    def reject(self):
        self.cleanup()
        super().reject()
        
    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)
        
        
def main(ddir='', iprobe=0, ishank=0):
    """ Run DS classification GUI """
    # allow user to select processed data folder, probe, and shank
    if not dp.validate_classification_ddir(ddir, iprobe, ishank):
        ddir = ephys.select_directory(init_ddir=ephys.base_dirs()[0], 
                                      title='Select recording folder')
        if not ddir or not dp.validate_processed_ddir(ddir):
            return None, (None,None,None)
        # get all valid shanks for the selected directory
        probe_group = ephys.read_probe_group(ddir)
        llist = []
        for iprb,prb in enumerate(probe_group.probes):
            for ishk,shk in enumerate(prb.get_shanks()):
                if dp.validate_classification_ddir(ddir, iprb, ishk):
                    llist.append([iprb, ishk])
        if len(llist) == 0:  # if directory contains no valid shanks, raise error message
            QtWidgets.QMessageBox.critical(None, 'Error', 'No qualifying shanks in directory.')
            return None, (None,None,None)
        if len(llist) > 1:   # if directory contains 2+ valid shanks, prompt user to select desired shank
            radio_btns = [QtWidgets.QRadioButton(f'probe {a}, shank {b}') for a,b in llist]
            radio_btns[0].setChecked(True)
            continue_btn = QtWidgets.QPushButton('Continue')
            dlg = pyfx.get_widget_container('v', *[*radio_btns, continue_btn], widget='dialog')
            continue_btn.clicked.connect(dlg.accept)
            res = dlg.exec()
            if not res: return None, (None,None,None)
            # update $iprobe and $ishank from selected radio button
            probe_txt, shank_txt = [b for b in radio_btns if b.isChecked()][0].text().split(', ')
            iprobe = int(probe_txt.split(' ')[1])
            ishank = int(shank_txt.split(' ')[1])
        else:  # if directory contains exactly one valid shank, automatically select it
            iprobe, ishank = llist[0]
    print(f'iprobe={iprobe}, ishank={ishank}')
    # launch window
    w = DS_CSDWindow(ddir, iprobe, ishank)
    w.show()
    w.raise_()
    w.exec()
    return w, (ddir, iprobe, ishank)

if __name__ == '__main__':
    app = pyfx.qapp()
    w, (ddir,iprobe,ishank) = main()
