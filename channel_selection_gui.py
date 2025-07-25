#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main analysis GUI

@author: amandaschott
"""
import os
from pathlib import Path
import scipy
import h5py
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import warnings
from copy import deepcopy
import bisect
import quantities as pq
from PyQt5 import QtWidgets, QtCore, QtGui
import pdb
# custom modules
import QSS
import pyfx
import ephys
import gui_items as gi
import data_processing as dp
import resources_rc


def str_fmt(ddict, key=None, key_top=True):
    """ Create structured annotation string for individual event plots """
    llist = [f'{k} = {v}' for k,v in ddict.items()]
    if key in ddict:
        i = list(ddict.keys()).index(key)
        keyval = llist.pop(i) + ' \u2605' # unicode star
        llist.insert(0 if key_top else i, keyval)
    fmt = os.linesep.join(llist)
    return fmt
       

class IFigLFP(QtWidgets.QWidget):
    """ Main analysis figure; scrollable LFPs with optional event markers """
    updated_noise_signal = QtCore.pyqtSignal(int, np.ndarray)
    
    SHOW_DS = True       # show DS event markers
    SHOW_SWR = True      # show ripple event markers
    ADD_DS = False       # enable addition of new DSs to dataset by clicking
    ADD_SWR = False      # enable addition of new ripples to dataset by clicking
    SHOW_DS_RM = False   # show markers for DSs excluded from dataset
    SHOW_SWR_RM = False  # show markers for ripples excluded from dataset
    
    edit_events_signal = QtCore.pyqtSignal(str, pd.DataFrame)
    
    def __init__(self, DATA, lfp_time, lfp_fs, PARAMS, **kwargs):
        super().__init__()
        self.DATA = DATA          # dictionary of raw and filtered LFP signals
        self.lfp_time = lfp_time  # recording times (s)
        self.lfp_fs = lfp_fs      # LFP sampling rate (Hz)
        self.PARAMS = PARAMS      # parameter dictionary
        # initialize event channels
        twin = kwargs.get('twin', 1)  # initial viewing window size (+/- X s)
        event_channels = kwargs.get('event_channels', [0,0,0])
        self.ch_cmap = pd.Series(['blue', 'green', 'red'], index=event_channels)
        #self.probe = kwargs.get('probe', None)
        self.shank = kwargs.get('shank', None)
        self.AUX = kwargs.get('AUX', np.full(len(self.lfp_time), np.nan))
        self.DS_ALL = kwargs.get('DS_ALL', None)
        self.SWR_ALL = kwargs.get('SWR_ALL', None)
        self.STD = kwargs.get('STD', None)
        self.NOISE_TRAIN = kwargs.get('NOISE_TRAIN', None)
        self.SWR_THRES = kwargs.get('SWR_THRES')
        self.init_event_items()  # create event indexes/trains/etc
        
        # create subplots and interactive widgets
        self.plot_height = pyfx.ScreenRect(perc_height=0.75).height()
        self.create_subplots(twin=twin)
        #self.fig.set_tight_layout(True)
        #self.fig_w.set_tight_layout(True)
        #self.fig_freq.set_tight_layout(True)
        # main canvas with toolbar and central LFP plot
        self.canvas = FigureCanvas(self.fig)  # main plot canvas
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setOrientation(QtCore.Qt.Vertical)
        self.toolbar.setMaximumWidth(30)
        for tbtn in self.toolbar.findChildren(QtWidgets.QToolButton):
            if tbtn.actions()[0].text() == 'Pan':
                tbtn.setObjectName('Pan')
            if tbtn.actions()[0].text() == 'Subplots':
                tbtn.setEnabled(False)
        self.canvas_freq = FigureCanvas(self.fig_freq) # freq plot canvas
        self.canvas_w = FigureCanvas(self.fig_w) # slider canvas
        self.canvas_w.setMaximumHeight(80)
        self.connect_mpl_widgets()
        # embed plots in QScrollArea for vertical zoom
        self.plot_row = QtWidgets.QWidget()
        self.plot_row.setFixedHeight(self.plot_height)
        self.plot_row_hlay = QtWidgets.QHBoxLayout(self.plot_row)
        self.plot_row_hlay.setContentsMargins(0,0,0,0)
        self.plot_row_hlay.addWidget(self.toolbar, stretch=0)
        self.plot_row_hlay.addWidget(self.canvas, stretch=6)
        self.plot_row_hlay.addWidget(self.canvas_freq, stretch=3)
        self.qscroll = QtWidgets.QScrollArea()
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setWidget(self.plot_row)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas_w)
        self.layout.addWidget(self.qscroll)
        self.canvas_freq.hide()
        
        self.channel_changed(*event_channels)
    
    def create_subplots(self, twin):
        """ Set up subplot axes for data plots and slider widgets """
        
        ### SLIDER AXES
        self.fig_w = matplotlib.figure.Figure(constrained_layout=True)
        gridspec = matplotlib.gridspec.GridSpec(2, 3, hspace=0, figure=self.fig_w)
        self.sax0 = self.fig_w.add_subplot(gridspec[0,0])
        self.sax1 = self.fig_w.add_subplot(gridspec[0,1])
        self.sax2 = self.fig_w.add_subplot(gridspec[0,2])
        self.tax = self.fig_w.add_subplot(gridspec[1,:])
        # set slider params
        iwin = int(twin*self.lfp_fs)
        i_kw = dict(valmin=iwin, valmax=len(self.lfp_time)-iwin-1, valstep=1, valfmt='%s s', 
                    valinit=int(len(self.lfp_time)/2))
        iwin_kw = dict(valmin=1, valmax=int(3*self.lfp_fs), valstep=1, valinit=iwin)
        ycoeff_kw = dict(valmin=-50, valmax=50, valstep=1, valinit=0,)
        yfig_kw = dict(valmin=0, valmax=int(5*self.plot_height), valstep=1, valinit=0)
        
        # create sliders for scaling/navigation
        iwin_sldr = gi.MainSlider(self.sax0, 'X', **iwin_kw)
        yfig_sldr = gi.MainSlider(self.sax1, 'Y', **yfig_kw)
        ycoeff_sldr = gi.MainSlider(self.sax2, 'Z', **ycoeff_kw)
        i_sldr = gi.MainSlider(self.tax, 'i', **i_kw)
        i_sldr.init_main_style()
        i_sldr.nsteps = int(iwin/2)
        
        # connect slider signals
        self.iw = pd.Series(dict(i=i_sldr, iwin=iwin_sldr, ycoeff=ycoeff_sldr, yfig=yfig_sldr))#, btns=radio_btns))
        self.iw.i.on_changed(self.plot_lfp_data)
        self.iw.iwin.on_changed(self.update_iwin)
        self.iw.ycoeff.on_changed(self.plot_lfp_data)
        self.iw.yfig.on_changed(lambda val: self.plot_row.setFixedHeight(int(self.plot_height + val)))
        
        ### FREQ BAND AXES
        self.fig_freq = matplotlib.figure.Figure(constrained_layout=True)
        #gridspec2 = matplotlib.gridspec.GridSpec(1, 3, figure=self.fig_freq)
        fax0 = self.fig_freq.add_subplot(131)
        fax0.autoscale(enable=True, axis='x', tight=True)
        fax1 = self.fig_freq.add_subplot(132, sharey=fax0)
        fax1.autoscale(enable=True, axis='x', tight=True)
        fax2 = self.fig_freq.add_subplot(133, sharey=fax1)
        fax2.autoscale(enable=True, axis='x', tight=True)
        self.faxs = [fax0, fax1, fax2]
        
        ### MAIN AXES
        self.fig = matplotlib.figure.Figure(constrained_layout=True)
        self.ax = self.fig.add_subplot()
        self.ax.sharey(fax0)
        self.xfmts = {'S':FuncFormatter(lambda x,pos: f'{x:.2f}'),
                      'M':FuncFormatter(lambda x,pos: f'{x/60:.2f}'),
                      'H':FuncFormatter(lambda x,pos: f'{x/3600:.2f}')}
        self.xax_formatter = self.xfmts['S']  # initialize x-units as seconds
        self.xu = 's'
    
    def connect_mpl_widgets(self):
        """ Connect keyboard/mouse inputs """
        
        def oncallback(xmin, xmax):
            """ Dynamically draw selection rectangle during mouse drag """
            self.canvas.draw_idle()
            
        def onselect(xmin, xmax):
            """ Show selected interval, get x-axis time span """
            self.xspan = xmax - xmin
            self.canvas.draw_idle()
            
        self.xspan = 0
        self.span = matplotlib.widgets.SpanSelector(ax=self.ax,
                                                    onselect=onselect,
                                                    onmove_callback=oncallback,
                                                    direction='horizontal',
                                                    button = 1,
                                                    useblit=True,
                                                    interactive=True,
                                                    drag_from_anywhere=True,
                                                    props=dict(fc='red', ec='black', lw=5, alpha=0.5))
        
        def on_click(event):
            """ Mouse press events 
            Right click -> context menu options for the nearest channel/timepoint
            Double click -> add a new DS or ripple event to the dataset
            """
            if event.xdata is None: return
            
            ### CONTEXT MENU ###
            
            if event.button == matplotlib.backend_bases.MouseButton.RIGHT:
                if self.toolbar.findChild(QtWidgets.QToolButton, 'Pan').isChecked():
                    return
                # get closest channel/timepoint to clicked point
                ch = -int(round(event.ydata))
                if ch not in self.shank_channels: return
                ich = list(self.shank_channels).index(ch)
                ###
                idx = pyfx.IdxClosest(event.xdata, self.lfp_time)
                tpoint = round(self.lfp_time[idx], 2)
                # highlight selected channel/timepoint
                xdata, ydata = self.ax.get_lines()[ich].get_data()
                shadow = self.ax.plot(xdata, ydata, color='yellow', zorder=-1, lw=5)[0]
                vline = self.ax.axvline(tpoint, color='indigo', zorder=-1, lw=2, ls='--')
                self.canvas.draw_idle()
                
                # create context menu and section headers
                menu = QtWidgets.QMenu()
                menu.setStyleSheet(pyfx.dict2ss(QSS.QMENU))
                headers = []
                for txt in [f'Time = {tpoint:.2f} s', f'Channel {ch}']:
                    hdr = QtWidgets.QWidgetAction(self)
                    lbl = QtWidgets.QLabel(txt)
                    hdr.setDefaultWidget(lbl)
                    headers.append(hdr)
                t_hdr, ch_hdr = headers
                t_hdr.defaultWidget().setObjectName('top_header')
                
                ### timepoint section
                menu.addAction(t_hdr)
                # copy timestamp or recording index to clipboard
                copyTAction = menu.addAction('Copy time point')
                copyIAction = menu.addAction('Copy index')
                
                ### channel section
                menu.addAction(ch_hdr)
                noise_txt = ["noise","clean"][self.NOISE_TRAIN[ch]]
                noiseAction = menu.addAction(f'Mark as {noise_txt}')
                noiseAction.setEnabled(ch not in self.event_channels)
                
                # execute menu
                res = menu.exec_(event.guiEvent.globalPos())
                if res == copyTAction:
                    QtWidgets.QApplication.clipboard().setText(str(tpoint))
                elif res == copyIAction:
                    QtWidgets.QApplication.clipboard().setText(str(idx))
                elif res == noiseAction:
                    # switch channel designation between "clean" and "noisy"
                    new_noise_train = np.array(self.NOISE_TRAIN)
                    new_noise_train[ch] = int(1-self.NOISE_TRAIN[ch])
                    self.updated_noise_signal.emit(ch, new_noise_train)
                    self.plot_lfp_data()
                shadow.remove()
                vline.remove()
                self.canvas.draw_idle()
                
            
            ### ADD EVENT ###
            
            elif event.dblclick and (self.ADD_DS or self.ADD_SWR):
                # get closest index to clicked point
                idx = pyfx.IdxClosest(event.xdata, self.lfp_time)
                opts = [('ds', self.hil_chan, 0.025, self.DS_ALL), 
                        ('swr', self.ripple_chan, 0.1, self.SWR_ALL)]
                EV, CHAN, WIN, EV_DF = opts[0 if self.ADD_DS else 1]
                lfp = self.DATA[EV][CHAN]
                iwin = int(self.lfp_fs * WIN)  # 50ms (DS) or 200ms (SWR) window
                tmpwin = int(iwin/2) if EV=='swr' else int(iwin)
                tmpwin0, tmpwin1 = min(idx, tmpwin), min(len(lfp)-idx, tmpwin)
                tmp = lfp[idx-tmpwin0 : idx+tmpwin1]
                if EV=='swr': 
                    tmp = np.abs(scipy.signal.hilbert(tmp)).astype('float32')
                # get index of peak LFP/envelope amplitude
                ipk = idx + (np.argmax(tmp) - tmpwin0)
                if ipk < iwin or ipk > len(lfp)-iwin:
                    gi.MsgboxError('Too close to recording bounds.', parent=self).exec()
                    self.span.set_visible(False)
                    return
                # get event-filtered and raw LFP within given window
                filt_lfp = lfp[ipk-iwin : ipk+iwin]
                raw_lfp = self.DATA['raw'][CHAN][ipk-iwin : ipk+iwin]
                ddict, error_msg = None, ''
                if self.ADD_DS:
                    # detect peaks
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', r'some peaks have a prominence of 0')
                        warnings.filterwarnings('ignore', r'some peaks have a width of 0')
                        props = scipy.signal.find_peaks(filt_lfp, height=max(filt_lfp), prominence=0)[1]
                        pws = np.squeeze(scipy.signal.peak_widths(filt_lfp, peaks=[iwin], rel_height=0.5))
                    istart, istop = ipk + (np.round(pws[2:4]).astype('int')-iwin)
                    imax = ipk + (np.argmax(raw_lfp)-iwin)
                    try:
                        amp,prom = [props[k][0] for k in ['peak_heights','prominences']]
                        assert istop > istart
                        ddict = dict(ch=self.hil_chan, time=self.lfp_time[imax], amp=amp,
                                     half_width=(pws[0]/self.lfp_fs)*1000, width_height=pws[1],
                                     asym=ephys.get_asym(ipk,istart,istop), prom=prom,
                                     start=self.lfp_time[istart], stop=self.lfp_time[istop],
                                     idx=imax, idx_peak=ipk, idx_start=istart, idx_stop=istop,
                                     status=2, is_valid=1, shank=int(self.shank.shank_id))
                    except IndexError:
                        error_msg = 'Error: No local LFP peak found.'
                        self.span.set_visible(False)
                    except AssertionError:
                        error_msg = 'Detected peak has a width of 0.'
                        self.span.set_visible(False)
                elif self.ADD_SWR:
                    hilb = scipy.signal.hilbert(filt_lfp)
                    env = np.abs(hilb).astype('float32')
                    # set thresholds for peak width calculation
                    if self.SWR_THRES is not None:
                        swr_min = self.SWR_THRES[self.ripple_chan].edge_height
                    else:
                        # quick 'n dirty estimation for envelope variance
                        env2 = np.abs(scipy.signal.hilbert(lfp[::10])).astype('float32')
                        swr_min = self.PARAMS.swr_min_thr * np.nanstd(env2)
                    env_clip = np.clip(env, swr_min, max(env))
                    # detect peaks
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', r'some peaks have a prominence of 0')
                        warnings.filterwarnings('ignore', r'some peaks have a width of 0')
                        props = scipy.signal.find_peaks(env, height=env[iwin])[1]
                        pws = np.squeeze(scipy.signal.peak_widths(env_clip, peaks=[iwin], rel_height=1))
                    istart, istop = ipk + (np.round(pws[2:4]).astype('int')-iwin)
                    # find largest oscillation in ripple
                    ampwin = int(round(self.PARAMS.swr_maxamp_win)/1000 * self.lfp_fs)
                    iosc = (np.argmax(filt_lfp[iwin-ampwin : iwin+ampwin]) + (iwin-ampwin))
                    imax = ipk + (iosc - iwin)
                    # inst. freq
                    fwin = int(round(self.PARAMS.swr_freq_win/1000 * self.lfp_fs))
                    ifreq = ephys.get_inst_freq(hilb, self.lfp_fs, self.PARAMS.swr_freq)
                    swr_ifreq = np.mean(ifreq[iwin-fwin : iwin+fwin])
                    try:
                        assert istop > istart
                        ddict = dict(ch=self.ripple_chan, time=self.lfp_time[imax], 
                                     amp=env[iwin], dur=(pws[0]/self.lfp_fs)*1000, freq=swr_ifreq,
                                     start=self.lfp_time[istart], stop=self.lfp_time[istop],
                                     idx=imax, idx_peak=ipk, idx_start=istart, idx_stop=istop,
                                     status=2, is_valid=1, shank=int(self.shank.shank_id))
                    except AssertionError:
                        error_msg = 'Detected envelope peak has a width of 0.'
                        self.span.set_visible(False)
                if ddict is None:
                    gi.MsgboxError(error_msg, parent=self).exec()
                else:
                    # add freq band power to data row, update n_valid column
                    ddict.update({**self.STD.loc[CHAN].to_dict(), 'n_valid':0})
                    ddf = pd.DataFrame(ddict, index=[CHAN])
                    if EV_DF.empty:
                        EV_DF = ddf
                    else:
                        EV_DF = pd.concat([EV_DF, ddf]).sort_values(['ch','time'])
                    EV_DF.loc[CHAN, 'n_valid'] = np.atleast_1d(EV_DF.loc[CHAN, 'is_valid']).sum()
                    if   EV == 'ds' : self.DS_ALL  = EV_DF
                    elif EV == 'swr': self.SWR_ALL = EV_DF
                    # update event trains, update event dataframe in main window
                    print(f'Added {EV.upper()}!')
                    self.update_event_items(event=EV)
                    self.edit_events_signal.emit(EV, deepcopy(EV_DF))
                    self.plot_lfp_data()
                    
            
        def on_press(event):
            """ Key press events
            Left/right arrows -> shift viewing window back/forward by 25%
            Backspace -> Exclude highlighted event(s) from analysis
            Spacebar -> Restore previously excluded events
            Escape -> Permanently erase event(s) from the dataset
            Enter -> Display CSD heatmap over the selected interval
            """
            ee = {'backspace':'delete',  # exclude events from dataset
                  'escape'   :'erase',   # permanently erase events
                  ' '        :'restore'} # restore events
            
            ###   SHIFT VIEWING WINDOW   ###
            
            if event.key   == 'left'  : self.iw.i.key_step(0)  # step backwards
            elif event.key == 'right' : self.iw.i.key_step(1)  # step forwards
            
            ###   CURATE EVENT DATASETS   ###
            
            elif event.key in ee and self.xspan > 0:
                # get selected time interval
                irg = [*map(lambda x: int(x*self.lfp_fs), self.span.extents)]
                mode = ee[event.key] # delete/erase/restore
                ds_kw = {'DF':self.DS_ALL, 'chan':self.hil_chan, 'irange':irg, 'mode':mode}
                swr_kw = {'DF':self.SWR_ALL, 'chan':self.ripple_chan, 'irange':irg, 'mode':mode}
                for ev,kw in zip(['ds','swr'],[ds_kw,swr_kw]):
                    kw['idx'] = self.get_visible_events(ev)
                    if len(kw['idx']) > 0 and kw['DF'] is not None:
                        res = self.edit_event_status(**kw)
                        if res: 
                            self.update_event_items(event=ev)
                            self.edit_events_signal.emit(ev, deepcopy(kw['DF']))
                self.span.set_visible(False)
                self.plot_lfp_data()
                
            ###   PLOT INSTANTANEOUS CSD   ###
                
            elif event.key=='enter' and self.xspan > 0 and self.coord_electrode is not None:
                imin, imax = [*map(lambda x: int(x*self.lfp_fs), self.span.extents)]
                # plot temporary CSD
                arr = self.DATA['raw'][self.shank_channels, imin:imax]
                arr2 = np.array(arr)
                noise_idx_rel = np.nonzero(self.NOISE_TRAIN[self.shank_channels])[0]
                clean_idx_rel = np.setdiff1d(np.arange(arr.shape[0]), noise_idx_rel)
                # replace noisy channels with the average of its two closest neighbors
                for i in noise_idx_rel:
                    if i == 0:
                        interp_sig = np.array(arr[min(clean_idx_rel)])
                    elif i > max(clean_idx_rel):
                        interp_sig = np.array(arr[max(clean_idx_rel)])
                    else:
                        sig1 = arr[pyfx.Closest(i, clean_idx_rel[clean_idx_rel < i])]
                        sig2 = arr[pyfx.Closest(i, clean_idx_rel[clean_idx_rel > i])]
                        interp_sig = np.nanmean([sig1, sig2], axis=0)
                    arr2[i,:] = interp_sig
                # calculate CSD
                csd_obj = ephys.get_csd_obj(arr2, self.coord_electrode, self.PARAMS)
                csd = ephys.csd_obj2arrs(csd_obj)[1]
                a,b = pyfx.Edges(self.shank_channels)
                yax = np.linspace(b+1, a-1,len(csd)) * -1 # (-43,-42...,0,1)
                csd = csd[::-1, :]  # arrange rows bottom to top to match $yax
                try:
                    _ = self.ax.pcolorfast(self.lfp_time[imin:imax], yax, csd, 
                                           cmap=plt.get_cmap('bwr'))
                except:
                    _ = self.ax.pcolorfast(pyfx.Edges(self.lfp_time[imin:imax]), 
                                                      pyfx.Edges(yax), csd, cmap=plt.get_cmap('bwr'))
                self.span.set_visible(False)
                self.canvas.draw_idle()
                
            ###   CHANGE TIME UNITS   ###
            
            elif event.key in ['S','M','H']:
                self.xax_formatter = self.xfmts[event.key]
                self.xu = event.key.lower()
                self.ax.xaxis.set_major_formatter(self.xax_formatter)
                self.ax.set_xlabel(f'Time ({self.xu})')
                self.canvas.draw_idle()
                
        self.cid = self.canvas.mpl_connect("key_press_event", on_press)
        self.cid2 = self.canvas.mpl_connect("button_press_event", on_click)
    
    def update_iwin(self, new_iwin):
        """ Use viewing window bounds to update range of index slider """
        imin, imax = (new_iwin, len(self.lfp_time)-new_iwin-1)
        self.iw.i.update_range(imin, imax)
        i = self.iw.i.val
        if i < imin or i > imax:
            self.iw.i.set_val(i)
        else:
            self.plot_lfp_data()
            
    def edit_event_status(self, DF, idx, irange, chan, mode=''):
        """ Delete or restore events within $irange for channel $chan in dataframe $DF """
        # find qualifying indices within range, map to DF rows
        qidx = np.intersect1d(np.arange(*irange), idx)
        irows = np.where((DF.ch == chan) & (np.in1d(DF.idx, qidx)))[0]
        if len(irows) == 0:
            return False
        if mode == 'erase':
            # delete event forever
            DF.reset_index(inplace=True)
            DF.drop(irows, axis=0, inplace=True)
            DF.set_index('index', inplace=True)
            DF.index.name = None
            if chan in DF.index:
                DF.loc[chan, 'n_valid'] = DF.loc[chan, 'is_valid'].sum()
            print(f'Permanently erased {len(irows)} event(s)!')
            return True
        # set status negative (1>>-1, -1>>-1) or positive (1>>1, -1>>1)
        if   mode == 'delete'  : fx = lambda x: x * np.sign(x) * -1
        elif mode == 'restore' : fx = lambda x: np.abs(x)
        else                   : fx = lambda x: print(f'WARNING: Mode {mode} not recognized')
        istatus = DF.columns.get_loc('status')
        DF.iloc[irows, istatus] = d = fx(DF.iloc[irows, istatus])
        # removed events (-1 or -2) are invalid; restored events are valid
        DF.iloc[irows, DF.columns.get_loc('is_valid')] = (d > 0).astype('int')
        DF.loc[chan, 'n_valid'] = DF.loc[chan, 'is_valid'].sum()
        print(f'{mode.capitalize()}d {len(irows)} event(s)!')
        return True
    
    def plot_freq_band_pwr(self):
        """ Plot distribution of channel amplitudes for defined frequency bands """
        _ = [ax.clear() for ax in self.faxs]
        self.freq_kw = dict(xytext=(4,4), xycoords=('axes fraction','data'), 
                            bbox=dict(facecolor='w', edgecolor='w', 
                                      boxstyle='square,pad=0.0'),
                            textcoords='offset points', va='bottom', 
                            fontweight='semibold', annotation_clip=True)
        # (channel, color) for each event
        (TH,THC), (RPL,RPLC), (HIL,HILC) = self.ch_cmap.items()
        noise_idx_rel = np.nonzero(self.NOISE_TRAIN[self.shank_channels])[0]
        if (len(noise_idx_rel) == 0) and (len(self.shank_channels) == len(self.STD)):
            tmp = self.STD[['norm_theta','norm_swr','slow_gamma','fast_gamma']].values.T
            norm_theta, norm_swr, slow_gamma, fast_gamma = tmp
        else:
            freq_data = self.STD.loc[self.shank_channels, ['theta','swr','slow_gamma','fast_gamma']].values.T
            if len(noise_idx_rel)>0:
                for i,d in enumerate(freq_data):
                    d[noise_idx_rel] = np.nan
            norm_theta = pyfx.Normalize(freq_data[0])
            norm_swr = pyfx.Normalize(freq_data[1])
            slow_gamma, fast_gamma = freq_data[2:]
        
        yax = np.array(self.shank_channels * -1)
        # plot theta power
        self.faxs[0].plot(norm_theta, yax, color='black')
        self.faxs[0].axhline(TH*-1, c=THC)
        self.faxs[0].annotate('Theta', xy=(0,TH*-1), color=THC, **self.freq_kw)
        
        # plot ripple power
        self.faxs[1].plot(norm_swr, yax, color='black')
        self.faxs[1].axhline(RPL*-1, c=RPLC)
        self.faxs[1].annotate('Ripple', xy=(0,RPL*-1), color=RPLC, **self.freq_kw)
        
        # plot fast and slow gamma power
        self.faxs[2].plot(slow_gamma, yax, color='gray', lw=2, label='slow')
        self.faxs[2].plot(fast_gamma, yax, color='indigo', lw=2, label='fast')
        self.faxs[2].axhline(HIL*-1, c=HILC)
        self.faxs[2].annotate('Hilus', xy=(0,HIL*-1), color=HILC, **self.freq_kw)
        
        # set axis titles, labels, legend
        self.faxs[0].set_title('Theta Power', va='bottom', y=0.97)
        self.faxs[1].set_title('Ripple Power', va='bottom', y=0.97)
        self.faxs[2].set_title('Gamma Power', va='bottom', y=0.97)
        _ = self.faxs[2].legend(loc='upper right', bbox_to_anchor=(1.1,.95))
        _ = [ax.set_xlabel('SD') for ax in self.faxs]
        _ = self.faxs[0].set_yticks(yax, labels=np.int16(abs(yax)).astype('str'))
        
        # set style, annotation kwargs
        self.faxs[1].spines['left'].set_visible(False)
        self.faxs[2].spines['left'].set_visible(False)
        self.canvas_freq.draw_idle()
        
    def switch_the_probe(self, **kwargs):
        """ Update local variables for a new probe """
        self.DATA = kwargs['DATA']
        self.DS_ALL = kwargs.get('DS_ALL', None)
        self.SWR_ALL = kwargs.get('SWR_ALL', None)
        self.STD = kwargs.get('STD', None)
        self.NOISE_TRAIN = kwargs.get('NOISE_TRAIN', None)
        #self.probe = kwargs.get('probe', None)
        _shank = kwargs.get('shank', None)
        _event_channels = kwargs.get('event_channels', [0,0,0])
        self.switch_the_shank(shank=_shank, event_channels=_event_channels)
    
    def switch_the_shank(self, **kwargs):
        """ Update local variables for a new shank """
        self.shank = kwargs.get('shank', None)
        event_channels = kwargs.get('event_channels', [0,0,0])
        self.ch_cmap = pd.Series(['blue', 'green', 'red'], index=event_channels)
        self.init_event_items()
        self.channel_changed(*event_channels)  # set event channels (and DS/SWR indices)
        self.plot_freq_band_pwr()
    
    def init_event_items(self):
        """ Initialize channels, geometry, and event trains for the given probe/shank """
        # try getting electrode geometry in meters
        self.coord_electrode = None
        if self.shank is None:
            self.shank_channels = np.arange(self.DATA['raw'].shape[0])
        else:
            self.shank_channels = np.array(self.shank.get_indices())
            ypos = np.array(sorted(self.shank.contact_positions[:, 1]))
            self.coord_electrode = pq.Quantity(ypos, self.shank.probe.si_units).rescale('m')  # um -> m
        # get data timecourses
        self.lfp_ampl = np.nansum(self.DATA['raw'][self.shank_channels, :], axis=0)
        
        # initialize event indexes (empty), event trains (zeros)
        self.DS_idx = np.array(())
        self.DS_valid = np.array(())
        self.SWR_idx = np.array(())
        self.SWR_valid = np.array(())
        self.DS_train = np.zeros(len(self.lfp_time))
        self.SWR_train = np.zeros(len(self.lfp_time))
    
    def update_event_items(self, event='all'):
        """
        Update event indices after one of the following actions:
        1) User changes the current hilus channel or ripple channel
        2) User switches between probes
        3) User adds or removes an event instance
        """
        if (event in ['all','ds']) and (self.DS_ALL is not None):
            self.DS_train[:] = 0
            DS_DF = self.DS_ALL[self.DS_ALL.ch == self.hil_chan]
            # valid events are either 1) auto-detected and non-user-removed or 2) user-added
            self.DS_idx, DS_status = np.array(DS_DF[['idx','status']], dtype='int').T
            self.DS_train[self.DS_idx] = DS_status # 1=auto-detected; 2=user-added; -1=user-removed
            self.DS_valid = np.array(DS_DF[DS_DF['is_valid']==1].idx, dtype='int')
        
        if (event in ['all','swr']) and (self.SWR_ALL is not None):
            self.SWR_train[:] = 0
            SWR_DF = self.SWR_ALL[self.SWR_ALL.ch == self.ripple_chan]
            self.SWR_idx, SWR_status = np.array(SWR_DF[['idx','status']], dtype='int').T
            self.SWR_train[self.SWR_idx] = SWR_status # 1=auto-detected; 2=user-added; -1=user-removed
            self.SWR_valid = np.array(SWR_DF[SWR_DF['is_valid']==1].idx, dtype='int')
            
    def channel_changed(self, theta_chan, ripple_chan, hil_chan):
        """ Update currently selected event channels """
        self.event_channels = [theta_chan, ripple_chan, hil_chan]
        self.theta_chan, self.ripple_chan, self.hil_chan = self.event_channels
        self.ch_cmap = self.ch_cmap.set_axis(self.event_channels)
        self.update_event_items()  # set DS/SWR indices for new channel
        self.plot_lfp_data()
        
    def get_visible_events(self, event):
        """ Get event instances with visible plot markers """
        if event == 'ds':
            if not self.SHOW_DS: # all events hidden from plot
                return np.array([], dtype='int')
            if not self.SHOW_DS_RM: # excluded events hidden from plot
                return self.DS_valid
            return self.DS_idx # all events visible
        elif event == 'swr':
            if not self.SHOW_SWR:
                return np.array([], dtype='int')
            if not self.SHOW_SWR_RM:
                return self.SWR_valid
            return self.SWR_idx
    
    def event_jump(self, sign, event):
        """ Set plot index to the next (or previous) instance of a given event """
        # get idx for given event type, return if empty
        idx = self.get_visible_events(event)
        if len(idx) == 0: return
        # find nearest event preceding (sign==0) or following (1) current idx
        i = self.iw.i.val
        idx_next = idx[idx < i][::-1] if sign==0 else idx[idx > i]
        if len(idx_next) == 0: return
        # set index slider to next event (automatically updates plot)
        self.iw.i.set_val(idx_next[0])
    
    def point_jump(self, val, unit):
        """ Shift viewing window to the given index or timepoint  """
        if   unit == 't' : new_idx = pyfx.IdxClosest(val, self.lfp_time)
        elif unit == 'i' : new_idx = val
        self.iw.i.set_val(new_idx)
    
    def plot_lfp_data(self, x=None):
        """ Update LFP signals and event annotations on central plot """
        self.ax.clear()
        i,iwin = self.iw.i.val, self.iw.iwin.val
        idx = np.arange(i-iwin, i+iwin)
        x = self.lfp_time[idx]
        arr = self.DATA['raw']#[self.shank_channels, :]
        self.iw.i.nsteps = int(iwin/2)
        
        # scale signals based on y-slider value
        if   self.iw.ycoeff.val < -1 : coeff = 1/np.abs(self.iw.ycoeff.val) * 2
        elif self.iw.ycoeff.val >  1 : coeff = self.iw.ycoeff.val / 2
        else                         : coeff = 1
        #for irow,y in enumerate(arr):
        for irow in self.shank_channels:
            y = arr[irow]
            clr = self.ch_cmap.get(irow, 'black')
            if isinstance(clr, pd.Series): clr = clr.values[0]
            # plot LFP signals (y - irow, set tick labels to show absolute values)
            if self.NOISE_TRAIN[irow] == 1:
                self.ax.plot(x, np.repeat(-irow, len(x)), color='lightgray', label=str(irow), lw=2)
            else:
                self.ax.plot(x, y[idx]*coeff - irow, color=clr, label=str(irow), lw=1)
        
        # mark ripple/DS events with red/green lines
        if self.SHOW_DS:
            ds_ii = np.where(self.DS_train[idx] != 0)[0]
            linekw = dict(color='red', lw=2, zorder=5, alpha=0.4)
            for ds_time, status in zip(x[ds_ii], self.DS_train[idx][ds_ii]):
                if status == 1:
                    self.ax.axvline(ds_time, **linekw)
                elif status == 2:
                    self.ax.axvline(ds_time, ls=':', **linekw)
                elif self.SHOW_DS_RM:
                    self.ax.axvline(ds_time, ls='--', **linekw)
        if self.SHOW_SWR:
            swr_ii = np.where(self.SWR_train[idx] != 0)[0]
            linekw = dict(color='green', lw=2, zorder=5, alpha=0.4)
            for swr_time, status in zip(x[swr_ii], self.SWR_train[idx][swr_ii]):
                if status == 1:
                    self.ax.axvline(swr_time, **linekw)
                elif status == 2:
                    self.ax.axvline(swr_time, ls=':', **linekw)
                elif self.SHOW_SWR_RM:
                    self.ax.axvline(swr_time, ls='--', **linekw)
        
        if self.canvas_freq.isVisible():
            self.plot_freq_band_pwr()
            
        yticks = -np.array(self.shank_channels)
        _ = self.ax.set_yticks(yticks, labels=np.int16(abs(yticks)).astype('str'))
        # set x-axis with appropriate units
        self.ax.xaxis.set_major_formatter(self.xax_formatter)
        self.ax.set(xlabel=f'Time ({self.xu})', ylabel='channel index')
        self.ax.set_xmargin(0.01)
        self.ax.set_ymargin(0.01)
        self.canvas.draw_idle()
        
    def closeEvent(self, event):
        """ Close plots """
        plt.close()
        super().closeEvent(event)
        
        
class IFigEvent(QtWidgets.QWidget):
    """ Base figure showing detected events from one LFP channel """
    
    pal = sns.cubehelix_palette(dark=0.2, light=0.9, rot=0.4, as_cmap=True)
    FLAG = 0  # 0=plot average waveform, 1=plot individual events
    EXCLUDE_NOISE = False
    annot_dict = dict(time='{time:.2f} s')
    CHC = pd.Series(pyfx.rand_hex(9999))
    
    def __init__(self, ch, channels, DF_ALL, DATA, lfp_time, lfp_fs, PARAMS, **kwargs):
        super().__init__()
        
        # initialize params from input arguments
        self.ch = ch
        self.channels = channels # shank_channels
        self.ch2irow = pd.Series(np.arange(len(channels)), index=channels)
        self.DF_ALL = DF_ALL[DF_ALL['is_valid']==1]
        self.DATA = DATA
        self.lfp_time = lfp_time
        self.lfp_fs = lfp_fs
        self.PARAMS = PARAMS
        twin = kwargs.get('twin', 0.2)
        self.thresholds = kwargs.get('thresholds', {})
        
        # get LFP and DF for the primary channel
        self.LFP_arr = self.DATA['raw']
        self.ich = int(self.ch2irow[ch])
        self.LFP = self.LFP_arr[self.ich]
        self.EVENT_NOISE_TRAIN = kwargs.get('EVENT_NOISE_TRAIN', np.zeros(len(self.channels), dtype='int'))
        self.EVENT_NOISE_IDX = np.nonzero(self.EVENT_NOISE_TRAIN)[0]
        ###
        # get DF means per channel, isolate DF of current channel
        self.DF_MEAN = self.DF_ALL.groupby('ch').agg('mean')
        self.DF_MEAN = ephys.replace_missing_channels(self.DF_MEAN, self.channels)
        self.DF_MEAN.insert(0, 'ch', self.DF_MEAN.index.values)
        # protect from shenanigans with empty or 1-item dataframes
        num_events = list(self.DF_ALL.ch).count(self.ch)
        if num_events == 0:
            self.DF = pd.DataFrame(columns=self.DF_ALL.columns)
        elif num_events == 1:
            self.DF = pd.DataFrame(self.DF_ALL.loc[self.ch]).T
            for k,v in self.DF_ALL.dtypes.items():
                self.DF[k] = self.DF[k].astype(v)
        else:
            self.DF = pd.DataFrame(self.DF_ALL.loc[self.ch])
        self.dt = self.lfp_time[1] - self.lfp_time[0]
        
        # get event indexes / event train for the primary channel
        self.iev = np.atleast_1d(self.DF.idx.values)
        self.ev_train = np.full(self.lfp_time.shape, np.nan)
        for idx,istart,iend in self.DF[['idx','idx_start','idx_stop']].values:
            self.ev_train[istart : iend] = idx
            
        self.CHC[self.ch] = 'red'
            
        # create structured annotation string
        self.annot_fmt = str_fmt(self.annot_dict, key='time', key_top=True)
        
        # initialize channel plotting list
        self.CH_ON_PLOT = [int(self.ch)]
        
        self.create_subplots(twin=twin)
        
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setOrientation(QtCore.Qt.Vertical)
        self.toolbar.setMaximumWidth(30)
        for tbtn in self.toolbar.findChildren(QtWidgets.QToolButton):
            if tbtn.actions()[0].text() == 'Pan':
                tbtn.setObjectName('Pan')
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.toolbar, stretch=0)
        self.layout.addWidget(self.canvas, stretch=2)
        
        self.connect_mpl_widgets()
        
        self.update_twin(twin)
        
    def create_subplots(self, twin=0.2):
        """ Set up grid of data and widget subplots """
        # create subplots
        self.fig = matplotlib.figure.Figure()
        subplot_kwargs = dict(height_ratios=[10,1,1,1,10], gridspec_kw=dict(hspace=0))
        self.axs = self.fig.subplot_mosaic([['gax0','gax1','gax2'],
                                            ['spacer','spacer','spacer'],
                                            ['sax0','sax0','sax0'],
                                            ['esax','esax','esax'],
                                            ['main','main','main']], **subplot_kwargs)
        divider = make_axes_locatable(self.axs['sax0'])
        self.axs['sax1'] = divider.append_axes('right', size="100%", pad=0.5)
        self.axs['spacer'].set_visible(False)
        
        self.ax = self.axs['main']
        self.ax.set_xmargin(0.01)
        
        self.EXCLUDE_NOISE=True
        
        ###   PLOT EVENTS FROM ALL CHANNELS   ###
        _ = ephys.plot_channel_events(self.DF_ALL, self.DF_MEAN,
                                      self.axs['gax0'], self.axs['gax1'], self.axs['gax2'],
                                      pal=self.pal, noise_train=self.EVENT_NOISE_TRAIN,
                                      exclude_noise=bool(self.EXCLUDE_NOISE), CHC=self.CHC)
            
        # initial colormap
        self.ch_gax0_artists = pd.Series(list(self.axs['gax0'].patches), index=self.channels) # bars
        collections = np.array([None] * len(self.channels)).astype('object')
        collections[self.DF_MEAN.n_valid > 0] = list(self.axs['gax1'].collections)
        self.ch_gax1_artists = pd.Series(collections, index=self.channels) # collections
        polygons = []
        for ch in self.channels:
            c = 'red' if ch == self.ch else self.CHC[ch]  # "highlight" given channel
            axv = self.axs['gax2'].axvspan(ch-0.25, ch+0.25, color=c, alpha=0.7, zorder=-5)
            axv.set_visible(False)
            polygons.append(axv)
        self.ch_gax2_polygons = pd.Series(polygons, index=self.channels)
        
        ### create threshold artists
        # X axis (horizontal line at amplitude 0)
        self.xax_line = self.ax.axhline(0, color='indigo', lw=2, alpha=0.7)
        self.xax_line.set_visible(False)
        # Y axis (vertical line at timepoint 0)
        self.yax_line = self.ax.axvline(0, color='darkred', lw=2, alpha=0.7)
        self.yax_line.set_visible(False)
        
        # event height threshold
        self.hThr_line = self.ax.axhline(color='darkblue', label='Peak height', lw=2, alpha=0.5)
        if 'peak_height' in self.thresholds:
            self.hThr_line.set_ydata([self.thresholds.peak_height]*2)
        self.hThr_line.set_visible(False)
        self.thres_items = [self.hThr_line]
        # set threshold legend params
        self.thres_leg_kw = dict(loc='lower right',  bbox_to_anchor=(1,0.5), 
                                 title='Thresholds', draggable=True)
        
        # create sliders for scaling/navigation
        twin_kw = dict(valmin=.01, valmax=0.5, valstep=.01, valfmt='%.2f s', valinit=twin)
        ywin_kw = dict(valmin=-0.5, valmax=1, valstep=0.05, valfmt='%.2f', valinit=0)
        idx_kw = dict(valmin=0, valmax=max(len(self.iev)-1,1), valstep=1, valinit=0, 
                   valfmt='%.0f%% / ' + str(len(self.iev)-1))
        
        twin_sldr = gi.MainSlider(self.axs['sax0'], 'X', **twin_kw)
        ywin_sldr = gi.MainSlider(self.axs['sax1'], 'Y', **ywin_kw)
        idx_sldr = gi.MainSlider(self.axs['esax'], 'event', **idx_kw)
        idx_sldr.init_main_style()
        self.iw = pd.Series(dict(idx=idx_sldr, twin=twin_sldr, ywin=ywin_sldr))
        self.iw.idx.on_changed(self.plot_event_data)
        self.iw.twin.on_changed(lambda x: self.plot_event_data(self.iw.idx.val, twin=x))
        self.iw.ywin.on_changed(lambda x: self.ax.set_ylim(pyfx.Limit(self.EY, pad=np.abs(x), 
                                                                      sign=np.sign(x))))
        
        # create radio buttons to toggle between raw/filtered signals
        self.radio_ax = self.ax.inset_axes([0.8, 0.8, 0.2, 0.2])
        self.radio_ax.set_axis_off()
        self.dbtns = matplotlib.widgets.RadioButtons(self.radio_ax, labels=['Raw','Filtered'], 
                                                     active=0, activecolor='black')
        _ = self.radio_ax.collections[0].set(sizes=[125,125])
        _ = [lbl.set(fontsize=12) for lbl in self.dbtns.labels]
        
    
    def connect_mpl_widgets(self):
        """ Connect keyboard/mouse inputs """
        
        def on_press(event):
            """ Left/right arrows -> iterate over individual event dataset """
            if self.FLAG == 1:
                i = self.iw.idx.val  # index of current event
                if event.key == 'left' and i > 0:
                    self.iw.idx.set_val(i-1)  # previous event
                elif event.key == 'right' and i < len(self.iev)-1: 
                    self.iw.idx.set_val(i+1)  # next event
        self.cid = self.canvas.mpl_connect("key_press_event", on_press)
        
        def on_click(event):
            """ Copy index of individual waveforms to view in main window """
            if event.xdata is None: return
            if event.button == matplotlib.backend_bases.MouseButton.RIGHT:
                if event.inaxes != self.ax: return
                if self.FLAG == 0: return
                if self.toolbar.findChild(QtWidgets.QToolButton, 'Pan').isChecked():
                    return
                menu = QtWidgets.QMenu()
                menu.setStyleSheet(pyfx.dict2ss(QSS.QMENU))
                # copy timestamp or recording index to clipboard
                copyTAction = menu.addAction('Copy time point')
                copyIAction = menu.addAction('Copy index')
                idx = self.iev[self.iw.idx.val]
                tpoint = round(self.lfp_time[idx], 2)
                # execute menu
                res = menu.exec_(event.guiEvent.globalPos())
                if res == copyTAction:
                    QtWidgets.QApplication.clipboard().setText(str(tpoint))
                elif res == copyIAction:
                    QtWidgets.QApplication.clipboard().setText(str(idx))
                self.canvas.draw_idle()
        self.cid2 = self.canvas.mpl_connect("button_press_event", on_click)
        
        
    def update_twin(self, twin):
        """ Update plot data window, set x-axis limits """
        # update data window, set x-limits
        self.iwin = int(twin * self.lfp_fs)  # window size
        self.ev_x = np.linspace(-twin, twin, self.iwin*2)
        self.ax.set_xlim(pyfx.Limit(self.ev_x, pad=0.01))
    
    
    def rescale_y(self):
        """ Automatically determine y-limits when updating data plot 
        (e.g. new data source, changed timescale, or toggled plot mode) """
        if len(self.iev) == 0:
            return
        
        if self.FLAG == 0:
            if len(self.CH_ON_PLOT) == 1:
                lfp_arr = ephys.getwaves(self.LFP, self.iev, self.iwin)
                yerrs, _ = ephys.getyerrs(lfp_arr, mode='sem')
                self.EY = pyfx.Limit(np.concatenate(yerrs), pad=0.05)
            else:
                waves = [ephys.getavg(self.LFP_arr[self.ch2irow[xch]],
                                      np.atleast_1d(self.DF_ALL.loc[xch].idx),
                                      self.iwin) for xch in self.CH_ON_PLOT]
                self.EY = pyfx.Limit(np.concatenate(waves), pad=0.05)
        elif self.FLAG == 1:
            self.EY = pyfx.Limit(ephys.getwaves(self.LFP, self.iev, self.iwin).flatten(), pad=0.05)
        ylim = pyfx.Limit(self.EY, pad=np.abs(self.iw.ywin.val), sign=np.sign(self.iw.ywin.val))
        if len(np.unique(ylim)) == 1:
            ylim = tuple(np.array(ylim) + (np.array(ylim) * 0.1 * np.array([-1,1])))
        self.ax.set_ylim(ylim)
        
        
    def sort_events(self, col):
        """ Sort individual events in dataframe by given parameter column $col """
        idx = self.iev[self.iw.idx.val]  # save idx of plotted event
        # sort event dataframe
        self.DF = self.DF.sort_values(col)
        self.iev = np.atleast_1d(self.DF.idx.values)
        self.annot_fmt = str_fmt(self.annot_dict, key=col, key_top=True)
        
        # set slider value to event index in sorted data
        event = list(self.iev).index(idx)
        self.iw.idx.set_val(event)
        
    # def colormap_plots(self):
    #     ax0,ax1,ax2 = [self.axs[f'gax{i}'] for i in range(3)]
    #     bar.set(lw=1)
    #     _ = [bar.set(color=c, lw=1) for bar,c in zip(ax0.ch_bars, ax0.CM)]
    #     _ = [coll.set(lw=0,ec='k',) for coll in ax1.ch_collections]
        
    #     ax0.ch_bars
        
        
    #     CMAPS = [ax0.cmap, ax1.cmaps, ax2.cmap]
    #     if self.EXCLUDE_NOISE:
    #         CMAPS = [ax0.cmapNE, ax1.cmapsNE, ax2.cmapNE]
        
    def new_label_ch_data(self, x):
        
        ax0, ax1, ax2 = [self.axs[f'gax{i}'] for i in range(3)]
        # clear all graph highlights
        _ = [bar.set(lw=1, ec=c) for bar,c in zip(ax0.ch_bars, ax0.cmap)]
        _ = [coll.set(lw=0, ec='black') for coll in ax1.ch_collections]
        _ = [ol.set(mew=3, mec=c) for ol,c in zip(ax2.outlines, ax2.cmap)]
        _ = [vl.set_visible(False) for vl in ax2.ch_vlines]
        # color-coded highlights show channels in comparison plot
        if x:
            for ch in self.CH_ON_PLOT[1:]:
                ax0.ch_bars[ch].set(lw=2, ec=self.CHC[ch])
                ax1.ch_collections[ch].set(lw=1, ec=self.CHC[ch])
                ax2.outlines[ch].set(mew=3, mec=self.CHC[ch])
                ax2.ch_vlines[ch].set_visible(True)
            # red highlights show primary event channel
            ax0.ch_bars[self.ch].set(lw=2, ec='red')
            ax1.ch_collections[self.ch].set(lw=1, ec='red')
            ax2.outlines[self.ch].set(mew=4, mec='red')
        self.canvas.draw_idle()
        
    
    def label_ch_data(self, x):
        """ Add (or remove) colored highlight on current channel data """
        #self.new_label_ch_data(x)
        #return
        for xch in self.CH_ON_PLOT[1:]:
            ch_bar = self.ch_gax0_artists[xch]
            ch_coll = self.ch_gax1_artists[xch]
            if x == True:
                if ch_bar is not None: ch_bar.set(lw=2, ec=self.CHC[xch])
                if ch_coll is not None: ch_coll.set(lw=1, ec=self.CHC[xch])
            else:
                if ch_bar is not None: ch_bar.set(lw=1, ec=ch_bar.get_fc())
                if ch_coll is not None: ch_coll.set(lw=0, ec='black')
            self.ch_gax2_polygons[xch].set_visible(x)
        # currently selected channel
        ch_bar = self.ch_gax0_artists[self.ch]
        ch_coll = self.ch_gax1_artists[self.ch]
        if x == True:
            if ch_bar is not None: ch_bar.set(lw=2, ec='red')
            if ch_coll is not None: ch_coll.set(lw=1, ec='red')
        else:
            if ch_bar is not None: ch_bar.set(lw=1, ec=ch_bar.get_fc())
            if ch_coll is not None: ch_coll.set(lw=0, ec='black')
        self.ch_gax2_polygons[self.ch].set_visible(x)
        self.canvas.draw_idle()
        
    
    def plot_event_data(self, event, twin=None):
        """ Update event data on graph; plot avg waveform or single events """
        #self.ax.clear()
        legs = self.ax.findobj(matplotlib.legend.Legend)
        _ = [x.remove() for x in self.ax.lines + self.ax.collections + self.ax.texts + legs]
        self.ax.set_title('', loc='left')
        if event is None: event = self.iw.idx.val
        # get waveform indexes for new time window
        if twin is not None:
            self.update_twin(twin)
        
        # add visible threshold items to plot
        visible_items = [item for item in self.thres_items if item.get_visible()]
        _ = [self.ax.add_artist(item) for item in visible_items]
        if len(visible_items) > 0:
            thres_legend = self.ax.legend(handles=visible_items, **self.thres_leg_kw)
        else: thres_legend = None
        # add X and Y axes to plot (but not the legend)
        if self.xax_line.get_visible(): self.ax.add_line(self.xax_line)
        if self.yax_line.get_visible(): self.ax.add_line(self.yax_line)
        
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlabel('Time (s)')
        
        if len(self.iev) == 0: return
        
        ### plot average waveform(s)
        if self.FLAG == 0:
            # plot mean waveform for primary channel
            lfp_arr = ephys.getwaves(self.LFP, self.iev, self.iwin)
            (yerr0, yerr1), self.ev_y = ephys.getyerrs(lfp_arr, mode='sem')
            line = self.ax.plot(self.ev_x, self.ev_y, color='black', lw=2, zorder=5)[0]
            
            # if only plotting primary channel, include y-error
            if len(self.CH_ON_PLOT) == 1:
                _ = self.ax.fill_between(self.ev_x, yerr0, yerr1, color='black', alpha=0.3, zorder=-2)
            else:
                # overlay other channel(s) for direct comparison
                comparison_lines = []
                for xch in self.CH_ON_PLOT[1:]:
                    xmean = ephys.getavg(self.LFP_arr[self.ch2irow[xch]], 
                                         np.atleast_1d(self.DF_ALL.loc[xch].idx),
                                         self.iwin)
                    line = self.ax.plot(self.ev_x, xmean, color=self.CHC[xch], lw=2, label=f'ch {xch}')[0]
                    comparison_lines.append(line)
                self.ax.legend(handles=comparison_lines, loc='upper right', bbox_to_anchor=(1,0.4),
                               title='Other Channels', draggable=True)
                if thres_legend is not None: self.ax.add_artist(thres_legend)
            
            title = f'Average Waveform for Channel {self.ch}\n(n = {len(self.iev)} events)'
            self.ax.set_title(title, loc='left', va='top', ma='center', x=0.01, y=0.98, 
                              fontdict=dict(fontweight='bold'))
            #self.ax.margins(0.01, self.iw.ywin.val)
            
        ### plot individual events
        else:
            self.idx  = self.iev[event]
            self.ii   = np.arange(self.idx-self.iwin, self.idx+self.iwin)  # plot window idx
            self.iii  = np.where(self.ev_train == self.idx)[0]     # event idx
            self.irel = np.nonzero(np.in1d(self.ii, self.iii))[0]  # event idx within plot window
            
            # update LFP signal plot
            self.ev_y = ephys.pad_lfp(self.LFP, self.idx, self.iwin, pad_val=0)
            _ = self.ax.plot(self.ev_x, self.ev_y, color='black', lw=1.5)[0]
            _ = self.ax.plot(self.ev_x[self.irel], self.ev_y[self.irel])[0]
            # update annotation
            self.E = pd.Series(self.DF.iloc[event,:])
            fmt = self.annot_fmt.format(**self.E[self.annot_dict.keys()])
            txt = 'QUANTIFICATION' + os.linesep + fmt
            self.ax.annotate(txt, xy=(0.02,1), xycoords='axes fraction', 
                                     ha='left', va='top', fontsize=12)
        self.canvas.draw_idle()


class IFigSWR(IFigEvent):
    """ Figure displaying sharp-wave ripple events """
    
    SHOW_ENV = False
    SHOW_DUR = False
    
    FLAG = 0
    annot_dict = dict(time='{time:.2f} s', amp='{amp:.2f} mV', dur='{dur:.0f} ms', freq='{freq:.0f} Hz')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hilb = scipy.signal.hilbert(self.LFP)
        self.env = np.abs(self.hilb).astype('float32')
    
    
    def create_subplots(self, **kwargs):
        """ Add artists to subplots """
        super().create_subplots(**kwargs)
        
        def toggle_data(label):
            """ Plot raw vs filtered LFP data """
            if label == 'Raw' : self.LFP_arr = self.DATA['raw']
            else              : self.LFP_arr = self.DATA['swr']
            self.LFP = self.LFP_arr[self.ich]
            self.hilb = scipy.signal.hilbert(self.LFP)
            self.env = np.abs(self.hilb).astype('float32')
            
            self.rescale_y()
            self.plot_event_data(event=None)
            self.canvas.draw_idle()
            
        self.dbtns.on_clicked(toggle_data)
            
        
        ### create additional threshold items
        # min height at edges of ripple event
        self.minThr_line = self.ax.axhline(color='purple', label='Min. height', lw=2, alpha=0.3)
        if 'edge_height' in self.thresholds:
            self.minThr_line.set_ydata([self.thresholds.edge_height]*2)
        self.minThr_line.set_visible(False)
        # min ripple duration
        self.minDur_box = self.ax.axvspan(xmin=-0.5, xmax=0.5, color='darkgreen', lw=2,
                                          label='Min. width', alpha=0.1, zorder=0)
        if 'dur' in self.thresholds:
            self.minDur_box.xy[[0,1,-1], 0] = -self.thresholds.dur/2
            self.minDur_box.xy[[2,3],    0] = self.thresholds.dur/2
        self.minDur_box.set_visible(False)
        self.thres_items.append(self.minThr_line)
        self.thres_items.append(self.minDur_box)
        
        
    def plot_event_data(self, event, twin=None):
        """ Add ripple envelope and duration to event plot """
        super().plot_event_data(event, twin=twin)
        
        if len(self.iev) == 0:
            self.ax.annotate(f'No ripples detected on channel {self.ch}', xy=(0.5,0.8), 
                             xycoords='axes fraction', ha='center', va='center', fontsize=25)
            self.canvas.draw_idle()
            return
        
        feat_items = []
        if self.FLAG == 0:
            # plot mean envelope for primary channel
            if self.SHOW_ENV == True:
                env_arr = ephys.getwaves(self.env, self.iev, self.iwin)
                env_mean = np.nanmean(env_arr, axis=0)
                env_line = self.ax.plot(self.ev_x, env_mean, color='black', lw=2.5, 
                                        ls=':', zorder=5, label='envelope')[0]
                feat_items.append(env_line)
                
            # plot mean ripple duration for primary channel
            if self.SHOW_DUR == True:
                dur_mean = self.DF.dur.mean() / 1000
                dur_x = [-dur_mean/2, dur_mean/2]
                dur_y = [pyfx.Limit(self.ev_y, mode=0)]*2
                dur_line = self.ax.plot(dur_x, dur_y, color='darkgreen', lw=3, 
                                        marker='|', ms=10, mew=3, label='duration')[0]
                feat_items.append(dur_line)
                
        elif self.FLAG == 1:
            # plot ripple envelope
            if self.SHOW_ENV == True:
                env_line = self.ax.plot(self.ev_x, self.env[self.ii], label='envelope')[0]
                feat_items.append(env_line)
            # plot ripple duration
            if self.SHOW_DUR == True:
                dur_x = [self.E.start-self.E.time, self.E.stop-self.E.time+self.dt]
                dur_y = [pyfx.Limit(self.ev_y, mode=0)]*2
                dur_line = self.ax.plot(dur_x, dur_y, color='darkgreen', lw=3, 
                                        marker='|', ms=10, mew=3, label='duration')[0]
                feat_items.append(dur_line)
        self.canvas.draw_idle()
    
    
class IFigDS(IFigEvent):
    """ Figure displaying dentate spike events """
    
    SHOW_HW = False
    SHOW_WH = False
    
    FLAG = 0
    annot_dict = dict(time='{time:.2f} s', amp='{amp:.2f} mV', asym='{asym:+.0f}%', 
                      half_width='{half_width:.2f} ms', width_height='{width_height:.2f} mV')
    
    def create_subplots(self, **kwargs):
        """ Add artists to subplots """
        super().create_subplots(**kwargs)
        
        def toggle_data(label):
            """ Plot raw vs filtered LFP data """
            if label == 'Raw' : self.LFP_arr = self.DATA['raw']
            else              : self.LFP_arr = self.DATA['ds']
            self.LFP = self.LFP_arr[self.ich]
            
            self.rescale_y()
            self.plot_event_data(event=None)
            self.canvas.draw_idle()
            
        self.dbtns.on_clicked(toggle_data)
        
        ### create additional feature artists
        self.hw_line_kw = dict(color='red', lw=3, zorder=0, solid_capstyle='butt', 
                               marker='|', ms=10, mew=3, label='half-width')
        self.wh_line_kw = dict(color='darkgreen', lw=3, zorder=-1,
                               marker='_', ms=10, mew=3, label='half-prom. height')
        
        
    def plot_event_data(self, event, twin=None):
        """ Add waveform height/width measurements to event plot """
        super().plot_event_data(event, twin=twin)
        
        if len(self.iev) == 0:
            self.ax.annotate(f'No dentate spikes detected on channel {self.ch}', xy=(0.5,0.8), 
                             xycoords='axes fraction', ha='center', va='center', fontsize=25)
            self.canvas.draw_idle()
            return
        
        feat_items = []
        if self.FLAG == 0:
            # run scipy.signal peak detection on mean waveform
            ipk = np.argmax(self.ev_y)
            wlen = int(round(self.lfp_fs * self.PARAMS['ds_wlen']))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'some peaks have a prominence of 0')
                warnings.filterwarnings('ignore', r'some peaks have a width of 0')
                pws = scipy.signal.peak_widths(self.ev_y, peaks=[ipk], rel_height=0.5, wlen=wlen)
            hw, wh, istart, istop = np.array(pws).flatten()
            
            # plot mean half-width for primary channel
            if self.SHOW_HW == True:
                hw_x = [(istart-ipk)/self.lfp_fs, (istop-ipk)/self.lfp_fs+self.dt]
                hw_line = self.ax.plot(hw_x, [wh, wh], **self.hw_line_kw)[0] 
                feat_items.append(hw_line)
            
            # plot mean width-height for primary channel
            if self.SHOW_WH == True:
                wh_line = self.ax.plot([0,0], [0,wh], **self.wh_line_kw)[0]
                feat_items.append(wh_line)
                
        elif self.FLAG == 1:
            # plot waveform half-width
            if self.SHOW_HW == True:
                # get waveform start, stop, and peak times (at rel_height=0.5)
                hw_x = [self.E.start-self.E.time, self.E.stop-self.E.time+self.dt]
                hw_y = [self.E.width_height]*2
                hw_line = self.ax.plot(hw_x, hw_y, **self.hw_line_kw)[0] 
                feat_items.append(hw_line)
            
            # plot waveform height at half its max prominence
            if self.SHOW_WH == True:
                wh_line = self.ax.plot([0, 0], [0, self.E.width_height], 
                                       **self.wh_line_kw)[0]
                feat_items.append(wh_line)
            
        self.canvas.draw_idle()


class EventViewPopup(QtWidgets.QDialog):
    """ Popup window containing interactive event plots """
    
    def __init__(self, ch, channels, DF_ALL, fig, parent=None):
        super().__init__(parent)        
        self.ch = ch
        self.channels = channels
        self.DF_ALL = DF_ALL
        self.event_fig = fig
        self.event_fig.setMinimumWidth(int(pyfx.ScreenRect().width() * 0.5))
        
        # initialize settings widget, populate channel dropdown
        self.evw = EventViewWidget(self.DF_ALL.columns, parent=self)
        self.evw.setMaximumWidth(300)
        self.reset_ch_dropdown()
        
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.event_fig)
        self.layout.addWidget(self.evw)
        self.setLayout(self.layout)
        
        # connect view widgets
        self.evw.view_grp.buttonToggled.connect(self.show_hide_plot_items)
        self.evw.chLabel_btn.toggled.connect(self.event_fig.label_ch_data)
        self.evw.plot_mode_bgrp.buttonToggled.connect(self.toggle_plot_mode)
        self.evw.ch_comp_widget.add_btn.clicked.connect(self.compare_channel)
        self.evw.ch_comp_widget.clear_btn.clicked.connect(self.clear_channels)
        self.evw.sort_bgrp.buttonToggled.connect(self.sort_events)
        
        
    def sort_events(self, btn, chk):
        """ Sort individual event waveforms by attribute (e.g. time, amplitude, width) """
        if chk:
            self.event_fig.sort_events(btn.column)
            
    def show_hide_plot_items(self, btn, chk):
        """ Show/hide event detection thresholds on plot """
        # thresholds
        if btn.text() == 'Peak height':
            self.event_fig.hThr_line.set_visible(chk)
        elif btn.text() == 'Min. height':
            self.event_fig.minThr_line.set_visible(chk)
        elif btn.text() == 'Min. width':
            self.event_fig.minDur_box.set_visible(chk)
            
        # features
        elif btn.text() == 'Ripple envelope':
            self.event_fig.SHOW_ENV = bool(chk)
        elif btn.text() == 'Ripple duration':
            self.event_fig.SHOW_DUR = bool(chk)
        elif btn.text() == 'Half-width':
            self.event_fig.SHOW_HW = bool(chk)
        elif btn.text() == 'Half-prom. height':
            self.event_fig.SHOW_WH = bool(chk)
            
        # reference points
        elif btn.text() == 'X (amplitude = 0)':
            self.event_fig.xax_line.set_visible(chk)
        elif btn.text() == 'Y (time = 0)':
            self.event_fig.yax_line.set_visible(chk)
        
        self.event_fig.plot_event_data(event=None)
        
        
    def toggle_plot_mode(self, btn, chk):
        """ Switch between "average" and "individual" viewing modes """
        if not chk:
            return
        mode = btn.group().id(btn)  # 0=average waveform, 1=individual events
        self.event_fig.FLAG = mode
        self.event_fig.rescale_y()
        self.event_fig.plot_event_data(event=None)
        iiterable = bool(mode and len(self.event_fig.iev)>1)
        self.event_fig.iw.idx.set_active(iiterable)    # slider active if mode==1
        self.event_fig.iw.idx.enable(iiterable)
        self.evw.sort_gbox.setEnabled(iiterable) # sort options enabled if mode==1
        self.evw.ch_comp_widget.setEnabled(not bool(mode))  # channel comparison disabled if mode==1
        
    
    def reset_ch_dropdown(self):
        """ Populate channel dropdown with all channels except primary """
        # remove all items
        for i in reversed(range(self.evw.ch_comp_widget.ch_dropdown.count())):
            self.evw.ch_comp_widget.ch_dropdown.removeItem(i)
            
        # repopulate dropdown with all channels, then remove primary channel
        channel_strings = list(np.array(self.channels, dtype='str'))
        self.evw.ch_comp_widget.ch_dropdown.addItems(channel_strings)
        ch_no_events = np.setdiff1d(self.event_fig.channels, np.unique(self.event_fig.DF_ALL.ch))
        model = self.evw.ch_comp_widget.ch_dropdown.model()
        for i,ch in enumerate(self.event_fig.channels):
            if ch in ch_no_events:
                model.item(i).setEnabled(False)  # disable items of channels with no events
        ich = channel_strings.index(str(self.ch))
        self.evw.ch_comp_widget.ch_dropdown.removeItem(ich)
        display_ch = str(self.ch-1) if self.ch > 0 else str(self.ch+1)
        self.evw.ch_comp_widget.ch_dropdown.setCurrentText(display_ch)
        
    def compare_channel(self):
        """ Add other channel events to plot """
        # get selected channel ID, remove from dropdown options
        idx = self.evw.ch_comp_widget.ch_dropdown.currentIndex()
        if not self.evw.ch_comp_widget.ch_dropdown.model().item(idx).isEnabled():
            return
        new_chan = int(self.evw.ch_comp_widget.ch_dropdown.itemText(idx))
        self.evw.ch_comp_widget.ch_dropdown.removeItem(idx)
        # add channel to plotting list, re-plot data
        self.event_fig.CH_ON_PLOT.append(new_chan)
        self.event_fig.label_ch_data(self.evw.chLabel_btn.isChecked())
        self.event_fig.rescale_y()
        self.event_fig.plot_event_data(event=None)
        
        
    def clear_channels(self):
        """ Clear all comparison channels from event plot """
        # clear channel plotting list (except for primary channel)
        self.event_fig.label_ch_data(False)
        self.event_fig.CH_ON_PLOT = [int(self.ch)]
        self.event_fig.label_ch_data(self.evw.chLabel_btn.isChecked())
        self.event_fig.rescale_y()
        self.event_fig.plot_event_data(event=None)
        # reset channel dropdown
        self.reset_ch_dropdown()
    
    def closeEvent(self, event):
        plt.close()
        super().closeEvent(event)
        
    
class EventViewWidget(QtWidgets.QFrame):
    """ Settings widget for popup event window """
    sort_labels = pd.Series(dict(time          = 'Time',
                                  amp          = 'Amplitude',
                                  dur          = 'Duration',
                                  freq         = 'Instantaneous freq',
                                  asym         = 'Asymmetry',
                                  half_width   = 'Half-width',
                                  width_height = 'Half-prom. height'))
    
    def __init__(self, sort_columns, parent=None):
        super().__init__(parent)
        # set widget frame
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setLineWidth(2)
        self.setMidLineWidth(2)
        
        self.vlay = QtWidgets.QVBoxLayout()
        self.vlay.setSpacing(10)
        
        ###   VIEW PLOT ITEMS
        self.view_gbox = QtWidgets.QGroupBox('VIEW')
        self.view_gbox.setStyleSheet(pyfx.dict2ss(QSS.EVENT_SETTINGS_GBOX))
        view_lay = pyfx.InterWidgets(self.view_gbox, 'v')[2]
        view_lay.setSpacing(10)
        ### show/hide thresholds
        self.thres_vbox = QtWidgets.QVBoxLayout()
        self.thres_vbox.setSpacing(5)
        thres_view_lbl = QtWidgets.QLabel('<u>Show thresholds</u>')
        self.thres_vbox.addWidget(thres_view_lbl)
        view_lay.addLayout(self.thres_vbox)
        view_line0 = pyfx.DividerLine(lw=2, mlw=2)
        view_lay.addWidget(view_line0)
        ### show/hide waveform features
        self.feat_vbox = QtWidgets.QVBoxLayout()
        self.feat_vbox.setSpacing(5)
        feat_view_lbl = QtWidgets.QLabel('<u>Show data features</u>')
        self.feat_vbox.addWidget(feat_view_lbl)
        view_lay.addLayout(self.feat_vbox)
        view_line1 = pyfx.DividerLine(lw=2, mlw=2)
        view_lay.addWidget(view_line1)
        ### show/hide X and Y axes
        self.ref_vbox = QtWidgets.QVBoxLayout()
        self.ref_vbox.setSpacing(5)
        ref_view_lbl = QtWidgets.QLabel('<u>Show axes</u>')
        self.ref_vbox.addWidget(ref_view_lbl)
        view_lay.addLayout(self.ref_vbox)
        view_line2 = pyfx.DividerLine(lw=2, mlw=2)
        view_lay.addWidget(view_line2)
        ### misc standalone checkboxes
        misc_hbox1 = QtWidgets.QHBoxLayout()
        misc_hbox1.setSpacing(0)
        self.chLabel_btn = QtWidgets.QCheckBox()
        chLabel_lbl = QtWidgets.QLabel('Highlight data from current channel?')
        chLabel_lbl.setWordWrap(True)
        misc_hbox1.addWidget(self.chLabel_btn)
        misc_hbox1.addWidget(chLabel_lbl)
        view_lay.addLayout(misc_hbox1)
        
        # create non-exclusive button group to handle all checkboxes
        self.view_grp = QtWidgets.QButtonGroup()
        self.view_grp.setExclusive(False)
        self.add_view_btns(['Peak height'], 'threshold')
        self.add_view_btns(['X (amplitude = 0)', 'Y (time = 0)'], 'reference')
        self.vlay.addWidget(self.view_gbox)
        
        line0 = pyfx.DividerLine()
        self.vlay.addWidget(line0)
        
        ###   DATA PLOT ITEMS (SINGLE VS AVERAGED EVENTS)
        self.data_gbox = QtWidgets.QGroupBox('MODE')
        self.data_gbox.setStyleSheet(pyfx.dict2ss(QSS.EVENT_SETTINGS_GBOX))
        data_lay = pyfx.InterWidgets(self.data_gbox, 'v')[2]
        data_lay.setSpacing(10)
        ### buttons for single vs averaged plot mode
        pm_vbox = QtWidgets.QVBoxLayout()
        pm_vbox.setSpacing(5)
        self.plot_mode_bgrp = QtWidgets.QButtonGroup(self.data_gbox)
        self.single_btn = QtWidgets.QPushButton('Single Events')
        self.single_btn.setCheckable(True)
        self.single_btn.setStyleSheet(pyfx.dict2ss(QSS.BOLD_INSET_BTN))
        self.avg_btn = QtWidgets.QPushButton('Averages')
        self.avg_btn.setCheckable(True)
        self.avg_btn.setChecked(True)
        self.avg_btn.setStyleSheet(pyfx.dict2ss(QSS.BOLD_INSET_BTN))
        self.plot_mode_bgrp.addButton(self.avg_btn, 0)
        self.plot_mode_bgrp.addButton(self.single_btn, 1)
        pm_vbox.addWidget(self.single_btn)
        pm_vbox.addWidget(self.avg_btn)
        data_lay.addLayout(pm_vbox)
        data_line0 = pyfx.DividerLine(lw=2, mlw=2)
        data_lay.addWidget(data_line0)
        
        ### channel comparison widget
        self.ch_comp_widget = gi.AddChannelWidget(add_btn_pos='left')
        data_lay.addWidget(self.ch_comp_widget)
        self.vlay.addWidget(self.data_gbox)
        
        line1 = pyfx.DividerLine()
        self.vlay.addWidget(line1)
        
        ###   SORT EVENTS
        self.sort_gbox = QtWidgets.QGroupBox('SORT')
        self.sort_gbox.setStyleSheet(pyfx.dict2ss(QSS.EVENT_SETTINGS_GBOX))
        sort_lay = pyfx.InterWidgets(self.sort_gbox, 'v')[2]
        sort_lay.setSpacing(0)
        self.sort_bgrp = QtWidgets.QButtonGroup(self.sort_gbox)
        sort_params = list(np.intersect1d(sort_columns, self.sort_labels.index.values))
        sort_params.remove('time'); sort_params.insert(0, 'time')  # "time" must be first param
        for i,param in enumerate(sort_params):
            lbl = self.sort_labels.get(param, param)
            btn = QtWidgets.QRadioButton(lbl)
            btn.column = param
            if i==0:
                btn.setChecked(True)
            self.sort_bgrp.addButton(btn)
            sort_lay.addWidget(btn)
        self.sort_gbox.setEnabled(False)
        self.vlay.addWidget(self.sort_gbox)
        
        self.setLayout(self.vlay)
    
    def add_view_btns(self, thres_lbls, category):
        """ Dynamically add viewing buttons for different event elements """
        for tl in thres_lbls:
            chk = QtWidgets.QCheckBox(tl)
            chk.setStyleSheet('QCheckBox {margin-left : 5px}')
            self.view_grp.addButton(chk)
            if category == 'threshold':
                self.thres_vbox.addWidget(chk)
            elif category == 'feature':
                self.feat_vbox.addWidget(chk)
            elif category == 'reference':
                self.ref_vbox.addWidget(chk)

class ChannelSelectionWidget(QtWidgets.QFrame):
    """ Settings widget for main analysis GUI """
    
    noise_signal = QtCore.pyqtSignal(np.ndarray)  # user changed noise channels
    
    def __init__(self, ddir, ntimes, lfp_time, lfp_fs, event_channels, parent=None):
        super().__init__(parent)
        
        self.ddir = ddir
        self.settings_layout = QtWidgets.QVBoxLayout(self)
        self.settings_layout.setSpacing(10)
        
        ###   TOGGLE PROBES   ###
        
        # toggle between probes and shanks
        qlists = []
        for _ in range(2):
            qlist = QtWidgets.QListWidget()
            qlist.setStyleSheet(pyfx.dict2ss(QSS.QLIST))
            qlist.setFocusPolicy(QtCore.Qt.NoFocus)
            qlist.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            qlist.setSizeAdjustPolicy(qlist.AdjustToContents)
            qlists.append(qlist)
        self.probes_qlist, self.shanks_qlist = qlists
        # ignore item deselection (one probe must always be selected)
        fx = lambda q: (len(q.selectedItems())==0, q.item(q.currentRow()))
        fx2 = lambda x,item: item.setSelected(True) if x else None
        self.probes_qlist.itemSelectionChanged.connect(lambda: fx2(*fx(self.probes_qlist)))
        self.shanks_qlist.itemSelectionChanged.connect(lambda: fx2(*fx(self.shanks_qlist)))
        qlist_hlay = QtWidgets.QHBoxLayout()
        qlist_hlay.setContentsMargins(0,0,0,0)
        qlist_hlay.addWidget(self.probes_qlist)
        qlist_hlay.addWidget(self.shanks_qlist)
        #################
        
        self.tab_widget = QtWidgets.QTabWidget(parent=self)
        self.tab_widget.setObjectName('tab_widget')
        #self.tab_widget.setMinimumWidth(25)
        self.tab_widget.setMovable(True)
        
        
        ######################################################
        ######################################################
        ############           TAB 1              ############
        ######################################################
        ######################################################
        
        #  Events tab
        self.tab1 = QtWidgets.QFrame()
        self.tab1.setObjectName('tab1')
        self.tab1.setFrameShape(QtWidgets.QFrame.Box)
        self.tab1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tab1.setLineWidth(2)
        self.tab1.setMidLineWidth(2)
        self.tab_widget.addTab(self.tab1, 'Events')
        self.vlay = QtWidgets.QVBoxLayout(self.tab1)
        self.vlay.setSpacing(10)
        
        ###   FREQUENCY POWER BANDS   ###
        
        self.plot_freq_pwr = gi.ShowHideBtn(init_show=False)
        self.plot_freq_pwr.setAutoDefault(False)
        self.plot_freq_pwr.setStyleSheet(pyfx.dict2ss(QSS.EXPAND_LEFT_BTN))
        #self.plot_freq_pwr.setStyleSheet(pyfx.dict2ss(QSS.FREQ_TOGGLE_BTN))
        self.plot_freq_pwr.setLayoutDirection(QtCore.Qt.LeftToRight)
        
        ###   EVENT WIDGETS   ###
        
        # hilus channel widgets
        self.ds_gbox = gi.EventGroupbox('ds')
        self.hil_input = self.ds_gbox.chan_input
        self.ds_reset = self.ds_gbox.chan_reset
        self.ds_event_btn = self.ds_gbox.chan_event_btn
        self.ds_show = self.ds_gbox.chan_show
        self.ds_arrows = self.ds_gbox.chan_arrows
        self.ds_add = self.ds_gbox.chan_add
        self.ds_show_rm = self.ds_gbox.chan_show_rm
        
        # ripple channel widgets
        self.ripple_gbox = gi.EventGroupbox('swr')
        self.swr_input = self.ripple_gbox.chan_input
        self.swr_reset = self.ripple_gbox.chan_reset
        self.swr_event_btn = self.ripple_gbox.chan_event_btn
        self.swr_show = self.ripple_gbox.chan_show
        self.swr_arrows = self.ripple_gbox.chan_arrows
        self.swr_add = self.ripple_gbox.chan_add
        self.swr_show_rm = self.ripple_gbox.chan_show_rm
        
        # theta channel widgets
        self.theta_gbox = gi.EventGroupbox('theta')
        self.theta_input = self.theta_gbox.chan_input
        self.theta_reset = self.theta_gbox.chan_reset
        self.theta_event_btn = self.theta_gbox.chan_event_btn
        self.theta_event_btn.hide()
        
        # set channels
        self.ch_inputs = [self.theta_input, self.swr_input, self.hil_input]
        self.set_channel_values(*event_channels)
        
        self.vlay.addWidget(self.plot_freq_pwr, stretch=0)
        self.vlay.addWidget(self.ds_gbox, stretch=3)
        self.vlay.addWidget(self.ripple_gbox, stretch=3)
        self.vlay.addWidget(self.theta_gbox, stretch=2)
        
        
        ######################################################
        ######################################################
        ############           TAB 2              ############
        ######################################################
        ######################################################
        
        # Recording tab
        self.tab2 = QtWidgets.QFrame()
        self.tab2.setObjectName('tab2')
        self.tab2.setFrameShape(QtWidgets.QFrame.Box)
        self.tab2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tab2.setLineWidth(2)
        self.tab2.setMidLineWidth(2)
        self.tab_widget.addTab(self.tab2, 'Recording')
        self.vlay2 = QtWidgets.QVBoxLayout(self.tab2)
        self.vlay2.setSpacing(10)
        
        ###   JUMP TO TIMEPOINT OR INDEX   ###
        
        trange = (tmin, tmax) = np.array(pyfx.Edges(lfp_time))
        irange = (imin, imax) = (trange * lfp_fs).astype('int')
        
        self.time_gbox = QtWidgets.QGroupBox()
        time_gbox_lay = QtWidgets.QVBoxLayout(self.time_gbox)
        self.time_w = gi.LabeledWidget(txt='<b><u>Jump to:</u></b>', spacing=5)
        time_lay = QtWidgets.QVBoxLayout(self.time_w.qw)
        time_lay.setContentsMargins(0,0,0,0)
        self.tjump = gi.LabeledSpinbox('Time', double=True, orientation='h', range=trange, spacing=5)
        self.ijump = gi.LabeledSpinbox('Index', orientation='h', range=irange, spacing=5)
        self.tjump.qw.setKeyboardTracking(False)
        self.ijump.qw.setKeyboardTracking(False)
        time_lay.addWidget(self.tjump)
        time_lay.addWidget(self.ijump)
        time_gbox_lay.addWidget(self.time_w)
        
        ###   ANNOTATE NOISE   ###
        
        self.noise_gbox = QtWidgets.QGroupBox()
        noise_layout = QtWidgets.QVBoxLayout(self.noise_gbox)
        noise_lbl = QtWidgets.QLabel('<b><u>Noise Channels</u></b>')
        self.save_noise_btn = QtWidgets.QToolButton()
        self.save_noise_btn.setIcon(QtGui.QIcon(':/icons/save.png'))
        self.save_noise_btn.setIconSize(QtCore.QSize(20,20))
        self.save_noise_btn.setMaximumWidth(self.save_noise_btn.minimumSizeHint().height())
        noise_hdr = QtWidgets.QHBoxLayout()
        noise_hdr.addWidget(noise_lbl)
        # noise_hdr.addWidget(self.save_noise_btn)
        # QList of channels currently designated as "noisy"
        self.noise_qlist = QtWidgets.QListWidget()
        self.noise_qlist.setStyleSheet(pyfx.dict2ss(QSS.QLIST))
        self.noise_qlist.setFocusPolicy(QtCore.Qt.NoFocus)
        self.noise_qlist.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.noise_qlist.setSizeAdjustPolicy(self.noise_qlist.AdjustToContents)
        # noise widget
        self.ch_noise_widget = gi.AddChannelWidget(add_btn_pos='left')
        self.ch_noise_widget.label.hide()
        self.ch_noise_widget.vlayout.setSpacing(10)
        #self.ch_noise_widget.vlayout.setStretchFactor(self.ch_noise_widget.clear_btn, 255)
        #self.ch_noise_widget.vlayout.setStretch(2, 5)
        # dropdown list of the remaining "clean" channels
        self.clean_dropdown = self.ch_noise_widget.ch_dropdown
        self.clean_dropdown.setPlaceholderText('- - -')
        self.clean2noise_btn = self.ch_noise_widget.add_btn
        #self.clean2noise_btn.setEnabled(False)
        self.noise2clean_btn = self.ch_noise_widget.clear_btn
        self.noise2clean_btn.setText('Restore\nchannel(s)')
        self.noise2clean_btn.setEnabled(False)
        policy = self.noise2clean_btn.sizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.MinimumExpanding)
        self.noise2clean_btn.setSizePolicy(policy)
        #self.clean_dropdown.currentIndexChanged.connect(lambda x: self.clean2noise_btn.setEnabled(x>-1))
        self.noise_qlist.itemSelectionChanged.connect(lambda: self.noise2clean_btn.setEnabled(len(self.noise_qlist.selectedItems())>0))
        
        noise_main = QtWidgets.QHBoxLayout()
        noise_main.addWidget(self.noise_qlist)
        noise_main.addWidget(self.ch_noise_widget)
        
        #noise_layout.addLayout(noise_hdr, stretch=0)
        noise_layout.addLayout(noise_hdr, stretch=0)
        #noise_layout.addWidget(noise_lbl)
        noise_layout.addLayout(noise_main, stretch=2)
        #noise_layout.addWidget(self.noise_qlist)
        #noise_layout.addWidget(self.ch_noise_widget)
        
        ###   RECORDING NOTES   ###
        
        self.notes_gbox = QtWidgets.QGroupBox()
        notes_layout = QtWidgets.QVBoxLayout(self.notes_gbox)
        notes_lbl = QtWidgets.QLabel('<b><u>NOTES</u></b>')
        self.save_notes_btn = QtWidgets.QToolButton()
        self.save_notes_btn.setIcon(QtGui.QIcon(':/icons/save.png'))
        self.save_notes_btn.setIconSize(QtCore.QSize(20,20))
        self.save_notes_btn.setMaximumWidth(self.save_notes_btn.minimumSizeHint().height())
        notes_hdr = QtWidgets.QHBoxLayout()
        notes_hdr.addWidget(notes_lbl)
        notes_hdr.addWidget(self.save_notes_btn)
        self.notes_qedit = QtWidgets.QTextEdit()
        # load notes
        notes_txt = ephys.read_notes(Path(self.ddir, 'notes.txt'))
        self.notes_qedit.setPlainText(notes_txt)
        self.last_saved_notes = str(notes_txt)
        notes_layout.addLayout(notes_hdr)
        notes_layout.addWidget(self.notes_qedit)
        self.save_notes_btn.clicked.connect(self.export_notes)
        
        self.vlay2.addWidget(self.time_gbox, stretch=0)
        self.vlay2.addWidget(self.noise_gbox, stretch=0)
        self.vlay2.addWidget(self.notes_gbox, stretch=2)
        
        # save changes
        bbox = QtWidgets.QVBoxLayout()
        bbox.setSpacing(2)
        self.save_btn = QtWidgets.QPushButton('  Save  ')
        self.save_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.save_btn.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.save_btn.setStyleSheet('QPushButton {padding : 5px 20px;}')
        self.save_btn.setDefault(False)
        self.save_btn.setAutoDefault(False)
        self.debug_btn = QtWidgets.QPushButton('  debug  ')
        self.debug_btn.setDefault(False)
        self.debug_btn.setAutoDefault(False)
        bbox.addWidget(self.save_btn)
        #bbox.addWidget(self.debug_btn)
        self.settings_layout.addLayout(qlist_hlay, stretch=0)
        self.settings_layout.addWidget(self.tab_widget, stretch=2)
        self.settings_layout.addLayout(bbox, stretch=0)
    
    def set_channel_values(self, theta_chan, ripple_chan, hil_chan):
        """ Update event channel inputs """
        pyfx.stealthy(self.theta_input, theta_chan)
        pyfx.stealthy(self.swr_input, ripple_chan)
        pyfx.stealthy(self.hil_input, hil_chan)
    
    def get_item_channels(self, qwidget):
        """ Get all items in a QListWidget or a QComboBox
            Returns indexes, item objects, and integer channels """
        if type(qwidget) == QtWidgets.QListWidget:
            qitems = [qwidget.item(i) for i in range(qwidget.count())]
        elif type(qwidget) in [QtWidgets.QComboBox, gi.ComboBox]:
            qitems = [qwidget.model().item(i) for i in range(qwidget.count())]
        qch = np.array([int(item.text()) for item in qitems], dtype='int')
        return np.arange(len(qitems)), np.array(qitems), qch

    def clean2noise(self):
        """ Label current channel in dropdown menu as "noisy" """
        if self.clean_dropdown.currentIndex() < 0:
            return
        ch = int(self.clean_dropdown.currentText())
        # remove current channel from dropdown, add to noise list
        self.clean_dropdown.removeItem(self.clean_dropdown.currentIndex())
        self.clean_dropdown.setCurrentIndex(-1)
        qlist_ch = self.get_item_channels(self.noise_qlist)[-1]
        if ch not in qlist_ch:
            self.noise_qlist.insertItem(bisect.bisect(qlist_ch, ch), str(ch))
        self.emit_noise_signal()
    
    def noise2clean(self):
        """ Restore selected channel(s) in noise list as "clean" """
        dropdown_ch = self.get_item_channels(self.clean_dropdown)[-1]
        tup_list = [*zip(*self.get_item_channels(self.noise_qlist))][::-1]
        # remove each selected channel from noise list and add to dropdown
        for i,item,ch in tup_list:
            if item.isSelected():
                self.noise_qlist.takeItem(i)
                self.clean_dropdown.insertItem(bisect.bisect(dropdown_ch, ch), str(ch))
        self.emit_noise_signal()
    
    @QtCore.pyqtSlot(int, np.ndarray)
    def update_noise_from_plot(self, ch, noise_train):
        """ User switches a channel from clean to noisy (or vice versa) in plot """
        is_noise = bool(noise_train[ch])
        if is_noise:
            # set current channel in dropdown, click button to convert to noise
            self.clean_dropdown.setCurrentText(str(ch))
            self.clean2noise_btn.click()
        else:
            # select current channel in noise list, click button to restore as clean
            for (i,_,qch) in [*zip(*self.get_item_channels(self.noise_qlist))]:
                self.noise_qlist.item(i).setSelected(bool(ch == qch))
            self.noise2clean_btn.click()
    
    def init_shanks(self, shank_list, ishank):
        """ Update the available shanks to match the current probe """
        self.shanks_qlist.blockSignals(True)
        self.shanks_qlist.clear()
        items = [f'shank {i}' for i in range(len(shank_list))]
        self.shanks_qlist.addItems(items)
        self.shanks_qlist.setCurrentRow(ishank)
        self.shanks_qlist.blockSignals(False)
        self.shanks_qlist.setVisible(len(shank_list) > 1)
        # adjust min/max event channel
        shank_channels = shank_list[ishank].get_indices()
        self.set_event_range(shank_channels)
        
    def set_event_range(self, shank_channels):
        """ Restrict event channel inputs to the current shank  """
        for sbox in self.ch_inputs:
            sbox.blockSignals(True)
            sbox.setRange(min(shank_channels), max(shank_channels))
            sbox.blockSignals(False)
    
    def set_noise_channels(self, noise_train):
        """ Update noise channel widgets """
        noise_channels = np.nonzero(noise_train)[0]
        clean_channels = np.setdiff1d(np.arange(len(noise_train)), noise_channels)
        self.noise_qlist.clear()
        self.clean_dropdown.clear()
        self.noise_qlist.addItems(noise_channels.astype(str))
        self.clean_dropdown.addItems(clean_channels.astype(str))
        self.disable_event_noise()
        self.emit_noise_signal()
    
    def disable_event_noise(self):
        """ Block current event channels from being annotated as "noise" """
        event_channels = [int(item.value()) for item in self.ch_inputs]
        # disable current event channels in dropdown menu, enable the rest 
        for i,item,ch in zip(*self.get_item_channels(self.clean_dropdown)):
            item.setEnabled(ch not in event_channels)
            if i == self.clean_dropdown.currentIndex() and ch in event_channels:
                # reset dropdown if current item is an event channel
                self.clean_dropdown.setCurrentIndex(-1)

    def emit_noise_signal(self):
        """ Emit signal with updated noise train """
        qlist_ch = self.get_item_channels(self.noise_qlist)[-1]
        dropdown_ch = self.get_item_channels(self.clean_dropdown)[-1]
        noise_train = np.zeros(len(qlist_ch) + len(dropdown_ch), dtype='int')
        noise_train[qlist_ch] = 1
        self.noise_signal.emit(noise_train)
    
    # self-contained notes saving
    def export_notes(self):
        """ Save contents of the "Notes" field to disk """
        txt = self.notes_qedit.toPlainText()
        ephys.write_notes(Path(self.ddir,'notes.txt'), txt)
        print('Notes saved!')
        # keep track of last save point, use for warning message
        self.last_saved_notes = str(txt)
        
        
class ChannelSelectionWindow(QtWidgets.QDialog):
    """ Main channel selection GUI """
    
    ch_signal = QtCore.pyqtSignal(int, int, int)  # user changed event channel
    
    def __init__(self, ddir, iprb=0, ishank=0, parent=None):
        super().__init__(parent)
        qrect = pyfx.ScreenRect(perc_width=0.9)
        self.setGeometry(qrect)
        
        self.init_data(ddir, iprb, ishank)
        self.init_figs()
        self.gen_layout()
        self.connect_signals()
        
        # update window title
        folder = os.path.basename(self.ddir)
        parent_folder = os.path.basename(os.path.dirname(self.ddir))
        self.setWindowTitle(str(os.path.join(parent_folder, folder)))
    
    def init_data(self, ddir, iprb, ishank=0):
        """ Initialize all recording variables """
        self.ddir = ddir
        self.probe_group = ephys.read_probe_group(ddir)
        self.probe_list = self.probe_group.probes
        self.iprobes = list(range(len(self.probe_list)))
        self.probe_shank_list = []
        for prb in self.probe_list:
            # initialize list of separate shanks within probe
            shank_list = prb.get_shanks()
            self.probe_shank_list.append(shank_list)
        ####################
        
        # load params
        self.PARAMS = ephys.load_recording_params(ddir)
        
        # load LFP signals, event dfs, and detection thresholds for all probes
        dmode = dp.validate_processed_ddir(ddir)
        if dmode == 1: # load HDF5 files
            self.FF = h5py.File(Path(ddir, 'DATA.hdf5'), 'r+')
            self.lfp_fs = self.FF.attrs['lfp_fs']
            self.lfp_time = ephys.load_h5_array(self.FF, 'lfp_time', in_memory=True)
            self.data_list = [ephys.load_h5_lfp(self.FF, iprb=i) for i in self.iprobes]
            self.swr_list = [ephys.load_h5_event_df(self.FF, 'swr', i)[0] for i in self.iprobes]
            self.ds_list = [ephys.load_h5_event_df(self.FF, 'ds', i)[0] for i in self.iprobes]
            self.threshold_list = [ephys.load_h5_thresholds(self.FF, i) for i in self.iprobes]
            self.noise_list = [ephys.load_h5_array(self.FF, 'NOISE', i, in_memory=True) for i in self.iprobes]
            self.std_list = [ephys.load_h5_df(self.FF, 'STD', i) for i in self.iprobes]
        elif dmode == 2:  # load NPZ and CSV files
            self.FF = None
            self.data_list, self.lfp_time, self.lfp_fs = ephys.load_lfp(ddir, '', -1)
            self.swr_list = [x[0] for x in ephys.load_csv_event_dfs(ddir, 'swr', -1)]
            self.ds_list = [x[0] for x in ephys.load_csv_event_dfs(ddir, 'ds', -1)]
            self.threshold_list = list(np.load(Path(ddir, 'THRESHOLDS.npy'), allow_pickle=True))
            self.noise_list = ephys.load_noise_channels(ddir, -1)
            self.std_list = ephys.csv2list(ddir, 'channel_bp_std')
        self.NTS = len(self.lfp_time)
        self.aux_array = ephys.load_aux(ddir)
        if len(self.aux_array) > 0:
            self.AUX = self.aux_array[0]
            self.AUX_TRAIN = np.array(self.AUX > 0.2, dtype='int')
        else:
            self.AUX = np.full(self.NTS, np.nan)
            self.AUX_TRAIN = np.full(self.NTS, np.nan)
        # load individual event channels for each probe/shank combination
        self.event_channel_list = [ephys.load_event_channels(ddir, i) for i in self.iprobes]
        for i,l in enumerate(self.event_channel_list):
            for ii,ll in enumerate(l):
                if ll == [None,None,None]:
                    shank_ch = self.probe_list[i].get_shanks()[ii].get_indices()
                    STD = self.std_list[i].loc[shank_ch]
                    DS_MEAN = ephys.get_mean_event_df(self.ds_list[i], self.std_list[i]).loc[shank_ch]
                    rel_noise_idx = np.nonzero(self.noise_list[i][shank_ch])[0]
                    itheta = ephys.estimate_theta_chan(STD, noise_idx=rel_noise_idx)
                    iripple = ephys.estimate_ripple_chan(STD, noise_idx=rel_noise_idx)
                    ihil = ephys.estimate_hil_chan(DS_MEAN, noise_idx=rel_noise_idx)
                    llist = [shank_ch[iii] for iii in [itheta, iripple, ihil]]
                    self.event_channel_list[i][ii] = llist
        self.orig_event_channel_list = deepcopy(self.event_channel_list)
        
        # initialize probe data
        self.load_probe_data(iprb, ishank)

    def load_probe_data(self, iprb, ishank):
        """ Update data for selected probe """
        self.iprb = iprb
        self.probe = self.probe_list[iprb]
        self.DATA = self.data_list[iprb]
        self.SWR_ALL = pd.DataFrame(self.swr_list[iprb])
        self.DS_ALL = pd.DataFrame(self.ds_list[iprb])
        self.SWR_THRES, self.DS_THRES = dict(self.threshold_list[iprb]).values()
        self.STD = pd.DataFrame(self.std_list[iprb])
        self.NOISE_TRAIN = np.array(self.noise_list[iprb])
        self.probe_shanks = self.probe_shank_list[iprb]
        self.probe_event_channels = list(self.event_channel_list[iprb])
        self.load_shank_data(ishank)
        ##############
    
    def load_shank_data(self, ishank):
        """ Update data for selected shank """
        self.ishank = ishank
        self.shank = self.probe_shanks[ishank]
        # load event channels for current shank
        self.event_channels = list(self.probe_event_channels[ishank])
        self.theta_chan, self.ripple_chan, self.hil_chan = self.event_channels
        self.orig_theta_chan, self.orig_ripple_chan, self.orig_hil_chan = self.orig_event_channel_list[self.iprb][ishank]
        
    def SWITCH_THE_PROBE(self, idx):
        """ Update main plot and sidebar with a different probe """
        self.load_probe_data(idx, ishank=0)
        kwargs = dict(shank=self.shank, DATA=self.DATA, event_channels=self.event_channels, 
                      DS_ALL=self.DS_ALL, SWR_ALL=self.SWR_ALL, STD=self.STD, 
                      NOISE_TRAIN=self.NOISE_TRAIN, SWR_THRES=self.SWR_THRES)
        self.main_fig.switch_the_probe(**kwargs)
        
        self.widget.init_shanks(self.probe_shanks, self.ishank)
        ##############
        self.widget.set_channel_values(*self.event_channels)
        self.widget.set_noise_channels(self.NOISE_TRAIN)
        
    def SWITCH_THE_SHANK(self, idx):
        """ Update main plot and sidebar with a different shank on the same probe """
        self.load_shank_data(idx)
        self.main_fig.switch_the_shank(shank=self.shank, event_channels=self.event_channels)
        
        self.widget.set_event_range(self.shank.get_indices())
        self.widget.set_channel_values(*self.event_channels)
    
    def init_figs(self):
        """ Create main figure, initiate event channel update """
        kwargs = dict(shank=self.shank, DATA=self.DATA, lfp_time=self.lfp_time, lfp_fs=self.lfp_fs, 
                      PARAMS=self.PARAMS, event_channels=self.event_channels, 
                      DS_ALL=self.DS_ALL, SWR_ALL=self.SWR_ALL, STD=self.STD, 
                      NOISE_TRAIN=self.NOISE_TRAIN, SWR_THRES=self.SWR_THRES,
                      AUX=self.AUX)
        
        # set up figures
        self.main_fig = IFigLFP(**kwargs)
        sns.despine(self.main_fig.fig)
        sns.despine(self.main_fig.fig_freq)
            
        
    def gen_layout(self):
        """ Set up layout """
        # create channel selection widget, initialize values
        self.widget = ChannelSelectionWidget(self.ddir, self.NTS, self.lfp_time, self.lfp_fs, self.event_channels)
        self.widget.setStyleSheet('QWidget:focus {outline : none;}')
        self.widget.setMaximumWidth(300)
        self.widget.tab_widget.setCurrentIndex(0)
        
        # update probe list in settings widget
        items = [f'probe {i}' for i in range(len(self.probe_list))]
        self.widget.probes_qlist.addItems(items)
        self.widget.probes_qlist.setCurrentRow(self.iprb)
        row_height = self.widget.probes_qlist.sizeHintForRow(0)
        self.widget.probes_qlist.setMaximumHeight(row_height * 2 + 5)
        # update shank list in settings widget
        self.widget.shanks_qlist.setMaximumHeight(row_height * 2 + 5)
        self.widget.init_shanks(self.probe_shanks, self.ishank)
        
        # update noise channel list in settings widget
        self.widget.set_noise_channels(self.NOISE_TRAIN)
        
        # set up layout
        self.centralWidget = QtWidgets.QWidget()
        self.centralLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.centralLayout.setContentsMargins(5,5,5,5)
        
        self.centralLayout.addWidget(self.main_fig)
        self.centralLayout.addWidget(self.widget)
        self.setLayout(self.centralLayout)
        
        
    def connect_signals(self):
        """ Connect GUI inputs """
        self.widget.probes_qlist.currentRowChanged.connect(self.SWITCH_THE_PROBE)
        self.widget.shanks_qlist.currentRowChanged.connect(self.SWITCH_THE_SHANK)
        # updated event channel inputs
        for item in self.widget.ch_inputs:
            item.valueChanged.connect(self.emit_ch_signal)
        self.ch_signal.connect(self.update_event_channels)
        # reset event channels to auto
        self.widget.theta_reset.clicked.connect(lambda x: self.reset_ch('theta'))
        self.widget.swr_reset.clicked.connect(lambda x: self.reset_ch('swr'))
        self.widget.ds_reset.clicked.connect(lambda x: self.reset_ch('ds'))
        # updated noise channels
        self.main_fig.updated_noise_signal.connect(self.widget.update_noise_from_plot)
        self.widget.clean2noise_btn.clicked.connect(self.widget.clean2noise)
        self.widget.noise2clean_btn.clicked.connect(self.widget.noise2clean)
        self.widget.noise_signal.connect(self.update_noise_channels)
        
        # view popup windows
        self.widget.swr_event_btn.clicked.connect(self.view_swr)
        self.widget.ds_event_btn.clicked.connect(self.view_ds)
        # show event lines
        self.widget.ds_show.toggled.connect(self.show_hide_events)
        self.widget.ds_show_rm.toggled.connect(self.show_hide_events)
        self.widget.swr_show.toggled.connect(self.show_hide_events)
        self.widget.swr_show_rm.toggled.connect(self.show_hide_events)
        # show given index/timepoint
        self.widget.tjump.qw.valueChanged.connect(lambda x: self.main_fig.point_jump(x, 't'))
        self.widget.ijump.qw.valueChanged.connect(lambda x: self.main_fig.point_jump(x, 'i'))
        # show next/previous event
        self.widget.ds_arrows.bgrp.idClicked.connect(lambda x: self.main_fig.event_jump(x, 'ds'))
        self.widget.swr_arrows.bgrp.idClicked.connect(lambda x: self.main_fig.event_jump(x, 'swr'))
        # toggle event type to add on double-click
        self.widget.ds_add.toggled.connect(lambda x: self.toggle_event_add(x, 'ds'))
        self.widget.swr_add.toggled.connect(lambda x: self.toggle_event_add(x, 'swr'))
        
        # show frequency band plots
        self.widget.plot_freq_pwr.toggled.connect(self.toggle_main_plot)
        # save event channels
        self.widget.save_btn.clicked.connect(self.save_channels)
        self.widget.debug_btn.clicked.connect(self.debug)
        self.widget.probes_qlist.setFocus(True)
        # update event datasets
        self.main_fig.edit_events_signal.connect(self.edit_events_slot)
        
    def emit_ch_signal(self, new_chan):
        """ Emit signal with all 3 current event channels """
        event_channels = [int(item.value()) for item in self.widget.ch_inputs]
        shank_channels = self.shank.get_indices()
        clean_ch = shank_channels[np.where(self.NOISE_TRAIN[shank_channels]==0)[0]]
        ichanged = event_channels.index(new_chan)
        prev_chan = int(self.event_channels[ichanged])
        if new_chan not in clean_ch: # prevent user from selecting noisy channel
            distances = [*map(lambda x:abs(x-new_chan), clean_ch)]
            closest_ch = clean_ch[np.where(distances==np.min(distances))[0]]
            if len(closest_ch)==2:
                closest_ch = [closest_ch[max(np.sign(new_chan-prev_chan), 0)]]
            event_channels[ichanged] = int(closest_ch[0])
            pyfx.stealthy(self.widget.ch_inputs[ichanged], int(closest_ch[0]))
        self.ch_signal.emit(*event_channels)
        self.widget.disable_event_noise()
    
    @QtCore.pyqtSlot(str, pd.DataFrame)
    def edit_events_slot(self, event, DF_ALL):
        """ Update event dataset from user input """
        if event == 'ds':
            self.ds_list[self.iprb] = DF_ALL
            self.DS_ALL = self.ds_list[self.iprb]
        elif event == 'swr':
            self.swr_list[self.iprb] = DF_ALL
            self.SWR_ALL = self.swr_list[self.iprb]
    
    def toggle_event_add(self, chk, event):
        """ Choose whether to add DS or SWR event on double-click """
        if event == 'ds' and chk:
            self.widget.swr_add.setChecked(False)
        elif event == 'swr' and chk:
            self.widget.ds_add.setChecked(False)
        self.main_fig.ADD_DS = bool(self.widget.ds_add.isChecked())
        self.main_fig.ADD_SWR = bool(self.widget.swr_add.isChecked())
        
    def show_hide_events(self):
        """ Set event markers visible or hidden """
        show_ds = bool(self.widget.ds_show.isChecked())         # show DS events
        show_ds_rm = bool(self.widget.ds_show_rm.isChecked())   # show deleted DS events
        show_swr = bool(self.widget.swr_show.isChecked())       # show SWR events
        show_swr_rm = bool(self.widget.swr_show_rm.isChecked()) # show deleted SWR events
        self.main_fig.SHOW_DS = bool(show_ds)
        self.main_fig.SHOW_DS_RM = bool(show_ds_rm)
        self.main_fig.SHOW_SWR = bool(show_swr)
        self.main_fig.SHOW_SWR_RM = bool(show_swr_rm)
        self.widget.ds_arrows.setEnabled(show_ds)
        if not show_ds: self.widget.ds_add.setChecked(False)
        self.widget.ds_add.setEnabled(show_ds)
        self.widget.ds_show_rm.setEnabled(show_ds)
        self.widget.swr_arrows.setEnabled(show_swr)
        if not show_swr: self.widget.swr_add.setChecked(False)
        self.widget.swr_add.setEnabled(show_swr)
        self.widget.swr_show_rm.setEnabled(show_swr)
        
        self.main_fig.plot_lfp_data()
        
    def toggle_main_plot(self, chk):
        """ Expand/hide frequency band plots """
        self.main_fig.canvas_freq.setVisible(chk)
        self.main_fig.plot_lfp_data()
        
    def update_noise_channels(self, noise_train):
        """ Pass updated noise channel annotation to figure """
        self.main_fig.NOISE_TRAIN = np.array(noise_train)
        self.NOISE_TRAIN = self.main_fig.NOISE_TRAIN
        self.noise_list[self.iprb] = np.array(self.NOISE_TRAIN)
        self.main_fig.plot_lfp_data()
        
    def update_event_channels(self, a, b, c):
        """ Pass updated event channels from settings widget to figure """
        self.main_fig.channel_changed(a,b,c)
        self.event_channels = self.main_fig.event_channels
        self.probe_event_channels[self.ishank] = list(self.event_channels)
        self.event_channel_list[self.iprb] = list(self.probe_event_channels)
        
    def reset_ch(self, k):
        """ User resets event channel to its original value """
        if k == 'theta':
            self.widget.theta_input.setValue(self.orig_theta_chan)
        elif k =='swr':
            self.widget.swr_input.setValue(self.orig_ripple_chan)
        elif k == 'ds':
            self.widget.hil_input.setValue(self.orig_hil_chan)
        
    def view_swr(self):
        """ Launch ripple analysis popup """
        RIPPLE_CHAN = self.widget.swr_input.value()
        shank_channels = np.array(self.shank.get_indices())
        irows = np.nonzero(np.in1d(self.SWR_ALL.ch, shank_channels))[0]
        SWR_ALL = self.SWR_ALL.iloc[irows, :]
        DATA = {k:d[shank_channels, :] for k,d in self.DATA.items()}
        # initialize figure
        kwargs = dict(ch=RIPPLE_CHAN, channels=shank_channels, DF_ALL=SWR_ALL, DATA=DATA, lfp_time=self.lfp_time,
                      lfp_fs=self.lfp_fs, PARAMS=self.PARAMS, EVENT_NOISE_TRAIN=self.NOISE_TRAIN[shank_channels],
                      thresholds=self.SWR_THRES[RIPPLE_CHAN])
        self.swr_fig = IFigSWR(**kwargs)
        sns.despine(self.swr_fig.fig)
        # initialize popup
        self.swr_popup = EventViewPopup(ch=RIPPLE_CHAN, channels=shank_channels, DF_ALL=SWR_ALL, fig=self.swr_fig, parent=self)
        self.swr_popup.setWindowTitle(f'Sharp-wave ripples on channel {RIPPLE_CHAN}')
        self.swr_popup.evw.add_view_btns(['Min. height', 'Min. width'], 'threshold')
        self.swr_popup.evw.add_view_btns(['Ripple envelope', 'Ripple duration'], 'feature')
        # set slider states, initialize plot
        n = int(self.swr_fig.DF_MEAN.loc[RIPPLE_CHAN].n_valid/2)
        self.swr_fig.iw.idx.set_val(n)
        self.swr_fig.iw.idx.set_active(False)
        self.swr_fig.iw.idx.enable(False)
        self.swr_fig.plot_event_data(event=n, twin=0.2)
        self.swr_fig.rescale_y()
        # disable settings widgets if channel has no events
        if len(self.swr_fig.iev) == 0:
            self.swr_popup.evw.view_gbox.setEnabled(False)
            self.swr_popup.evw.data_gbox.setEnabled(False)
            self.swr_popup.evw.sort_gbox.setEnabled(False)
            self.swr_fig.iw.twin.enable(False)
            self.swr_fig.iw.ywin.enable(False)
        self.swr_popup.show()
    
    def view_ds(self):
        """ Launch DS analysis popup """
        HIL_CHAN = self.widget.hil_input.value()
        shank_channels = np.array(self.shank.get_indices())
        irows = np.nonzero(np.in1d(self.DS_ALL.ch, shank_channels))[0]
        DS_ALL = self.DS_ALL.iloc[irows, :]
        DATA = {k:d[shank_channels, :] for k,d in self.DATA.items()}
        # initialize figure
        kwargs = dict(ch=HIL_CHAN, channels=shank_channels, DF_ALL=DS_ALL, DATA=DATA, lfp_time=self.lfp_time,
                      lfp_fs=self.lfp_fs, PARAMS=self.PARAMS, EVENT_NOISE_TRAIN=self.NOISE_TRAIN[shank_channels],
                      thresholds=self.DS_THRES[HIL_CHAN])
        self.ds_fig = IFigDS(**kwargs)
        sns.despine(self.ds_fig.fig)
        # initialize popup
        self.ds_popup = EventViewPopup(ch=HIL_CHAN, channels=shank_channels, DF_ALL=DS_ALL, fig=self.ds_fig, parent=self)
        self.ds_popup.setWindowTitle(f'Dentate spikes on channel {HIL_CHAN}')
        self.ds_popup.evw.add_view_btns(['Half-width', 'Half-prom. height'], 'feature')
        # set slider states, initialize plot
        n = int(self.ds_fig.DF_MEAN.loc[HIL_CHAN].n_valid/2)
        self.ds_fig.iw.idx.set_val(n)
        self.ds_fig.iw.idx.set_active(False)
        self.ds_fig.iw.idx.enable(False)
        self.ds_fig.plot_event_data(event=n, twin=0.2)
        self.ds_fig.rescale_y()
        # disable settings widgets if channel has no events
        if len(self.ds_fig.iev) == 0:
            self.ds_popup.evw.view_gbox.setEnabled(False)
            self.ds_popup.evw.data_gbox.setEnabled(False)
            self.ds_popup.evw.sort_gbox.setEnabled(False)
            self.ds_fig.iw.twin.enable(False)
            self.ds_fig.iw.ywin.enable(False)
        self.ds_popup.show()
    
    def save_channels(self):
        """ Save event channels, datasets, and noise channels for current probe/shank """
        # save event dataframes for the current probe
        if self.FF is not None:
            ephys.save_h5_df(self.SWR_ALL.drop('ch', axis=1), self.FF, name='ALL_SWR', iprb=self.iprb)
            ephys.save_h5_df(self.DS_ALL.drop('ch', axis=1), self.FF, name='ALL_DS', iprb=self.iprb)
            self.FF[f'{self.iprb}']['NOISE'][:] = np.array(self.noise_list[self.iprb]) # save noise annotation
        else:
            for event,DF in zip(['swr','ds'], [self.SWR_ALL, self.DS_ALL]):
                ephys.save_csv_event_dfs(self.ddir, event, DF, iprb=self.iprb)
            np.save(Path(self.ddir, 'noise_channels.npy'), self.noise_list)
        
        # save event channels for the current probe/shank
        event_channels = list(self.event_channel_list[self.iprb][self.ishank]) # 3 event channels for current shank
        self.orig_event_channel_list[self.iprb][self.ishank] = event_channels
        ephys.save_event_channels(self.ddir, iprb=self.iprb, ishank=self.ishank, 
                                  new_channels=event_channels)
        
        # save ripple/DS event dataframes
        theta_chan, ripple_chan, hil_chan = event_channels
        if hil_chan in self.DS_ALL.ch:
            DS_DF = pd.DataFrame(self.DS_ALL.loc[hil_chan])
            DS_DF = DS_DF[DS_DF['is_valid']==1].reset_index(drop=True)
        else:
            DS_DF = pd.DataFrame(columns=self.DS_ALL.columns)
        DS_DF.to_csv(Path(self.ddir, f'DS_DF_probe{self.iprb}-shank{self.ishank}'), index_label=False)
        if ripple_chan in self.SWR_ALL.ch:
            SWR_DF = pd.DataFrame(self.SWR_ALL.loc[ripple_chan])
            SWR_DF = SWR_DF[SWR_DF['is_valid']==1].reset_index(drop=True)
        else:
            SWR_DF = pd.DataFrame(columns=self.SWR_ALL.columns)
        SWR_DF.to_csv(Path(self.ddir, f'SWR_DF_probe{self.iprb}-shank{self.ishank}'), index_label=False)
        
        # pop-up messagebox appears when save is complete
        res = gi.MsgboxSave('Event channels saved!\nExit window?', parent=self).exec()
        if res == QtWidgets.QMessageBox.Yes:
            self.widget.export_notes()  # automatically save notes
            self.accept()
        
    def check_unsaved_notes(self):
        """ Check for notes updates, prompt user to save changes """
        a = self.widget.notes_qedit.toPlainText()
        b = self.widget.last_saved_notes
        if a==b: 
            return True  # continue closing
        else:
            msg = 'Save changes to your notes?'
            res = gi.MsgboxWarning.unsaved_changes_warning(msg=msg, sub_msg='', parent=self)
            if res == False:
                return False  # "Cancel" to abort closing attempt
            else:
                if res == -1: # save notes before closing
                    self.widget.export_notes()
                return True   # close dialog
        
    def reject(self):
        """ Close window without saving changes """
        if self.check_unsaved_notes():
            super().reject()
    
    def closeEvent(self, event):
        plt.close()
        if self.FF is not None:
            self.FF.close()
        super().closeEvent(event)
    
    def debug(self):
        pdb.set_trace()
        
def main(ddir=''):
    """ Run channel selection GUI """
    # allow user to select processed data folder
    if not dp.validate_processed_ddir(ddir):
        ddir = ephys.select_directory(init_ddir=ephys.base_dirs()[0], 
                                      title='Select recording folder')
        if not ddir or not dp.validate_processed_ddir(ddir):
            return None, None
    
    # launch window
    w = ChannelSelectionWindow(ddir, 0)
    w.show()
    w.raise_()
    w.exec()
    return w, ddir
        
if __name__ == '__main__':
    app = pyfx.qapp()
    w, ddir = main()
