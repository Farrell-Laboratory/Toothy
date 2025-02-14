#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:45:06 2024

@author: amandaschott
"""
import os
from pathlib import Path
import scipy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from copy import deepcopy
import bisect
import quantities as pq
from PyQt5 import QtWidgets, QtCore, QtGui
import probeinterface as prif
import pdb
# custom modules
import pyfx
import ephys
import gui_items as gi
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
    """ Main channel selection figure; scrollable LFPs with optional event markers """
    updated_noise_signal = QtCore.pyqtSignal(int, np.ndarray)
    
    SHOW_DS = True
    SHOW_SWR = True
    ADD_DS = False
    ADD_SWR = False
    SHOW_DS_RM = False
    SHOW_SWR_RM = False
    SHOW_SZR = False
    
    def __init__(self, DATA, lfp_time, lfp_fs, PARAMS, **kwargs):
        super().__init__()
        self.DATA = DATA
        self.lfp_time = lfp_time
        self.lfp_fs = lfp_fs
        self.PARAMS = PARAMS
        self.channels = np.arange(self.DATA['raw'].shape[0])
        
        twin = kwargs.get('twin', 1)
        event_channels = kwargs.get('event_channels', [0,0,0])
        self.ch_cmap = pd.Series(['blue', 'green', 'red'], index=event_channels)
        
        self.probe = kwargs.get('probe', None)
        self.AUX = kwargs.get('AUX', np.full(len(self.lfp_time), np.nan))
        self.DS_ALL = kwargs.get('DS_ALL', None)
        self.SWR_ALL = kwargs.get('SWR_ALL', None)
        self.STD = kwargs.get('STD', None)
        self.NOISE_TRAIN = kwargs.get('NOISE_TRAIN', None)
        self.seizures = kwargs.get('seizures', [])
        self.SWR_THRES = kwargs.get('SWR_THRES')
        self.init_event_items()  # create event indexes/trains/etc
        
        # create subplots and interactive widgets
        self.plot_height = pyfx.ScreenRect(perc_height=0.75).height()
        self.create_subplots(twin=twin)
        
        self.fig.set_tight_layout(True)
        self.fig_w.set_tight_layout(True)
        self.fig_freq.set_tight_layout(True)
        
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setOrientation(QtCore.Qt.Vertical)
        self.toolbar.setMaximumWidth(30)
        self.canvas_freq = FigureCanvas(self.fig_freq)
        self.canvas_w = FigureCanvas(self.fig_w)
        self.canvas_w.setMaximumHeight(80)
        
        self.connect_mpl_widgets()
        
        self.plot_row = QtWidgets.QWidget()
        self.plot_row.setFixedHeight(self.plot_height)
        self.plot_row_hlay = QtWidgets.QHBoxLayout(self.plot_row)
        self.plot_row_hlay.addWidget(self.toolbar, stretch=0)
        self.plot_row_hlay.addWidget(self.canvas, stretch=5)
        self.plot_row_hlay.addWidget(self.canvas_freq, stretch=3)
        
        self.qscroll = QtWidgets.QScrollArea()
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setWidget(self.plot_row)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.canvas_w)
        self.layout.addWidget(self.qscroll)
        #self.layout.addLayout(self.plot_row)
        self.canvas_freq.hide()
        
        self.channel_changed(*event_channels)
    
    
    def create_subplots(self, twin):
        """ Set up data and widget subplot axes """
        
        ### SLIDER AXES
        self.fig_w = matplotlib.figure.Figure()
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
        self.iw.iwin.on_changed(self.plot_lfp_data)
        self.iw.ycoeff.on_changed(self.plot_lfp_data)
        self.iw.yfig.on_changed(lambda val: self.plot_row.setFixedHeight(int(self.plot_height + val)))
        
        ### FREQ BAND AXES
        self.fig_freq = matplotlib.figure.Figure()
        #gridspec2 = matplotlib.gridspec.GridSpec(1, 3, figure=self.fig_freq)
        fax0 = self.fig_freq.add_subplot(131)
        fax0.autoscale(enable=True, axis='x', tight=True)
        fax1 = self.fig_freq.add_subplot(132, sharey=fax0)
        fax1.autoscale(enable=True, axis='x', tight=True)
        fax2 = self.fig_freq.add_subplot(133, sharey=fax1)
        fax2.autoscale(enable=True, axis='x', tight=True)
        self.faxs = [fax0, fax1, fax2]
        
        ### MAIN AXES
        self.fig = matplotlib.figure.Figure()
        self.ax = self.fig.add_subplot()
        self.ax.sharey(fax0)
    
    
    def connect_mpl_widgets(self):
        
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
            if event.xdata is None: return
            ### SELECT CHANNEL ###
            if event.button == matplotlib.backend_bases.MouseButton.RIGHT:
                # get closest channel/timepoint to clicked point
                ch = abs(int(round(event.ydata)))
                idx = pyfx.IdxClosest(event.xdata, self.lfp_time)
                tpoint = round(self.lfp_time[idx], 2)
                # highlight selected channel/timepoint
                xdata, ydata = self.ax.get_lines()[ch].get_data()
                shadow = self.ax.plot(xdata, ydata, color='yellow', zorder=-1, lw=5)[0]
                vline = self.ax.axvline(tpoint, color='indigo', zorder=-1, lw=2, ls='--')
                self.canvas.draw_idle()
                
                # create context menu and section headers
                menu = QtWidgets.QMenu()
                headers = []
                for txt in [f'Time = {tpoint} s', f'Channel {ch}']:
                    hdr = QtWidgets.QWidgetAction(self)
                    lbl = QtWidgets.QLabel(txt)
                    hdr.setDefaultWidget(lbl)
                    headers.append(hdr)
                t_hdr, ch_hdr = headers
                
                ### timepoint section
                menu.addAction(t_hdr)
                # copy timestamp to clipboard
                copyTAction = menu.addAction('Copy timepoint')
                
                ### channel section
                menu.addAction(ch_hdr)
                noise_txt = ["noise","clean"][self.NOISE_TRAIN[ch]]
                noiseAction = menu.addAction(f'Mark as {noise_txt}')
                noiseAction.setEnabled(ch not in self.event_channels)
                
                # execute menu
                res = menu.exec_(event.guiEvent.globalPos())
                if res == copyTAction:
                    QtWidgets.QApplication.clipboard().setText(str(tpoint))
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
                iwin = int(self.lfp_fs * WIN)  # 50ms (DS) or 200ms (SWR) window
                # filtered LFP for hilus/ripple channel
                lfp = self.DATA[EV][CHAN]
                
                if self.ADD_DS:
                    # get 50ms window surrounding peak index (max filtered LFP amplitude)
                    ipk = idx + (np.argmax(lfp[idx-iwin : idx+iwin]) - iwin)
                    ds_lfp = lfp[ipk-iwin : ipk+iwin]
                    # detect peaks
                    props = scipy.signal.find_peaks(ds_lfp, height=max(ds_lfp), prominence=0)[1]
                    pws = np.squeeze(scipy.signal.peak_widths(ds_lfp, peaks=[iwin], rel_height=0.5))
                    istart, istop = ipk + (np.round(pws[2:4]).astype('int')-iwin)
                    ds_raw_lfp = self.DATA['raw'][self.hil_chan][ipk-iwin : ipk+iwin]
                    imax = ipk + (np.argmax(ds_raw_lfp)-iwin)
                    
                    ddict = dict(ch=self.hil_chan, time=self.lfp_time[imax], amp=props['peak_heights'][0],
                                 half_width=(pws[0]/self.lfp_fs)*1000, width_height=pws[1],
                                 asym=ephys.get_asym(ipk,istart,istop), prom=props['prominences'][0],
                                 start=self.lfp_time[istart], stop=self.lfp_time[istop],
                                 idx=imax, idx_peak=ipk, idx_start=istart, idx_stop=istop,
                                 status=2, is_valid=1) #ffindme
                elif self.ADD_SWR:
                    half_win = int(iwin/2)
                    # get SWR envelope peak in central half of window
                    tmp = np.abs(scipy.signal.hilbert(lfp[idx-half_win:idx+half_win])).astype('float32')
                    ipk = idx + (np.argmax(tmp) - half_win)
                    swr_lfp = lfp[ipk-iwin : ipk+iwin]
                    hilb = scipy.signal.hilbert(swr_lfp)
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
                    props = scipy.signal.find_peaks(env, height=env[iwin])[1]
                    pws = np.squeeze(scipy.signal.peak_widths(env_clip, peaks=[iwin], rel_height=1))
                    istart, istop = ipk + (np.round(pws[2:4]).astype('int')-iwin)
                    # find largest oscillation in ripple
                    ampwin = int(round(self.PARAMS.swr_maxamp_win)/1000 * self.lfp_fs)
                    iosc = (np.argmax(swr_lfp[iwin-ampwin : iwin+ampwin]) + (iwin-ampwin))
                    imax = ipk + (iosc - iwin)
                    # inst. freq
                    fwin = int(round(self.PARAMS.swr_freq_win/1000 * self.lfp_fs))
                    ifreq = ephys.get_inst_freq(hilb, self.lfp_fs, self.PARAMS.swr_freq)
                    swr_ifreq = np.mean(ifreq[iwin-fwin : iwin+fwin])
                    
                    ddict = dict(ch=self.ripple_chan, time=self.lfp_time[imax], 
                                 amp=env[iwin], dur=(pws[0]/self.lfp_fs)*1000, freq=swr_ifreq,
                                 start=self.lfp_time[istart], stop=self.lfp_time[istop],
                                 idx=imax, idx_peak=ipk, idx_start=istart, idx_stop=istop,
                                 status=2, is_valid=1)
                
                # add freq band power to data row, update n_valid column
                ddict.update({**self.STD.loc[CHAN].to_dict(), 'n_valid':0})
                ddf = pd.DataFrame(ddict, index=[CHAN])
                EV_DF = pd.concat([EV_DF, ddf]).sort_values(['ch','time'])
                EV_DF.loc[CHAN, 'n_valid'] = EV_DF.loc[CHAN, 'is_valid'].sum()
                if   EV == 'ds' : self.DS_ALL  = EV_DF
                elif EV == 'swr': self.SWR_ALL = EV_DF
                self.update_event_items(event=EV)
                self.plot_lfp_data()
                print(f'added {EV}!')
                
            
        def on_press(event):
            """ Handle keypress events """
            # update current timepoint with arrow keys 
            if event.key   == 'left'  : self.iw.i.key_step(0)  # step backwards
            elif event.key == 'right' : self.iw.i.key_step(1)  # step forwards
            
            elif event.key in ['backspace', 'escape', ' ', 'enter'] and self.xspan > 0:
                """
                backspace: remove event from dataset
                escape: delete event permanently
                space: restore hidden event
                enter: plot instantaneous CSD
                """
                irg = (imin, imax) = list(map(lambda x: int(x*self.lfp_fs), self.span.extents))
                
                ###   PERMANENTLY DELETE EVENTS   ###
                
                if event.key=='escape':
                    # delete detected DS events
                    if self.SHOW_DS:
                        res = self.edit_event_status(self.DS_ALL, idx=self.DS_valid, irange=irg, 
                                                     chan=self.hil_chan, mode='erase')
                        if res:
                            self.update_event_items(event='ds')
                            
                    # delete detected SWR events
                    if self.SHOW_SWR and self.SWR_ALL is not None:
                        res = self.edit_event_status(self.SWR_ALL, idx=self.SWR_valid, irange=irg, 
                                                     chan=self.ripple_chan, mode='erase')
                        if res:
                            self.update_event_items(event='swr')
                    
                    self.span.set_visible(False)
                    self.plot_lfp_data()
                
                
                ###   DELETE EVENTS
                
                if event.key=='backspace':
                    # remove detected DS events
                    if self.SHOW_DS and self.DS_ALL is not None:
                        res = self.edit_event_status(self.DS_ALL, idx=self.DS_valid, irange=irg, 
                                                     chan=self.hil_chan, mode='delete')
                        if res:
                            self.update_event_items(event='ds')
                    
                    # remove detected SWR events
                    if self.SHOW_SWR and self.SWR_ALL is not None:
                        res = self.edit_event_status(self.SWR_ALL, idx=self.SWR_valid, irange=irg, 
                                                     chan=self.ripple_chan, mode='delete')
                        if res:
                            self.update_event_items(event='swr')
                            
                    self.span.set_visible(False)
                    self.plot_lfp_data()
                    
                ###   RESTORE EVENTS
                
                elif event.key==' ':
                    # replace previously deleted DS events
                    if self.SHOW_DS and self.SHOW_DS_RM and self.DS_ALL is not None:
                        res = self.edit_event_status(self.DS_ALL, idx=self.DS_idx, irange=irg, 
                                                     chan=self.hil_chan, mode='restore')
                        if res:
                            self.update_event_items(event='ds')
                        
                    # replace previously deleted DS events
                    if self.SHOW_SWR and self.SHOW_SWR_RM and self.SWR_ALL is not None:
                        res = self.edit_event_status(self.SWR_ALL, idx=self.SWR_idx, irange=irg, 
                                                     chan=self.ripple_chan, mode='restore')
                        if res:
                            self.update_event_items(event='swr')
                    self.span.set_visible(False)
                    self.plot_lfp_data()
                    
                elif event.key=='enter' and self.coord_electrode is not None:
                    # plot temporary CSD
                    arr = self.DATA['raw'][:, imin:imax]
                    arr2 = np.array(arr)
                    noise_idx = np.nonzero(self.NOISE_TRAIN)[0]
                    clean_idx = np.setdiff1d(np.arange(arr.shape[0]), noise_idx)
                    # replace noisy channels with the average of its two closest neighbors
                    for i in noise_idx:
                        if i == 0:
                            interp_sig = np.array(arr[min(clean_idx)])
                        elif i > max(clean_idx):
                            interp_sig = np.array(arr[max(clean_idx)])
                        else:
                            sig1 = arr[pyfx.Closest(i, clean_idx[clean_idx < i])]
                            sig2 = arr[pyfx.Closest(i, clean_idx[clean_idx > i])]
                            interp_sig = np.nanmean([sig1, sig2], axis=0)
                        arr2[i,:] = interp_sig
                    # calculate CSD
                    csd_obj = ephys.get_csd_obj(arr2, self.coord_electrode, self.PARAMS)
                    csd = ephys.csd_obj2arrs(csd_obj)[1]
                    yax  = np.linspace(-1, len(csd), len(csd)) * -1
                    try:
                        _ = self.ax.pcolorfast(self.lfp_time[imin:imax], yax, csd, 
                                               cmap=plt.get_cmap('bwr'))
                    except:
                        _ = self.ax.pcolorfast(pyfx.Edges(self.lfp_time[imin:imax]), 
                                                          pyfx.Edges(yax), csd, cmap=plt.get_cmap('bwr'))
                    
                    self.span.set_visible(False)
                    self.canvas.draw_idle()  
        self.cid = self.canvas.mpl_connect("key_press_event", on_press)
        self.cid2 = self.canvas.mpl_connect("button_press_event", on_click)
        
        
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
            DF.loc[chan, 'n_valid'] = DF.loc[chan, 'is_valid'].sum()
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
        return True
    
    
    def plot_freq_band_pwr(self):
        _ = [ax.clear() for ax in self.faxs]
        
        self.freq_kw = dict(xytext=(4,4), xycoords=('axes fraction','data'), 
                            bbox=dict(facecolor='w', edgecolor='w', 
                                      boxstyle='square,pad=0.0'),
                            textcoords='offset points', va='bottom', 
                            fontweight='semibold', annotation_clip=True)
        yax = self.channels * -1
        # (channel, color) for each event
        (TH,THC), (RPL,RPLC), (HIL,HILC) = self.ch_cmap.items()
        noise_idx = np.nonzero(self.NOISE_TRAIN)[0] #findme
        if len(noise_idx) == 0:  # no noisy channels, use original freq data
            tmp = self.STD[['norm_theta','norm_swr','slow_gamma','fast_gamma']].values.T
            norm_theta, norm_swr, slow_gamma, fast_gamma = tmp
        else:                    # re-normalize to exclude 1+ noisy channels
            freq_data = self.STD[['theta','swr','slow_gamma','fast_gamma']].values.T
            for i,d in enumerate(freq_data):
                d[noise_idx] = np.nan
            norm_theta = pyfx.Normalize(freq_data[0])
            norm_swr = pyfx.Normalize(freq_data[1])
            slow_gamma, fast_gamma = freq_data[2:]
        
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
        leg = self.faxs[2].legend(loc='upper right', bbox_to_anchor=(1.1,.95))
        _ = [ax.set_xlabel('SD') for ax in self.faxs]
        
        # set style, annotation kwargs
        self.faxs[1].spines['left'].set_visible(False)
        self.faxs[2].spines['left'].set_visible(False)
        
        
    def switch_the_probe(self, **kwargs):
        self.DATA = kwargs['DATA']
        event_channels = kwargs.get('event_channels', [0,0,0])
        self.ch_cmap = pd.Series(['blue', 'green', 'red'], index=event_channels)
        
        self.probe = kwargs.get('probe', None)
        self.DS_ALL = kwargs.get('DS_ALL', None)
        self.SWR_ALL = kwargs.get('SWR_ALL', None)
        self.STD = kwargs.get('STD', None)
        self.NOISE_TRAIN = kwargs.get('NOISE_TRAIN', None)
        self.seizures = kwargs.get('seizures', [])
        
        self.init_event_items()
        self.channel_changed(*event_channels)  # set event channels (and DS/SWR indices)
        
    
    def init_event_items(self):
        # initialize event indexes (empty), event trains (zeros)
        self.DS_idx = np.array(())
        self.DS_valid = np.array(())
        self.SWR_idx = np.array(())
        self.SWR_valid = np.array(())
        self.DS_train = np.zeros(len(self.lfp_time))
        self.SWR_train = np.zeros(len(self.lfp_time))
        self.SZR_train = np.zeros(len(self.lfp_time))
        for iseq in self.seizures:
            self.SZR_train[iseq] = 1
        try: self.seizures_mid = np.array(list(map(pyfx.Center, self.seizures)))
        except: self.seizures_mid = np.array(())
        
        # get data timecourses
        self.lfp_ampl = np.nansum(self.DATA['raw'], axis=0)
        
        # try getting electrode geometry in meters
        self.coord_electrode = None
        if self.probe is not None:
            ypos = np.array(sorted(self.probe.contact_positions[:, 1]))
            self.coord_electrode = pq.Quantity(ypos, self.probe.si_units).rescale('m')  # um -> m
    
    
    def update_event_items(self, event='all'):
        """
        Update event indices after one of the following actions:
        1) User changes the current hilus channel or ripple channel
        2) User switches between probes
        3) User adds or removes an event instance
        """
        #findme
        if (event in ['all','ds']) and (self.DS_ALL is not None):
            DS_DF = self.DS_ALL[self.DS_ALL.ch == self.hil_chan]
            # valid events are either 1) auto-detected and non-user-removed or 2) user-added
            self.DS_idx = DS_DF.idx.values
            self.DS_valid = DS_DF[DS_DF['is_valid'] == 1].idx.values
            self.DS_train[:] = 0
            # 1=auto-detected; 2=user-added; -1=user-removed
            self.DS_train[DS_DF['idx'].values] = DS_DF['status'].values
        
        if (event in ['all','swr']) and (self.SWR_ALL is not None):
            SWR_DF = self.SWR_ALL[self.SWR_ALL.ch == self.ripple_chan]
            self.SWR_idx = SWR_DF.idx.values
            self.SWR_valid = SWR_DF[SWR_DF['is_valid'] == 1].idx.values
            self.SWR_train[:] = 0
            self.SWR_train[SWR_DF['idx'].values] = SWR_DF['status'].values
        
    
    def channel_changed(self, theta_chan, ripple_chan, hil_chan):
        """ Update currently selected event channels """
        self.event_channels = [theta_chan, ripple_chan, hil_chan]
        self.theta_chan, self.ripple_chan, self.hil_chan = self.event_channels
        self.ch_cmap = self.ch_cmap.set_axis(self.event_channels)
        self.update_event_items()  # set DS/SWR indices for new channel
        self.plot_lfp_data()
    
    
    def event_jump(self, sign, event):
        """ Set plot index to the next (or previous) instance of a given event """
        # get idx for given event type, return if empty
        if event == 'ds':
            idx = self.DS_idx if self.SHOW_DS_RM else self.DS_valid
        elif event == 'swr': 
            idx = self.SWR_idx if self.SHOW_SWR_RM else self.SWR_valid
        elif event == 'szr': 
            idx = np.array(self.seizures_mid)
        if len(idx) == 0: return
        
        # find nearest event preceding (sign==0) or following (1) current idx
        i = self.iw.i.val
        idx_next = idx[idx < i][::-1] if sign==0 else idx[idx > i]
        if len(idx_next) == 0: return
        
        # set index slider to next event (automatically updates plot)
        self.iw.i.set_val(idx_next[0])
    
    
    def point_jump(self, val, unit):
        """ Set plot index to the given index or timepoint  """
        if   unit == 't' : new_idx = pyfx.IdxClosest(val, self.lfp_time)
        elif unit == 'i' : new_idx = val
        self.iw.i.set_val(new_idx)
    
        
    def plot_lfp_data(self, x=None):
        """ Update LFP signals on graph """
        self.ax.clear()
        i,iwin = self.iw.i.val, self.iw.iwin.val
        idx = np.arange(i-iwin, i+iwin)
        x = self.lfp_time[idx]
        arr = self.DATA['raw']
        self.iw.i.nsteps = int(iwin/2)
        
        #self.ax.plot(x, self.AUX[idx] + 1, color='blue')
        
        # scale signals based on y-slider value
        if   self.iw.ycoeff.val < -1 : coeff = 1/np.abs(self.iw.ycoeff.val) * 2
        elif self.iw.ycoeff.val >  1 : coeff = self.iw.ycoeff.val / 2
        else                         : coeff = 1
        for irow,y in enumerate(arr):
            clr = self.ch_cmap.get(irow, 'black')
            if isinstance(clr, pd.Series): clr = clr.values[0]
            # plot LFP signals (-y + irow, then invert y-axis)
            if self.NOISE_TRAIN[irow] == 1:
                self.ax.plot(x, np.repeat(-irow, len(x)), color='lightgray', label=str(irow), lw=2)
            else:
                self.ax.plot(x, y[idx]*coeff - irow, color=clr, label=str(irow), lw=1)
        #self.ax.invert_yaxis()
        
        # mark ripple/DS events with red/green lines
        if self.SHOW_DS:
            ds_ii = np.where(self.DS_train[idx] != 0)[0]
            
            for ds_time, status in zip(x[ds_ii], self.DS_train[idx][ds_ii]):
                if status == 1:
                    self.ax.axvline(ds_time, color='red', lw=2, zorder=-5, alpha=0.4)
                elif status == 2:
                    self.ax.axvline(ds_time, color='red', lw=2, ls=':', zorder=-5, alpha=0.4)
                elif self.SHOW_DS_RM:
                    self.ax.axvline(ds_time, color='red', lw=2, ls='--', zorder=-5, alpha=0.4)
                    
        if self.SHOW_SWR:
            # swr_ii = np.where(self.SWR_train[idx] != 0)[0]
            
            # for swr_time, status in zip(x[swr_ii], self.SWR_train[idx][swr_ii]):
            #     if status == 1:
            #         self.ax.axvline(swr_time, color='green', lw=2, zorder=-5, alpha=0.4)
            #     elif status == 2:
            #         self.ax.axvline(swr_time, color='green', lw=2, ls=':', zorder=-5, alpha=0.4)
            #     elif self.SHOW_SWR_RM:
            #         self.ax.axvline(swr_time, color='green', lw=2, ls='--', zorder=-5, alpha=0.4)
            
            swr_times = x[np.where(self.SWR_train[idx] > 0)[0]]
            for swrt in swr_times:
                self.ax.axvline(swrt, color='green', zorder=-5, alpha=0.4)
            if self.SHOW_SWR_RM:
                swr_rm_times = x[np.where(self.SWR_train[idx] < 0)[0]]
                for swrrt in swr_rm_times:
                    self.ax.axvline(swrrt, color='green', ls='--', zorder=-5, alpha=0.4)
                    
        if self.SHOW_SZR:
            szr_seqs = [self.lfp_time[seq] for seq in map(lambda x: np.intersect1d(x, idx), 
                                                          self.seizures) if len(seq) > 0]
            for seq in szr_seqs:
                self.ax.axvspan(*pyfx.Edges(seq), color='purple', zorder=-6, alpha=0.2)
        self.ax.set(xlabel='Time (s)', ylabel='channel index')
        self.ax.set_xmargin(0.01)
        self.ax.set_ymargin(0.05)
        
        if self.canvas_freq.isVisible():
            self.plot_freq_band_pwr()
        
        yticks = np.array([y for y in self.ax.get_yticks() if pyfx.InRange(y, *self.ax.get_ybound())])
        self.ax.set_yticks(yticks, labels=np.int16(abs(yticks)).astype('str'))
        self.canvas.draw_idle()
        
    def closeEvent(self, event):
        plt.close()
        event.accept()
        
        
class IFigEvent(QtWidgets.QWidget):
    """ Base figure showing detected events from one LFP channel """
    
    pal = sns.cubehelix_palette(dark=0.2, light=0.9, rot=0.4, as_cmap=True)
    FLAG = 0  # 0=plot average waveform, 1=plot individual events
    EXCLUDE_NOISE = False
    annot_dict = dict(time='{time:.2f} s')
    CHC = pd.Series(pyfx.rand_hex(96))
    
    def __init__(self, ch, DF_ALL, DATA, lfp_time, lfp_fs, PARAMS, **kwargs):
        super().__init__()
        
        # initialize params from input arguments
        self.ch = ch
        self.DF_ALL = DF_ALL
        self.DATA = DATA
        self.lfp_time = lfp_time
        self.lfp_fs = lfp_fs
        self.PARAMS = PARAMS
        twin = kwargs.get('twin', 0.2)
        self.thresholds = kwargs.get('thresholds', {})
        
        # get LFP and DF for the primary channel
        self.LFP_arr = self.DATA['raw']
        self.LFP = self.LFP_arr[self.ch]
        self.channels = np.arange(self.LFP_arr.shape[0])
        self.NOISE_TRAIN = kwargs.get('NOISE_TRAIN', np.zeros(len(self.channels), dtype='int'))
        self.NOISE_IDX = np.nonzero(self.NOISE_TRAIN)[0]
        
        # get DF means per channel, isolate DF of current channel
        self.DF_MEAN = self.DF_ALL.groupby('ch').agg('mean')
        self.DF_MEAN = ephys.replace_missing_channels(self.DF_MEAN, self.channels)
        self.DF_MEAN.insert(0, 'ch', self.DF_MEAN.index.values)
        if self.ch in self.DF_ALL.ch:
            self.DF = self.DF_ALL.loc[self.ch]
        else:
            self.DF = pd.DataFrame(columns=self.DF_ALL.columns)
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
        self.CH_ON_PLOT = [int(self.ch)]  # ffindme (noise)
        
        self.create_subplots(twin=twin)
        
        self.fig.set_tight_layout(True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setOrientation(QtCore.Qt.Vertical)
        self.toolbar.setMaximumWidth(30)
        self.layout = QtWidgets.QHBoxLayout(self)
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
                                      pal=self.pal, noise_train=self.NOISE_TRAIN,
                                      exclude_noise=bool(self.EXCLUDE_NOISE), CHC=self.CHC)
        # for ax in [self.axs[f'gax{i}'] for i in range(3)]:
        #     if self.EXCLUDE_NOISE: ax.CM = ax.cmapNE
        #     else: 
            
        # initial colormap
        
        self.ch_gax0_artists = list(self.axs['gax0'].patches)
        self.ch_gax1_artists = np.array([None] * len(self.channels)).astype('object')
        self.ch_gax1_artists[self.DF_MEAN.n_valid > 0] = list(self.axs['gax1'].collections)
        self.ch_gax2_polygons = []
        for ch in self.channels:
            c = 'red' if ch == self.ch else self.CHC[ch]  # "highlight" given channel
            axv = self.axs['gax2'].axvspan(ch-0.25, ch+0.25, color=c, alpha=0.7, zorder=-5)
            axv.set_visible(False)
            self.ch_gax2_polygons.append(axv)
        
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
        self.thres_leg_kw = dict(loc='lower right',  bbox_to_anchor=(1,1), 
                                 title='Thresholds', draggable=True)
        
        # create sliders for scaling/navigation
        twin_kw = dict(valmin=.01, valmax=0.5, valstep=.01, valfmt='%.2f s', valinit=twin)
        ywin_kw = dict(valmin=-0.5, valmax=1, valstep=0.05, valfmt='%.2f', valinit=0)
        idx_kw = dict(valmin=0, valmax=len(self.iev)-1, valstep=1, valinit=0, 
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
        def on_press(event):
            """ Update current event with left/right arrow keys """
            if self.FLAG == 1:
                i = self.iw.idx.val  # index of current event
                if event.key == 'left' and i > 0:
                    self.iw.idx.set_val(i-1)  # previous event
                elif event.key == 'right' and i < len(self.iev)-1: 
                    self.iw.idx.set_val(i+1)  # next event
        self.cid = self.canvas.mpl_connect("key_press_event", on_press)
        
        
    def update_twin(self, twin):
        """ Update plot data window, set x-axis limits """
        # update data window, set x-limits
        self.iwin = int(twin * self.lfp_fs)  # window size
        self.ev_x = np.linspace(-twin, twin, self.iwin*2)
        self.ax.set_xlim(pyfx.Limit(self.ev_x, pad=0.01))
    
    
    def rescale_y(self):
        """ Automatically determine y-limits when updating data plot 
        (e.g. new data source, changed timescale, or toggled plot mode) """
        if self.FLAG == 0:
            if len(self.CH_ON_PLOT) == 1:
                lfp_arr = ephys.getwaves(self.LFP, self.iev, self.iwin)
                yerrs, _ = ephys.getyerrs(lfp_arr, mode='sem')
                self.EY = pyfx.Limit(np.concatenate(yerrs), pad=0.05)
            else:
                waves = [ephys.getavg(self.LFP_arr[xch],
                                      np.atleast_1d(self.DF_ALL.loc[xch].idx),
                                      self.iwin) for xch in self.CH_ON_PLOT]
                self.EY = pyfx.Limit(np.concatenate(waves), pad=0.05)
        elif self.FLAG == 1:
            self.EY = pyfx.Limit(ephys.getwaves(self.LFP, self.iev, self.iwin).flatten(),
                                 pad=0.05)
        ylim = pyfx.Limit(self.EY, pad=np.abs(self.iw.ywin.val), sign=np.sign(self.iw.ywin.val))
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
            thres_legend.setDraggable(True)
        else: thres_legend = None
        # add X and Y axes to plot (but not the legend)
        if self.xax_line.get_visible(): self.ax.add_line(self.xax_line)
        if self.yax_line.get_visible(): self.ax.add_line(self.yax_line)
        
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlabel('Time (s)')
        
        if len(self.iev) == 0: return
        
        ### average waveform(s)
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
                    xmean = np.nanmean(ephys.getwaves(self.LFP_arr[xch],
                                                self.DF_ALL.loc[xch].idx,
                                                self.iwin), axis=0)
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
            self.LFP = self.LFP_arr[self.ch]
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
    annot_dict = dict(time='{time:.2f} s', amp='{amp:.2f} mV', asym='{asym:+.0f} \%', 
                      half_width='{half_width:.2f} ms', width_height='{width_height:.2f} mV')
    
    def create_subplots(self, **kwargs):
        super().create_subplots(**kwargs)
        
        def toggle_data(label):
            """ Plot raw vs filtered LFP data """
            if label == 'Raw' : self.LFP_arr = self.DATA['raw']
            else              : self.LFP_arr = self.DATA['ds']
            self.LFP = self.LFP_arr[self.ch]
            
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
        self.evw.setMaximumWidth(200)
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
        #self.event_fig.plot_event_data(self.event_fig.iw.idx.val, self.event_fig.iw.twin.val)
        
        
    def toggle_plot_mode(self, btn, chk):
        """ Switch between figure views """
        if not chk:
            return
        mode = btn.group().id(btn)  # 0=average waveform, 1=individual events
        self.event_fig.FLAG = mode
        self.event_fig.rescale_y()
        self.event_fig.plot_event_data(event=None)#(self.event_fig.iw.idx.val, self.event_fig.iw.twin.val)
        
        self.event_fig.iw.idx.set_active(bool(mode))    # slider active if mode==1
        self.event_fig.iw.idx.enable(bool(mode))
        self.evw.sort_gbox.setEnabled(bool(mode)) # sort options enabled if mode==1
        self.evw.ch_comp_widget.setEnabled(not bool(mode))  # channel comparison disabled if mode==1
        
    
    def reset_ch_dropdown(self):
        """ Populate channel dropdown with all channels except primary """
        # remove all items
        for i in reversed(range(self.evw.ch_comp_widget.ch_dropdown.count())):
            self.evw.ch_comp_widget.ch_dropdown.removeItem(i)
            
        # repopulate dropdown with all channels, then remove primary channel
        channel_strings = list(np.array(self.channels, dtype='str'))
        self.evw.ch_comp_widget.ch_dropdown.addItems(channel_strings)
        for idx in np.setdiff1d(self.event_fig.channels, np.unique(self.event_fig.DF_ALL.ch)):
            self.evw.ch_comp_widget.ch_dropdown.model().item(idx).setEnabled(False)  # disable items of channels with no events
        self.evw.ch_comp_widget.ch_dropdown.removeItem(self.ch)
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
        event.accept()
        
    
class EventViewWidget(QtWidgets.QFrame):
    """ Settings widget for popup event window """
    sort_labels = pd.Series(dict(time         = 'Time',
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
        
        gbox_ss = ('QGroupBox {'
                   'border : 1px solid gray;'
                   'border-radius : 8px;'
                   'font-size : 16pt;'
                   'font-weight : bold;'
                   'margin-top : 10px;'
                   'padding : 10px 5px 10px 5px;'
                   '}'
                   
                   'QGroupBox::title {'
                   'subcontrol-origin : margin;'
                   'subcontrol-position : top left;'
                   'padding : 2px 5px;' # top, right, bottom, left
                   '}')
        
        mode_btn_ss = ('QPushButton {'
                       'background-color : whitesmoke;'
                       'border : 2px outset gray;'
                       'border-radius : 2px;'
                       'color : black;'
                       'padding : 3px;'
                       'font-weight : bold;'
                       '}'
                       
                       'QPushButton:pressed {'
                       'background-color : dimgray;'
                       'border : 2px inset gray;'
                       'color : white;'
                       '}'
                       
                       'QPushButton:checked {'
                       'background-color : darkgray;'
                       'border : 2px inset gray;'
                       'color : black;'
                       '}'
                       
                       'QPushButton:disabled {'
                       'background-color : gainsboro;'
                       'border : 2px outset darkgray;'
                       'color : gray;'
                       '}'
                       
                       'QPushButton:disabled:checked {'
                       'background-color : darkgray;'
                       'border : 2px inset darkgray;'
                       'color : dimgray;'
                       '}'
                       )
        
        ###   VIEW PLOT ITEMS
        self.view_gbox = QtWidgets.QGroupBox('VIEW')
        self.view_gbox.setStyleSheet(gbox_ss)
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
        self.data_gbox.setStyleSheet(gbox_ss)
        data_lay = pyfx.InterWidgets(self.data_gbox, 'v')[2]
        data_lay.setSpacing(10)
        ### buttons for single vs averaged plot mode
        pm_vbox = QtWidgets.QVBoxLayout()
        pm_vbox.setSpacing(5)
        self.plot_mode_bgrp = QtWidgets.QButtonGroup(self.data_gbox)
        self.single_btn = QtWidgets.QPushButton('Single Events')
        self.single_btn.setCheckable(True)
        self.single_btn.setStyleSheet(mode_btn_ss)
        self.avg_btn = QtWidgets.QPushButton('Averages')
        self.avg_btn.setCheckable(True)
        self.avg_btn.setChecked(True)
        self.avg_btn.setStyleSheet(mode_btn_ss)
        self.plot_mode_bgrp.addButton(self.avg_btn, 0)
        self.plot_mode_bgrp.addButton(self.single_btn, 1)
        pm_vbox.addWidget(self.single_btn)
        pm_vbox.addWidget(self.avg_btn)
        data_lay.addLayout(pm_vbox)
        data_line0 = pyfx.DividerLine(lw=2, mlw=2)
        data_lay.addWidget(data_line0)
        
        ### channel comparison widget
        self.ch_comp_widget = gi.AddChannelWidget()
        data_lay.addWidget(self.ch_comp_widget)
        self.vlay.addWidget(self.data_gbox)
        
        line1 = pyfx.DividerLine()
        self.vlay.addWidget(line1)
        
        ###   SORT EVENTS
        self.sort_gbox = QtWidgets.QGroupBox('SORT')
        self.sort_gbox.setStyleSheet(gbox_ss)
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
    """ Settings widget for main channel selection GUI """
    
    ch_signal = QtCore.pyqtSignal(int, int, int)  # user changed event channel
    noise_signal = QtCore.pyqtSignal(np.ndarray)  # user changed noise channels
    
    def __init__(self, ddir, nchannels, ntimes, lfp_time, lfp_fs, event_channels, parent=None):
        super().__init__(parent)
        
        self.ddir = ddir
        
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
        
        qlist_ss_main = ('QListWidget {'
                         'background-color : rgba(255,255,255,50);'
                         #'background-color : rgba(220,220,220,100);'  # gainsboro
                         'border : 2px groove rgb(150,150,150);'
                         'border-radius : 2px;'
                         'padding : 0px;'
                         '}'
                      
                        'QListWidget::item {'
                        'background-color : rgb(255,255,255);'
                        'color : black;'
                        'border : 1px solid rgb(200,200,200);'
                        'border-radius : 1px;'
                        'padding : 4px;'
                        '}'
                      
                        'QListWidget::item:selected {'
                        'background-color : rgba(85,70,160,200);'
                        
                        'color : white;'
                        #'background-color : rgba(179,179,191);'
                        '}')
                      
        self.settings_layout = QtWidgets.QVBoxLayout(self)
        self.settings_layout.setSpacing(10)
        
        ###   TOGGLE PROBES   ###
        
        # toggle between probes
        self.probes_qlist = QtWidgets.QListWidget()
        #self.probes_qlist.setFixedHeight(50)
        self.probes_qlist.setStyleSheet(qlist_ss_main)
        self.probes_qlist.setFocusPolicy(QtCore.Qt.NoFocus)
        self.probes_qlist.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.probes_qlist.setSizeAdjustPolicy(self.probes_qlist.AdjustToContents)
        # ignore item deselection (one probe must always be selected)
        fx = lambda q: (len(q.selectedItems())==0, q.item(q.currentRow()))
        fx2 = lambda x,item: item.setSelected(True) if x else None
        self.probes_qlist.itemSelectionChanged.connect(lambda: fx2(*fx(self.probes_qlist)))
        
        self.tab_widget = QtWidgets.QTabWidget(parent=self)
        self.tab_widget.setObjectName('tab_widget')
        #self.tab_widget.setMinimumWidth(25)
        self.tab_widget.setMovable(True)
        
        
        ######################################################
        ######################################################
        ############           TAB 1              ############
        ######################################################
        ######################################################
        
        
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
        #self.plot_freq_pwr.setStyleSheet('QPushButton {padding : 5px 10px;}')
        self.plot_freq_pwr.setStyleSheet('QPushButton {'
                                          'background-color : whitesmoke;'
                                          'border : 2px outset gray;'
                                          'color : black;'
                                          'image : url(:/icons/double_chevron_left.png);'
                                          'image-position : left;'
                                          'padding : 10px 5px;'
                                          '}'
                                          
                                        'QPushButton:checked {'
                                        'background-color : gainsboro;'
                                        'border : 2px inset gray;'
                                        'image : url(:/icons/double_chevron_right.png);'
                                        '}'
                                        )
        self.plot_freq_pwr.setLayoutDirection(QtCore.Qt.LeftToRight)
        
        ###   EVENT WIDGETS   ###
        
        cline_ss = ('QLabel {'
                    'border : 1px solid transparent;'
                    'border-bottom : 3px solid %s;'
                    'max-height : 2px;'
                    '}')
        # hilus channel widgets
        self.ds_gbox = QtWidgets.QGroupBox('DG SPIKES')
        self.ds_gbox.setStyleSheet(gbox_ss_main)
        ds_lay = pyfx.InterWidgets(self.ds_gbox, 'v')[2]
        ds_lay.setSpacing(5)
        ds_row1 = QtWidgets.QHBoxLayout()
        self.hil_lbl = QtWidgets.QLabel('Hilus channel:')
        self.hil_input = gi.ReverseSpinBox()
        self.ds_reset = QtWidgets.QPushButton('\u27F3')
        ds_row1.addWidget(self.hil_lbl)
        ds_row1.addWidget(self.hil_input)
        ds_row1.addWidget(self.ds_reset)
        # buttons
        ds_row2 = QtWidgets.QHBoxLayout()
        self.ds_event_btn = QtWidgets.QPushButton('View DS')
        ds_row2.addWidget(self.ds_event_btn)
        
        ds_row3 = QtWidgets.QVBoxLayout()
        ds_row3_0 = QtWidgets.QHBoxLayout()
        ds_row3_0.setSpacing(10)
        ds_row3_0.setContentsMargins(0,0,0,0)
        self.ds_show = QtWidgets.QPushButton()
        self.ds_arrows = gi.EventArrows()
        self.ds_add = QtWidgets.QCheckBox('Add')
        self.ds_add.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.ds_show_rm = QtWidgets.QCheckBox('Show deleted events')
        self.ds_show_rm.setChecked(False)
        ds_vline1, ds_vline2 = pyfx.DividerLine('v'), pyfx.DividerLine('v')
        ds_row3_0.addWidget(self.ds_show)
        ds_row3_0.addWidget(ds_vline1)
        ds_row3_0.addWidget(self.ds_arrows)
        ds_row3_0.addWidget(ds_vline2)
        ds_row3_0.addWidget(self.ds_add)
        ds_row3.addLayout(ds_row3_0)
        ds_row3.addWidget(self.ds_show_rm)
        # colorcode
        ds_cc = QtWidgets.QLabel()
        ds_cc.setStyleSheet(cline_ss % 'red')
        ds_lay.addSpacing(10)
        ds_lay.addLayout(ds_row1)
        ds_lay.addLayout(ds_row2)
        ds_lay.addLayout(ds_row3)
        ds_lay.addWidget(ds_cc)
        
        # ripple channel widgets
        self.ripple_gbox = QtWidgets.QGroupBox('RIPPLES')
        self.ripple_gbox.setStyleSheet(gbox_ss_main)
        swr_lay = pyfx.InterWidgets(self.ripple_gbox, 'v')[2]
        swr_lay.setSpacing(5)
        swr_row1 = QtWidgets.QHBoxLayout()
        self.swr_lbl = QtWidgets.QLabel('Ripple channel:')
        self.swr_input = gi.ReverseSpinBox()
        self.swr_reset = QtWidgets.QPushButton('\u27F3')
        swr_row1.addWidget(self.swr_lbl)
        swr_row1.addWidget(self.swr_input)
        swr_row1.addWidget(self.swr_reset)
        # buttons
        swr_row2 = QtWidgets.QHBoxLayout()
        self.swr_event_btn = QtWidgets.QPushButton('View ripples')
        swr_row2.addWidget(self.swr_event_btn)
        swr_row3 = QtWidgets.QVBoxLayout()
        swr_row3_0 = QtWidgets.QHBoxLayout()
        swr_row3_0.setSpacing(10)
        swr_row3_0.setContentsMargins(0,0,0,0)
        self.swr_show = QtWidgets.QPushButton()
        self.swr_arrows = gi.EventArrows()
        self.swr_add = QtWidgets.QCheckBox('Add')
        self.swr_add.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.swr_show_rm = QtWidgets.QCheckBox('Show deleted events')
        self.swr_show_rm.setChecked(False)
        swr_vline1, swr_vline2 = pyfx.DividerLine('v'), pyfx.DividerLine('v')
        swr_row3_0.addWidget(self.swr_show)
        swr_row3_0.addWidget(swr_vline1)
        swr_row3_0.addWidget(self.swr_arrows)
        swr_row3_0.addWidget(swr_vline2)
        swr_row3_0.addWidget(self.swr_add)
        swr_row3.addLayout(swr_row3_0)
        swr_row3.addWidget(self.swr_show_rm)
        # colorcode
        swr_cc = QtWidgets.QLabel()
        swr_cc.setStyleSheet(cline_ss % 'green')
        swr_lay.addSpacing(10)
        swr_lay.addLayout(swr_row1)
        swr_lay.addLayout(swr_row2)
        swr_lay.addLayout(swr_row3)
        swr_lay.addWidget(swr_cc)
        
        # theta channel widgets
        self.theta_gbox = QtWidgets.QGroupBox('THETA')
        self.theta_gbox.setStyleSheet(gbox_ss_main)
        theta_lay = pyfx.InterWidgets(self.theta_gbox, 'v')[2]
        theta_lay.setSpacing(5)
        theta_row1 = QtWidgets.QHBoxLayout()
        self.theta_lbl = QtWidgets.QLabel('Theta channel:')
        self.theta_input = gi.ReverseSpinBox()
        self.theta_reset = QtWidgets.QPushButton('\u27F3')
        theta_row1.addWidget(self.theta_lbl)
        theta_row1.addWidget(self.theta_input)
        theta_row1.addWidget(self.theta_reset)
        # buttons
        theta_row2 = QtWidgets.QHBoxLayout()
        self.theta_event_btn = QtWidgets.QPushButton('View theta')
        self.theta_event_btn.setEnabled(False)
        theta_row2.addWidget(self.theta_event_btn)
        # colorcode
        theta_cc = QtWidgets.QLabel()
        theta_cc.setStyleSheet(cline_ss % 'blue')
        theta_lay.addSpacing(10)
        theta_lay.addLayout(theta_row1)
        theta_lay.addLayout(theta_row2)
        theta_lay.addWidget(theta_cc)
        # set min/max channel values
        self.ch_inputs = [self.theta_input, self.swr_input, self.hil_input]
        for sbox in self.ch_inputs:
            sbox.setRange(0, nchannels-1)
        for reset_btn in [self.ds_reset, self.swr_reset, self.theta_reset]:
            reset_btn.setStyleSheet('QPushButton {font-size:25pt; padding:0px 0px 2px 2px;}')
            reset_btn.setMaximumSize(20,20)
            reset_btn.setAutoDefault(False)
        for show_btn in [self.ds_show, self.swr_show]:
            show_btn.setCheckable(True)
            show_btn.setChecked(True)
            show_btn.setStyleSheet('QPushButton {'
                                   'background-color : whitesmoke;'
                                   'border : 2px outset gray;'
                                   'image : url(:/icons/show_outline.png);'
                                   '}'
                                   
                                   'QPushButton:checked {'
                                   'background-color : gainsboro;'
                                   'border : 2px inset gray;'
                                   'image : url(:/icons/hide_outline.png);'
                                   '}')
        # set channels
        self.set_channel_values(*event_channels)
        
        self.vlay.addWidget(self.plot_freq_pwr, stretch=0)
        self.vlay.addWidget(self.ds_gbox, stretch=2)
        self.vlay.addWidget(self.ripple_gbox, stretch=2)
        self.vlay.addWidget(self.theta_gbox, stretch=1)
        
        
        ######################################################
        ######################################################
        ############           TAB 2              ############
        ######################################################
        ######################################################
        
        
        self.tab2 = QtWidgets.QFrame()
        self.tab2.setObjectName('tab2')
        self.tab2.setFrameShape(QtWidgets.QFrame.Box)
        self.tab2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tab2.setLineWidth(2)
        self.tab2.setMidLineWidth(2)
        self.tab_widget.addTab(self.tab2, 'Notes')
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
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward)
        jbtns = [self.tjump_btn, self.ijump_btn] = [QtWidgets.QPushButton(), 
                                                    QtWidgets.QPushButton()]
        for btn in jbtns:
            btn.setIcon(icon)
            btn.setMaximumWidth(btn.minimumSizeHint().height())
            btn.setFlat(True)
        self.tjump.layout.addWidget(self.tjump_btn)
        self.ijump.layout.addWidget(self.ijump_btn)
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
        self.noise_qlist.setStyleSheet(qlist_ss_main)
        self.noise_qlist.setFocusPolicy(QtCore.Qt.NoFocus)
        self.noise_qlist.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.noise_qlist.setSizeAdjustPolicy(self.noise_qlist.AdjustToContents)
        
        # self.noise_qlist.setStyleSheet('QListWidget {'
        #                                           #'background-color : rgba(255,255,255,50);'
        #                                           'border : 2px solid gray;'
        #                                           'padding : 0px;'
        #                                           '}'
                                                  
        #                                           'QListWidget::item {'
        #                                           'border : 2px solid rgb(200,200,200);'
        #                                           #'border-radius : 1px;'
        #                                           #'min-height : 12px;'
        #                                           #'max-height : 12px;'
        #                                           'padding : 5px;'
        #                                           '}'
        #                                           )
        #ffindme
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
        
        
        ######################################################
        ######################################################
        ############           TAB 3              ############
        ######################################################
        ######################################################
        
        
        self.tab3 = QtWidgets.QFrame()
        self.tab3.setObjectName('tab3')
        self.tab3.setFrameShape(QtWidgets.QFrame.Box)
        self.tab3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.tab3.setLineWidth(2)
        self.tab3.setMidLineWidth(2)
        self.tab_widget.addTab(self.tab3, 'Seizures')
        self.vlay3 = QtWidgets.QVBoxLayout(self.tab3)
        self.vlay3.setSpacing(10)
        
        ###   SEIZURE WIDGET   ###
        
        # seizures
        self.szr_gbox = QtWidgets.QGroupBox()
        szr_row = QtWidgets.QGridLayout(self.szr_gbox)
        self.szr_show = QtWidgets.QRadioButton('Show seizures')
        self.szr_arrows = gi.EventArrows()
        self.szr_arrows.setEnabled(False)
        self.szr_elim = QtWidgets.QPushButton('X')
        self.szr_elim.setMaximumWidth(self.szr_elim.minimumSizeHint().height())
        self.szr_elim.setEnabled(False)
        szr_row.addWidget(self.szr_show, 0, 0)
        szr_row.addWidget(self.szr_elim, 0, 1)
        szr_row.addWidget(self.szr_arrows, 1, 0, 1, 2)
        self.vlay3.addWidget(self.szr_gbox, stretch=0)
        
        # save changes
        bbox = QtWidgets.QVBoxLayout()
        bbox.setSpacing(2)
        self.save_btn = QtWidgets.QPushButton('  Save channels  ')
        self.save_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.save_btn.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.save_btn.setStyleSheet('QPushButton {padding : 5px 20px;}')
        self.save_btn.setDefault(False)
        self.save_btn.setAutoDefault(False)
        self.debug_btn = QtWidgets.QPushButton('  debug  ')
        self.debug_btn.setDefault(False)
        self.debug_btn.setAutoDefault(False)
        bbox.addWidget(self.save_btn)
        bbox.addWidget(self.debug_btn)
        
        self.settings_layout.addWidget(self.probes_qlist, stretch=0)
        self.settings_layout.addWidget(self.tab_widget, stretch=2)
        self.settings_layout.addLayout(bbox, stretch=0)
    
    
    def set_channel_values(self, theta_chan, ripple_chan, hil_chan):
        """ Update channel selection widgets """
        self.theta_input.setValue(theta_chan)
        self.swr_input.setValue(ripple_chan)
        self.hil_input.setValue(hil_chan)
    
    
    def emit_ch_signal(self):
        """ Emit signal with all 3 current event channels """
        event_channels = [int(item.value()) for item in self.ch_inputs]
        self.ch_signal.emit(*event_channels)
        self.disable_event_noise()
    
    
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
        try:
            noise_train[qlist_ch] = 1
        except:
            pdb.set_trace()
        self.noise_signal.emit(noise_train)
    
    # self-contained notes saving
    def export_notes(self):
        txt = self.notes_qedit.toPlainText()
        ephys.write_notes(Path(self.ddir,'notes.txt'), txt)
        print('Notes saved!')
        # keep track of last save point, use for warning message
        self.last_saved_notes = str(txt)
        
        
class ChannelSelectionWindow(QtWidgets.QDialog):
    """ Main channel selection GUI """
    
    def __init__(self, ddir, probe_list, iprb=0, parent=None):
        super().__init__(parent)
        qrect = pyfx.ScreenRect(perc_width=0.9)
        self.setGeometry(qrect)
        
        self.init_data(ddir, probe_list, iprb)
        self.init_figs()
        self.gen_layout()
        self.connect_signals()
        
        # update window title
        title = f'{os.path.basename(self.ddir)}'
        self.setWindowTitle(title)
        self.show()
    
    
    def init_data(self, ddir, probe_list, iprb):
        """ Initialize all recording variables """
        self.ddir = ddir
        self.probe_list = probe_list
        self.iprb = iprb
        
        # load params
        self.PARAMS = ephys.load_recording_params(ddir)
        #self.INFO = ephys.load_recording_info(ddir)
        
        # load LFP signals, event dfs, and detection thresholds for all probes
        self.data_list, self.lfp_time, self.lfp_fs = ephys.load_lfp(ddir, '', -1)
        self.swr_list = [x[0] for x in ephys.load_event_dfs(ddir, 'swr', -1)]
        self.swr_list_orig = deepcopy(self.swr_list)  # originally loaded version 
        self.swr_list_file = deepcopy(self.swr_list)  # version saved to file
        self.ds_list = [x[0] for x in ephys.load_event_dfs(ddir, 'ds', -1)]
        self.ds_list_orig = deepcopy(self.ds_list)
        self.ds_list_file = deepcopy(self.ds_list)
        self.threshold_list = list(np.load(Path(ddir, 'THRESHOLDS.npy'), allow_pickle=True))
        self.noise_list = ephys.load_noise_channels(ddir, -1)
        self.noise_list_orig = deepcopy(self.noise_list)
        self.std_list = ephys.csv2list(ddir, 'channel_bp_std')
        self.NCH, self.NTS = self.data_list[0]['raw'].shape
        
        
        self.aux_array = ephys.load_aux(ddir)
        if len(self.aux_array) > 0:
            self.AUX = self.aux_array[0]
            self.AUX_TRAIN = np.array(self.AUX > 0.2, dtype='int')
        else:
            self.AUX = np.full(self.NTS, np.nan)
            self.AUX_TRAIN = np.full(self.NTS, np.nan)
            #ilsr = np.nonzero(AUX)[0]
            #seqs = pyfx.get_sequences(np.nonzero(AUX)[0])
        #self.seizures, _, _ = ephys.load_seizures(ddir)
        
        # load (or estimate) event channels for all probes
        self.event_channel_list = []
        for i in range(len(self.probe_list)):
            event_channels = ephys.load_event_channels(ddir, i, DATA=self.data_list[i], 
                                                       STD=self.std_list[i], 
                                                       NOISE_TRAIN=self.noise_list[i])
            self.event_channel_list.append(event_channels)
        self.orig_event_channel_list = deepcopy(self.event_channel_list)
        
        # # load AUX channel(s) if they exist
        # self.aux_mx = np.array(())
        # if os.path.exists(Path(ddir, 'AUX.npy')):
        #     self.aux_mx = np.load(Path(ddir, 'AUX.npy'))
        
        # select initial probe data
        self.load_probe_data(self.iprb)


    def load_probe_data(self, iprb):
        """ Set data corresponding to the given probe """
        self.iprb = iprb
        self.probe = self.probe_list[iprb]
        self.DATA = dict(self.data_list[iprb])
        self.SWR_ALL = pd.DataFrame(self.swr_list[iprb])
        self.DS_ALL = pd.DataFrame(self.ds_list[iprb])
        self.SWR_THRES, self.DS_THRES = dict(self.threshold_list[iprb]).values()
        self.STD = pd.DataFrame(self.std_list[iprb])
        self.NOISE_TRAIN = np.array(self.noise_list[iprb])
        self.channels = np.arange(self.DATA['raw'].shape[0])
        self.seizures, _, _ = ephys.load_iis(self.ddir, iprb)
        self.event_channels = list(self.event_channel_list[iprb])
        self.theta_chan, self.ripple_chan, self.hil_chan = self.event_channels
        self.orig_theta_chan, self.orig_ripple_chan, self.orig_hil_chan = self.orig_event_channel_list[iprb]
    
    
    def SWITCH_THE_PROBE(self, idx):
        # save dataframe with added/removed events
        self.swr_list[self.iprb] = pd.DataFrame(self.main_fig.SWR_ALL)
        self.ds_list[self.iprb]  = pd.DataFrame(self.main_fig.DS_ALL)
        
        self.load_probe_data(idx)
        
        kwargs = dict(probe=self.probe, DATA=self.DATA, event_channels=self.event_channels, 
                      DS_ALL=self.DS_ALL, SWR_ALL=self.SWR_ALL, STD=self.STD, 
                      NOISE_TRAIN=self.NOISE_TRAIN, seizures=self.seizures, SWR_THRES=self.SWR_THRES)
        self.main_fig.switch_the_probe(**kwargs)
        self.widget.set_channel_values(*self.event_channels)
        self.widget.set_noise_channels(self.NOISE_TRAIN)
        self.widget.tab_widget.setTabVisible(2, len(self.seizures) > 0)
    
    def init_figs(self):
        """ Create main figure, initiate event channel update """
        kwargs = dict(probe=self.probe, DATA=self.DATA, lfp_time=self.lfp_time, lfp_fs=self.lfp_fs, 
                      PARAMS=self.PARAMS, event_channels=self.event_channels, 
                      DS_ALL=self.DS_ALL, SWR_ALL=self.SWR_ALL, STD=self.STD, 
                      NOISE_TRAIN=self.NOISE_TRAIN, seizures=self.seizures, SWR_THRES=self.SWR_THRES,
                      AUX=self.AUX)
        
        # set up figures
        self.main_fig = IFigLFP(**kwargs)
        sns.despine(self.main_fig.fig)
        sns.despine(self.main_fig.fig_freq)
            
        
    def gen_layout(self):
        """ Set up layout """
        # create channel selection widget, initialize values
        self.widget = ChannelSelectionWidget(self.ddir, self.NCH, self.NTS, self.lfp_time, self.lfp_fs, self.event_channels)
        self.widget.setStyleSheet('QWidget:focus {outline : none;}')
        self.widget.setMaximumWidth(250)
        has_szr = bool(len(self.seizures) > 0)
        self.widget.tab_widget.setTabVisible(2, has_szr)
        
        self.widget.tab_widget.setCurrentIndex(0)
        
        # update probe list in settings widget
        items = [f'probe {i}' for i in range(len(self.probe_list))]
        self.widget.probes_qlist.addItems(items)
        self.widget.probes_qlist.setCurrentRow(self.iprb)
        row_height = self.widget.probes_qlist.sizeHintForRow(0)
        self.widget.probes_qlist.setMaximumHeight(row_height * 2 + 5)
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
        # SWITCH_THE_PROBE
        self.widget.probes_qlist.currentRowChanged.connect(self.SWITCH_THE_PROBE)
        # updated event channel inputs
        for item in self.widget.ch_inputs:
            item.valueChanged.connect(self.widget.emit_ch_signal)
        self.widget.ch_signal.connect(self.update_event_channels)
        # reset event channels to auto
        self.widget.theta_reset.clicked.connect(lambda x: self.reset_ch('theta'))
        self.widget.swr_reset.clicked.connect(lambda x: self.reset_ch('swr'))
        self.widget.ds_reset.clicked.connect(lambda x: self.reset_ch('ds'))
        # updated noise channels
        self.main_fig.updated_noise_signal.connect(self.widget.update_noise_from_plot)
        self.widget.clean2noise_btn.clicked.connect(self.widget.clean2noise)
        self.widget.noise2clean_btn.clicked.connect(self.widget.noise2clean)
        self.widget.save_noise_btn.clicked.connect(lambda: print('save_noise_btn clicked'))
        self.widget.noise_signal.connect(self.update_noise_channels)
        
        # view popup windows
        self.widget.swr_event_btn.clicked.connect(self.view_swr)
        self.widget.ds_event_btn.clicked.connect(self.view_ds)
        # show event lines
        self.widget.szr_show.toggled.connect(self.show_hide_events)
        self.widget.ds_show.toggled.connect(self.show_hide_events)
        self.widget.ds_show_rm.toggled.connect(self.show_hide_events)
        self.widget.swr_show.toggled.connect(self.show_hide_events)
        self.widget.swr_show_rm.toggled.connect(self.show_hide_events)
        # show given index/timepoint
        self.widget.tjump_btn.clicked.connect(lambda: self.main_fig.point_jump(self.widget.tjump.value(), 't'))
        self.widget.ijump_btn.clicked.connect(lambda: self.main_fig.point_jump(self.widget.ijump.value(), 'i'))
        # show next/previous event
        self.widget.ds_arrows.bgrp.idClicked.connect(lambda x: self.main_fig.event_jump(x, 'ds'))
        self.widget.swr_arrows.bgrp.idClicked.connect(lambda x: self.main_fig.event_jump(x, 'swr'))
        self.widget.szr_arrows.bgrp.idClicked.connect(lambda x: self.main_fig.event_jump(x, 'szr'))
        # toggle event type to add on double-click
        self.widget.ds_add.toggled.connect(lambda x: self.toggle_event_add(x, 'ds'))
        self.widget.swr_add.toggled.connect(lambda x: self.toggle_event_add(x, 'swr'))
        
        # show frequency band plots
        self.widget.plot_freq_pwr.toggled.connect(self.toggle_main_plot)
        # save event channels
        self.widget.save_btn.clicked.connect(self.save_channels)
        self.widget.debug_btn.clicked.connect(self.debug)
        self.widget.probes_qlist.setFocus(True)
    
    
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
        show_szr = bool(self.widget.szr_show.isChecked())       # show seizures
        self.main_fig.SHOW_DS = bool(show_ds)
        self.main_fig.SHOW_DS_RM = bool(show_ds_rm)
        self.main_fig.SHOW_SWR = bool(show_swr)
        self.main_fig.SHOW_SWR_RM = bool(show_swr_rm)
        self.main_fig.SHOW_SZR = bool(show_szr)
        
        self.widget.ds_arrows.setEnabled(show_ds)
        if not show_ds: self.widget.ds_add.setChecked(False)
        self.widget.ds_add.setEnabled(show_ds)
        self.widget.ds_show_rm.setEnabled(show_ds)
        self.widget.swr_arrows.setEnabled(show_swr)
        if not show_swr: self.widget.swr_add.setChecked(False)
        self.widget.swr_add.setEnabled(show_swr)
        self.widget.swr_show_rm.setEnabled(show_swr)
        self.widget.szr_arrows.setEnabled(show_szr)
        a = bool(self.main_fig.iw.i.val in self.main_fig.seizures_mid)
        self.widget.szr_elim.setEnabled(bool(show_szr and a))
        
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
        self.event_channel_list[self.iprb] = list(self.event_channels)
        
        
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
        self.ripple_chan = self.widget.swr_input.value()
        self.swr_fig = IFigSWR(self.ripple_chan, self.SWR_ALL, self.DATA, #self.DATA['swr'], 
                               self.lfp_time, self.lfp_fs, PARAMS=self.PARAMS, 
                               NOISE_TRAIN=self.NOISE_TRAIN, thresholds=self.SWR_THRES[self.ripple_chan])
        sns.despine(self.swr_fig.fig)
        
        self.swr_popup = EventViewPopup(self.ripple_chan, self.channels, self.SWR_ALL, self.swr_fig, parent=self)
        self.swr_popup.setWindowTitle(f'Sharp-wave ripples on channel {self.ripple_chan}')
        # add SWR thresholds to event view widget
        self.swr_popup.evw.add_view_btns(['Min. height', 'Min. width'], 'threshold')
        self.swr_popup.evw.add_view_btns(['Ripple envelope', 'Ripple duration'], 'feature')
        # set slider states, initialize plot
        n = int(self.swr_fig.DF_MEAN.loc[self.ripple_chan].n_valid/2)
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
        self.swr_popup.show()
       
    
    def view_ds(self):
        """ Launch DS analysis popup """
        self.hil_chan = self.widget.hil_input.value()
        
        self.ds_fig = IFigDS(self.hil_chan, self.DS_ALL, self.DATA,
                             self.lfp_time, self.lfp_fs, PARAMS=self.PARAMS, 
                             NOISE_TRAIN=self.NOISE_TRAIN, thresholds=self.DS_THRES[self.hil_chan])
        sns.despine(self.ds_fig.fig)
        
        self.ds_popup = EventViewPopup(self.hil_chan, self.channels, self.DS_ALL, self.ds_fig, parent=self)
        self.ds_popup.setWindowTitle(f'Dentate spikes on channel {self.hil_chan}')
        self.ds_popup.evw.add_view_btns(['Half-width', 'Half-prom. height'], 'feature')
        # set slider states, initialize plot
        n = int(self.ds_fig.DF_MEAN.loc[self.hil_chan].n_valid/2)
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
        self.ds_popup.show()
        
    
    def save_channels(self):
        """ Save event channels to .npy file """
        #ffindme
            
        # save dataframes listed by probe
        self.swr_list[self.iprb] = pd.DataFrame(self.main_fig.SWR_ALL)
        self.ds_list[self.iprb]  = pd.DataFrame(self.main_fig.DS_ALL)
        # update to match current file
        self.swr_list_file[self.iprb] = pd.DataFrame(self.main_fig.SWR_ALL)
        self.ds_list_file[self.iprb] = pd.DataFrame(self.main_fig.DS_ALL)
        
        # load ALL_DS and ALL_SWR event dataframes, update with current probe
        ALL_SWR = pd.concat(self.swr_list_file, keys=range(len(self.swr_list_file)), ignore_index=False)
        ALL_DS = pd.concat(self.ds_list_file, keys=range(len(self.ds_list_file)), ignore_index=False)
        ALL_SWR = ALL_SWR.drop('ch', axis=1)
        ALL_DS = ALL_DS.drop('ch', axis=1)
        ALL_SWR.to_csv(Path(self.ddir, 'ALL_SWR'), index_label=False)
        ALL_DS.to_csv(Path(self.ddir, 'ALL_DS'), index_label=False)
        
        # save event DFs for current probe
        self.DS_ALL = pd.DataFrame(self.ds_list[self.iprb])
        self.SWR_ALL = pd.DataFrame(self.swr_list[self.iprb])
        
        # save specific event DFs for current probe
        DS_DF  = pd.DataFrame(self.DS_ALL.loc[self.hil_chan])
        SWR_DF = pd.DataFrame(self.SWR_ALL.loc[self.ripple_chan])
        
        SWR_DF.to_csv(Path(self.ddir, f'SWR_DF_{self.iprb}'), index_label=False)
        DS_DF.to_csv(Path(self.ddir, f'DS_DF_{self.iprb}'), index_label=False)
        
        # save event channels and noise annotation for current probe
        np.save(Path(self.ddir, f'theta_ripple_hil_chan_{self.iprb}.npy'), self.event_channels)
        np.save(Path(self.ddir, 'noise_channels.npy'), self.noise_list)
        
        # pop-up messagebox appears when save is complete
        msgbox = gi.MsgboxSave('Event channels saved!\nExit window?', parent=self)
        res = msgbox.exec()
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
        if self.check_unsaved_notes():
            super().reject()
    
    def debug(self):
        pdb.set_trace()
        


if __name__ == '__main__':
    #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/JG007_nosort'
    #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/shank3_saved/JG035_het'
    #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/shank3_saved/JG032_het'
    #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/jh201_stanford'
    ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/stanford/JF513_saved'
    #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/stanford/JF512_new'
    ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/neuralynx_saved'
    #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/stanford/deepak_saved2'
    #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/Chucky10mW'
    pyfx.qapp()
    #w = hippos()
    probe_group = prif.read_probeinterface(Path(ddir, 'probe_group'))
            
    w = ChannelSelectionWindow(ddir, probe_group.probes, 0)
   #w.widget.ds_event_btn.click()
    
    w.show()
    w.raise_()
    w.exec()
