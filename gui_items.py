#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:45:34 2024

@author: amandaschott
"""
import sys
import os
from pathlib import Path
import shutil
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtCore, QtGui
import probeinterface as prif
from probeinterface.plotting import plot_probe
import pdb
# custom modules
import pyfx
import ephys

# widget_list = ('QPushButton, QRadioButton, QLabel, QComboBox, QSpinBox, '
#                'QDoubleSpinBox, QLineEdit, QTextEdit')
basic_widget_style = ('QPushButton, QRadioButton, QCheckBox, QLabel, QComboBox, '
                      'QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QAbstractItemView'
                      '{padding : 2px;}'
                      'QWidget:focus {outline : none}')
basic_popup_style = 'QWidget {font-size : 15pt}' #+ basic_widget_style

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
          
          # 'QPushButton:checked {'
          # 'background-color : rgba%s;'  # dark
          # #'outline : 2px solid red;'
          # 'border : 3px solid red;'
          # 'border-radius : 2px;'
          # '}'
          )

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

def iter2str(v):
    if not hasattr(v, '__iter__'): 
        return str(v)
    return ', '.join(np.array(v,dtype='str').tolist())

def val2str(v):
    if np.ndim(v) < 2:
        txt = iter2str(v)
    elif np.ndim(v) == 2:
        txt = ', '.join([f'({x})' for x in map(iter2str, v)])
    return txt

def info2text2(info):
    keys, vals = zip(*info.items())
    vstr = [*map(str,vals)]
    klens = [*map(len, keys)]
    kmax=max(klens)
    padk = [*map(lambda k: k + '_'*(kmax-len(k)), keys)]
    rows = ['<tr><td align="center"><pre>'+pdk+' : '+'</pre></td><td><pre>'+v+'</pre></td></tr>' for pdk,v in zip(padk,vstr)]
    text = '<body><h2><pre>Recording Info</pre></h2><hr><table>' + ''.join(rows) + '</table></body>'
    return text
    #text = ''.join(rows)
    
def info2text(info, rich=True):
    sep = '<br>' if rich else os.linesep
    fmt = ('<tr>'
               '<td align="center"; style="background-color:#f2f2f2; border:2px solid #e6e6e6; white-space:nowrap;"><tt>%s</tt></td>'
               '<td align="center"><font size="4"></font></td>'
               '<td style="background-color:#f2f2f2; border:2px solid #e6e6e6;"><tt>%s</tt></td>'
           '</tr>')
    
    div_row = '<tr><td colspan="3"><hr></td></tr>'
    
    gen_rows = [fmt % (k,info[k]) for k in ['raw_data_path','recording_system','units']]
    gen_rows[1] = gen_rows[1].replace('recording_system','system')
    # probe rows
    keys = ['ports', 'nprobes', 'probe_nch']
    vals = [val2str(info[k]) for k in keys]
    probe_rows = [fmt % (a,b) for a,b in zip(keys,vals)]
    # recording rows
    rec_rows = [fmt % ('fs',        '%s Hz'        % info['fs']),  #    f"{info['nchannels']} Hz"  f'%s Hz'
                fmt % ('nchannels', '%s primary el.' % info['nchannels']),
                fmt % ('nsamples',  '%.1E bins' % info['nsamples']),
                fmt % ('tstart',    '%.2f s' % info['tstart']),
                fmt % ('tend',      '%.2f s' % info['tend']),
                fmt % ('duration',  '%.2f s' % info['dur'])]
    header = (
              '<h2 align="center"; style="background-color:#f2f2f2; border:2px solid red; padding:100px;"><tt>Recording Info</tt></h2>'
              )
    
    info_text = ('<body style="background-color:#e6e6e6;">' + header + 
                 '<br style="line-height:30%">' + 
                 '<table border-collapse="collapse"; cellspacing="0"; cellpadding="3">' +
                ''.join(gen_rows)   + str(div_row)     + #'</table>')#str(div_row) +
                ''.join(probe_rows) + str(div_row)     + #'</table>')
                ''.join(rec_rows)   + '</table>' + '</body')
    
    return info_text


def unique_fname(ddir, base_name):
    existing_files = os.listdir(ddir)
    fname = str(base_name)
    i = 1
    while fname in existing_files:
        new_name = fname + f' ({i})'
        if new_name not in existing_files:
            fname = str(new_name)
            break
        i += 1
    return fname


class MainSlider(matplotlib.widgets.Slider):
    """ Slider with enable/disable function """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsteps = 500
        self.init_style()
        
    def init_style(self):
        self._handle._markersize = 12
        self._handle._markeredgewidth = 1
        self.vline.set_color('none')
        self.valtext.set_visible(False)
        
        # baseline
        self.track_fc = '#d3d3d3'     # 'lightgray', (211, 211, 211)
        self.poly_fc  = '#3366cc'     # medium-light blue (51, 102, 204)
        self.handle_mec = '#a9a9a9'   # 'darkgray', (169, 169, 169)
        self.handle_mfc = '#ffffff'   # 'white', (255, 255, 255)
        # disabled
        self.track_fc_off = '#d3d3d3' # 'lightgray', (211, 211, 211)
        self.poly_fc_off  = '#d3d3d3' # 'lightgray', (211, 211, 211)
        self.handle_mec_off = '#a9a9a9'   # 'darkgray', (169, 169, 169)
        self.handle_mfc_off = '#dcdcdc'   # 'gainsboro', (220, 220, 220)
        self.track_alpha_off   = 0.3
        self.poly_alpha_off    = 0.5
        self.handle_alpha_off  = 0.5
        self.valtext_alpha_off = 0.2
        self.label_alpha_off   = 0.2
        self.set_style()
    
    def init_main_style(self, **kwargs):
        self._handle._markersize = 20
        self.poly_fc = '#4b0082'  # 'indigo', (75, 0, 130)
        if 'nsteps' in kwargs: self.nsteps = kwargs['nsteps']
        self.set_style()
    
    def set_style(self):
        self.track.set_facecolor(self.track_fc)
        self.poly.set_facecolor(self.poly_fc)
        self._handle.set_markeredgecolor(self.handle_mec)
        self._handle.set_markerfacecolor(self.handle_mfc)
        self.track.set_alpha(1)
        self.poly.set_alpha(1)
        self._handle.set_alpha(1)
        self.valtext.set_alpha(1)
        self.label.set_alpha(1)
    
    def set_style_off(self):
        self.track.set_facecolor(self.track_fc_off)
        self.poly.set_facecolor(self.poly_fc_off)
        self._handle.set_markeredgecolor(self.handle_mec_off)
        self._handle.set_markerfacecolor(self.handle_mfc_off)
        self.track.set_alpha(self.track_alpha_off)
        self.poly.set_alpha(self.poly_alpha_off)
        self._handle.set_alpha(self.handle_alpha_off)
        self.valtext.set_alpha(self.valtext_alpha_off)
        self.label.set_alpha(self.label_alpha_off)
    
    def key_step(self, x):
        if x==1:
            self.set_val(self.val + self.nsteps)
        elif x==0:
            self.set_val(self.val - self.nsteps)
    
    def enable(self, x):
        if x: self.set_style()
        else: self.set_style_off()
        

class CSlider(matplotlib.widgets.Slider):
    """ Slider with enable/disable function """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handle._markersize = 20
        self._handle._markeredgewidth = 2
        self.nsteps = 500
    
    def key_step(self, x):
        if x==1:
            self.set_val(self.val + self.nsteps)
        elif x==0:
            self.set_val(self.val - self.nsteps)
        
    def enable(self, x):
        if x:
            self.track.set_facecolor('lightgray')
            self.track.set_alpha(1)
            self.poly.set_facecolor('indigo')
            self.poly.set_alpha(1)
            self._handle.set_markeredgecolor('darkgray')
            self._handle.set_markerfacecolor('white')
            self._handle.set_alpha(1)
            self.valtext.set_alpha(1)
            self.label.set_alpha(1)
            
        else:
            self.track.set_facecolor('lightgray')
            self.track.set_alpha(0.3)
            self.poly.set_facecolor('lightgray')
            self.poly.set_alpha(0.5)
            self._handle.set_markeredgecolor('darkgray')
            self._handle.set_markerfacecolor('gainsboro')
            self._handle.set_alpha(0.5)
            self.valtext.set_alpha(0.2)
            self.label.set_alpha(0.2)


class EventArrows(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QPushButton {font-weight:bold; padding:2px;}')
        self.left = QtWidgets.QPushButton('\u2190') # unicode ← and →
        self.right = QtWidgets.QPushButton('\u2192')
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.setSpacing(1)
        hbox.setContentsMargins(0,0,0,0)
        hbox.addWidget(self.left)
        hbox.addWidget(self.right)
        self.bgrp = QtWidgets.QButtonGroup(self)
        self.bgrp.addButton(self.left, 0)
        self.bgrp.addButton(self.right, 1)
    
    
class ShowHideBtn(QtWidgets.QPushButton):
    def __init__(self, text_shown='Hide freq. band power', 
                 text_hidden='Show freq. band power', init_show=False, parent=None):
        #\u00BB , \u00AB
        super().__init__(parent)
        self.setCheckable(True)
        self.TEXTS = [text_hidden, text_shown]
        # set checked/visible or unchecked/hidden
        self.setChecked(init_show)
        self.setText(self.TEXTS[int(init_show)])
        
        self.toggled.connect(self.update_state)

    def update_state(self, show):
        self.setText(self.TEXTS[int(show)])


class ReverseSpinBox(QtWidgets.QSpinBox):
    """ Spin box with reversed increments (down=+1) to match LFP channels """
    def stepEnabled(self):
        if self.wrapping() or self.isReadOnly():
            return super().stepEnabled()
        ret = QtWidgets.QAbstractSpinBox.StepNone
        if self.value() > self.minimum():
            ret |= QtWidgets.QAbstractSpinBox.StepUpEnabled
        if self.value() < self.maximum():
            ret |= QtWidgets.QAbstractSpinBox.StepDownEnabled
        return ret

    def stepBy(self, steps):
        return super().stepBy(-steps)

class LabeledWidget(QtWidgets.QWidget):
    def __init__(self, widget=QtWidgets.QWidget, txt='', orientation='v', 
                 label_pos=0, **kwargs):
        super().__init__()
        assert orientation in ['h','v'] and label_pos in [0,1]
        self.setContentsMargins(0,0,0,0)
        if orientation == 'h': 
            self.layout = QtWidgets.QHBoxLayout(self)
        else: 
            self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(kwargs.get('spacing', 1))
        self.label = QtWidgets.QLabel(txt)
        self.qw = widget()
        self.layout.addWidget(self.qw, stretch=2)
        self.layout.insertWidget(label_pos, self.label, stretch=0)
    
    def text(self):
        return self.label.text()
    
    def setText(self, txt):
        self.label.setText(txt)
        
class LabeledSpinbox(LabeledWidget):
    def __init__(self, txt='', double=False, **kwargs):
        widget = QtWidgets.QDoubleSpinBox if double else QtWidgets.QSpinBox
        super().__init__(widget, txt, **kwargs)
        if 'suffix' in kwargs: self.qw.setSuffix(kwargs['suffix'])
        if 'minimum' in kwargs: self.qw.setMinimum(kwargs['minimum'])
        if 'maximum' in kwargs: self.qw.setMaximum(kwargs['maximum'])
        if 'range' in kwargs: self.qw.setRange(*kwargs['range'])
        if 'decimals' in kwargs: self.qw.setDecimals(kwargs['decimals'])
    
    def value(self):
        return self.qw.value()
    
    def setValue(self, val):
        self.qw.setValue(val)


class LabeledCombobox(LabeledWidget):
    def __init__(self, txt='', **kwargs):
        super().__init__(QtWidgets.QComboBox, txt, **kwargs)
    
    def addItems(self, items):
        return self.qw.addItems(items)
    
    def currentText(self): 
        return self.qw.currentText()
    
    def currentIndex(self):
        return self.qw.currentIndex()
    
    def setCurrentText(self, txt):
        self.qw.setCurrentText(txt)
    
    def setCurrentIndex(self, idx):
        self.qw.setCurrentIndex(idx)


class LabeledPushbutton(LabeledWidget):
    def __init__(self, txt='', orientation='h', label_pos=1, spacing=10, **kwargs):
        super().__init__(QtWidgets.QPushButton, txt, orientation, label_pos, spacing=spacing, **kwargs)
        if 'btn_txt' in kwargs: self.qw.setText(kwargs['btn_txt'])
        if 'icon' in kwargs: self.qw.setIcon(kwargs['icon'])
        if 'icon_size' in kwargs: self.qw.setIconSize(QtCore.QSize(kwargs['icon_size']))
        if 'ss' in kwargs: self.qw.setStyleSheet(kwargs['ss'])
    
    def isChecked(self):
        return self.qw.isChecked()
    
    def setCheckable(self, x):
        self.qw.setCheckable(x)


class SpinboxRange(QtWidgets.QWidget):
    range_changed_signal = QtCore.pyqtSignal()
    
    def __init__(self, double=False, alignment=QtCore.Qt.AlignLeft, parent=None, **kwargs):
        super().__init__(parent)
        if double:
            self.box0 = QtWidgets.QDoubleSpinBox()
            self.box1 = QtWidgets.QDoubleSpinBox()
        else:
            self.box0 = QtWidgets.QSpinBox()
            self.box1 = QtWidgets.QSpinBox()
        for box in [self.box0, self.box1]:
            box.setAlignment(alignment)
            if 'suffix' in kwargs: box.setSuffix(kwargs['suffix'])
            if 'minimum' in kwargs: box.setMinimum(kwargs['minimum'])
            if 'maximum' in kwargs: box.setMaximum(kwargs['maximum'])
            if 'decimals' in kwargs: box.setDecimals(kwargs['decimals'])
            if 'step' in kwargs: box.setSingleStep(kwargs['step'])
            box.valueChanged.connect(lambda: self.range_changed_signal.emit())
            
        self.dash = QtWidgets.QLabel(' — ')
        self.dash.setAlignment(QtCore.Qt.AlignCenter)
        
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.box0, stretch=2)
        self.layout.addWidget(self.dash, stretch=0)
        self.layout.addWidget(self.box1, stretch=2)
    
    def get_values(self):
        return [self.box0.value(), self.box1.value()]


class QEdit_HBox(QtWidgets.QHBoxLayout):
    def __init__(self, simple=False, colors=['gray','darkgreen'], parent=None):
        super().__init__(parent)
        self.simple_mode = simple
        self.c0, self.c1 = colors
        
        self.setContentsMargins(0,0,0,0)
        self.setSpacing(0)
        
        # ellipsis (...)
        self.ellipsis = QtWidgets.QLineEdit()
        self.ellipsis.setAlignment(QtCore.Qt.AlignCenter)
        self.ellipsis.setTextMargins(0,4,0,4)
        self.ellipsis.setReadOnly(True)
        self.ellipsis.setText('...')
        self.ellipsis.setMaximumWidth(20)
        # base path to directory (resizable)
        self.path = QtWidgets.QLineEdit()
        self.path.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.path.setTextMargins(0,4,0,4)
        self.path.setReadOnly(True)
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                        QtWidgets.QSizePolicy.Fixed)
        self.path.setSizePolicy(policy)
        # directory name (gets size priority according to text length)
        self.folder = QtWidgets.QLineEdit()
        self.folder.setTextMargins(0,4,0,4)
        self.folder.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.folder.setReadOnly(True)
        policy2 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)
        self.folder.setSizePolicy(policy2)
        # set attributes for all QLineEdit items
        self.qedits = [self.ellipsis, self.path, self.folder]
        ss = ('QLineEdit {'
              f'border : 2px groove {self.c0};'
              'border-left : %s;'
              'border-right : %s;'
              'font-weight : %s;'
              'padding : 0px;}'
              
              'QLineEdit:disabled {'
              'border-color : gainsboro;}'
              )
        self.ellipsis.setStyleSheet(ss % (f'2px groove {self.c0}', 'none', 'normal'))
        self.folder.setStyleSheet(ss % ('none', f'2px groove {self.c0}', 'bold'))
        if self.simple_mode:
            self.path.setStyleSheet(ss % (f'2px groove {self.c0}', f'2px groove {self.c0}', 'normal'))
        else:
            self.path.setStyleSheet(ss % ('none', 'none', 'normal'))
        
        self.addWidget(self.path)
        if not self.simple_mode:
            self.insertWidget(0, self.ellipsis)
            self.addWidget(self.folder)
    
    
    def update_qedit(self, ddir, x=False):
        # update QLineEdit text
        if self.simple_mode:
            self.path.setText(ddir)
            return
        
        dirs = ddir.split(os.sep)
        folder_txt = dirs.pop(-1)
        path_txt = os.sep.join(dirs) + os.sep
        self.qedits[1].setText(path_txt)
        self.qedits[2].setText(folder_txt)
        fm = self.qedits[2].fontMetrics()
        width = fm.horizontalAdvance(folder_txt) + int(fm.maxWidth()/2)
        self.qedits[2].setFixedWidth(width)
        
        c0, c1 = [self.c0, self.c1] if x else [self.c1, self.c0]
        for qedit in self.qedits:
            qedit.setStyleSheet(qedit.styleSheet().replace(c0, c1))
            

class StatusIcon(QtWidgets.QPushButton):
    def __init__(self, init_state=0):
        super().__init__()
        self.icons = [QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogNoButton),
                      QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogYesButton)]
        self.new_status(init_state)
        self.setStyleSheet('QPushButton,'
                            'QPushButton:default,'
                            'QPushButton:hover,'
                            'QPushButton:selected,'
                            'QPushButton:disabled,'
                            'QPushButton:pressed {'
                            'background-color: none;'
                               'border: none;'
                               'color: none;}')
    def new_status(self, x):
        self.setIcon(self.icons[int(x)])  # status icon
        

class AnalysisBtns(QtWidgets.QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # View/save event channels; always available for all processed data folders
        self.option1_widget = self.create_widget('Select event channels', 'green')
        self.option1_btn = self.option1_widget.findChild(QtWidgets.QPushButton)
        # Classify dentate spikes; requires event channels and probe DS_DF files
        self.option2_widget = self.create_widget('Classify dentate spikes', 'blue')
        self.option2_btn = self.option2_widget.findChild(QtWidgets.QPushButton)
        
        self.option1_widget.setEnabled(False)
        self.option2_widget.setEnabled(False)
    
    def create_widget(self, txt, c):
        widget = QtWidgets.QWidget()
        widget.setContentsMargins(0,0,0,0)
        hlay = QtWidgets.QHBoxLayout(widget)
        hlay.setContentsMargins(0,0,0,0)
        hlay.setSpacing(8)
        btn = QtWidgets.QPushButton()
        #btn.setCheckable(True)
        #clight = pyfx.hue(c, 0.7, 1); cdark = pyfx.hue(c, 0.4, 0)#; cdull = pyfx.hue(clight, 0.8, 0.5, alpha=0.5)
        btn.setStyleSheet(btn_ss % (pyfx.hue(c, 0.7, 1),  pyfx.hue(c, 0.4, 0)))
        lbl = QtWidgets.QLabel(txt)
        hlay.addWidget(btn)
        hlay.addWidget(lbl)
        #self.btn_grp.addButton(btn)
        return widget
        
    
    def ddir_toggled(self, ddir, probe_idx=0):
        self.option1_widget.setEnabled(False)
        self.option2_widget.setEnabled(False)
        
        if not os.path.isdir(ddir):
            return
        
        files = os.listdir(ddir)
        # required: basic LFP files
        if all([bool(f in files) for f in ['lfp_bp.npz', 'lfp_time.npy', 'lfp_fs.npy']]):
            self.option1_widget.setEnabled(True)  # req: basic LFP data
        
        # required: event channels file, DS_DF file for current probe
        if f'DS_DF_{probe_idx}' in files and f'theta_ripple_hil_chan_{probe_idx}.npy' in files:
            self.option2_widget.setEnabled(True)
            

class ComboBox(QtWidgets.QComboBox):
    
    def paintEvent(self, event):
        painter = QtWidgets.QStylePainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.Text))

        # draw the combobox frame, focusrect and selected etc.
        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QtWidgets.QStyle.CC_ComboBox, opt)

        if self.currentIndex() < 0:
            opt.palette.setBrush(
                QtGui.QPalette.ButtonText,
                opt.palette.brush(QtGui.QPalette.ButtonText).color().lighter(),
            )
            painter.setOpacity(0.5)
            if self.placeholderText():
                opt.currentText = self.placeholderText()

        # draw the icon and text
        painter.drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, opt)


class AddChannelWidget(QtWidgets.QWidget):
    def __init__(self, orientation='h', add_btn_pos='right',parent=None):
        super().__init__(parent)
        
        ### create widgets
        self.label = QtWidgets.QLabel('<u>Add channel</u>')
        self.ch_dropdown = ComboBox()  # channel dropdown menu
        self.add_btn = QtWidgets.QPushButton()  # annotate channel as "noise"
        self.add_btn.setFixedSize(25,25)
        self.add_btn.setStyleSheet('QPushButton {padding : 2px 0px 0px 2px;}')
        # set arrow icon direction
        fmt = 'QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_Arrow%s)'
        icon = eval(fmt % {'left':'Back','right':'Forward'}[add_btn_pos])
        
        #ffindme
        
        self.add_btn.setIcon(icon)
        self.add_btn.setIconSize(QtCore.QSize(18,18))
        self.clear_btn = QtWidgets.QPushButton('Clear channels')
        
        self.vlayout = QtWidgets.QVBoxLayout(self)
        self.vlayout.setContentsMargins(0,0,0,0)
        if orientation == 'h':
            self.horiz_layout(add_btn_pos=add_btn_pos)
        elif orientation == 'v':
            self.vert_layout()
        else:
            pass
        
    def horiz_layout(self, add_btn_pos):
        """ Layout used for DS/ripple event analysis popup """
        # dropdown next to add button
        hlay = QtWidgets.QHBoxLayout()
        hlay.setContentsMargins(0,0,0,0)
        hlay.setSpacing(1)
        hlay.addWidget(self.ch_dropdown)
        hlay.addWidget(self.add_btn) if add_btn_pos=='right' else hlay.insertWidget(0, self.add_btn)
        self.vlayout.addWidget(self.label)
        self.vlayout.addLayout(hlay)
        self.vlayout.addWidget(self.clear_btn)
        
    def vert_layout(self):
        """ Layout used for noise channel annotations """
        pass
    


class TableWidget(QtWidgets.QTableWidget):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.load_df(df)
        self.verticalHeader().hide()
        
        self.itemChanged.connect(self.print_update)
    
    def print_update(self, item):
        print('itemChanged')
        self.df.iloc[item.row(), item.column()] = int(item.text())
        
    
    def init_table(self, selected_columns = []):
        nRows = len(self.df.index)
        nColumns = len(selected_columns) or len(self.df.columns)
        self.setRowCount(nRows)
        self.setColumnCount(nColumns)

        self.setHorizontalHeaderLabels(selected_columns or self.df.columns)
        self.setVerticalHeaderLabels(self.df.index.astype(str))
        
        # Display an empty table
        if self.df.empty:
            self.clearContents()
            return

        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                item = QtWidgets.QTableWidgetItem(str(self.df.iat[row, col]))
                self.setItem(row, col, item)
        # Enable sorting on the table
        self.setSortingEnabled(True)
        # Enable column moving by drag and drop
        self.horizontalHeader().setSectionsMovable(True)
    
    def load_df(self, df, selected_columns = []):
        self.df = df
        self.init_table(selected_columns)


class TTabWidget(QtWidgets.QTabWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.currentChanged.connect(self.updateGeometry)

    def minimumSizeHint(self):
        return self.sizeHint()

    def sizeHint(self):
        lc = QtCore.QSize(0, 0)
        rc = QtCore.QSize(0, 0)
        opt = QtWidgets.QStyleOptionTabWidgetFrame()
        self.initStyleOption(opt)
        if self.cornerWidget(QtCore.Qt.TopLeftCorner):
            lc = self.cornerWidget(QtCore.Qt.TopLeftCorner).sizeHint()
        if self.cornerWidget(QtCore.Qt.TopRightCorner):
            rc = self.cornerWidget(QtCore.Qt.TopRightCorner).sizeHint()
        layout = self.findChild(QtWidgets.QStackedLayout)
        layoutHint = layout.currentWidget().sizeHint()
        tabHint = self.tabBar().sizeHint()
        if self.tabPosition() in (self.North, self.South):
            size = QtCore.QSize(
                max(layoutHint.width(), tabHint.width() + rc.width() + lc.width()), 
                layoutHint.height() + max(rc.height(), max(lc.height(), tabHint.height()))
            )
        else:
            size = QtCore.QSize(
                layoutHint.width() + max(rc.width(), max(lc.width(), tabHint.width())), 
                max(layoutHint.height(), tabHint.height() + rc.height() + lc.height())
            )
        return size

class MsgWindow(QtWidgets.QDialog):
    def __init__(self, msg='A message!', sub_msg='A smaller message!', title='A title!', parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.icon_btn = QtWidgets.QPushButton()
        self.icon_btn.setFixedSize(50, 50)
        self.icon_btn.setFlat(True)
        self.icon_btn.setIcon(QtGui.QIcon(':/icons/info_blue.png'))
        self.icon_btn.setIconSize(QtCore.QSize(45, 45))
        self.text_label = QtWidgets.QLabel(msg)
        self.text_label.setAlignment(QtCore.Qt.AlignCenter)
        self.text_label.setStyleSheet('QLabel {'
                                      'font-size : 15pt;'
                                      '}')
        self.subtext_label = QtWidgets.QLabel()
        self.subtext_label.setStyleSheet('QLabel {'
                                        'font-size : 12pt;'
                                        '}')
        self.main_grid = QtWidgets.QGridLayout()
        self.main_grid.addWidget(self.icon_btn, 0, 0, 2, 1)
        self.main_grid.addWidget(self.text_label, 0, 1)
        self.main_grid.addWidget(self.subtext_label, 1, 1)
        self.bbox = QtWidgets.QHBoxLayout()
        self.accept_btn = QtWidgets.QPushButton('Ok')
        self.reject_btn = QtWidgets.QPushButton('Close')
        self.bbox.addWidget(self.accept_btn)
        self.bbox.addWidget(self.reject_btn)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addLayout(self.main_grid)
        self.layout.addLayout(self.bbox)
        
        self.accept_btn.clicked.connect(self.accept)
        self.reject_btn.clicked.connect(self.reject)
        
        self.show()
        self.raise_()
        

class Msgbox(QtWidgets.QMessageBox):
    def __init__(self, msg='', sub_msg='', title='', no_buttons=False, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        # get icon label
        self.icon_label = self.findChild(QtWidgets.QLabel, 'qt_msgboxex_icon_label')
        # set main text
        self.setText(msg)
        self.label = self.findChild(QtWidgets.QLabel, 'qt_msgbox_label')
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        # set sub text
        self.setInformativeText('tmp')  # make sure label shows up in widget children 
        self.setInformativeText(sub_msg)
        self.sub_label = self.findChild(QtWidgets.QLabel, 'qt_msgbox_informativelabel')
        # locate button box
        self.bbox = self.findChild(QtWidgets.QDialogButtonBox, 'qt_msgbox_buttonbox')
    
    @classmethod
    def run(cls, *args, **kwargs):
        pyfx.qapp()
        msgbox = cls(*args, **kwargs)
        msgbox.show()
        res = msgbox.exec()
        return res
    
    
class MsgboxSave(Msgbox):
    def __init__(self, msg='Save successful!', sub_msg='', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        # pop-up messagebox appears when save is complete
        self.setIcon(QtWidgets.QMessageBox.Information)
        self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        # set check icon
        chk_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        px_size = self.icon_label.pixmap().size()
        self.setIconPixmap(chk_icon.pixmap(px_size))
        
        

class MsgboxError(Msgbox):
    def __init__(self, msg='Something went wrong!', sub_msg='', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        # pop-up messagebox appears when save is complete
        self.setIcon(QtWidgets.QMessageBox.Critical)
        self.setStandardButtons(QtWidgets.QMessageBox.Close)
        

class MsgboxInvalid(MsgboxError):
    def __init__(self, msg='Invalid file!', sub_msg='', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
    
    @classmethod
    def invalid_file(cls, filepath='', filetype='probe', parent=None):
        fopts =  ['probe', 'param', 'array']
        assert filetype in fopts
        ftxt = ['PROBE','PARAMETER','DATA'][fopts.index(filetype)]
        sub_msg = ''
        #findme
        if not os.path.isfile(filepath):
            msg = f'<h3><u>{ftxt} FILE DOES NOT EXIST</u>:</h3><br><nobr><code>{filepath}</code></nobr>'
        else:
            msg = f'<h3><u>INVALID {ftxt} FILE</u>:</h3><br><nobr><code>{filepath}</code></nobr>'
            if filetype == 'param':
                params, invalid_keys = ephys.read_param_file(filepath)
                sub_msg = f'<hr><code><u>MISSING PARAMS</u>: {", ".join(invalid_keys)}</code>'
        # launch messagebox
        msgbox = cls(msg=msg, sub_msg=sub_msg, parent=parent)
        msgbox.show()
        msgbox.raise_()
        res = msgbox.exec()
        if res == QtWidgets.QMessageBox.Open:
            return True   # keep file dialog open for another selection
        elif res == QtWidgets.QMessageBox.Close:
            return False  # close file dialog
    


class MsgboxWarning(Msgbox):
    def __init__(self, msg='Warning!', sub_msg='', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    
    
    @classmethod
    def overwrite_warning(cls, ppath, parent=None):
        # check whether selected folder contains any contents/selected file already exists
        ddir_ovr = bool(os.path.isdir(ppath) and len(os.listdir(ppath)) > 0)
        f_ovr = bool(os.path.isfile(ppath))
        if not any([ddir_ovr, f_ovr]):
            return True
        
        msgbox = cls(parent=parent)
        if ddir_ovr:
            n = len(os.listdir(ppath))
            msgbox.setText(f'The directory <code>{os.path.basename(ppath)}</code> contains '
                         f'<code>{n}</code> items.')#'<br><br>Overwrite existing files?')
            msgbox.setInformativeText('Overwrite existing files?')
            msgbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
                                      | QtWidgets.QMessageBox.Apply)
            merge_btn = msgbox.button(QtWidgets.QMessageBox.Apply)
            merge_btn.setText(merge_btn.text().replace('Apply','Merge'))
        elif f_ovr:
            msgbox.setText(f'The file <code>{os.path.basename(ppath)}</code> already exists.')
            msgbox.setInformativeText('Do you want to replace it?')
            msgbox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)    
        yes_btn = msgbox.button(QtWidgets.QMessageBox.Yes)
        yes_btn.setText(yes_btn.text().replace('Yes','Overwrite'))
        res = msgbox.exec()
        
        if res == QtWidgets.QMessageBox.Yes: 
            if os.path.isdir(ppath):
                shutil.rmtree(ppath) # delete any existing directory files
                os.makedirs(ppath)
            return True    # continue overwriting
        elif res == QtWidgets.QMessageBox.Apply:
            return True    # add new files to the existing directory contents
        else: return False # abort save attempt
        
    @classmethod
    def unsaved_changes_warning(cls, msg='Unsaved changes', sub_msg='Do you want to save your work?', parent=None):
        msgbox = cls(msg, sub_msg, parent=parent)
        msgbox.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard)
        msgbox.show()
        msgbox.raise_()
        res = msgbox.exec()
        if res == QtWidgets.QMessageBox.Discard:
            return True   # don't worry about changes
        elif res == QtWidgets.QMessageBox.Cancel:
            return False  # abort close attempt
        elif res == QtWidgets.QMessageBox.Save:
            return -1     # save changes and then close
        
        
class AuxDialog(QtWidgets.QDialog):
    def __init__(self, n, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle('AUX channels')
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(10)
        qlabel = QtWidgets.QLabel('Set AUX file names (leave blank to ignore)')
        # load list of previously saved aux files
        self.auxf = list(np.load('.aux_files.npy'))
        completer = QtWidgets.QCompleter(self.auxf, self)
        grid = QtWidgets.QGridLayout()
        self.qedits = []
        # create QLineEdit for each AUX channel
        for i in range(n):
            lbl = QtWidgets.QLabel(f'AUX {i}')
            qedit = QtWidgets.QLineEdit()
            qedit.setCompleter(completer)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(qedit, i, 1)
            self.qedits.append(qedit)
        # action button
        bbox = QtWidgets.QHBoxLayout()
        self.continue_btn = QtWidgets.QPushButton('Continue')
        self.continue_btn.clicked.connect(self.accept)
        self.clear_btn = QtWidgets.QPushButton('Clear all')
        self.clear_btn.clicked.connect(self.clear_files)
        bbox.addWidget(self.continue_btn)
        bbox.addWidget(self.clear_btn)
        # set up layout
        self.layout.addWidget(qlabel)
        line = pyfx.DividerLine()
        self.layout.addWidget(line)
        self.layout.addLayout(grid)
        self.layout.addLayout(bbox)
    
    def update_files(self):
        for i,qedit in enumerate(self.qedits):
            txt = qedit.text()
            if txt != '':
                if not txt.endswith('.npy'):
                    txt += '.npy'
            self.aux_files[i] = txt
    
    def clear_files(self):
        for qedit in self.qedits:
            qedit.setText('')
    
    def accept(self):
        self.aux_files = []
        for qedit in self.qedits:
            txt = qedit.text()
            if txt.endswith('.npy'):
                txt = txt[0:-4]
            if txt not in self.auxf:
                self.auxf.append(txt)
            fname = txt + ('' if txt == '' else '.npy')
            self.aux_files.append(fname)
        np.save('.aux_files.npy', self.auxf)
        super().accept()
        

class ParamObject(QtWidgets.QWidget):
    update_signal = QtCore.pyqtSignal(dict)
    
    def __init__(self, params={}, data_processing=True, ds_classification=True, 
                 el_geom=True, parent=None):
        super().__init__(parent)
        self.data_processing = data_processing
        self.ds_classification = ds_classification
        self.el_geom = el_geom
        
        self.gen_layout()
        self.update_gui_from_ddict(params)
        self.connect_signals()
        
        QtCore.QTimer.singleShot(50, self.adjust_labels)
        self.setStyleSheet('QWidget {'
                           'font-size : 15pt;'
                           '}'
                           'QToolTip {'
                           'background-color : lightyellow;'
                           'border : 2px solid black;'
                           'font-size : 15pt;'
                           'padding : 4px;'
                           '}')
        
    def gen_layout(self):
        
        ###   DATA PROCESSING   ###
        
        # labels
        self.lfp_fs_lbl = QtWidgets.QLabel('Downsampled FS')
        self.theta_lbl = QtWidgets.QLabel('Theta range')
        self.slow_gamma_lbl = QtWidgets.QLabel('Slow gamma range')
        self.fast_gamma_lbl = QtWidgets.QLabel('Fast gamma range')
        self.ds_freq_lbl = QtWidgets.QLabel('DS bandpass filter')
        self.ds_height_thr_lbl = QtWidgets.QLabel('DS height')
        self.ds_dist_thr_lbl = QtWidgets.QLabel('DS separation')
        self.ds_prom_thr_lbl = QtWidgets.QLabel('DS prominence')
        self.ds_wlen_lbl = QtWidgets.QLabel('DS window length')
        self.swr_freq_lbl = QtWidgets.QLabel('Ripple bandpass filter')
        self.swr_ch_bound_lbl = QtWidgets.QLabel('Ripple window')
        self.swr_height_thr_lbl = QtWidgets.QLabel('Ripple height')
        self.swr_min_thr_lbl = QtWidgets.QLabel('Ripple min. height')
        self.swr_dist_thr_lbl = QtWidgets.QLabel('Ripple separation')
        self.swr_min_dur_lbl = QtWidgets.QLabel('Ripple min. duration')
        self.swr_freq_thr_lbl = QtWidgets.QLabel('Ripple frequency')
        self.swr_freq_win_lbl = QtWidgets.QLabel('Ripple freq window')
        self.swr_maxamp_win_lbl = QtWidgets.QLabel('Ripple peak window')
        # descriptions
        self.lfp_fs_info = 'Target sampling rate (Hz) for downsampled LFP data.'
        self.theta_info = 'Bandpass filter cutoff frequencies (Hz) in the theta frequency range.'
        self.slow_gamma_info = 'Bandpass filter cutoff frequencies (Hz) in the slow gamma frequency range.'
        self.fast_gamma_info = 'Bandpass filter cutoff frequencies (Hz) in the fast gamma frequency range.'
        self.ds_freq_info = 'Bandpass filter cutoff frequencies (Hz) for detecting dentate spikes.'
        self.ds_height_thr_info = 'Minimum peak height (standard deviations) of a dentate spike.'
        self.ds_dist_thr_info = 'Minimum distance (ms) between neighboring dentate spikes.'
        self.ds_prom_thr_info = 'Minimum dentate spike prominence (relative to the surrounding signal).'
        self.ds_wlen_info = 'Window size (ms) for evaluating dentate spikes.'
        self.swr_freq_info = 'Bandpass filter cutoff frequencies (Hz) for detecting sharp-wave ripples.'
        self.swr_ch_bound_info =' Number of channels on either side of the ripple LFP to include in ripple profile.'
        self.swr_height_thr_info = 'Minimum height (standard deviations) at the peak of a ripple envelope.'
        self.swr_min_thr_info = 'Minimum height (standard deviations) at the edges of a ripple envelope.'
        self.swr_dist_thr_info = 'Minimum distance (ms) between neighboring ripples.'
        self.swr_min_dur_info = 'Minimum duration (ms) of a ripple.'
        self.swr_freq_thr_info = 'Minimum instantaneous frequency (Hz) of a ripple.'
        self.swr_freq_win_info = 'Window size (ms) for calculating ripple instantaneous frequency.'
        self.swr_maxamp_win_info = 'Window size (ms) to look for maximum ripple LFP amplitude.'
        # widgets
        self.lfp_fs = QtWidgets.QDoubleSpinBox()
        self.lfp_fs.setDecimals(1)
        self.lfp_fs.setSingleStep(0.5)
        self.lfp_fs.setSuffix(' Hz')
        kwargs = dict(double=True, decimals=1, step=0.5, maximum=999999, suffix=' Hz')
        self.theta = SpinboxRange(**kwargs)
        self.slow_gamma = SpinboxRange(**kwargs)
        self.fast_gamma = SpinboxRange(**kwargs)
        self.ds_freq = SpinboxRange(**kwargs)
        self.swr_freq = SpinboxRange(**kwargs)
        self.ds_height_thr = QtWidgets.QDoubleSpinBox()
        self.ds_height_thr.setSuffix(' S.D.')
        self.ds_dist_thr = QtWidgets.QSpinBox()
        self.ds_dist_thr.setSuffix(' ms')
        self.ds_prom_thr = QtWidgets.QDoubleSpinBox()
        self.ds_prom_thr.setSuffix(' a.u.')
        self.ds_wlen = QtWidgets.QSpinBox()
        self.ds_wlen.setSuffix(' ms')
        self.swr_ch_bound = QtWidgets.QSpinBox()
        self.swr_ch_bound.setSuffix(' channels')
        self.swr_height_thr = QtWidgets.QDoubleSpinBox()
        self.swr_height_thr.setSuffix(' S.D.')
        self.swr_min_thr = QtWidgets.QDoubleSpinBox()
        self.swr_min_thr.setSuffix(' S.D.')
        self.swr_dist_thr = QtWidgets.QSpinBox()
        self.swr_dist_thr.setSuffix(' ms')
        self.swr_min_dur = QtWidgets.QSpinBox()
        self.swr_min_dur.setSuffix(' ms')
        self.swr_freq_thr = QtWidgets.QDoubleSpinBox()
        self.swr_freq_thr.setDecimals(1)
        self.swr_freq_thr.setSuffix(' Hz')
        self.swr_freq_win = QtWidgets.QSpinBox()
        self.swr_freq_win.setSuffix(' ms')
        self.swr_maxamp_win = QtWidgets.QSpinBox()
        self.swr_maxamp_win.setSuffix(' ms')
        # row widgets
        self.lfp_fs_row = self.create_row(self.lfp_fs_lbl, self.lfp_fs, self.lfp_fs_info)
        self.theta_row = self.create_row(self.theta_lbl, self.theta, self.theta_info)
        self.slow_gamma_row = self.create_row(self.slow_gamma_lbl, self.slow_gamma, self.slow_gamma_info)
        self.fast_gamma_row = self.create_row(self.fast_gamma_lbl, self.fast_gamma, self.fast_gamma_info)
        self.ds_freq_row = self.create_row(self.ds_freq_lbl, self.ds_freq, self.ds_freq_info)
        self.ds_height_thr_row = self.create_row(self.ds_height_thr_lbl, self.ds_height_thr, self.ds_height_thr_info)
        self.ds_dist_thr_row = self.create_row(self.ds_dist_thr_lbl, self.ds_dist_thr, self.ds_dist_thr_info)
        self.ds_prom_thr_row = self.create_row(self.ds_prom_thr_lbl, self.ds_prom_thr, self.ds_prom_thr_info)
        self.ds_wlen_row = self.create_row(self.ds_wlen_lbl, self.ds_wlen, self.ds_wlen_info)
        self.swr_freq_row = self.create_row(self.swr_freq_lbl, self.swr_freq, self.swr_freq_info)
        self.swr_ch_bound_row = self.create_row(self.swr_ch_bound_lbl, self.swr_ch_bound, self.swr_ch_bound_info)
        self.swr_height_thr_row = self.create_row(self.swr_height_thr_lbl, self.swr_height_thr, self.swr_height_thr_info)
        self.swr_min_thr_row = self.create_row(self.swr_min_thr_lbl, self.swr_min_thr, self.swr_min_thr_info)
        self.swr_dist_thr_row = self.create_row(self.swr_dist_thr_lbl, self.swr_dist_thr, self.swr_dist_thr_info)
        self.swr_min_dur_row = self.create_row(self.swr_min_dur_lbl, self.swr_min_dur, self.swr_min_dur_info)
        self.swr_freq_thr_row = self.create_row(self.swr_freq_thr_lbl, self.swr_freq_thr, self.swr_freq_thr_info)
        self.swr_freq_win_row = self.create_row(self.swr_freq_win_lbl, self.swr_freq_win, self.swr_freq_win_info)
        self.swr_maxamp_win_row = self.create_row(self.swr_maxamp_win_lbl, self.swr_maxamp_win, self.swr_maxamp_win_info)
        
        ###   DS CLASSIFICATION   ###
        
        # labels
        self.csd_method_lbl = QtWidgets.QLabel('CSD mode')
        self.f_type_lbl = QtWidgets.QLabel('CSD filter')
        self.f_order_lbl = QtWidgets.QLabel('Filter order')
        self.f_sigma_lbl = QtWidgets.QLabel('Filter sigma (\u03C3)')
        self.vaknin_el_lbl = QtWidgets.QLabel('Vaknin electrode')
        self.tol_lbl = QtWidgets.QLabel('Tolerance')
        self.spline_nsteps_lbl = QtWidgets.QLabel('Spline steps')
        self.src_diam_lbl = QtWidgets.QLabel('Source diameter')
        self.src_h_lbl = QtWidgets.QLabel('Source thickness')
        self.cond_lbl = QtWidgets.QLabel('Conductivity')
        self.cond_top_lbl = QtWidgets.QLabel('Conductivity (top)')
        self.clus_algo_lbl = QtWidgets.QLabel('Clustering method')
        self.nclusters_lbl = QtWidgets.QLabel('# clusters')
        self.eps_lbl = QtWidgets.QLabel('Epsilon (\u03B5)')
        self.min_clus_samples_lbl = QtWidgets.QLabel('Min. cluster samples')
        # descriptions
        self.csd_method_info = 'Current source density (CSD) estimation method.'
        self.f_type_info = 'Spatial filter for estimated CSD.'
        self.f_order_info = 'CSD spatial filter settings (passed to scipy.signal method).'
        self.f_sigma_info = 'Sigma (or standard deviation) parameter; applies to Gaussian filter only.'
        self.vaknin_el_info = "Calculate CSD with or without Vaknin's method of duplicating endpoint electrodes."
        self.tol_info = 'Tolerance of numerical integration in CSD estimation; applies to step and spline methods only.'
        self.spline_nsteps_info = 'Number of upsampled data points in CSD estimation; applies to spline method only.'
        self.src_diam_info = 'Diameter (mm) of the assumed circular current sources.'
        self.src_h_info = 'Thickness (mm) of the assumed cylindrical current sources.'
        self.cond_info = 'Conductivity (Siemens per meter) through brain tissue.'
        self.cond_top_info = 'Conductivity (Siemens per meter) on top of brain tissue.'
        self.clus_algo_info = 'Clustering algorithm used to classify dentate spikes.'
        self.nclusters_info = 'Number of target clusters; applies to K-means algorithm only.'
        self.eps_info = 'Maximum distance between points in the same cluster; applies to DBSCAN algorithm only.'
        self.min_clus_samples_info = 'Minimum number of samples per cluster; applies to DBSCAN algorithm only.'
        # widgets
        self.csd_method = QtWidgets.QComboBox()
        self.csd_method.addItems(['Standard','Delta','Step','Spline'])
        self.f_type = QtWidgets.QComboBox()
        self.f_type.addItems(['Gaussian','Identity','Boxcar','Hamming','Triangular'])
        self.f_order = QtWidgets.QSpinBox()
        self.f_order.setMinimum(1)
        self.f_sigma = QtWidgets.QDoubleSpinBox()
        self.f_sigma.setDecimals(1)
        self.f_sigma.setSingleStep(0.1)
        self.vaknin_el = QtWidgets.QComboBox()
        self.vaknin_el.addItems(['True','False'])
        self.tol = QtWidgets.QDoubleSpinBox()
        self.tol.setDecimals(7)
        self.tol.setSingleStep(0.0000001)
        self.spline_nsteps = QtWidgets.QSpinBox()
        self.spline_nsteps.setMaximum(2500)
        self.src_diam = QtWidgets.QDoubleSpinBox()
        self.src_diam.setDecimals(3)
        self.src_diam.setSingleStep(0.01)
        self.src_diam.setSuffix(' mm')
        self.src_h = QtWidgets.QDoubleSpinBox()
        self.src_h.setDecimals(3)
        self.src_h.setSingleStep(0.01)
        self.src_h.setSuffix(' mm')
        self.cond = QtWidgets.QDoubleSpinBox()
        self.cond.setDecimals(3)
        self.cond.setSingleStep(0.01)
        self.cond.setSuffix(' S/m')
        self.cond_top = QtWidgets.QDoubleSpinBox()
        self.cond_top.setDecimals(3)
        self.cond_top.setSingleStep(0.01)
        self.cond_top.setSuffix(' S/m')
        self.clus_algo = QtWidgets.QComboBox()
        self.clus_algo.addItems(['K-means','DBSCAN'])
        self.nclusters = QtWidgets.QSpinBox()
        self.nclusters.setMinimum(1)
        self.nclusters.setSuffix(' clusters')
        self.eps = QtWidgets.QDoubleSpinBox()
        self.eps.setDecimals(2)
        self.eps.setSingleStep(0.1)
        self.eps.setSuffix(' a.u.')
        self.min_clus_samples = QtWidgets.QSpinBox()
        self.min_clus_samples.setMinimum(1)
        # row widgets
        self.csd_method_row = self.create_row(self.csd_method_lbl, self.csd_method, self.csd_method_info)
        self.f_type_row = self.create_row(self.f_type_lbl, self.f_type, self.f_type_info)
        self.f_order_row = self.create_row(self.f_order_lbl, self.f_order, self.f_order_info)
        self.f_sigma_row = self.create_row(self.f_sigma_lbl, self.f_sigma, self.f_sigma_info)
        self.vaknin_el_row = self.create_row(self.vaknin_el_lbl, self.vaknin_el, self.vaknin_el_info)
        self.tol_row = self.create_row(self.tol_lbl, self.tol, self.tol_info)
        self.spline_nsteps_row = self.create_row(self.spline_nsteps_lbl, self.spline_nsteps, self.spline_nsteps_info)
        self.src_diam_row = self.create_row(self.src_diam_lbl, self.src_diam, self.src_diam_info)
        self.src_h_row = self.create_row(self.src_h_lbl, self.src_h, self.src_h_info)
        self.cond_row = self.create_row(self.cond_lbl, self.cond, self.cond_info)
        self.cond_top_row = self.create_row(self.cond_top_lbl, self.cond_top, self.cond_top_info)
        self.clus_algo_row = self.create_row(self.clus_algo_lbl, self.clus_algo, self.clus_algo_info)
        self.nclusters_row = self.create_row(self.nclusters_lbl, self.nclusters, self.nclusters_info)
        self.eps_row = self.create_row(self.eps_lbl, self.eps, self.eps_info)
        self.min_clus_samples_row = self.create_row(self.min_clus_samples_lbl, self.min_clus_samples, self.min_clus_samples_info)
        
        ###   PROBE GEOMETRY   ###
        
        # labels
        self.el_w_lbl = QtWidgets.QLabel('Contact width')
        self.el_h_lbl = QtWidgets.QLabel('Contact height')
        self.el_shape_lbl = QtWidgets.QLabel('Contact shape')
        # descriptions
        self.el_w_info = 'Default width (\u00B5m) of probe electrode contacts.'
        self.el_h_info = 'Default height (\u00B5m) of probe electrode contacts.'
        self.el_shape_info = 'Default shape of probe electrode contacts.'
        # widgets
        self.el_w = QtWidgets.QDoubleSpinBox()
        self.el_w.setDecimals(1)
        self.el_w.setSuffix(' \u00B5m')
        self.el_h = QtWidgets.QDoubleSpinBox()
        self.el_h.setDecimals(1)
        self.el_h.setSuffix(' \u00B5m')
        self.el_shape = QtWidgets.QComboBox()
        self.el_shape.addItems(['Circle', 'Square', 'Rectangle'])
        # row widgets
        self.el_w_row = self.create_row(self.el_w_lbl, self.el_w, self.el_w_info)
        self.el_h_row = self.create_row(self.el_h_lbl, self.el_h, self.el_h_info)
        self.el_shape_row = self.create_row(self.el_shape_lbl, self.el_shape, self.el_shape_info)
        
        for sbox in [self.lfp_fs, self.ds_height_thr, self.ds_dist_thr, self.ds_prom_thr, self.ds_wlen, 
                     self.swr_ch_bound, self.swr_height_thr, self.swr_min_thr, self.swr_dist_thr, 
                     self.swr_min_dur, self.swr_freq_thr, self.swr_freq_win, self.swr_maxamp_win,
                     self.f_order, self.f_sigma, self.tol, self.src_diam, self.src_h, self.cond,
                     self.cond_top, self.nclusters, self.eps, self.min_clus_samples, self.el_w, self.el_h]:
            sbox.setMaximum(999999)
            
        self.LABELS = [self.lfp_fs_lbl,
                       self.theta_lbl,
                       self.slow_gamma_lbl,
                       self.fast_gamma_lbl,
                       self.ds_freq_lbl,
                       self.ds_height_thr_lbl,
                       self.ds_dist_thr_lbl,
                       self.ds_prom_thr_lbl,
                       self.ds_wlen_lbl,
                       self.swr_freq_lbl,
                       self.swr_ch_bound_lbl,
                       self.swr_height_thr_lbl,
                       self.swr_min_thr_lbl,
                       self.swr_dist_thr_lbl,
                       self.swr_min_dur_lbl,
                       self.swr_freq_thr_lbl,
                       self.swr_freq_win_lbl,
                       self.swr_maxamp_win_lbl,
                       self.csd_method_lbl,
                       self.f_type_lbl,
                       self.f_order_lbl,
                       self.f_sigma_lbl,
                       self.vaknin_el_lbl,
                       self.tol_lbl,
                       self.spline_nsteps_lbl,
                       self.src_diam_lbl,
                       self.src_h_lbl,
                       self.cond_lbl,
                       self.cond_top_lbl,
                       self.clus_algo_lbl,
                       self.nclusters_lbl,
                       self.eps_lbl,
                       self.min_clus_samples_lbl,
                       self.el_w_lbl,
                       self.el_h_lbl,
                       self.el_shape_lbl]
        self.WIDGETS = [self.lfp_fs,
                        self.theta,
                        self.slow_gamma,
                        self.fast_gamma,
                        self.ds_freq,
                        self.ds_height_thr,
                        self.ds_dist_thr,
                        self.ds_prom_thr,
                        self.ds_wlen,
                        self.swr_freq,
                        self.swr_ch_bound,
                        self.swr_height_thr,
                        self.swr_min_thr,
                        self.swr_dist_thr,
                        self.swr_min_dur,
                        self.swr_freq_thr,
                        self.swr_freq_win,
                        self.swr_maxamp_win,
                        self.csd_method,
                        self.f_type,
                        self.f_order,
                        self.f_sigma,
                        self.vaknin_el,
                        self.tol,
                        self.spline_nsteps,
                        self.src_diam,
                        self.src_h,
                        self.cond,
                        self.cond_top,
                        self.clus_algo,
                        self.nclusters,
                        self.eps,
                        self.min_clus_samples,
                        self.el_w,
                        self.el_h,
                        self.el_shape]
        # disable scroll wheel input for spinbox/dropdown widgets
        for w in self.WIDGETS:
            if w.__class__ == SpinboxRange:
                w.box0.wheelEvent = lambda event: None
                w.box1.wheelEvent = lambda event: None
            else:
                w.wheelEvent = lambda event: None
            
        ###   LAYOUT   ###
        
        self.layout = QtWidgets.QVBoxLayout(self)
        if self.data_processing == True:
            self.layout.addWidget(self.lfp_fs_row)
            self.layout.addWidget(self.theta_row)
            self.layout.addWidget(self.slow_gamma_row)
            self.layout.addWidget(self.fast_gamma_row)
            self.layout.addWidget(self.ds_freq_row)
            self.layout.addWidget(self.ds_height_thr_row)
            self.layout.addWidget(self.ds_dist_thr_row)
            self.layout.addWidget(self.ds_prom_thr_row)
            self.layout.addWidget(self.ds_wlen_row)
            self.layout.addWidget(self.swr_freq_row)
            self.layout.addWidget(self.swr_ch_bound_row)
            self.layout.addWidget(self.swr_height_thr_row)
            self.layout.addWidget(self.swr_min_thr_row)
            self.layout.addWidget(self.swr_dist_thr_row)
            self.layout.addWidget(self.swr_min_dur_row)
            self.layout.addWidget(self.swr_freq_thr_row)
            self.layout.addWidget(self.swr_freq_win_row)
            self.layout.addWidget(self.swr_maxamp_win_row) # 18 items
        if self.ds_classification == True:
            self.layout.addWidget(self.csd_method_row)
            self.layout.addWidget(self.f_type_row)
            self.layout.addWidget(self.f_order_row)
            self.layout.addWidget(self.f_sigma_row)
            self.layout.addWidget(self.vaknin_el_row)
            self.layout.addWidget(self.tol_row)
            self.layout.addWidget(self.spline_nsteps_row)
            self.layout.addWidget(self.src_diam_row)
            self.layout.addWidget(self.src_h_row)
            self.layout.addWidget(self.cond_row)
            self.layout.addWidget(self.cond_top_row)
            self.layout.addWidget(self.clus_algo_row)
            self.layout.addWidget(self.nclusters_row)
            self.layout.addWidget(self.eps_row)
            self.layout.addWidget(self.min_clus_samples_row) # 15 items
        if self.el_geom == True:
            self.layout.addWidget(self.el_w_row)
            self.layout.addWidget(self.el_h_row)
            self.layout.addWidget(self.el_shape_row) # 3 items
    
    def connect_signals(self):
        self.lfp_fs.valueChanged.connect(self.emit_signal)
        self.theta.range_changed_signal.connect(self.emit_signal)
        self.slow_gamma.range_changed_signal.connect(self.emit_signal)
        self.fast_gamma.range_changed_signal.connect(self.emit_signal)
        self.ds_freq.range_changed_signal.connect(self.emit_signal)
        self.swr_freq.range_changed_signal.connect(self.emit_signal)
        self.ds_height_thr.valueChanged.connect(self.emit_signal)
        self.ds_dist_thr.valueChanged.connect(self.emit_signal)
        self.ds_prom_thr.valueChanged.connect(self.emit_signal)
        self.ds_wlen.valueChanged.connect(self.emit_signal)
        self.swr_ch_bound.valueChanged.connect(self.emit_signal)
        self.swr_height_thr.valueChanged.connect(self.emit_signal)
        self.swr_min_thr.valueChanged.connect(self.emit_signal)
        self.swr_dist_thr.valueChanged.connect(self.emit_signal)
        self.swr_min_dur.valueChanged.connect(self.emit_signal)
        self.swr_freq_thr.valueChanged.connect(self.emit_signal)
        self.swr_freq_win.valueChanged.connect(self.emit_signal)
        self.swr_maxamp_win.valueChanged.connect(self.emit_signal)
        self.csd_method.currentIndexChanged.connect(self.emit_signal)
        self.f_type.currentIndexChanged.connect(self.emit_signal)
        self.f_order.valueChanged.connect(self.emit_signal)
        self.f_sigma.valueChanged.connect(self.emit_signal)
        self.vaknin_el.currentIndexChanged.connect(self.emit_signal)
        self.tol.valueChanged.connect(self.emit_signal)
        self.spline_nsteps.valueChanged.connect(self.emit_signal)
        self.src_diam.valueChanged.connect(self.emit_signal)
        self.src_h.valueChanged.connect(self.emit_signal)
        self.cond.valueChanged.connect(self.emit_signal)
        self.cond_top.valueChanged.connect(self.emit_signal)
        self.clus_algo.currentIndexChanged.connect(self.emit_signal)
        self.nclusters.valueChanged.connect(self.emit_signal)
        self.eps.valueChanged.connect(self.emit_signal)
        self.min_clus_samples.valueChanged.connect(self.emit_signal)
        self.el_w.valueChanged.connect(self.emit_signal)
        self.el_h.valueChanged.connect(self.emit_signal)
        self.el_shape.currentIndexChanged.connect(self.emit_signal)
    
    def create_row(self, label, widget, info=''):
        """ Return row widget containing parameter label and input item """
        label.setToolTip(info)
        w = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(w)
        hbox.setContentsMargins(0,0,0,0)
        hbox.addWidget(label, stretch=0)
        hbox.addWidget(widget, stretch=2)
        return w
    
    def adjust_labels(self):
        mw = max([lbl.width() for lbl in self.LABELS])
        for lbl in self.LABELS:
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setFixedWidth(mw)
    
    def emit_signal(self):
        PARAMS = self.ddict_from_gui()
        self.update_signal.emit(PARAMS)
    
    def ddict_from_gui(self):
        """ Return GUI widget values as parameter dictionary """
        ddict = dict(lfp_fs           = self.lfp_fs.value(),
                     theta            = self.theta.get_values(),
                     slow_gamma       = self.slow_gamma.get_values(),
                     fast_gamma       = self.fast_gamma.get_values(),
                     ds_freq          = self.ds_freq.get_values(),
                     swr_freq         = self.swr_freq.get_values(),
                     ds_height_thr    = self.ds_height_thr.value(),
                     ds_dist_thr      = self.ds_dist_thr.value(),
                     ds_prom_thr      = self.ds_prom_thr.value(),
                     ds_wlen          = self.ds_wlen.value(),
                     swr_ch_bound     = self.swr_ch_bound.value(),
                     swr_height_thr   = self.swr_height_thr.value(),
                     swr_min_thr      = self.swr_min_thr.value(),
                     swr_dist_thr     = self.swr_dist_thr.value(),
                     swr_min_dur      = self.swr_min_dur.value(),
                     swr_freq_thr     = self.swr_freq_thr.value(),
                     swr_freq_win     = self.swr_freq_win.value(),
                     swr_maxamp_win   = self.swr_maxamp_win.value(),
                     csd_method       = self.csd_method.currentText().lower(),
                     f_type           = self.f_type.currentText().lower(),
                     f_order          = self.f_order.value(),
                     f_sigma          = self.f_sigma.value(),
                     vaknin_el        = not bool(self.vaknin_el.currentIndex()),
                     tol              = self.tol.value(),
                     spline_nsteps    = self.spline_nsteps.value(),
                     src_diam         = self.src_diam.value(),
                     src_h            = self.src_h.value(),
                     cond             = self.cond.value(),
                     cond_top         = self.cond_top.value(),
                     clus_algo        = self.clus_algo.currentText().replace('-','').lower(),
                     nclusters        = self.nclusters.value(),
                     eps              = self.eps.value(),
                     min_clus_samples = self.min_clus_samples.value(),
                     el_w             = self.el_w.value(),
                     el_h             = self.el_h.value(),
                     el_shape         = self.el_shape.currentText().replace('angle','').lower()
                     )
        return ddict
        
    def update_gui_from_ddict(self, ddict):
        """ Set GUI widget values from input ddict """
        param_dict = dict(self.ddict_from_gui())
        param_dict.update(ddict)
        # data processing
        self.lfp_fs.setValue(param_dict['lfp_fs'])
        self.theta.box0.setValue(param_dict['theta'][0])
        self.theta.box1.setValue(param_dict['theta'][1])
        self.slow_gamma.box0.setValue(param_dict['slow_gamma'][0])
        self.slow_gamma.box1.setValue(param_dict['slow_gamma'][1])
        self.fast_gamma.box0.setValue(param_dict['fast_gamma'][0])
        self.fast_gamma.box1.setValue(param_dict['fast_gamma'][1])
        self.ds_freq.box0.setValue(param_dict['ds_freq'][0])
        self.ds_freq.box1.setValue(param_dict['ds_freq'][1])
        self.swr_freq.box0.setValue(param_dict['swr_freq'][0])
        self.swr_freq.box1.setValue(param_dict['swr_freq'][1])
        self.ds_height_thr.setValue(param_dict['ds_height_thr'])
        self.ds_dist_thr.setValue(int(param_dict['ds_dist_thr']))
        self.ds_prom_thr.setValue(param_dict['ds_prom_thr'])
        self.ds_wlen.setValue(int(param_dict['ds_wlen']))
        self.swr_ch_bound.setValue(int(param_dict['swr_ch_bound']))
        self.swr_height_thr.setValue(param_dict['swr_height_thr'])
        self.swr_min_thr.setValue(param_dict['swr_min_thr'])
        self.swr_dist_thr.setValue(int(param_dict['swr_dist_thr']))
        self.swr_min_dur.setValue(int(param_dict['swr_min_dur']))
        self.swr_freq_thr.setValue(param_dict['swr_freq_thr'])
        self.swr_freq_win.setValue(int(param_dict['swr_freq_win']))
        self.swr_maxamp_win.setValue(int(param_dict['swr_maxamp_win']))
        # DS classification
        self.csd_method.setCurrentText(param_dict['csd_method'].capitalize())
        self.f_type.setCurrentText(param_dict['f_type'].capitalize())
        self.f_order.setValue(int(param_dict['f_order']))
        self.f_sigma.setValue(param_dict['f_sigma'])
        self.vaknin_el.setCurrentIndex(int(not bool(param_dict['vaknin_el'])))
        self.tol.setValue(param_dict['tol'])
        self.spline_nsteps.setValue(int(param_dict['spline_nsteps']))
        self.src_diam.setValue(param_dict['src_diam'])
        self.src_h.setValue(param_dict['src_h'])
        self.cond.setValue(param_dict['cond'])
        self.cond_top.setValue(param_dict['cond_top'])
        self.clus_algo.setCurrentIndex(['kmeans','dbscan'].index(param_dict['clus_algo']))
        self.nclusters.setValue(int(param_dict['nclusters']))
        self.eps.setValue(param_dict['eps'])
        self.min_clus_samples.setValue(int(param_dict['min_clus_samples']))
        self.el_w.setValue(param_dict['el_w'])
        self.el_h.setValue(param_dict['el_h'])
        self.el_shape.setCurrentIndex(['circle','square','rect'].index(param_dict['el_shape']))

class MsgboxParams(Msgbox):
    PARAM_FILE = None
    def __init__(self, filepath='', title='Select parameter file', parent=None):
        # try loading parameters
        params, invalid_keys = ephys.read_param_file(filepath=filepath)
        fname = os.path.basename(filepath)
        if len(invalid_keys) == 0:
            msg = f'<h3>Parameter file <code>{fname}</code> not found.</h3>'
            sub_msg = ''
        else:
            msg = f'<h3>Parameter file <code>{fname}</code> contains invalid value(s).</h3>'
            sub_msg = f'<hr><code><u>MISSING PARAMS</u>: {", ".join(invalid_keys)}</code>'
            
        super().__init__(msg, sub_msg, title, no_buttons=False, parent=parent)
        #self.setStandardButtons(QtWidgets.QMessageBox.Close)
        self.open_btn = QtWidgets.QPushButton('Select existing file')
        self.save_btn = QtWidgets.QPushButton('Create new file')
        self.bbox.layout().addWidget(self.open_btn)
        self.bbox.layout().addWidget(self.save_btn)
        
        self.open_btn.clicked.connect(self.choose_param_file)
        self.save_btn.clicked.connect(self.create_param_file)
    
    def choose_param_file(self):
        params, fpath = FileDialog.load_file(filetype='param', parent=self)
        if params is not None:
            print(f'params = {params}')
            self.PARAM_FILE = str(fpath)
            self.accept()
    
    def create_param_file(self):
        self.param_dlg = ParamSettings(ddict=ephys.get_original_defaults(), parent=self)
        self.param_dlg.show()
        self.param_dlg.raise_()
        res = self.param_dlg.exec()
        if res:
            self.PARAM_FILE = str(self.param_dlg.SAVE_LOCATION)
            self.accept()
        
        
class ParamSettings(QtWidgets.QDialog):
    
    def __init__(self, ddict={}, parent=None):
        super().__init__(parent)
        
        # initialize parameter input widget
        self.main_widget = ParamObject(ddict)
        self.PARAMS = self.main_widget.ddict_from_gui()
        self.PARAMS_ORIG = dict(self.PARAMS)
        
        self.gen_layout()
        self.connect_signals()
        
        QtCore.QTimer.singleShot(50, lambda: self.qscroll.setMinimumWidth(int(self.main_widget.width())))
        
    def gen_layout(self):
        # embed main parameter widget in scroll area
        self.main_widget.setContentsMargins(0,0,15,0)
        self.qscroll = QtWidgets.QScrollArea()
        self.qscroll.horizontalScrollBar().hide()
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setWidget(self.main_widget)
        left_hash = QtWidgets.QLabel('#####')
        right_hash = QtWidgets.QLabel('#####')
        title = QtWidgets.QLabel('Default Parameters')
        for lbl in [left_hash, right_hash, title]:
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet('QLabel {'
                              'font-size : 18pt;'
                              'font-weight : bold;'
                              'text-decoration : none;'
                              '}')
        title_widget = QtWidgets.QFrame()
        title_widget.setFrameShape(QtWidgets.QFrame.Panel)
        title_widget.setFrameShadow(QtWidgets.QFrame.Sunken)
        title_widget.setLineWidth(3)
        title_widget.setMidLineWidth(3)
        title_row = QtWidgets.QHBoxLayout(title_widget)
        title_row.addWidget(left_hash, stretch=0)
        title_row.addWidget(title, stretch=2)
        title_row.addWidget(right_hash, stretch=0)
        bbox = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton('Save')
        #self.save_btn.setEnabled(False)
        self.reset_btn = QtWidgets.QPushButton('Reset parameters')
        self.reset_btn.setEnabled(False)
        self.close_btn = QtWidgets.QPushButton('Close')
        bbox.addWidget(self.save_btn)
        bbox.addWidget(self.reset_btn)
        bbox.addWidget(self.close_btn)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        #self.layout.setSpacing(20)
        self.layout.addWidget(title_widget, stretch=0)
        self.layout.addWidget(self.qscroll, stretch=2)
        self.layout.addLayout(bbox, stretch=0)
    
    def connect_signals(self):
        """ Connect widget signals to functions """
        self.main_widget.update_signal.connect(self.update_slot)
        self.save_btn.clicked.connect(self.save_param_file)
        self.reset_btn.clicked.connect(self.reset_params)
        self.close_btn.clicked.connect(self.reject)
    
    def update_slot(self, PARAMS):
        """ Update parameter dictionary based on user input """
        self.PARAMS.update(PARAMS)
        x = not all([self.PARAMS[k] == self.PARAMS_ORIG[k] for k in PARAMS.keys()])
        #self.save_btn.setEnabled(x)
        self.reset_btn.setEnabled(x)
    
    def reset_params(self):
        """ Reset parameters to original values """
        self.main_widget.update_gui_from_ddict(self.PARAMS_ORIG)
    
    def save_param_file(self):
        fpath = FileDialog.save_file(data_object=self.PARAMS, filetype='param', parent=self)
        if fpath:
            self.SAVE_LOCATION = fpath
            print('accept!')
            self.accept()


class FileDialog(QtWidgets.QFileDialog):
    array_exts = ['.npy', '.mat']#, '.csv']
    probe_exts = ['.json', '.prb', '.mat']
    LOADED_DATA = None
    
    def __init__(self, init_ddir='', load_or_save='load', 
                 is_directory=True, is_probe=False, is_array=False, is_param=False, 
                 parent=None, **kwargs):
        """
        init_ddir: optional starting directory
        load_or_save: "load" in existing directory/file or "save" new one
        is_directory: allow selection of all files (False) or only directories (True)
        is_probe: load probe object from configuration file
        is_array: load 2D array from raw data file
        is_param: load parameter dictionary from .txt file
        """
        super().__init__(parent)
        self.load_or_save = load_or_save
        self.is_probe, self.is_array, self.is_param = is_probe, is_array, is_param
        self.probe_exts = kwargs.get('probe_exts', self.probe_exts)
        self.array_exts = kwargs.get('array_exts', self.array_exts)
        
        options = self.Options()
        options |= self.DontUseNativeDialog
        
        self.setViewMode(self.List)
        self.setAcceptMode(self.AcceptOpen)  # open file
        
        # filter for array/probe files
        fx = lambda llist: ' '.join([*map(lambda x: '*'+x, llist)])
        if is_probe or is_array or is_param:
            is_directory = False
            if is_probe:
                ffilter = f'Probe files ({fx(self.probe_exts)})'
            elif is_array:
                ffilter = f'Data files ({fx(self.array_exts)})'
            elif is_param:
                ffilter = 'Text file (*.txt)'
            self.setNameFilter(ffilter)
            if self.load_or_save=='save': 
                self.setAcceptMode(self.AcceptSave)
        self.is_directory = is_directory
            
        # allow directories only
        if self.is_directory:
            self.setFileMode(self.Directory)
        else:
            if self.load_or_save == 'load':
                self.setFileMode(self.ExistingFile)  # allow existing files
            elif self.load_or_save == 'save':
                self.setFileMode(self.AnyFile)       # allow any file name
        
        try: self.setDirectory(init_ddir)
        except: print('init_ddir argument in FileDialog is invalid')
        self.setOptions(options)
        self.connect_signals()
        self.show()
    
    def connect_signals(self):
        self.btn = self.findChild(QtWidgets.QPushButton)
        self.lineEdit = self.findChild(QtWidgets.QLineEdit)
        self.stackedWidget = self.findChild(QtWidgets.QStackedWidget)
        self.view = self.stackedWidget.findChild(QtWidgets.QListView)
        self.view.selectionModel().selectionChanged.connect(self.updateText)
    
    def updateText(self, selected, deselected):
        if selected.indexes() == []:
            return
        # update contents of the line edit widget with the selected files
        txt = self.view.selectionModel().currentIndex().data()
        self.lineEdit.setText(txt)
    
    def file_validation(self):
        f = self.selectedFiles()[0]
        if self.is_probe:    # probe object (or None)
            self.LOADED_DATA = ephys.read_probe_file(f)
            filetype='probe'
        elif self.is_array:  # data array (or None)
            self.LOADED_DATA = ephys.read_array_file(f)
            filetype='array'
        elif self.is_param:  # parameter dictionary (or None)
            self.LOADED_DATA = ephys.read_param_file(f)[0]
            filetype='param'
        is_valid = bool(self.LOADED_DATA is not None)
        return is_valid, f, filetype
    
    def accept(self):
        if self.load_or_save=='load' and self.is_directory==False:
            is_valid, filepath, filetype = self.file_validation()
            if not is_valid:
                _ = MsgboxInvalid.invalid_file(filepath=filepath, filetype=filetype)
                self.LOADED_DATA = None
                return
        elif self.load_or_save == 'save':
            ddir = self.directory().path()
            if self.fileMode()==self.Directory and len(os.listdir(ddir))>0:
                res = MsgboxWarning.overwrite_warning(ddir, parent=self)
            else:
                res = MsgboxWarning.overwrite_warning(self.selectedFiles()[0], parent=self)
            if not res: return
        QtWidgets.QDialog.accept(self)
    
    @classmethod
    def get_file_kwargs(cls, filetype, init_ddir=None):
        """ Set keyword args based on file type/initial directory """
        kwargs = dict(init_ddir='', is_probe=False, is_array=False, is_param=False)
        if filetype == 'probe':
            kwargs['init_ddir'] = ephys.base_dirs()[2]
            kwargs['is_probe'] = True
        elif filetype == 'array':
            kwargs['init_ddir'] = ephys.base_dirs()[0]
            kwargs['is_array'] = True
        elif filetype == 'param':
            kwargs['init_ddir'] = str(os.getcwd())
            kwargs['is_param'] = True
        if init_ddir is not None:
            kwargs['init_ddir'] = init_ddir
        return kwargs
        
    @classmethod
    def load_file(cls, filetype, init_ddir=None, parent=None):
        assert filetype in ['probe','array','param']
        kwargs = cls.get_file_kwargs(filetype, init_ddir=init_ddir)
        # initialize file dialog
        dlg = cls(parent=parent, **kwargs)
        res = dlg.exec()
        if res:
            return (dlg.LOADED_DATA, dlg.selectedFiles()[0])
        else:
            return (None, None)
        

    @classmethod
    def save_file(cls, data_object, filetype, init_ddir=None, parent=None):
        assert filetype in ['probe','param']
        kwargs = cls.get_file_kwargs(filetype, init_ddir=init_ddir)
        
        # initialize file dialog
        pyfx.qapp()
        dlg = cls(load_or_save='save', parent=parent, **kwargs)
        if filetype == 'probe':   # filter for .json, .prb, and .mat extensions
            init_fname = f'{data_object.name}_config.json'
            dlg.fx = lambda txt: any(map(lambda ext: txt.endswith(ext), dlg.probe_exts))
        elif filetype == 'param': # filter for .txt extension
            init_fname = 'default_params.txt'
            dlg.fx = lambda txt: bool(txt.endswith('.txt'))
        dlg.lineEdit.textChanged.connect(lambda txt: dlg.btn.setEnabled(dlg.fx(txt)))
        dlg.lineEdit.setText(init_fname)
        dlg.show()
        dlg.raise_()
        res = dlg.exec()
        # return filepath for successful save, None otherwise
        if res:
            fpath = str(dlg.selectedFiles()[0])
            if filetype == 'probe':    # write probe file
                res2 = ephys.write_probe_file(data_object, fpath)
            elif filetype == 'param':  # write parameter file
                res2 = ephys.write_param_file(data_object, fpath)
            if res2:
                print(f'{filetype.capitalize()} file saved!')
                return fpath
        return None


class Popup(QtWidgets.QDialog):
    """ Simple popup window to display any widget(s) """
    def __init__(self, widgets=[], orientation='v', title='', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        if orientation   == 'v': self.layout = QtWidgets.QVBoxLayout(self)
        elif orientation == 'h': self.layout = QtWidgets.QHBoxLayout(self)
        for widget in widgets:
            self.layout.addWidget(widget)
        
    def center_window(self):
        qrect = self.frameGeometry()  # proxy rectangle for window with frame
        screen_rect = pyfx.ScreenRect()
        qrect.moveCenter(screen_rect.center())  # move center of qr to center of screen
        self.move(qrect.topLeft())
        
        
class MatplotlibPopup(Popup):
    """ Simple popup window to display Matplotlib figure """
    def __init__(self, fig, toolbar_pos='top', title='', parent=None):
        super().__init__(widgets=[], orientation='h', title=title, parent=parent)
        # create figure and canvas
        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        # create toolbar
        if toolbar_pos != 'none':
            self.toolbar = NavigationToolbar(self.canvas, self)
        if toolbar_pos in ['top','bottom', 'none']:
            self.canvas_layout = QtWidgets.QVBoxLayout()
        elif toolbar_pos in ['left','right']:
            self.canvas_layout = QtWidgets.QHBoxLayout()
            self.toolbar.setOrientation(QtCore.Qt.Vertical)
            self.toolbar.setMaximumWidth(30)
        # populate layout with canvas and toolbar (if indicated)
        self.canvas_layout.addWidget(self.canvas)
        if toolbar_pos != 'none':
            idx = 0 if toolbar_pos in ['top','left'] else 1
            self.canvas_layout.insertWidget(idx, self.toolbar)
            
        #self.layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.canvas_layout)
        #self.setLayout(self.layout)
    
    def closeEvent(self, event):
        plt.close()
        self.deleteLater()
        
        
class InfoView(QtWidgets.QDialog):
    def __init__(self, info=None, ddir=None, parent=None):
        super().__init__(parent)
        qrect = pyfx.ScreenRect(perc_height=0.5, perc_width=0.3, keep_aspect=False)
        self.setGeometry(qrect)
        self.info_text = 'Sorry - no text here :('
        
        # create QTextEdit for displaying file
        self.textwidget = QtWidgets.QTextEdit(self.info_text)
        self.textwidget.setAlignment(QtCore.Qt.AlignCenter)
        self.textwidget.setReadOnly(True)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.textwidget)


def create_filepath_row(txt, base_dir):
    fmt = '<code>{}</code>'
    w = QtWidgets.QWidget()
    vlay = QtWidgets.QVBoxLayout(w)
    row0 = QtWidgets.QHBoxLayout()
    row0.setContentsMargins(0,0,0,0)
    header = QtWidgets.QLabel(fmt.format(txt))
    row0.addWidget(header)
    row0.addStretch()
    row1 = QtWidgets.QHBoxLayout()
    row1.setContentsMargins(0,0,0,0)
    row1.setSpacing(5)
    qlabel = QtWidgets.QLabel(base_dir.format(base_dir))
    qlabel.setStyleSheet('QLabel {background-color:white;'
                         'border:1px solid gray; border-radius:4px; padding:5px;}')
    btn = QtWidgets.QPushButton()
    btn.setIcon(QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
    btn.setMinimumSize(30,30)
    btn.setIconSize(QtCore.QSize(20,20))
    row1.addWidget(qlabel, stretch=2)
    row1.addWidget(btn, stretch=0)
    vlay.addLayout(row0)
    vlay.addLayout(row1)
    return w,header,qlabel,btn


class SpinBoxDelegate(QtWidgets.QStyledItemDelegate):
    """
    Edit value of QTableView cell using a spinbox widget
    """
    def createEditor(self, parent, option, index):
        """ Create spinbox for table cell """
        spinbox = QtWidgets.QSpinBox(parent)
        spinbox.valueChanged.connect(lambda: self.commitData.emit(spinbox))
        return spinbox
    
    def setEditorData(self, editor, index):
        """ Initialize spinbox value from model data """
        editor.setValue(index.data())
    
    def setModelData(self, editor, model, index):
        """ Update model data with new spinbox value """
        if editor.value() != index.data():
            model.setData(index, editor.value(), QtCore.Qt.EditRole)
            
            
class QtWaitingSpinner(QtWidgets.QWidget):
    # initialize class variables
    mColor = QtGui.QColor(QtCore.Qt.blue)
    mRoundness = 100.0
    mMinimumTrailOpacity = 31.4159265358979323846
    mTrailFadePercentage = 50.0
    mRevolutionsPerSecond = 1
    mNumberOfLines = 20
    mLineLength = 15
    mLineWidth = 5
    mInnerRadius = 50
    mCurrentCounter = 0
    mIsSpinning = False

    def __init__(self, centerOnParent=True, disableParentWhenSpinning=True, disabledWidget=None, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        self.mCenterOnParent = centerOnParent
        self.mDisableParentWhenSpinning = disableParentWhenSpinning
        self.disabledWidget = disabledWidget
        self.initialize()
    
    def initialize(self):
        # connect timer to rotate function
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.updateSize()
        self.updateTimer()
        self.hide()


    @QtCore.pyqtSlot()
    def rotate(self):
        self.mCurrentCounter += 1
        if self.mCurrentCounter > self.numberOfLines():
            self.mCurrentCounter = 0
        self.update()

    def updateSize(self):
        # adjust widget size based on input params for spinner radius/arm length
        size = (self.mInnerRadius + self.mLineLength) * 2
        self.setFixedSize(size, size)
        
    def updateTimer(self):
        # set timer to rotate spinner every revolution
        self.timer.setInterval(int(1000 / (self.mNumberOfLines * self.mRevolutionsPerSecond)))

    def updatePosition(self):
        # adjust widget position to stay in the center of parent window
        if self.parentWidget() and self.mCenterOnParent:
            self.move(int(self.parentWidget().width() / 2 - self.width() / 2),
                      int(self.parentWidget().height() / 2 - self.height() / 2))

    def lineCountDistanceFromPrimary(self, current, primary, totalNrOfLines):
        # calculate distance between a given line and the "primary" line
        distance = primary - current
        if distance < 0:
            distance += totalNrOfLines
        return distance

    def currentLineColor(self, countDistance, totalNrOfLines, trailFadePerc, minOpacity, color):
        # adjust color shading on a line by distance from the primary line
        if countDistance == 0:
            return color

        minAlphaF = minOpacity / 100.0

        distanceThreshold = np.ceil((totalNrOfLines - 1) * trailFadePerc / 100.0)
        if countDistance > distanceThreshold:
            color.setAlphaF(minAlphaF)
        # color interpolation
        else:
            alphaDiff = self.mColor.alphaF() - minAlphaF
            gradient = alphaDiff / distanceThreshold + 1.0
            resultAlpha = color.alphaF() - gradient * countDistance
            resultAlpha = min(1.0, max(0.0, resultAlpha))
            color.setAlphaF(resultAlpha)
        return color

    def paintEvent(self, event):
        # initialize painter
        self.updatePosition()
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.transparent)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self.mCurrentCounter > self.mNumberOfLines:
            self.mCurrentCounter = 0
        painter.setPen(QtCore.Qt.NoPen)

        for i in range(self.mNumberOfLines):
            # draw & angle rounded rectangle evenly between lines based on current distance
            painter.save()
            painter.translate(self.mInnerRadius + self.mLineLength,
                              self.mInnerRadius + self.mLineLength)
            rotateAngle = 360.0 * i / self.mNumberOfLines
            painter.rotate(rotateAngle)
            painter.translate(self.mInnerRadius, 0)
            distance = self.lineCountDistanceFromPrimary(i, self.mCurrentCounter,
                                                          self.mNumberOfLines)
            color = self.currentLineColor(distance, self.mNumberOfLines,
                                          self.mTrailFadePercentage, self.mMinimumTrailOpacity, self.mColor)
            painter.setBrush(color)
            painter.drawRoundedRect(QtCore.QRect(0, -self.mLineWidth // 2, self.mLineLength, self.mLineLength),
                                    self.mRoundness, QtCore.Qt.RelativeSize)
            painter.restore()


    def start(self):
        # set spinner visible, disable parent widget if requested
        self.updatePosition()
        self.mIsSpinning = True  # track spinner activity
        self.show()
        
        if self.mDisableParentWhenSpinning:
            if self.parentWidget() and self.disabledWidget is None:
                self.parentWidget.setEnabled(False)
            elif self.disabledWidget is not None:
                self.disabledWidget.setEnabled(False)
        # start timer
        if not self.timer.isActive():
            self.timer.start()
            self.mCurrentCounter = 0

    def stop(self):
        # hide spinner, re-enable parent widget, stop timer
        self.mIsSpinning = False
        self.hide()
        
        if self.mDisableParentWhenSpinning:
            if self.parentWidget() and self.disabledWidget is None:
                self.parentWidget.setEnabled(True)
            elif self.disabledWidget is not None:
                self.disabledWidget.setEnabled(True)
        
        # if self.parentWidget() and self.mDisableParentWhenSpinning:
        #     self.parentWidget().setEnabled(True)

        if self.timer.isActive():
            self.timer.stop()
            self.mCurrentCounter = 0

    def setNumberOfLines(self, lines):
        self.mNumberOfLines = lines
        self.updateTimer()

    def setLineLength(self, length):
        self.mLineLength = length
        self.updateSize()

    def setLineWidth(self, width):
        self.mLineWidth = width
        self.updateSize()

    def setInnerRadius(self, radius):
        self.mInnerRadius = radius
        self.updateSize()

    def color(self):
        return self.mColor

    def roundness(self):
        return self.mRoundness

    def minimumTrailOpacity(self):
        return self.mMinimumTrailOpacity

    def trailFadePercentage(self):
        return self.mTrailFadePercentage

    def revolutionsPersSecond(self):
        return self.mRevolutionsPerSecond

    def numberOfLines(self):
        return self.mNumberOfLines

    def lineLength(self):
        return self.mLineLength

    def lineWidth(self):
        return self.mLineWidth

    def innerRadius(self):
        return self.mInnerRadius

    def isSpinning(self):
        return self.mIsSpinning

    def setRoundness(self, roundness):
        self.mRoundness = min(0.0, max(100, roundness))

    def setColor(self, color):
        self.mColor = color

    def setRevolutionsPerSecond(self, revolutionsPerSecond):
        self.mRevolutionsPerSecond = revolutionsPerSecond
        self.updateTimer()

    def setTrailFadePercentage(self, trail):
        self.mTrailFadePercentage = trail

    def setMinimumTrailOpacity(self, minimumTrailOpacity):
        self.mMinimumTrailOpacity = minimumTrailOpacity


class SpinnerWindow(QtWidgets.QWidget):
    def __init__(self, parent=None, show_label=True):
        super(SpinnerWindow, self).__init__(parent)
        self.win = parent
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(10)
        # create spinner object
        self.spinner_widget = QtWidgets.QWidget()
        self.spinner = QtWaitingSpinner(disabledWidget=self.win)
        self.spinner.setParent(self.spinner_widget)
        # create label to display updates
        self.spinner_label = QtWidgets.QLabel('')
        self.spinner_label.setAlignment(QtCore.Qt.AlignCenter)
        self.spinner_label.setWordWrap(True)
        self.spinner_label.setStyleSheet('QLabel'
                                         '{'
                                         'background-color : rgba(230,230,255,200);'
                                         'border : 2px double darkblue;'
                                         'border-radius : 8px;'
                                         'color : darkblue;'
                                         'font-size : 12pt;'
                                         'font-weight : 900;'
                                         'padding : 5px'
                                         '}')
        self.spinner_label.setFixedSize(int(self.spinner.width()*2), int(self.spinner.height()*0.75))
        # add label and spinner to layout, set size
        if show_label:
            self.layout.addWidget(self.spinner_label, alignment=QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.spinner_widget)
        self.setLayout(self.layout)
        self.setFixedSize(int(self.spinner.width()*2), int(self.spinner.height()*2))
        self.hide()
    
    def adjust_labelSize(self, lw=2, lh=0.75, ww=2, wh=2):
        self.spinner_label.setFixedSize(int(self.spinner.width()*lw), int(self.spinner.height()*lh))
    #def adjust_widgetSize(self, w=2, h=2):
        self.setFixedSize(int(self.spinner.width()*ww), int(self.spinner.height()*wh))
    
    def start_spinner(self):
        xpos = int(self.win.width() / 2 - self.width() / 2)
        ypos = int(self.win.height() / 2 - self.height() / 2)
        self.move(xpos, ypos)
        self.show()
        self.spinner.start()
    
    def stop_spinner(self):
        self.spinner.stop()
        self.spinner_label.setText('')
        self.hide()
        
    
    @QtCore.pyqtSlot(str)
    def report_progress_string(self, txt):
        self.spinner_label.setText(txt)