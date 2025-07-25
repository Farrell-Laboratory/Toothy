#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Toothy widgets

@author: amandaschott
"""
import os
import re
import shutil
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtCore, QtGui
import pdb
# custom modules
import QSS
import pyfx
import qparam
import ephys
import data_processing as dp
import resources_rc


##############################################################################
##############################################################################
################                                              ################
################              GENERAL PYQT WIDGETS            ################
################                                              ################
##############################################################################
##############################################################################


class ComboBox(QtWidgets.QComboBox):
    """ QComboBox with placeholder text """
    
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
        
        
class SpinBoxDelegate(QtWidgets.QStyledItemDelegate):
    """ Edit value of QTableView cell using a spinbox widget """
    
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


class LabeledWidget(QtWidgets.QWidget):
    """ Base widget grouped with a QLabel """
    
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
        """ Return label text """
        return self.label.text()
    
    def setText(self, txt):
        """ Set label text """
        self.label.setText(txt)
        
        
class LabeledSpinbox(LabeledWidget):
    """ QSpinBox or QDoubleSpinBox grouped with a QLabel """
    
    def __init__(self, txt='', double=False, **kwargs):
        widget = QtWidgets.QDoubleSpinBox if double else QtWidgets.QSpinBox
        super().__init__(widget, txt, **kwargs)
        if 'prefix' in kwargs: self.qw.setSuffix(kwargs['prefix'])
        if 'suffix' in kwargs: self.qw.setSuffix(kwargs['suffix'])
        if 'minimum' in kwargs: self.qw.setMinimum(kwargs['minimum'])
        if 'maximum' in kwargs: self.qw.setMaximum(kwargs['maximum'])
        if 'range' in kwargs: self.qw.setRange(*kwargs['range'])
        if 'decimals' in kwargs: self.qw.setDecimals(kwargs['decimals'])
        if 'step' in kwargs: self.qw.setSingleStep(kwargs['step'])
    
    def value(self):
        """ Return spinbox value """
        return self.qw.value()
    
    def setValue(self, val):
        """ Set spinbox value """
        self.qw.setValue(val)


class LabeledCombobox(LabeledWidget):
    """ QComboBox grouped with a QLabel """
    
    def __init__(self, txt='', **kwargs):
        super().__init__(QtWidgets.QComboBox, txt, **kwargs)
    
    def addItems(self, items):
        """ Add items to dropdown menu """
        return self.qw.addItems(items)
    
    def currentText(self): 
        """ Return currently selected item """
        return self.qw.currentText()
    
    def currentIndex(self):
        """ Return currently selected index """
        return self.qw.currentIndex()
    
    def setCurrentText(self, txt):
        """ Set current dropdown item by text """
        self.qw.setCurrentText(txt)
    
    def setCurrentIndex(self, idx):
        """ Set current dropdown item by index """
        self.qw.setCurrentIndex(idx)


class LabeledPushbutton(LabeledWidget):
    """ QPushButton grouped with a QLabel """
    
    def __init__(self, txt='', orientation='h', label_pos=1, spacing=10, **kwargs):
        super().__init__(QtWidgets.QPushButton, txt, orientation, label_pos, spacing=spacing, **kwargs)
        if 'btn_txt' in kwargs: self.qw.setText(kwargs['btn_txt'])
        if 'icon' in kwargs: self.qw.setIcon(kwargs['icon'])
        if 'icon_size' in kwargs: self.qw.setIconSize(QtCore.QSize(kwargs['icon_size']))
        if 'ss' in kwargs: self.qw.setStyleSheet(kwargs['ss'])
    
    def isChecked(self):
        """ Check whether button is checkable """
        return self.qw.isChecked()
    
    def setCheckable(self, x):
        """ Set button to "check" mode (True) or "click" mode (False) """
        self.qw.setCheckable(x)


class SpinboxRange(QtWidgets.QWidget):
    """ Linked pair of spinboxes representing a numeric range """
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
        """ Return values for both spinboxes """
        return [self.box0.value(), self.box1.value()]
    

##############################################################################
##############################################################################
################                                              ################
################               MATPLOTLIB WIDGETS             ################
################                                              ################
##############################################################################
##############################################################################


class MainSlider(matplotlib.widgets.Slider):
    """ Matplotlib slider with enable/disable and step functions """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsteps = 500
        self.init_style()
        
    def init_style(self):
        """ Initialize slider color/size/text """
        self._handle._markersize = 12
        self._handle._markeredgewidth = 1
        self.vline.set_color('none')
        self.valtext.set_visible(False)
        #self.label.set(bbox=dict(fc='w', ec='w', boxstyle='square,pad=0.05'))
        
        # baseline
        self.track_fc = '#d3d3d3'   # 'lightgray', (211, 211, 211)
        self.poly_fc  = '#3366cc'   # medium-light blue (51, 102, 204)
        self.handle_mec = '#a9a9a9' # 'darkgray', (169, 169, 169)
        self.handle_mfc = '#ffffff' # 'white', (255, 255, 255)
        # disabled
        self.track_fc_off = '#d3d3d3'   # 'lightgray', (211, 211, 211)
        self.poly_fc_off  = '#d3d3d3'   # 'lightgray', (211, 211, 211)
        self.handle_mec_off = '#a9a9a9' # 'darkgray', (169, 169, 169)
        self.handle_mfc_off = '#dcdcdc' # 'gainsboro', (220, 220, 220)
        self.track_alpha_off   = 0.3
        self.poly_alpha_off    = 0.5
        self.handle_alpha_off  = 0.5
        self.valtext_alpha_off = 0.2
        self.label_alpha_off   = 0.2
        self.set_style()
    
    def init_main_style(self, **kwargs):
        """ Set color/size for a "main" slider """
        self._handle._markersize = 15
        self.poly_fc = '#4b0082'  # 'indigo', (75, 0, 130)
        if 'nsteps' in kwargs: self.nsteps = kwargs['nsteps']
        self.set_style()
    
    def set_style(self):
        """ Update slider appearance when enabled """
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
        """ Update slider appearance when disabled """
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
        """ Increase/decrease slider value by $nsteps """
        if x==1:
            self.set_val(min(self.val + self.nsteps, self.valmax))
        elif x==0:
            self.set_val(max(self.val - self.nsteps, self.valmin))
    
    def enable(self, x):
        """ Enable/disable slider """
        if x: self.set_style()
        else: self.set_style_off()
        
    def set_val(self, val):
        """ Set slider value between minimum and maximum bounds """
        val = min(max(val, self.valmin), self.valmax)
        super().set_val(val)
    
    def update_range(self, valmin, valmax):
        """ Update minimum and maximum bounds """
        self.valmin = valmin
        self.valmax = valmax
        self.ax.set_xlim(valmin,valmax)
        verts = self.poly.get_xy()
        verts[[0,1,4],0] = valmin
        self.poly.set_xy(verts)
        self.canvas.draw_idle()


##############################################################################
##############################################################################
################                                              ################
################              SETTINGS BAR WIDGETS            ################
################                                              ################
##############################################################################
##############################################################################


class EventArrows(QtWidgets.QWidget):
    """ Pair of left/right arrow buttons """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QPushButton {font-weight:bold; padding:2px;}')
        self.left = QtWidgets.QPushButton('\u2190') # unicode ← and →
        self.right = QtWidgets.QPushButton('\u2192')
        _ = [btn.setAutoDefault(False) for btn in [self.left, self.right]]
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.setSpacing(1)
        hbox.setContentsMargins(0,0,0,0)
        hbox.addWidget(self.left)
        hbox.addWidget(self.right)
        self.bgrp = QtWidgets.QButtonGroup(self)
        self.bgrp.addButton(self.left, 0)
        self.bgrp.addButton(self.right, 1)
        
        
class StatusIcon(QtWidgets.QPushButton):
    """ QPushButton icon to visually indicate status (True=green, False=red) """
    
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
        """ Update icon for given status $x """
        self.setIcon(self.icons[int(x)])  # status icon
    
    
class ShowHideBtn(QtWidgets.QPushButton):
    """ Toggle button with automatic text updates """
    
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
        """ Update text for "shown" or "hidden" state """
        self.setText(self.TEXTS[int(show)])


class ReverseSpinBox(QtWidgets.QSpinBox):
    """ Spin box with reversed increments (down=+1) to match LFP channels """
    
    def stepEnabled(self):
        """ Reimplement stepEnabled function in opposite direction """
        if self.wrapping() or self.isReadOnly():
            return super().stepEnabled()
        ret = QtWidgets.QAbstractSpinBox.StepNone
        if self.value() > self.minimum():
            ret |= QtWidgets.QAbstractSpinBox.StepUpEnabled
        if self.value() < self.maximum():
            ret |= QtWidgets.QAbstractSpinBox.StepDownEnabled
        return ret

    def stepBy(self, steps):
        """ Reimplement stepBy function in opposite direction """
        return super().stepBy(-steps)
    

class EventGroupbox(QtWidgets.QGroupBox):
    """ Event boxes for channel selection, event viewing, and dataset curation """
    
    def __init__(self, event):
        event_dict = {'ds' : ['DSs', 'DG SPIKES', 'Hilus', 'red'],
                      'swr': ['ripples', 'RIPPLES', 'Ripple', 'green'],
                      'theta': ['theta', 'THETA', 'Fissure', 'blue']}
        assert event in event_dict
        evs, title, region, color = event_dict[event]
        super().__init__(title)
        self.setStyleSheet(pyfx.dict2ss(QSS.EVENT_GBOX))
        # row 1: interactive event channel inputs
        chan_input_lbl = QtWidgets.QLabel(f'{region} channel:')
        self.chan_input = ReverseSpinBox()
        self.chan_input.setKeyboardTracking(False)
        self.chan_reset = QtWidgets.QPushButton('\u27F3')
        self.chan_reset.setStyleSheet('QPushButton {font-size:25pt; padding:0px 0px 2px 2px;}')
        self.chan_reset.setMaximumSize(20,20)
        self.chan_reset.setAutoDefault(False)
        row1_widgets = [chan_input_lbl, self.chan_input, self.chan_reset]
        row1 = pyfx.get_widget_container('h', *row1_widgets, spacing=5)
        # row 2: launch event-specific popup window
        self.chan_event_btn = QtWidgets.QPushButton(f'View {evs}')
        self.chan_event_btn.setAutoDefault(False)
        self.redetect_btn = QtWidgets.QPushButton('Detect')
        self.redetect_btn.setAutoDefault(False)
        row2 = pyfx.get_widget_container('h', self.chan_event_btn, self.redetect_btn,
                                         stretch_factors=[5,2])
        self.redetect_btn.hide() # in progress
        # row 3: toggle event visibility and edit datasets
        self.chan_show = QtWidgets.QPushButton()
        self.chan_show.setCheckable(True)
        self.chan_show.setChecked(True)
        self.chan_show.setAutoDefault(False)
        self.chan_show.setStyleSheet(pyfx.dict2ss(QSS.EVENT_SHOW_BTN))
        self.chan_arrows = EventArrows()
        self.chan_add = QtWidgets.QCheckBox('Add')
        self.chan_add.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.chan_show_rm = QtWidgets.QCheckBox('Show deleted events')
        self.chan_show_rm.setChecked(False)
        l1, l2 = pyfx.DividerLine('v'), pyfx.DividerLine('v')
        row3a_widgets = [self.chan_show, l1, self.chan_arrows, l2, self.chan_add]
        row3a = pyfx.get_widget_container('h', *row3a_widgets, spacing=10)
        row3 = pyfx.get_widget_container('v', row3a, self.chan_show_rm)
        if event == 'theta':
            _ = [w.hide() for w in [*row3a_widgets, self.chan_show_rm]]
        # colorcoded line (DS=red)
        cc_dict = dict(QSS.EVENT_GBOX_LINE)
        cc_dict['QLabel'].update({'border-bottom-color':color})
        self.cc_line = QtWidgets.QLabel()
        self.cc_line.setStyleSheet(pyfx.dict2ss(cc_dict))
        # set layout
        widget = pyfx.get_widget_container('v', row1, row2, row3, self.cc_line, 
                                           widget='widget')
        widget.layout().insertSpacing(0, 10)
        central_layout = pyfx.get_widget_container('v', widget)
        central_layout.setContentsMargins(10,0,10,0)
        self.setLayout(central_layout)


class AddChannelWidget(QtWidgets.QWidget):
    """ Channel dropdown for comparing mean event waveforms """
    
    def __init__(self, orientation='h', add_btn_pos='right',parent=None):
        super().__init__(parent)
        
        ### create widgets
        self.label = QtWidgets.QLabel('<u>Add channel</u>')
        self.ch_dropdown = ComboBox()  # channel dropdown menu
        self.add_btn = QtWidgets.QPushButton()  # plot channel button
        self.add_btn.setFixedSize(25,25)
        self.add_btn.setStyleSheet('QPushButton {padding : 2px 0px 0px 2px;}')
        # set arrow icon direction
        fmt = 'QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_Arrow%s)'
        icon = eval(fmt % {'left':'Back','right':'Forward'}[add_btn_pos])
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


##############################################################################
##############################################################################
################                                              ################
################                 MODULE WIDGETS               ################
################                                              ################
##############################################################################
##############################################################################


class FileSelectionWidget(QtWidgets.QWidget):
    """ Base widget for interactive file selection and validation """
    
    signal = QtCore.pyqtSignal(bool)
    VALID_PPATH = False
    le_styledict = {'QLineEdit' : {'border-width' : '2px',
                                   'border-style' : 'groove',
                                   'border-color' : '%s',
                                   'padding' : '0px'},
                    'QLineEdit:disabled' : {'border-color' : 'gainsboro'}}
    
    def __init__(self, title='', parent=None):
        super().__init__(parent)
        
        self.ppath_lbl = QtWidgets.QLabel(title) # widget title
        self.icon_btn = StatusIcon(init_state=0) # filepath status icon
        self.le = QtWidgets.QLineEdit()          # filepath
        self.le.setTextMargins(0,4,0,4)
        self.le.setReadOnly(True)
        self.ppath_btn = QtWidgets.QPushButton() # file dialog launch button
        self.ppath_btn.setIcon(QtGui.QIcon(':/icons/folder.png'))
        self.ppath_btn.setMinimumSize(30,30)
        self.ppath_btn.setIconSize(QtCore.QSize(20,20))
        self.ppath_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        
        top_row = pyfx.get_widget_container('h', self.icon_btn, self.ppath_lbl, stretch_factors=[0,2])
        bottom_row = pyfx.get_widget_container('h', self.le, self.ppath_btn, stretch_factors=[2,0])
        self.vlay = pyfx.get_widget_container('v', top_row, bottom_row, spacing=5)
        self.setLayout(self.vlay)
        
    def get_init_ddir(self):
        """ Set initial directory of file dialog """
        init_ddir = self.le.text()
        if os.path.isfile(init_ddir):
            init_ddir = os.path.dirname(init_ddir)
        return init_ddir
    
    def update_filepath(self, ppath):
        """ Handle selection of a new filepath """
        x = self.validate_ppath(ppath)
        if x is None: return  # keep current filepath
        # update QLineEdit text and border color
        self.le.setText(ppath)
        self.update_status(x)
        self.signal.emit(self.VALID_PPATH)
    
    def update_status(self, x):
        """ Update widget state with "valid" or "invalid" filepath """
        c = ['maroon','darkgreen'][int(x)]
        self.le.setStyleSheet(pyfx.dict2ss(self.le_styledict) % c)
        self.icon_btn.new_status(x)  # update status icon
        self.VALID_PPATH = bool(x)
    
    def validate_ppath(self, ppath):
        """ Check if filepath meets some criteria (see subclasses) """
        return True


class NumericDelegate(QtWidgets.QStyledItemDelegate):
    """ Allows only digits (0-9) as inputs to editable table """
    
    def createEditor(self, parent, option, index):
        editor = super(NumericDelegate, self).createEditor(parent, option, index)
        if isinstance(editor, QtWidgets.QLineEdit):
            reg_ex = QtCore.QRegExp('[0-9]+')#+.?[0-9]{,2}")
            validator = QtGui.QRegExpValidator(reg_ex, editor)
            editor.setValidator(validator)
        return editor
    
    
class TableWidget(QtWidgets.QTableWidget):
    """ Display table for dynamically updated central dataframe """
    
    def __init__(self, df, static_columns=[], parent=None):
        super().__init__(parent)
        self.static_columns = static_columns
        self.load_df(df)
        self.verticalHeader().hide()
        self.itemChanged.connect(self.print_update)
    
    def print_update(self, item):
        """ Update dataframe from model """
        self.df.iloc[item.row(), item.column()] = int(item.text())
        
    def init_table(self, selected_columns=[]):
        """ Update table widget from dataframe """
        nRows = len(self.df.index)
        nColumns = len(selected_columns) or len(self.df.columns)
        self.setRowCount(nRows)
        self.setColumnCount(nColumns)
        self.setHorizontalHeaderLabels(selected_columns or self.df.columns)
        self.setVerticalHeaderLabels(self.df.index.astype(str))
        # display an empty table
        if self.df.empty:
            self.clearContents()
            return
        # set item values and flags
        col_names = list(self.df.columns)
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                item = QtWidgets.QTableWidgetItem(str(self.df.iat[row, col]))
                if col_names[col] in self.static_columns:
                    item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.setItem(row, col, item)
        # enable sorting and column moving
        self.setSortingEnabled(True)
        self.horizontalHeader().setSectionsMovable(True)
    
    def load_df(self, df, static_columns=None, selected_columns=[]):
        """ Set $df as central dataframe """
        if static_columns is not None:
            self.static_columns = static_columns
        self.df = df
        self.init_table(selected_columns)
    
    def keyPressEvent(self, event):
        """ Enable copy (Ctrl+C) and paste (Ctrl+V) keyboard shortcuts """
        super().keyPressEvent(event)
        is_ctrl = (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        if event.key() == QtCore.Qt.Key.Key_C and is_ctrl:
            # copy text from table cells
            copied_cells = sorted(self.selectedIndexes())
            copy_text = ''
            max_column = copied_cells[-1].column()
            for c in copied_cells:
                copy_text += self.item(c.row(), c.column()).text()
                if c.column() == max_column:
                    copy_text += '\n'
                else:
                    copy_text += '\t'
            QtWidgets.QApplication.clipboard().setText(copy_text)
        elif event.key() == QtCore.Qt.Key.Key_V and is_ctrl:
            # paste text into table cells
            copied_text = QtWidgets.QApplication.clipboard().text()
            digits_only = re.split(r'\D+', copied_text)
            vals = [d for d in digits_only if d != '']
            if len(vals) == 0: return
            top_row = sorted(self.selectedIndexes())[0].row()
            bottom_row = min([top_row+len(vals), self.rowCount()])
            icol = self.selectedIndexes()[0].column()
            for i,irow in enumerate(range(top_row, bottom_row)):
                self.setItem(irow, icol, QtWidgets.QTableWidgetItem(vals[i]))


##############################################################################
##############################################################################
################                                              ################
################                  MESSAGEBOXES                ################
################                                              ################
##############################################################################
##############################################################################


class Msgbox(QtWidgets.QMessageBox):
    """ Base MessageBox widget """
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
        """ Execute messagebox """
        pyfx.qapp()
        msgbox = cls(*args, **kwargs)
        msgbox.show()
        res = msgbox.exec()
        return res

class MsgboxQuestion(Msgbox):
    """ Question Messagebox """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QtWidgets.QMessageBox.Question)
        self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    
class MsgboxSave(Msgbox):
    """ Save Messagebox """
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
    """ Error Messagebox """
    def __init__(self, msg='Something went wrong!', sub_msg='', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        # pop-up messagebox appears when save is complete
        self.setIcon(QtWidgets.QMessageBox.Critical)
        self.setStandardButtons(QtWidgets.QMessageBox.Close)
        
class MsgboxInvalid(MsgboxError):
    """ Error Messagebox for invalid file """
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
                params, invalid_keys = qparam.read_param_file(filepath)
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
    """ Warning Messagebox """
    def __init__(self, msg='Warning!', sub_msg='', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
    
    @classmethod
    def overwrite_warning(cls, ppath, parent=None):
        """ Ask user to confirm overwrite of existing directory or file """
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
    def unsaved_changes_warning(cls, msg='Unsaved changes', 
                                sub_msg='Do you want to save your work?', parent=None):
        """ Ask user to save or discard unsaved changes """
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
        
# class MsgboxParams(Msgbox):
#     PARAM_FILE = None
#     def __init__(self, filepath='', title='Select parameter file', parent=None):
#         # try loading parameters
#         params, invalid_keys = qparam.read_param_file(filepath=filepath)
#         fname = os.path.basename(filepath)
#         if len(invalid_keys) == 0:
#             msg = f'<h3>Parameter file <code>{fname}</code> not found.</h3>'
#             sub_msg = ''
#         else:
#             msg = f'<h3>Parameter file <code>{fname}</code> contains invalid value(s).</h3>'
#             sub_msg = f'<hr><code><u>MISSING PARAMS</u>: {", ".join(invalid_keys)}</code>'
            
#         super().__init__(msg, sub_msg, title, no_buttons=False, parent=parent)
#         #self.setStandardButtons(QtWidgets.QMessageBox.Close)
#         self.open_btn = QtWidgets.QPushButton('Select existing file')
#         self.save_btn = QtWidgets.QPushButton('Create new file')
#         self.bbox.layout().addWidget(self.open_btn)
#         self.bbox.layout().addWidget(self.save_btn)
        
#         self.open_btn.clicked.connect(self.choose_param_file)
#         self.save_btn.clicked.connect(self.create_param_file)
    
#     def choose_param_file(self):
#         params, fpath = FileDialog.load_file(filetype='param', init_fname='')
#         if params is not None:
#             print(f'params = {params}')
#             self.PARAM_FILE = str(fpath)
#             self.accept()
    
#     def create_param_file(self):
#         self.param_dlg = ParameterPopup(ddict=qparam.get_original_defaults(), parent=self)
#         self.param_dlg.show()
#         self.param_dlg.raise_()
#         res = self.param_dlg.exec()
#         if res:
#             self.PARAM_FILE = str(self.param_dlg.SAVE_LOCATION)
#             self.accept()

class MsgWindow(QtWidgets.QDialog):
    """ QDialog window masquerading as a messagebox """
    pixmaps = dict(info     = QtWidgets.QStyle.SP_MessageBoxInformation,
                   critical = QtWidgets.QStyle.SP_MessageBoxCritical,
                   warning  = QtWidgets.QStyle.SP_MessageBoxWarning,
                   question = QtWidgets.QStyle.SP_MessageBoxQuestion,
                   check    = QtWidgets.QStyle.SP_DialogApplyButton)
    
    def __init__(self, icon='info', msg='A message!', sub_msg='', 
                 title='', btns=['Ok','Close'], parent=None):
        super().__init__(parent=parent)
        if isinstance(btns, str): btns = [btns]
        self.setWindowTitle(title)
        self.icon_btn = QtWidgets.QPushButton()
        self.icon_btn.setFixedSize(50, 50)
        self.icon_btn.setFlat(True)
        pixmap = self.pixmaps.get(str(icon), QtWidgets.QStyle.SP_MessageBoxInformation)
        qicon = self.style().standardIcon(pixmap)
        self.icon_btn.setIcon(QtGui.QIcon(qicon))
        self.icon_btn.setIconSize(QtCore.QSize(40, 40))
        self.text_label = QtWidgets.QLabel(msg)
        self.text_label.setAlignment(QtCore.Qt.AlignCenter)
        self.text_label.setWordWrap(True)
        #self.text_label.setStyleSheet('QLabel {font-size : 15pt;}')
        self.subtext_label = QtWidgets.QLabel(sub_msg)
        self.subtext_label.setAlignment(QtCore.Qt.AlignCenter)
        #self.subtext_label.setStyleSheet('QLabel {font-size : 12pt;}')
        if sub_msg == '': self.subtext_label.hide()
        self.main_grid = QtWidgets.QGridLayout()
        self.main_grid.addWidget(self.icon_btn, 0, 0, 2, 1)
        self.main_grid.addWidget(self.text_label, 0, 1)
        self.main_grid.addWidget(self.subtext_label, 1, 1)
        self.bbox = QtWidgets.QHBoxLayout()
        self.accept_btn = QtWidgets.QPushButton(btns[0])
        self.reject_btn = QtWidgets.QPushButton()
        if len(btns) > 1: self.reject_btn.setText(btns[1])
        else            : self.reject_btn.hide()
        self.bbox.addWidget(self.accept_btn)
        self.bbox.addWidget(self.reject_btn)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addLayout(self.main_grid)
        self.layout.addLayout(self.bbox)
        
        self.accept_btn.clicked.connect(self.accept)
        self.reject_btn.clicked.connect(self.reject)
        
        self.show()
        self.raise_()


##############################################################################
##############################################################################
################                                              ################
################                HELPER INTERFACES             ################
################                                              ################
##############################################################################
##############################################################################

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
        
        
class ButtonPopup(QtWidgets.QDialog):
    def __init__(self, *items, orientation='v', label='Select an option', 
                 title='', parent=None):
        super().__init__(parent)
        self.label = QtWidgets.QLabel(label)
        self.btns = [QtWidgets.QPushButton(x) for x in items]
        for btn in self.btns:
            btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
            btn.clicked.connect(self.select_btn)
        layout = pyfx.get_widget_container(orientation, self.label, 5, *self.btns, cm=None)
        self.setLayout(layout)
        self.setWindowTitle(title)
        self.show(); self.raise_()
    
    def select_btn(self):
        self.result = str(self.sender().text())
        self.accept()
    
    @classmethod
    def run(cls, *args, **kwargs):
        """ Execute button popup """
        pyfx.qapp()
        dlg = cls(*args, **kwargs)
        if dlg.exec():
            return str(dlg.result)
        
        
class AuxDialog(QtWidgets.QDialog):
    """ Interface for viewing and saving AUX channels (not implemented) """
    
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
        

##############################################################################
##############################################################################
################                                              ################
################                SPINNER ANIMATION             ################
################                                              ################
##############################################################################
##############################################################################
        
        
class QtWaitingSpinner(QtWidgets.QWidget):
    """ Dynamic "loading" spinner icon """
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

    def __init__(self, centerOnParent=True, disableParentWhenSpinning=True, 
                 disabledWidget=None, *args, **kwargs):
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
    """ Widget for controlling the spinner icon and displaying progress reports """
    
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
                                         'background-color : rgba(230,230,255,220);'
                                         'border : 2px double darkblue;'
                                         'border-radius : 8px;'
                                         'color : darkblue;'
                                         'font-size : 10pt;'
                                         'font-weight : 900;'
                                         'padding : 3px'
                                         '}')
        self.spinner_label.setFixedSize(int(self.spinner.width()*2.5), int(self.spinner.height()*1.25))
        # add label and spinner to layout, set size
        if show_label:
            self.layout.addWidget(self.spinner_label, alignment=QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.spinner_widget)
        self.setLayout(self.layout)
        self.setFixedSize(int(self.spinner.width()*3), int(self.spinner.height()*2))
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
        
        
##############################################################################
##############################################################################
################                                              ################
################                MAIN INTERFACES               ################
################                                              ################
##############################################################################
##############################################################################


class BaseFolderWidget(QtWidgets.QWidget):
    """ Base folders GUI for adjusting default data locations and files """
    
    path_updated_signal = QtCore.pyqtSignal(int)
    saved_signal = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.BASE_FOLDERS, self.flabels = self.init_folders()
        self.gen_layout()
        self.connect_signals()
    
    def init_folders(self):
        BASE_FOLDERS = [str(x) for x in ephys.base_dirs()]
        flabels = ['Raw Data Folder', 'Probe Configuration Folder',
                        'Default Probe File', 'Parameter File']
        return BASE_FOLDERS, flabels
        
    def makebtn(self, icon):
        """ Return icon button """
        btn = QtWidgets.QPushButton()
        btn.setFocusPolicy(QtCore.Qt.NoFocus)
        btn.setFixedSize(30,30)
        btn.setIcon(icon)
        btn.setIconSize(QtCore.QSize(20,20))
        return btn
    
    def makerow(self, i, icon2=None):
        """ Return widgets for a given row """
        ppath_label_ss = ('QLabel {background-color:white;'
                                  'border:1px solid gray;'
                                  'border-radius:4px;'
                                  'padding:5px;}')
        # create filepath label
        ppath_label = QtWidgets.QLabel(f'<code>{self.BASE_FOLDERS[i]}</code>')
        ppath_label.setStyleSheet(ppath_label_ss)
        # create title and button
        qlabel = QtWidgets.QLabel(f'<b>{self.flabels[i]}</b>')
        ftype = self.flabels[i].split(' ')[-1].lower() # "folder" or "file"
        url = f':/icons/{dict(folder="folder", file="load")[ftype]}.png'
        res = (ppath_label,btn) = [ppath_label, self.makebtn(QtGui.QIcon(url))]
        if icon2 is not None: # add a second button
            res.append(self.makebtn(QtGui.QIcon(f':/icons/{icon2}.png')))
        row1 = pyfx.get_widget_container('h', *res, spacing=5)
        
        #if i==2: res.append(self.makebtn(QtGui.QIcon(':/icons/trash.png')))
        #if i==3: res.append(self.makebtn(QtGui.QIcon(':/icons/load_txt.png')))
        w = pyfx.get_widget_container('v', qlabel, row1, spacing=2,
                                      widget='widget', cm=None)
        return res, w
    
    def gen_layout(self):
        """ Set up layout """
        # create folder labels and selection buttons
        #(self.raw_ppath_label, self.raw_btn), self.raw_w = self.makerow(0)
        (self.raw_ppath_label, self.raw_btn), self.raw_w = self.makerow(0)
        (self.prb_ppath_label, self.prb_btn), self.prb_w = self.makerow(1)
        (self.prbf_ppath_label, self.prbf_btn, self.prbf_clear), self.prbf_w = self.makerow(2, 'trash')
        (self.param_ppath_label, self.param_btn, self.param_auto), self.param_w = self.makerow(3, 'load_txt')
        self.param_auto.setToolTip('<big>Auto-generate parameter file from default values.</big>')
        self.ppath_labels = [self.raw_ppath_label, self.prb_ppath_label,
                             self.prbf_ppath_label, self.param_ppath_label]
        layout = pyfx.get_widget_container('v', self.raw_w, self.prb_w, self.prbf_w, 
                                           self.param_w, spacing=20)
        self.setLayout(layout)
        # action buttons
        self.save_btn = QtWidgets.QPushButton('Save')
        #self.save_btn.setStyleSheet(blue_btn_ss)
        self.save_btn.setEnabled(False)
    
    def connect_signals(self):
        """ Connect GUI inputs """
        self.raw_btn.clicked.connect(lambda: self.choose_base_ddir(0))
        self.prb_btn.clicked.connect(lambda: self.choose_base_ddir(1))
        self.prbf_btn.clicked.connect(self.choose_probe_file)
        self.param_btn.clicked.connect(self.choose_param_file)
        self.prbf_clear.clicked.connect(self.clear_probe_file)
        self.param_auto.clicked.connect(self.auto_param_file)
        self.path_updated_signal.connect(lambda i: self.update_base_ddir(i))
        self.save_btn.clicked.connect(self.save_base_folders)
    
    def choose_base_ddir(self, i):
        """ Select base folder for raw data or probe config files """
        init_ddir = str(self.BASE_FOLDERS[i])
        fmt = 'Base folder for %s'
        titles = [fmt % x for x in ['raw data', 'probe files']]
        # when activated, initialize at ddir and save new base folder at index i
        ddir = ephys.select_directory(init_ddir, title=titles[i], parent=self)
        if ddir:
            self.BASE_FOLDERS[i] = str(ddir)
            self.path_updated_signal.emit(i)
    
    def choose_probe_file(self):
        """ Select default probe file """
        init_ppath = str(self.BASE_FOLDERS[2])
        if not os.path.isfile(init_ppath):
            init_ppath = str(self.BASE_FOLDERS[1])
        probe, fpath = ephys.select_load_probe_file(init_ppath=init_ppath, parent=self)
        if probe is not None:
            self.BASE_FOLDERS[2] = str(fpath)
            self.path_updated_signal.emit(2)
        
    def clear_probe_file(self):
        """ Clear default probe file """
        self.BASE_FOLDERS[2] = ''
        self.path_updated_signal.emit(2)
    
    def choose_param_file(self):
        """ Select parameter file """
        init_ppath = str(self.BASE_FOLDERS[3])
        param_dict, fpath = ephys.select_load_param_file(init_ppath=init_ppath, parent=self)
        if param_dict is not None:
            self.BASE_FOLDERS[3] = str(fpath)
            self.path_updated_signal.emit(3)
            
    def auto_param_file(self):
        """ Save new parameter file with default values """
        fpath = ephys.select_save_param_file(qparam.get_original_defaults(), 
                                             title='Save default parameter file',
                                             parent=self)
        if fpath:
            self.BASE_FOLDERS[3] = str(fpath)
            self.path_updated_signal.emit(3)
            
    def update_base_ddir(self, i):
        """ Update filepath text to the selected location """
        txt = f'<code>{self.BASE_FOLDERS[i]}</code>'
        self.ppath_labels[i].setText(txt)
        self.save_btn.setEnabled(True)
    
    def save_base_folders(self):
        """ Save filepaths to "default_folders.txt" """
        ddir_list = list(self.BASE_FOLDERS)
        ephys.write_base_dirs(ddir_list)
        self.saved_signal.emit()
    
    
class BaseFolderPopup(QtWidgets.QDialog):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Base Folders')
        self.widget = BaseFolderWidget()
        self.widget.saved_signal.connect(self.accept)
        bbox = QtWidgets.QHBoxLayout()
        bbox.addWidget(self.widget.save_btn)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.widget)
        layout.addLayout(bbox)
        

class ParameterPopup(QtWidgets.QDialog):
    """ Settings GUI for editing and saving parameter values """
    
    SAVE_LOCATION = None
    
    def __init__(self, ddict, mode='all', hide_params=['el_shape','el_area','el_h'], 
                 parent=None):
        super().__init__(parent)
        # initialize parameter input widget
        self.main_widget = qparam.ParamObject(ddict, mode=mode)
        for param in hide_params:
            if param in self.main_widget.ROWS.keys():
                self.main_widget.ROWS[param].hide()
        
        self.PARAMS, _ = self.main_widget.ddict_from_gui()
        self.PARAMS_ORIG = dict(self.PARAMS)
        
        self.gen_layout()
        self.connect_signals()
        self.setWindowTitle('Input Parameters')
        
    def gen_layout(self):
        """ Set up layout """
        # embed main parameter widget in scroll area
        self.main_widget.setContentsMargins(0,0,15,0)
        self.qscroll = QtWidgets.QScrollArea()
        self.qscroll.horizontalScrollBar().hide()
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setWidget(self.main_widget)
        left_hash = QtWidgets.QLabel('##########')
        right_hash = QtWidgets.QLabel('##########')
        title = QtWidgets.QLabel('Parameters')
        for lbl in [left_hash, right_hash, title]:
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet('QLabel {'
                              'font-size : 14pt;'
                              'font-weight : bold;'
                              'text-decoration : none;'
                              '}')
        bbox = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton('Save')
        self.save_btn.setAutoDefault(False)
        #self.save_btn.setEnabled(False)
        self.reset_btn = QtWidgets.QPushButton('Reset parameters')
        self.reset_btn.setAutoDefault(False)
        self.reset_btn.setEnabled(False)
        bbox.addWidget(self.save_btn)
        bbox.addWidget(self.reset_btn)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        #self.layout.setSpacing(20)
        self.layout.addWidget(self.qscroll, stretch=2)
        self.layout.addLayout(bbox, stretch=0)
    
    def connect_signals(self):
        """ Connect widget signals to functions """
        self.main_widget.update_signal.connect(self.update_slot)
        self.save_btn.clicked.connect(self.save_param_file)
        self.reset_btn.clicked.connect(self.reset_params)
    
    def update_slot(self, PARAMS):
        """ Update parameter dictionary based on user input """
        self.PARAMS.update(PARAMS)
        x = not all([self.PARAMS[k] == self.PARAMS_ORIG[k] for k in PARAMS.keys()])
        #self.save_btn.setEnabled(x)
        self.reset_btn.setEnabled(x)
    
    def reset_params(self):
        """ Reset parameters to original values """
        self.main_widget.update_gui_from_ddict(self.PARAMS_ORIG)
        #self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
    
    def save_param_file(self):
        fpath = ephys.select_save_param_file(self.PARAMS, parent=self)
        if fpath:
            self.SAVE_LOCATION = fpath
            self.accept()

class RawMeta(QtWidgets.QWidget):
    meta = dict.fromkeys(['ElectricalSeries','FS','nsamples','duration','total_ch','ch_names'])
    
    def __init__(self, recording=None, parent=None):
        super().__init__(parent)
        self.gen_layout()
        self.update_recording(recording)
    
    def gen_layout(self):
        self.qform = QtWidgets.QFormLayout(self)
        self.qform.setContentsMargins(11,0,11,0)
        self.meta_w = {}
        self.meta_w['ElectricalSeries'] = QtWidgets.QLineEdit()
        self.meta_w['FS']       = QtWidgets.QLineEdit()
        self.meta_w['nsamples'] = QtWidgets.QLineEdit()
        self.meta_w['duration'] = QtWidgets.QLineEdit()
        self.meta_w['total_ch'] = QtWidgets.QLineEdit()
        self.meta_w['ch_names'] = QtWidgets.QPlainTextEdit()
        for k,mw in self.meta_w.items():
            mw.setReadOnly(True)
            self.qform.addRow(k+':', mw)
        # te_w = self.meta_w['FS'].sizeHint().width()
        # te_h = self.meta_w['ch_names'].sizeHint().height()
        # self.meta_w['ch_names'].sizeHint = lambda: QtCore.QSize(te_w, te_h)
        # hide channel names
        self.qform.itemAt(5, QtWidgets.QFormLayout.LabelRole).widget().setVisible(False)
        self.qform.itemAt(5, QtWidgets.QFormLayout.FieldRole).widget().setVisible(False)
    
    def update_recording(self, recording):
        self.recording = recording
        if self.recording is None:
            self.meta = dict.fromkeys(list(self.meta.keys()))
        else:
            self.meta['FS'] = self.recording.get_sampling_frequency()
            self.meta['nsamples'] = self.recording.get_num_samples()
            self.meta['duration'] = self.recording.get_duration()
            self.meta['ch_names'] = self.recording.get_channel_ids().astype('str')
            self.meta['total_ch'] = len(self.meta['ch_names'])
            is_nwb = self.recording.__class__.__name__ == 'NwbRecordingExtractor'
            if is_nwb:
                es_name = os.path.basename(self.recording.electrical_series_path)
                self.meta['ElectricalSeries'] = es_name
            else:
                self.meta['ElectricalSeries'] = None
        self.update_gui_from_ddict(self.meta)
        #txt = os.linesep.join([f'{k} = {v}' for k,v in self.meta.items()])
        #self.setPlainText(txt)
        #self.setVisible(self.recording is not None)
    
    def update_gui_from_ddict(self, ddict):
        if ddict['FS'] is None:
            for mw in self.meta_w.values():
                mw.clear()
        else:
            self.meta_w['FS'].setText(f"{ddict['FS']:.1f} Hz")
            self.meta_w['nsamples'].setText(f"{ddict['nsamples']:.0f}")
            self.meta_w['duration'].setText(f"{ddict['duration']:.2f} s")
            self.meta_w['total_ch'].setText(f"{ddict['total_ch']:.0f}")
            self.meta_w['ch_names'].setPlainText(str(ddict['ch_names'])[1:-1])
            es_name = ddict['ElectricalSeries']
            is_nwb = bool(es_name is not None)
            self.meta_w['ElectricalSeries'].setText(f'{str(es_name)}')
            self.qform.itemAt(0, QtWidgets.QFormLayout.LabelRole).widget().setVisible(is_nwb)
            self.qform.itemAt(0, QtWidgets.QFormLayout.FieldRole).widget().setVisible(is_nwb)
            
if __name__ == '__main__':
    import sys
    app = pyfx.qapp()
    w = BaseFolderPopup()
    w.show()
    w.raise_()
    sys.exit(app.exec())
        