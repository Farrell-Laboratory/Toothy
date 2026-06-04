"""
Parameter management

Implements:
- Default Toothy-provided parameter values, optimized for mouse hippocampal data
- Defines parameter names and descriptions
- Functions for checking "validity" of a given parameter dictionary (i.e. type correctness of each parameter; does not check for inter-parameter dependency correctness)


Authors: Amanda Schott, Kathleen Esfahany
"""
# Import standard libraries
import os
import json

# Import third-party libraries
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

# Import custom modules
from . import QSS
from . import pyfx

##############################################################################
##############################################################################
################                                              ################
################             PARAMETER MANAGEMENT             ################
################                                              ################
##############################################################################
##############################################################################

def get_toothy_default_parameter_dict():
    """
    Return default parameter dictionary.
    These are values selected by the creators of Toothy, optimized for mouse hippocampal data.
    """
    
    PARAMS = {
    # 'lfp_fs' : 1000.0,  # TODO remove
    ##### Ingestion (includes decimation factor, time range) #####
    'lfp_decimation_factor': 1,
    'time_range' : [0.0, -1.0],

    'theta' : [6.0, 10.0],
    'slow_gamma' : [25.0, 55.0],
    'fast_gamma' : [60.0, 100.0],
    'ds_freq' : [5.0, 100.0],
    'swr_freq' : [120.0, 180.0],
    'ds_height_thr' : 4.5,
    'ds_abs_thr'  : 0.3,
    'ds_dist_thr' : 100.0,
    'ds_prom_thr' : 0.0,
    'ds_wlen' : 125.0,
    'swr_ch_bound' : 5.0,
    'swr_height_thr' : 5.0,
    'swr_min_thr' : 3.0,
    'swr_dist_thr' : 100.0,
    'swr_min_dur' : 25.0,
    'swr_freq_thr' : 125.0,
    'swr_freq_win' : 8.0,
    'swr_maxamp_win' : 40.0,
    'csd_method' : 'standard',
    'f_type' : 'gaussian',
    'f_order' : 3.0,
    'f_sigma' : 1.0,
    'vaknin_el' : True, 
    'tol' : 1e-06,
    'spline_nsteps' : 200.0,
    'src_diam' : 0.5,
    'src_h' : 0.1,
    'cond' : 0.3,
    'cond_top' : 0.3,
    'clus_algo' : 'kmeans',
    'nclusters' : 2.0,
    'eps' : 0.2,
    'min_clus_samples' : 3.0,
    'el_shape' : 'circle',
    'el_area' : 225.0,
    'el_h' : 12.5
    }
    
    return PARAMS


def get_param_info():
    """ Return dictionary of parameter labels and descriptions """

    PARAM_INFO = {
        
    # 'lfp_fs'     : ['Downsampled FS', 'Target sampling rate (Hz) for downsampled LFP data.'],
    'lfp_decimation_factor' : [
        "Decimation factor", 
        "<b>Decimation factor</b>: Factor by which data is decimated during downsampling.<br><br>"
        "Downsampling is performed using <code>scipy.signal.decimate</code>, which applies an 8th-order Chebyshev Type I (IIR) anti-aliasing filter before decimating.<br><br>"
        "If decimation factor is set to 1, anti-aliasing and decimation are not performed; the uploaded file is used directly for the subsequent signal processing steps."
        ],

    'time_range' : [
        'Time range', 
        '<b>Time range</b>: Start and end timestamps (s) defining a range of the recording to process.'
        ],

    'theta'      : ['Theta range', 'Bandpass filter cutoff frequencies (Hz) in the theta frequency range.'],
    'slow_gamma' : ['Slow gamma range', 'Bandpass filter cutoff frequencies (Hz) in the slow gamma frequency range.'],
    'fast_gamma' : ['Fast gamma range', 'Bandpass filter cutoff frequencies (Hz) in the fast gamma frequency range.'],
    'ds_freq'    : ['DS bandpass filter', 'Bandpass filter cutoff frequencies (Hz) for detecting dentate spikes.'],
    'swr_freq'   : ['Ripple bandpass filter', 'Bandpass filter cutoff frequencies (Hz) for detecting sharp-wave ripples.'],
                  
    'ds_height_thr' : ['DS height', 'Minimum peak height (standard deviations) of a dentate spike.'],
    'ds_abs_thr'    : ['DS voltage', 'Minimum absolute peak height (mV) of a dentate spike.'],
    'ds_dist_thr'   : ['DS separation', 'Minimum distance (ms) between neighboring dentate spikes.'],
    'ds_prom_thr'   : ['DS prominence', 'Minimum dentate spike prominence (relative to the surrounding signal).'],
    'ds_wlen'       : ['DS window length', 'Window size (ms) for evaluating dentate spikes.'],
                 
    'swr_ch_bound'   : ['Ripple window', 'Number of channels on either side of the ripple LFP to include in ripple profile.'],
    'swr_height_thr' : ['Ripple height', 'Minimum height (standard deviations) at the peak of a ripple envelope.'],
    'swr_min_thr'    : ['Ripple min. height', 'Minimum height (standard deviations) at the edges of a ripple envelope.'],
    'swr_dist_thr'   : ['Ripple separation', 'Minimum distance (ms) between neighboring ripples.'],
    'swr_min_dur'    : ['Ripple min. duration', 'Minimum duration (ms) of a ripple.'],
    'swr_freq_thr'   : ['Ripple frequency', 'Minimum instantaneous frequency (Hz) of a ripple.'],
    'swr_freq_win'   : ['Ripple freq window', 'Window size (ms) for calculating ripple instantaneous frequency.'],
    'swr_maxamp_win' : ['Ripple peak window', 'Window size (ms) to look for maximum ripple LFP amplitude.'],
                        
    'csd_method'     : ['CSD mode', 'Current source density (CSD) estimation method.'],
    'f_type'         : ['CSD filter', 'Spatial filter for estimated CSD.'],
    'f_order'        : ['Filter order', 'CSD spatial filter settings (passed to scipy.signal method).'],
    'f_sigma'        : ['Filter sigma (\u03C3)', 'Sigma (or standard deviation) parameter; applies to Gaussian filter only.'],
    'vaknin_el'      : ['Vaknin electrode', "Calculate CSD with or without Vaknin's method of duplicating endpoint electrodes."],
    'tol'            : ['Tolerance', 'Tolerance of numerical integration in CSD estimation; applies to step and spline methods only.'],
    'spline_nsteps'  : ['Spline steps', 'Number of upsampled data points in CSD estimation; applies to spline method only.'],
    'src_diam'       : ['Source diameter', 'Diameter (mm) of the assumed circular current sources.'],
    'src_h'          : ['Source thickness', 'Thickness (mm) of the assumed cylindrical current sources.'],
    'cond'           : ['Conductivity', 'Conductivity (Siemens per meter) through brain tissue.'],
    'cond_top'       : ['Conductivity (top)', 'Conductivity (Siemens per meter) on top of brain tissue.'],
                  
    'clus_algo'        : ['Clustering method', 'Clustering algorithm used to classify dentate spikes.'],
    'nclusters'        : ['# clusters', 'Number of target clusters; applies to K-means algorithm only.'],
    'eps'              : ['Epsilon (\u03B5)', 'Maximum distance between points in the same cluster; applies to DBSCAN algorithm only.'],
    'min_clus_samples' : ['Min. cluster samples', 'Minimum number of samples per cluster; applies to DBSCAN algorithm only.'],
    
    'el_shape' : ['Contact shape', 'Default shape of probe electrode contacts.'],
    'el_area'  : ['Contact area', 'Default area (\u00B5m\u00B2) of probe electrode contacts.'],
    'el_h'     : ['Contact height', 'Default height (\u00B5m) of probe electrode contacts.']
    }
    
    return PARAM_INFO
    

def get_parameter_dict_validation_status(parameter_dict):
    """ 
    Determine whether input dictionary "param_dict" is a valid parameter dictionary.

    For each parameter, checks for 1) presence in dictionary and 2) correct value type. For strings, ensures value is one of the accepted options.

    Returns:
        overall_status: Boolean value (True if all critical parameters are valid, False if not)
        parameter_status_dict: dictionary with validation results for each critical parameter
    """
    def is_numerical_type(variable):
        # Check if a variable is a float or integer
        return bool(type(variable) in [float, int])

    # Numerical parameter
    def is_number(key): 
        # Try to get the parameter
        try: 
            val = parameter_dict[key]
        # If it does not exist, return False (invalid value)
        except: 
            return False
        # If it does exist, return True if it is a float or integer; False otherwise
        return is_numerical_type(val)
    
    # List of two numerical parameter values
    def is_range(key):
        # Try to get the parameter
        try: 
            val = parameter_dict[key]
        # If it does not exist, return False (invalid value)
        except: 
            return False
        # If it does exist, return True if it is a list, has length 2, and both values are numerical; False otherwise
        return bool(isinstance(val,list) and len(val)==2 and all(map(is_numerical_type, val)))
    
    # Categorical (string) parameter value
    def is_category(key, options):
        # Try to get the parameter
        try:
            val = parameter_dict[key]
        # If it does not exist, return False (invalid value)
        except: 
            return False
        return bool(val in options)
    
    # make sure each parameter 1) is present in dictionary and 2) has a valid value
    parameter_status_dict = {
                #   'lfp_fs' : is_number('lfp_fs'),
                  'lfp_decimation_factor': is_number("lfp_decimation_factor"),
                  'time_range' : is_range('time_range'),
                  'theta' : is_range('theta'),
                  'slow_gamma' : is_range('slow_gamma'),
                  'fast_gamma' : is_range('fast_gamma'),
                  'ds_freq' : is_range('ds_freq'),
                  'swr_freq' : is_range('swr_freq'),
                  'ds_height_thr' : is_number('ds_height_thr'),
                  'ds_abs_thr'  : is_number('ds_abs_thr'),
                  'ds_dist_thr' : is_number('ds_dist_thr'),
                  'ds_prom_thr' : is_number('ds_prom_thr'),
                  'ds_wlen' : is_number('ds_wlen'),
                  'swr_ch_bound' : is_number('swr_ch_bound'),
                  'swr_height_thr' : is_number('swr_height_thr'),
                  'swr_min_thr' : is_number('swr_min_thr'),
                  'swr_dist_thr' : is_number('swr_dist_thr'),
                  'swr_min_dur' : is_number('swr_min_dur'),
                  'swr_freq_thr' : is_number('swr_freq_thr'),
                  'swr_freq_win' : is_number('swr_freq_win'),
                  'swr_maxamp_win' : is_number('swr_maxamp_win'),
                  'csd_method' : is_category('csd_method', ['standard','delta',
                                                            'step','spline']), 
                  'f_type' : is_category('f_type', ['gaussian','identity','boxcar',
                                                    'hamming','triangular']),
                  'f_order' : is_number('f_order'),
                  'f_sigma' : is_number('f_sigma'),
                  'vaknin_el' : is_category('vaknin_el', [True, False]), 
                  'tol' : is_number('tol'),
                  'spline_nsteps' : is_number('spline_nsteps'),
                  'src_diam' : is_number('src_diam'),
                  'src_h' : is_number('src_h'),
                  'cond' : is_number('cond'),
                  'cond_top' : is_number('cond_top'),
                  'clus_algo' : is_category('clus_algo', ['kmeans','dbscan']),
                  'nclusters' : is_number('nclusters'),
                  'eps' : is_number('eps'),
                  'min_clus_samples' : is_number('min_clus_samples'),
                  'el_shape' : is_category('el_shape', ['circle', 'square', 'rectangle']),
                  'el_area' : is_number('el_area'),
                  'el_h' : is_number('el_h')
                  }
    is_valid = bool(all(parameter_status_dict.values()))
    return is_valid, parameter_status_dict
    

# TODO remove this; change to alerting the user that there are invalid values but do not overwrite
def fix_params(ddict, default_dict=None):
    """ Fill in missing/invalid parameters with default values """
    print("DEBUG", ddict)
    # set "default" values to use in place of missing/invalid parameters
    if default_dict is None:
        default_dict = get_toothy_default_parameter_dict()
    assert get_parameter_dict_validation_status(default_dict)[0] == True
    # populate new dictionary with hard-coded parameter keys
    keys = list(get_toothy_default_parameter_dict().keys())
    valid_ddict = get_parameter_dict_validation_status(dict(ddict))[1]
    new_dict = {}
    for k in keys:
        if valid_ddict[k] == True:
            new_dict[k] = ddict[k]
        else:
            print(k, "INVALID!!!")
            new_dict[k] = default_dict[k]
    return new_dict


##############################################################################
##############################################################################
################                                              ################
################               PARAMETER FILE I/O             ################
################                                              ################
##############################################################################
##############################################################################

def read_param_file(file_path, parent = None, fail_quietly = False):
    """
    Load parameters from a JSON file at <file_path>.

    Returns a tuple of: 
    1. a dictionary of parameter key/value pairs
    2. a list of parameter keys that are missing or invalid.

    "fail_quietly": Boolean which allows for damaged or missing JSON files to fail without error message boxes (used in starting up Toothy)
    """
    # Try to read in the file at "filepath" and parse into a {key: value} dictionary
    # Check the filepath points to a real file; if so, load the JSON into a dictionary
    
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            with open(file_path, "r") as file:
                param_dict = json.load(file)
        # If the JSON file is broken in some way.
        except:
            if not fail_quietly:
                msg = f"<b>Error: Cannot load file.</b><br><br>JSON file syntax may be incorrect."
                QtWidgets.QMessageBox.critical(parent, 'Error loading JSON file', msg)
            return None, None
    else:
        if not fail_quietly:
            msg = f"<b>Error: Selected path not found."
            QtWidgets.QMessageBox.critical(parent, 'Error loading JSON file', msg)
        return None, None

   # Check for missing/invalid parameters
    overall_status, parameter_status_dict = get_parameter_dict_validation_status(param_dict)
    
    # If the entire parameter dictionary is valid, return it along with an empty list
    if overall_status:
        return param_dict, []  # return parameter dictionary and empty list
    
    # Othewise, get the list of invalid parametrs
    else:
        invalid_parameter_keys = [param_key for param_key, status in parameter_status_dict.items() if status == False]
        
        # Return parameter dictionary and list of the invalid parameter keys
        return param_dict, invalid_parameter_keys

def write_param_file(parameter_dictionary, file_path):#✅
    """
    Write parameter dictionary to a JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(parameter_dictionary, file, indent = 4)


##############################################################################
##############################################################################
################                                              ################
################               PARAMETER WIDGETS              ################
################                                              ################
##############################################################################
##############################################################################


class BaseParamWidget(QtWidgets.QWidget):
    """
    Base functions, signals, and attributes for managing parameter inputs
    """
    # Define custom signals to indicate when a parameter value has:
        # 1. changed ("param_changed_signal"): emitted by each child class when the value changes
        # 2. invalid value ("param_warning_signal")
    param_changed_signal = QtCore.pyqtSignal()
    param_warning_signal = QtCore.pyqtSignal()
    
    def __init__(self, key, isvalid=True, parent=None):
        super().__init__(parent)
        self.key = key
        self.setObjectName(key)
        self.validation_status = isvalid
        self.wheelEvent = lambda event: None
        
        # Define a list with stylesheet options for when parameters are invalid (False/0) or valid (True/1)
        self.stylesheet_list = [pyfx.dict2ss(QSS.PARAM_INPUTS_OFF), pyfx.dict2ss(QSS.PARAM_INPUTS)]
        
    def set_widget_valid(self, status):
        """
        Update widget validation status and corresponding style sheet

        "status": Boolean (False = invalid, True = valid)
        """
        self.validation_status = bool(status)
        self.setStyleSheet(self.stylesheet_list[int(status)])
        
    def set_warning(self, x, qitem=None, tooltip='Potential input error.'):
        print("WARNING TIME")
        print(self.objectName())
        print("x = ", x)
        print("qitem", type(qitem), qitem)
        """ 
        Flag potential errors with a warning icon and message
        """
        if qitem is None: 
            return
        qact = qitem.qact
        print("qact", qitem.qact)
    
        # If True, add a "caution" icon with a warning message upon hover
        if x:
            caution_icon = QtGui.QIcon(':/icons/warning_yellow.png')
            qact.setIcon(caution_icon)
            qact.setToolTip(tooltip)

        # Otherwise, put the normal icon back and remove the warning message
        else:
            qact.setIcon(QtGui.QIcon())
            qact.setToolTip(None)

        qitem.warning = bool(x)
        self.param_warning_signal.emit()
            
    def sbox_from_kwargs(self, **kwargs):
        """ QAbstractSpinBox for numeric inputs """
        double = kwargs.get('double', True)
        alignment = kwargs.get('alignment', QtCore.Qt.AlignLeft)
        if double: box = QtWidgets.QDoubleSpinBox()
        else:      box = QtWidgets.QSpinBox()
        box.setAlignment(alignment)
        box.wheelEvent = lambda event: None
        box.setKeyboardTracking(False)
        box.setSuffix(kwargs.get('suffix'))
        rg = [kwargs.get(k, box.property(k)) for k in ['minimum','maximum']]
        step = kwargs.get('step', box.singleStep())
        if not double: rg = [*map(int, rg)]; step=int(step)
        box.setRange(*rg)
        box.setSingleStep(step)
        if double:
            box.setDecimals(kwargs.get('decimals', box.decimals()))
        if 'icon' in kwargs:
            box.qact = box.lineEdit().addAction(kwargs['icon'],
                                                QtWidgets.QLineEdit.TrailingPosition)
        return box
    
    def cbox_from_kwargs(self, **kwargs):
        """ QComboBox for categorical inputs """
        cbox = QtWidgets.QComboBox()
        if 'items' in kwargs:
            cbox.addItems(kwargs['items'])
        cbox.wheelEvent = lambda event: None
        cbox.qact = QtWidgets.QPushButton()
        cbox.qact.setObjectName('params_warning')
        cbox.qact.setStyleSheet(pyfx.dict2ss(QSS.ICON_BTN))
        return cbox
    
    
class Spinbox(BaseParamWidget):
    """
    Individual numeric inputs (e.g. detection thresholds).

    Extended by SpinboxLfpFs
    """
    
    def __init__(self, key, isvalid=True, parent=None, **kwargs):
        super().__init__(key, isvalid=isvalid, parent=parent)
        kwargs['icon'] = QtGui.QIcon()
        self.box = self.sbox_from_kwargs(**kwargs)
        self.box.setObjectName('box')
        self.box.warning = False
        self.box.valueChanged.connect(lambda x: self.param_changed_signal.emit())
        hlay = pyfx.get_widget_container('h', self.box)
        self.setLayout(hlay)
        
    def set_warning(self, x, qitem=None, tooltip='Potential input error.'):
        super().set_warning(x, qitem=self.box, tooltip=tooltip)
    
    def get_param_value(self):
        """ Return spinbox value """
        return self.box.value()
    
    def update_param(self, val):
        """
        Set spinbox value
        """
        self.box.setValue(val)
        # pyfx.stealthy(self.box, val)


class SpinboxLfpFS(Spinbox):
    """
    Decimation factor to apply to LFP before further processing steps.

    Downsampling via decimation is achieved using scipy.
    """
    
    def __init__(self, *args, fs = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = fs
        self.box.valueChanged.connect(self.check_downsampling_factor)
    
    def check_downsampling_factor(self):
        """
        Check if downsample factor is an integer >= 1
        """
        if self.fs is None: 
            return
        else:
            # lfp_fs = self.box.value() # desired Fs
            ##### OLD CODE WHICH TOOK A DESIRED LFP_FS AND CHECKED IF IT WAS A FACTOR #####
            #ds_factor = self.fs / lfp_fs
            #x = (self.fs/lfp_fs != int(self.fs/lfp_fs))
            ##########

            #### REPLACED WITH CODE WHICH CHECKS IF DESIRED FS IS GREATER THAN THE CURRENT FS
            # x = self.fs < lfp_fs # this would be a problem...
            # print(lfp_fs, self.fs)
            # tt = f'WARNING: Downsampled FS cannot be greater than the original sampling rate ({self.fs:.2f}).'
            # self.set_warning(x, tooltip=tt)

            factor = self.box.value()
            print("factor", factor)
            x = factor >= 3 #  TODO PUT BACK TO <=TODO also add a check that it produces a value within the freq ranges for further processing
            tt = f'WARNING: blah blahe ({self.fs:.2f}).'
            self.set_warning(x, tooltip=tt)

    
    def set_fs(self, fs):
        """ Update original recording FS """
        self.fs = fs
        self.check_downsampling_factor()
        if self.fs is None:
            self.set_warning(False)
        
    def update_param(self, val):
        """ Update current downsampling factor """
        super().update_param(val)
        self.check_downsampling_factor()


class SpinboxRange(BaseParamWidget):
    """ Paired numeric inputs (e.g. parameter ranges) """
    
    def __init__(self, key, isvalid=True, parent=None, **kwargs):
        super().__init__(key, isvalid=isvalid, parent=parent)
        kwargs['icon'] = QtGui.QIcon()
        self.box0 = self.sbox_from_kwargs(**kwargs)
        self.box1 = self.sbox_from_kwargs(**kwargs)
        self.boxes = [self.box0, self.box1]
        self.dash = QtWidgets.QLabel(' — ')
        self.dash.setAlignment(QtCore.Qt.AlignCenter)
        hlay = pyfx.get_widget_container('h', self.box0, self.dash, self.box1, 
                                         stretch_factors=[2,0,2], spacing=0)
        self.setLayout(hlay)
        # input 2 must always be greater than input 1
        self.min_separation = self.get_min_separation(kwargs.get('min_separation', 0))
        self.box1.setMinimum(self.box0.minimum() + self.min_separation)
        self.box0.setMaximum(self.box1.maximum() - self.min_separation)
        for i,box in enumerate(self.boxes):
            box._min = box.minimum()
            box._max = box.maximum()
            box.setObjectName(f'box{i}')
            box.warning = False
        
        self.box0.valueChanged.connect(lambda _: self.range_updated(self.box0))
        self.box1.valueChanged.connect(lambda _: self.range_updated(self.box1))
        
    def get_min_separation(self, minsep):
        """ Minimum distance between range bounds """
        self.dec = self.box0.decimals() if hasattr(self.box0, 'decimals') else 0
        min_separation = np.ceil(minsep * 10**self.dec) / 10**self.dec
        return min_separation
        
    def range_updated(self, *boxes, block=False):
        """ Dynamically adjust min/max limits to maintain valid range """
        for box in boxes:
            obox = self.boxes[1-self.boxes.index(box)]
            if box == self.box0:
                self.stealthy_minmax(obox, mmin=box.value()+self.min_separation)
            elif box == self.box1:
                self.stealthy_minmax(obox, mmax=box.value()-self.min_separation)
        if not block:
            self.param_changed_signal.emit()
       
    def reset_minmaxes(self):
        """ Reset min/max limits to original parameter range """
        for box in self.boxes:
            self.stealthy_minmax(box, box._min, box._max)
        
    def stealthy_minmax(self, box, mmin=None, mmax=None):
        """ Adjust input minimum and/or maximum """
        box.blockSignals(True)
        if mmin is not None: box.setMinimum(mmin)
        if mmax is not None: box.setMaximum(mmax)
        box.blockSignals(False)
        
    def get_param_value(self):
        """ Return parameter range """
        return [self.box0.value(), self.box1.value()]
    
    def update_param(self, val):
        """ Set parameter range """
        self.reset_minmaxes()  # reset full spinbox range and update values
        _ = [pyfx.stealthy(box, x) for box,x in zip(self.boxes, val)]
        self.range_updated(*self.boxes, block=True)


class SpinboxFreqRange(SpinboxRange):
    """ Filtering frequency bands """
    
    def __init__(self, *args, lfp_fs=None, **kwargs):
        kwargs['min_separation'] = kwargs.get('min_separation', 0.5)
        super().__init__(*args, **kwargs)
        self.lfp_fs = lfp_fs
        
    def check_filter_bands(self):
        """ Check if downsampled FS is twice the highest frequency component """
        if self.lfp_fs is None: return
        f0, f1 = self.get_param_value()
        maxf = 0.5*self.lfp_fs
        tt = f'WARNING: Frequency component exceeds the Nyquist limit of {maxf} Hz.'
        self.set_warning(f0/maxf > 1, qitem=self.box0, tooltip=tt)
        self.set_warning(f1/maxf > 1, qitem=self.box1, tooltip=tt)
    
    def set_lfp_fs(self, lfp_fs):
        """ Update target downsampling rate """
        self.lfp_fs = lfp_fs
        self.check_filter_bands()
        
    def range_updated(self, *boxes, block=False):
        """ Update frequency range """
        super().range_updated(*boxes, block=block)
        #if self.lfp_fs is None: return
        self.check_filter_bands()
    
    
class SpinboxTimeRange(SpinboxRange):
    """
    Time range (start and end) to specify the subset of a recording to be processed.
    """
    
    def __init__(self, *args, dur=np.inf, **kwargs):
        # positive values = X seconds from the start of the recording
        # negative values = X seconds back from the end of the recording
        # e.g. [10, -10] is everything but the first and last 10 seconds
        
        self.dec = kwargs.get('decimals', 2) # always 0?
        self.dur = dur

        kwargs.update({
            'double': True,
            'minimum': -self.dur,
            'maximum': self.dur, 
            'decimals':self.dec,
            'min_separation': kwargs.get('min_separation', 1)
            }
        )

        super().__init__(*args, **kwargs)
        for i, box in enumerate(self.boxes):
            box.abs_btn = QtWidgets.QPushButton()
            box.abs_btn.setStyleSheet(pyfx.dict2ss(QSS.TRANGE_TOGGLE_BTN))
            box.abs_btn.setFixedWidth(15)
        self.enable_disable_toggles()
        # use the end of the recording as the end of the interval
        self.end_chk = QtWidgets.QCheckBox('End')
        self.end_chk.toggled.connect(self.set_endpoint)
        self.layout().insertWidget(1, self.box0.abs_btn, stretch=0)
        self.layout().addWidget(self.box1.abs_btn, stretch=0)
        self.layout().addSpacing(5)
        self.layout().addWidget(self.end_chk, stretch=0)
        
        self.box0.abs_btn.clicked.connect(lambda: self.toggle_abs(self.box0))
        self.box1.abs_btn.clicked.connect(lambda: self.toggle_abs(self.box1))
        
        pyfx.stealthy(self.box1, -1)
        self.box0._t = self.box0.value()
        self.box1._t = self.box1.value()
        
    def range_updated(self, box, block=False):
        """ Update timepoints to maintain valid interval """
        if box == self.box1 and self.box1.value() == 0:
            if self.box1._t == -1:
                pyfx.stealthy(self.box1, 1)
            else:
                pyfx.stealthy(self.box1, -1)
        t0, t1 = self.get_param_value()
        if (t0 >= 0) == (t1 > 0):
            tgap = t0 - (t1-self.min_separation)
        else:
            t0a, t1a = map(self.get_abs, self.boxes)
            tgap = t0a - (t1a-self.min_separation)
        if 0 < tgap < np.inf:
            if box==self.box0:
                pyfx.stealthy(box, t0 - tgap)
            else:
                pyfx.stealthy(box, t1 + tgap)
        for box in self.boxes:
            box._t = box.value()
        if not block:
            self.param_changed_signal.emit()
    
    def set_endpoint(self, x):
        """ Adjust interval end or lock to the recording end (check box) """
        self.box1.setEnabled(not x)
        if x:
            pyfx.stealthy(self.box1, -1)
    
    def toggle_abs(self, box):
        """ Convert between relative and absolute timepoints """
        thr = self.boxes.index(box)
        val = box.value()
        if val >= thr:
            box.setValue(val - self.dur) # from the end --> from the start
        else:
            box.setValue(self.get_abs(box))
            
    def get_abs(self, box):
        """ Get absolute timepoints """
        thr = self.boxes.index(box) # below 0 for tstart, including 0 for tend
        val = box.value()
        if val >= thr: # return absolute timepoints
            return val
        if val == -1:
            val += thr
        return self.dur + val
    
    def enable_disable_toggles(self):
        """ Disable timepoint conversion if duration of recording is unknown """
        for i,box in enumerate(self.boxes):
            box.abs_btn.setVisible(self.dur < np.inf)
    
    def update_param(self, val):
        """ Set time range """
        # reset full spinbox range and update values
        _ = [pyfx.stealthy(box, x) for box,x in zip(self.boxes, val)]
        is_end = (-1 <= self.box1.value() <= 0)
        pyfx.stealthy(self.end_chk, is_end)
        self.set_endpoint(is_end)
        self.range_updated(self.box0, block=True)
    
    def set_duration(self, dur):
        """ Update recording duration (np.inf if no recording is loaded) """
        self.dur = np.ceil(dur * 10**self.dec) / 10**self.dec
        self.stealthy_minmax(self.box0, -self.dur, self.dur-self.min_separation)
        self.stealthy_minmax(self.box1, -self.dur+self.min_separation, self.dur)
        for box in self.boxes:
            box._min = box.minimum()
            box._max = box.maximum()
        self.range_updated(self.box0, block=True)
        self.enable_disable_toggles()


class Combobox(BaseParamWidget):
    """ QComboBox for categorical parameters (e.g. clustering algorithms) """
    
    def __init__(self, key, isvalid=True, parent=None, **kwargs):
        #super().__init__(parent)
        super().__init__(key, isvalid=isvalid, parent=parent)
        self.cbox = self.cbox_from_kwargs(**kwargs)
        self.cbox.setObjectName('cbox')
        self.cbox.warning = False
        
        # convert item labels to variable names
        if key in ['csd_method', 'f_type', 'el_shape']:
            self.get_valtxt = lambda val: val.capitalize()
            self.export_valtxt = lambda txt: txt.lower()
        elif key == 'vaknin_el':
            self.get_valtxt = lambda val: str(bool(val))
            self.export_valtxt = lambda txt: bool(txt)
        elif key == 'clus_algo':
            self.get_valtxt = lambda val: dict(kmeans='K-means', dbscan='DBSCAN')[val]
            self.export_valtxt = lambda txt: txt.replace('-','').lower()
        
        hlay = pyfx.get_widget_container('h', self.cbox, self.cbox.qact,
                                         stretch_factors=[2,0], spacing=0)
        self.cbox.qact.hide()
        self.setLayout(hlay)
        
        self.cbox.currentIndexChanged.connect(lambda x: self.param_changed_signal.emit())
        self.cbox.activated.connect(lambda i: self.set_widget_valid(True))
        
    def get_param_value(self):
        """ Return selected item """
        res = self.export_valtxt(self.cbox.currentText())
        return res
    
    def update_param(self, val):
        """ Update parameter item """
        txt = self.get_valtxt(val)
        self.cbox.blockSignals(True)
        self.cbox.setCurrentText(txt)
        self.cbox.blockSignals(False)
    
    def set_warning(self, x, qitem=None, tooltip='Potential input error.'):
        """ Show warning icon outside menu area """
        super().set_warning(x, qitem=self.cbox, tooltip=tooltip)
        self.cbox.qact.setVisible(x)

#########################################
#########################################
############   Main Widget   ############
#########################################
#########################################


class ParameterWidget(QtWidgets.QWidget):
    """
    QWidget containing input widgets for all or a subset of parameters
    """
    
    update_signal = QtCore.pyqtSignal(dict, list)
    warning_update_signal = QtCore.pyqtSignal(int, dict)
    
    def __init__(self, params={}, mode='all', parent=None):
        """
        params: dictionary of initial parameter values
        mode: set visible parameters for different analysis modes TODO edit
            'signal_processing': LFP downsampling and filtering params
            'ds_detection': thresholds for detecting DS events
            'swr_detection': thresholds for detecting SWR events
            'data_processing': includes all the above params
            'ds_classification': CSD calculation and PCA clustering params
            'all': all parameters
        """
        super().__init__(parent)

        ##### Set "default values" based on input parameter dictionary #####
        
        # To start, get the Toothy-defined default parameter menu
        parameter_dict = get_toothy_default_parameter_dict()                  

        # Get the "validation status" of each value in the provided set of parameters (note: this checks for type correctness, not inter-parameter validity)      
        _, validation_status_dict = get_parameter_dict_validation_status(params)
        
        # Update the parameter dict (full of Toothy's default values) with valid values from the input dictionary
        for parameter_name, validation_status in validation_status_dict.items():
            if validation_status:
                parameter_dict[parameter_name] = params[parameter_name]
        
        self.DEFAULTS = dict(parameter_dict)
        
        self.gen_layout()
        self.init_widgets()
        self.set_mode(mode)
        self.update_gui_from_param_dict(params)
        self.connect_signals()
        
    def gen_layout(self):
        """
        Set up layout
        """
        self.setContentsMargins(0, 0, 15, 0) # Add space to the right margin (left, top, right, bottom)
        
        ######################
        #####  WIDGETS   #####
        ######################
        
        ##### "signal_processing" ######
        # self.lfp_fs = SpinboxLfpFS(key='lfp_fs', minimum=1, decimals=10, step=0.5, suffix=' Hz') # TODO remove: old lfp_fs

        # LFP decimation factor: Spinbox with integer inputs > 1; maximum can't be set but later validation should check compatibility with signal processing filtering ranges
        self.lfp_decimation_factor = SpinboxLfpFS(key='lfp_decimation_factor', minimum=1, decimals=0, step=1)
        
        # Time range: Two spinboxes 
        self.time_range = SpinboxTimeRange(key='time_range', double=True, decimals=0,
                                       minimum=0, maximum=999999, 
                                       suffix=' s', min_separation=1)
        kwargs = dict(double=True, decimals=1, step=0.5, maximum=999999, 
                      suffix=' Hz', min_separation=0.5)
        self.theta = SpinboxFreqRange(key='theta', **kwargs)
        self.slow_gamma = SpinboxFreqRange(key='slow_gamma', **kwargs)
        self.fast_gamma = SpinboxFreqRange(key='fast_gamma', **kwargs)
        self.ds_freq = SpinboxFreqRange(key='ds_freq', **kwargs)
        self.swr_freq = SpinboxFreqRange(key='swr_freq', **kwargs)
        ### ds_detection
        self.ds_height_thr = Spinbox(key='ds_height_thr', suffix=' S.D.')
        self.ds_abs_thr = Spinbox(key='ds_abs_thr', suffix=' mV')
        self.ds_dist_thr = Spinbox(key='ds_dist_thr', suffix=' ms', decimals=0)
        self.ds_prom_thr = Spinbox(key='ds_prom_thr', suffix=' a.u.')
        self.ds_wlen = Spinbox(key='ds_wlen', suffix=' ms', decimals=0)
        ### swr_detection
        self.swr_ch_bound = Spinbox(key='swr_ch_bound', suffix=' channels', decimals=0)
        self.swr_height_thr = Spinbox(key='swr_height_thr', suffix=' S.D.')
        self.swr_min_thr = Spinbox(key='swr_min_thr', suffix=' S.D.')
        self.swr_dist_thr = Spinbox(key='swr_dist_thr', suffix=' ms', decimals=0)
        self.swr_min_dur = Spinbox(key='swr_min_dur', suffix=' ms', decimals=0)
        self.swr_freq_thr = Spinbox(key='swr_freq_thr', suffix=' Hz', decimals=1)
        self.swr_freq_win = Spinbox(key='swr_freq_win', suffix=' ms', decimals=0)
        self.swr_maxamp_win = Spinbox(key='swr_maxamp_win', suffix=' ms', decimals=0)
        ### csd_calculation
        self.csd_method = Combobox(key='csd_method',
                                   items=['Standard','Delta','Step','Spline'])
        self.f_type = Combobox(key='f_type', items=['Gaussian','Identity','Boxcar',
                                                    'Hamming','Triangular'])
        self.f_order = Spinbox(key='f_order', minimum=1, decimals=1)
        self.f_sigma = Spinbox(key='f_sigma', decimals=1, step=0.1)
        self.vaknin_el = Combobox(key='vaknin_el', items=['True','False'])
        self.tol = Spinbox(key='tol', decimals=7, step=0.0000001)
        self.spline_nsteps = Spinbox(key='spline_nsteps', maximum=2500, decimals=0)
        self.src_diam = Spinbox(key='src_diam', suffix=' mm', decimals=3, step=0.01)
        self.src_h = Spinbox(key='src_h', suffix=' mm', decimals=3, step=0.01)
        self.cond = Spinbox(key='cond', suffix=' S/m', decimals=3, step=0.01)
        self.cond_top = Spinbox(key='cond_top', suffix=' S/m', decimals=3, step=0.01)
        ### csd_clustering
        self.clus_algo = Combobox(key='clus_algo', items=['K-means','DBSCAN'])
        self.nclusters = Spinbox(key='nclusters', suffix=' clusters', minimum=1, decimals=0)
        self.eps = Spinbox(key='eps', suffix=' a.u.', decimals=2, step=0.1)
        self.min_clus_samples = Spinbox(key='min_clus_samples', minimum=1, decimals=0)
        ### probe_geometry
        self.el_shape = Combobox(key='el_shape', items=['Circle', 'Square', 'Rectangle'])
        self.el_area = Spinbox(key='el_area', suffix=' \u00B5m\u00B2', decimals=1)
        self.el_h = Spinbox(key='el_h', suffix=' \u00B5m', decimals=1)
        for sbox in [
            # self.lfp_fs, 
            self.ds_height_thr, self.ds_abs_thr, self.ds_dist_thr, self.ds_prom_thr, 
                     self.ds_wlen, self.swr_ch_bound, self.swr_height_thr, self.swr_min_thr, self.swr_dist_thr, 
                     self.swr_min_dur, self.swr_freq_thr, self.swr_freq_win, self.swr_maxamp_win,
                     self.f_order, self.f_sigma, self.tol, self.src_diam, self.src_h, self.cond,
                     self.cond_top, self.nclusters, self.eps, self.min_clus_samples, self.el_area, self.el_h]:
            sbox.box.setMaximum(999999)
        # organize widgets by analysis step
        tups = [
            # (self.lfp_fs, 'signal_processing'),
                (self.lfp_decimation_factor, "signal_processing"),
                (self.time_range, 'signal_processing'),
                (self.theta, 'signal_processing'),
                (self.slow_gamma, 'signal_processing'),
                (self.fast_gamma, 'signal_processing'),
                (self.ds_freq, 'signal_processing'),
                (self.swr_freq, 'signal_processing'),
                
                (self.ds_height_thr, 'ds_detection'),
                (self.ds_abs_thr, 'ds_detection'),
                (self.ds_dist_thr, 'ds_detection'),
                (self.ds_prom_thr, 'ds_detection'),
                (self.ds_wlen, 'ds_detection'),
                
                (self.swr_ch_bound, 'swr_detection'),
                (self.swr_height_thr, 'swr_detection'),
                (self.swr_min_thr, 'swr_detection'),
                (self.swr_dist_thr, 'swr_detection'),
                (self.swr_min_dur, 'swr_detection'),
                (self.swr_freq_thr, 'swr_detection'),
                (self.swr_freq_win, 'swr_detection'),
                (self.swr_maxamp_win, 'swr_detection'),
                
                (self.csd_method, 'csd_calculation'),
                (self.f_type, 'csd_calculation'),
                (self.f_order, 'csd_calculation'),
                (self.f_sigma, 'csd_calculation'),
                (self.vaknin_el, 'csd_calculation'),
                (self.tol, 'csd_calculation'),
                (self.spline_nsteps, 'csd_calculation'),
                (self.src_diam, 'csd_calculation'),
                (self.src_h, 'csd_calculation'),
                (self.cond, 'csd_calculation'),
                (self.cond_top, 'csd_calculation'),
                
                (self.clus_algo, 'csd_clustering'),
                (self.nclusters, 'csd_clustering'),
                (self.eps, 'csd_clustering'),
                (self.min_clus_samples, 'csd_clustering'),
                
                (self.el_shape, 'probe_geometry'),
                (self.el_area, 'probe_geometry'),
                (self.el_h, 'probe_geometry')]
        
        # get widget items and corresponding modes, labels, and info text
        widgets,modes = map(list, zip(*tups))
        info_dict = get_param_info()
        
        ###   ORGANIZE PARAMETERS BY MODE   ###
        
        # map analysis modes to parameter variable names
        tmp = {m:[] for i,m in enumerate(modes) if modes.index(m)==i}
        mode2key = {**tmp, 'data_processing':[], 'ds_classification':[], 'all':[]}
        for key,m in zip(info_dict, modes):
            mode2key['all'].append(key)
            mode2key[m].append(key)
            if m in ['signal_processing','ds_detection','swr_detection']:
                mode2key['data_processing'].append(key)
            if m in ['csd_calculation','csd_clustering','probe_geometry']:
                mode2key['ds_classification'].append(key)
        self.mode_to_key_dict = mode2key
            
        ###   LAYOUT   ###
        
        self.vlay = QtWidgets.QVBoxLayout(self)
        self.ROWS, self.WIDGETS = {},{}
        for (key,(lbl,info)),widget in zip(info_dict.items(),widgets):
            # make QLabel with hover info
            qlabel = QtWidgets.QLabel(lbl)
            qlabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            qlabel.setMinimumWidth(120)  # Adjust as needed

            qlabel.setToolTip(info)
            # create row widget, add to layout
            row = QtWidgets.QWidget()
            hbox = QtWidgets.QHBoxLayout(row)
            hbox.setSpacing(1)
            hbox.setContentsMargins(0,0,0,0)
            hbox.addWidget(qlabel, stretch=0)
            hbox.addWidget(widget, stretch=1)
            self.ROWS[key] = row
            self.WIDGETS[key] = widget
            self.vlay.addWidget(row)
    
    def connect_signals(self):
        """ Connect GUI inputs """
        for widget in self.WIDGETS.values():
            widget.param_changed_signal.connect(self.emit_signal)
            widget.param_warning_signal.connect(self.handle_warnings)
        
        # FIX THIS
        # for fbk in ['theta', 'slow_gamma', 'fast_gamma', 'ds_freq', 'swr_freq']:
        #     self.lfp_fs.box.valueChanged.connect(self.WIDGETS[fbk].set_lfp_fs)
    
    def init_widgets(self):
        """
        Initialize widget values from parameter dictionary
        """
        # Note:
            # self.WIDGETS is dictionary of key/widget pairs, defined in gen_layout()
        for parameter_key, parameter_widget in self.WIDGETS.items():
            parameter_value = self.DEFAULTS[parameter_key]
            parameter_widget.update_param(parameter_value)
            parameter_widget.set_widget_valid(True)

    def handle_warnings(self):
        """ Pass parameter input warnings to main window """
        warn_dict = self.get_warn_dict()
        n = sum([*map(lambda l: sum(l)>0, warn_dict.values())])
        self.warning_update_signal.emit(n, warn_dict)
        
    def get_warn_dict(self):
        """ Return warning status for each input widget """
        ddict = {}
        for k in self.KEYS:
            item = self.WIDGETS[k]
            if isinstance(item, Spinbox):
                ddict[k] = [bool(item.box.warning)]
            elif isinstance(item, SpinboxRange):
                ddict[k] = [bool(item.box0.warning), bool(item.box1.warning)]
            elif isinstance(item, Combobox):
                ddict[k] = [bool(item.cbox.warning)]
        return ddict
        
    def get_param_dict_from_gui(self):
        """
        Return currently-displayed GUI widget values as parameter dictionary.

        Note: This returns a dictionary with n elements (where n is the number of parameters for the object "mode") of key/value pairs and a list of length m where m is the number of keys with "invalid" values.
        """
        # Initialize an empty dictionary to hold the values from the parameter widgets 
        parameter_dict_from_gui = {}

        # Initialize an empty list to store parameter keys with invalid values
        invalid_keys = []

        # Iterate over each parameter currently displayed in the GUI
        # Note: "self.KEYS" is a list of the currently-displayed parameter keys, given the mode (defined for object when set_mode() is called)
        for key in self.KEYS:
            # Get the corresponding parameter widget
            parameter_widget = self.WIDGETS[key]

            # If the parameter value is valid, add it to the dictionary
            if parameter_widget.validation_status == True:
                parameter_dict_from_gui[key] = parameter_widget.get_param_value()

            # If the item is invalid, export the original parameter value TODO assess if this is necessary
            # Note: self.PARAM_DICT_ORIGINAL is defined in "update_gui_from_param_dict" called when object is initialized TODO could just epxlicity define outside oft hat function?
            else:
                if key in self.PARAM_DICT_ORIGINAL: 
                    parameter_dict_from_gui[key] = self.PARAM_DICT_ORIGINAL[key]
                invalid_keys.append(key)
        return parameter_dict_from_gui, invalid_keys
    
    def update_gui_from_param_dict(self, param_dict):#, block=False):
        """
        Set GUI widget values from parameter dictionary
        """
        self.PARAM_DICT_ORIGINAL = dict(param_dict)

        # check for valid $ddict values for each current param (e.g. lfp_fs:False, time_range=True)
        validation_dict = get_parameter_dict_validation_status(param_dict)[1]
        validation_dict_filtered = {param_key: validation_dict[param_key] for param_key in self.KEYS}

        # valid_ddict = {k:b for k,b in get_parameter_dict_validation_status(param_dict)[1].items() if k in self.KEYS}
        
        validated_param_dict = {param_key: param_dict[param_key] if validation_dict_filtered[param_key] else self.DEFAULTS[param_key] for param_key in self.KEYS}
        
        # validated_param_dict = {k:param_dict[k] if x else self.DEFAULTS[k] for k,x in valid_ddict.items()}
        
        for param_key, param_val in validated_param_dict.items():
            param_widget = self.WIDGETS[param_key]
            param_widget.update_param(param_val)
            param_widget.set_widget_valid(validation_dict_filtered[param_key])

        # # FIX THIS
        # for fbk in ['theta', 'slow_gamma', 'fast_gamma', 'ds_freq', 'swr_freq']:
        #     self.WIDGETS[fbk].set_lfp_fs(self.lfp_fs.box.value())
        
    
    def emit_signal(self, *args):
        """
        Emit two data variables:
        (1) currently-displayed parameter dictionary with vaues from the GUI
        (2) list of keys of parameters with invalid values

        Slot connected to signal "param_changed_signal" emitted by each parameter widget when their values change.
        """
        # B) user uploads a new set of params -> validity depends on the content
        # * try to prevent emit_signal during B

        # Get the sender of the signal, i.e. the parameter widget for which a value was changed
        changed_parameter_widget = self.sender()

        # Set the value to be valid
        changed_parameter_widget.set_widget_valid(True)

        # Get a dictionary of parameter key/value pairs from the GUI and a list of keys with invalid values
        gui_param_dict, invalid_key_list = self.get_param_dict_from_gui()

        print("\n emit signal slot!")
        print(gui_param_dict)
        print("\n invalid")
        print(invalid_key_list)


        # Emit these values as part of "update_signal"; "update_signal" often used exterally (signal connected to external slots)
        self.update_signal.emit(gui_param_dict, invalid_key_list)
    
    ################################
    #####   Display Settings   #####
    ################################

    def set_mode(self, mode='all'):#✅
        """
        Filter display to show only input widgets for the given mode

        Takes a string mode, pulls the corresponding parameter keys, and filters the visibility of each parameter widget to show only the relevant parameters
        """
        # Get the list of parameter keys for the given mode
        self.KEYS = list(self.mode_to_key_dict[mode]) 

        # Show rows for parameters of the given mode and hide the rest
        for param_key, param_widget_row in self.ROWS.items():
            param_widget_row.setVisible(param_key in self.KEYS)

        # Adjust the display after 50ms to ensure the labels are all aligned and visible
        QtCore.QTimer.singleShot(50, self.adjust_labels)
        
    def adjust_labels(self):#✅
        """
        Left-align parameter labels (for example "Decimation factor").

        Set the text label widths to the longest visible QLabel, plus a buffer of 10 pixels.
        """
        # Note
            # self.KEYS: list parameter keys currently displayed, given the mode (defined for object when set_mode is called)
            # self.ROWS: dictionary of parameter widgets (defined in gen_layout() when object is initialzied)
        visible_qlabels = [self.ROWS[param_key].findChild(QtWidgets.QLabel) for param_key in self.KEYS]
        maximum_label_width = max([label.width() for label in visible_qlabels])
        for label in visible_qlabels:
            label.setFixedWidth(maximum_label_width + 10)
    

            
            