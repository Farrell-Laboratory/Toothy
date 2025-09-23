"""
Toothy (https://github.com/Farrell-Laboratory/Toothy)

Purpose: Home interface for launching Toothy GUI

Authors: Amanda Schott, Kathleen Esfahany

Last updated: 2025-08-26
"""
# Import standard libraries
import sys
import os
from pathlib import Path

# Import PyQt5 (PyQt5 is a Python version of the Qt framework for building GUI applications)
from PyQt5 import QtWidgets, QtCore, QtGui

# Set app folder as current working directory
    # Notes:
    # __file__ is a special Python variable that holds the path of the current file (.../toothy.py)
    # .parent returns the path to the directory containing the file (i.e. Toothy-main if repo is downloaded off GitHub)
    # os.chdir() changes the current working directory to the specified parent path
    # Why? To ensure the custom modules are imported correctly
app_dir = Path(__file__).parent 
os.chdir(app_dir)

# Import custom modules
import QSS      # Stylesheet definitions for GUI elements
import pyfx     # Misc. helper functions
import ephys    # File I/O and event detection
import gui_items as gi  #  GUI helpers
import resources_v2     # Compiled resources (images) referenced with ":/..."

# Import QDialog widgets for each part of the Toothy workflow
from raw_data_pipeline import InputDataSelectionPopup               # Step 1 (Selection of input data, triggers processing pipeline)
from processed_data_hub import ProcessedRecordingSelectionPopup     # Step 2 (Interact with the processed data to select channels and classify events)
from gui_set_paths import SetPathsPopup                             # Convenience: Set paths for input data, probe configuration files, etc.
from gui_set_parameters import SetParametersPopup                   # Convenience: Set parameters
from probe_handler import ProbeObjectPopup                          # Convenience: Create probe configuration files

# Define a QMainWindow class "toothy"
class toothy(QtWidgets.QMainWindow):
    
    def __init__(self):
        # Initalize parent class
        super().__init__()

        # Load base data directories
        if not os.path.exists('default_folders.txt'):
            ephys.init_default_folders()
        ephys.clean_base_dirs()
        
        self.init_processed_ddir = None
        
        # Set up QMainWindow
        self.gen_layout()       # Create button widgets and add to layout
        self.connect_signals()  # Connect buttons to imported QDialogs
        
        # Initialize variables for all the future QDialog windows
        self.rawdata_popup          = None
        self.analysis_popup         = None
        self.set_paths_popup        = None
        self.set_parameters_popup   = None
        self.probe_popup            = None
        
        self.show()
        self.center_window()
        
    def gen_layout(self):
        """
        Set up main window
        """
        # Set window title
        self.setWindowTitle('Toothy')

        # Set icon for upper left corner and toolbar
        self.setWindowIcon(QtGui.QIcon(':/resources/logo.png'))

        # Set margins between window edge and inner contents
        margin_px = 25
        self.setContentsMargins(margin_px, margin_px, margin_px, margin_px)

        # Create central widget and layout
        self.centralWidget = QtWidgets.QWidget()
        self.centralLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.centralLayout.setSpacing(25) # Spacing between vertical elements
        
        # Add a title/main heading ("Toothy")
        self.heading_label = QtWidgets.QLabel("Toothy")
        heading_stylesheet = """
        QLabel {
            font-size: 30px;
            font-weight: bold;
            color: #a9594a; /* pastel maroon */
            padding: 6px;
        }
        """
        self.heading_label.setStyleSheet(heading_stylesheet)
        self.heading_label.setAlignment(QtCore.Qt.AlignHCenter| QtCore.Qt.AlignTop) # Align center horizontally, Align to top vertically
        self.heading_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed) # Policy for horizonal (expand) and vertical (fixed) expansion within the layout; ensures "Toothy" is centered horizontally in the window

        # Subheadings
        self.subhead1 = QtWidgets.QLabel("Event detection pipeline:")
        self.subhead2 = QtWidgets.QLabel("Convenience:")
        self.subhead_btns = [self.subhead1, self.subhead2]
        subhead_stylesheet = """
        QLabel {
            font-size: 20px;
            color: black;
            padding: 3px;
        }
        """
        for btn in self.subhead_btns:
            btn.setStyleSheet(subhead_stylesheet)
            btn.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Fixed)

        # Create buttons for main event detection pipeline
        self.process_btn = QtWidgets.QPushButton('Step 1: Process data')
        self.analyze_btn = QtWidgets.QPushButton('Step 2: Analyze events')

        # Create buttons for convenience functions
        self.set_paths_btn = QtWidgets.QPushButton('Set paths')
        self.set_parameters_btn = QtWidgets.QPushButton('Set parameters')
        self.probe_btn = QtWidgets.QPushButton('Create probe configuration')

        # Style the buttons
        self.home_btns = [
             self.process_btn,        # Process input recording (file ingestion, filtering, event detection)
             self.analyze_btn,        # Analyze detected events
             self.set_paths_btn,      # Update default directories to data/probes/parameters
             self.set_parameters_btn, # View/edit analysis parameter
             self.probe_btn           # Create probe configuration files
        ]
        home_button_stylesheet = """
        QPushButton {
            background-color: #f5f5f5;
            color: #333333;
            border: 2px solid #cccccc;
            border-radius: 6px;
            padding: 8px 14px;
            font-weight: bold;

        }"""
        for btn in self.home_btns:
            btn.setStyleSheet(home_button_stylesheet)
            
        # Add headings/subheadings/buttons to layout in order
        self.centralLayout.addWidget(self.heading_label)    # "Toothy"
        self.centralLayout.addWidget(self.subhead1)         # "Event detection pipeline"
        self.centralLayout.addWidget(self.process_btn)      # Button for step 1
        self.centralLayout.addWidget(self.analyze_btn)      # Button for step 2
        self.centralLayout.addWidget(self.subhead2)         # "Convenience"
        self.centralLayout.addWidget(self.set_paths_btn)    # Button for setting paths
        self.centralLayout.addWidget(self.set_parameters_btn)   # Button for setting parametes
        self.centralLayout.addWidget(self.probe_btn)        # Button for creating probe configuration files

        # # Set a minimum width (otherwise the window ends up very skinny)
        self.centralWidget.setMinimumWidth(350)

        # Set the QMainWindow's central widget to be the QWidget to which we just added all the buttons
        self.setCentralWidget(self.centralWidget)
    
    def connect_signals(self):
        """
        Connect buttons to their corresponding QDialog widgets
        """
        # Connect button "clicked" signals to slots (functions that open QDialogs)
        self.process_btn.clicked.connect(self.raw_data_popup)
        self.analyze_btn.clicked.connect(self.processed_data_popup)
        self.set_paths_btn.clicked.connect(self.set_paths_popup_signal)
        self.set_parameters_btn.clicked.connect(self.set_parameters_popup_signal)
        self.probe_btn.clicked.connect(self.probe_popup)
    
    def raw_data_popup(self, *args):
        """ Selection of input data processing pipeline """
        self.rawdata_popup = InputDataSelectionPopup()
        self.rawdata_popup.exec()
        # set analysis hub to the most recently processed recording folder
        if self.rawdata_popup.last_saved_ddir is not None:
            self.init_processed_ddir = str(self.rawdata_popup.last_saved_ddir)
        
    def processed_data_popup(self, *args):
        """ Show processed data options """
        # create popup window for processed data
        self.analysis_popup = ProcessedRecordingSelectionPopup(init_ppath=self.init_processed_ddir, parent=self)
        self.analysis_popup.exec()
        self.init_processed_ddir = str(self.analysis_popup.ddir)

    def set_paths_popup_signal(self):
        """
        View or change paths for input data, probe configurations, analysis parameters
        """
        self.set_paths_popup = SetPathsPopup()              # Instantiates a QDialog
        self.set_paths_popup.widget.setMinimumWidth(500)    # Sets the minimum width of the QDialog
        self.set_paths_popup.exec()                         # Creates the local event loop
    
    def set_parameters_popup_signal(self):
        """
        View or change analysis parameters
        """
        current_param_dict = ephys.read_params()
        self.set_parameters_popup = SetParametersPopup(param_dict = current_param_dict) # Instantiate a QDialog
        self.set_parameters_popup.setMinimumWidth(600)                                  # Set the minimum width of the QDialog
        self.set_parameters_popup.exec()                                                # Creates local event loop
        
    def probe_popup(self, *args, init_probe=None):
        """ Build probe objects """
        self.probeobj_popup = ProbeObjectPopup(probe=init_probe)
        self.probeobj_popup.setModal(True)
        self.probeobj_popup.show()
    
    def center_window(self):
        """
        Move the GUI window to the center of the screen
        """
        # Get the screen that is under the user's mouse (important for users with multiple screens)
        screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())

        # Default to the "primary screen" if needed
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()

        # Get the screen's geometry (x, y, width, height) where x, y are coordinates of the top left corner
        # Note: "availableGeometry" excludes taskbar/dock; just "QtWidgets.QDesktopWidget().screenGeometry()" will return the entire screen
        screen_rect = screen.availableGeometry() 
        
        # Get the widget window's geometry (x, y, width, height) where x, y are coordinates of the top left corner
        # Note: "frameGeometry" includes the window frame; just "geometry" returns only the widget section
        widget_rect = self.frameGeometry()

        # Move the widget rectangle's center to match the center of the screen
        widget_rect.moveCenter(screen_rect.center())

        # Move the actual window to the newly-centered "widget_rect", specified by the top left corner coordinates
        self.move(widget_rect.topLeft())
    
if __name__ == '__main__':
    """Runs when toothy.py is executed as a script. Launches the Toothy GUI."""
    # Initilize QApplication
    app = pyfx.qapp()
    
    # Create and show QMainWindow widget
    # Note: widgets implicitly attach to the QApplication instance (app) created above
    ToothyWindow = toothy()

    # Bring window to front
    ToothyWindow.raise_()
    
    # After creating window, start the event loop
    sys.exit(app.exec())