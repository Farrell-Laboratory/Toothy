"""
GUI classes for parameter-setting QDialog (opens from main menu button "Set parameters").

Defines class "SetParametersPopup" (QDialog) which embeds a "ParamObject" (QWidget, defined elsewhere).

Authors: Amanda Schott, Kathleen Esfahany

Last updated: 09-09-2025
"""
# Import standard libraries
import os

# Import packages
from PyQt5 import QtWidgets, QtCore, QtGui

# Import custom modules
import qparam
import ephys
import pyfx
import gui_items as gi
import resources_v2     # For icons (":/resources/...") see resources.qrc

class SetParametersPopup(QtWidgets.QDialog):
    """
    Widget for setting parameters for analysis.
    Convenience widget; users can edit the parameters in later steps as well.
    """
        
    def __init__(self, param_dict, mode='all', hide_params=['el_shape','el_area','el_h']):
        # Initialize the QDialog
        super().__init__()

        # Initialize parameter input widget (ParamObject)
        self.main_widget = qparam.ParamObject(param_dict, mode=mode)

        self.main_widget.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)

        # Hide the widgets corresponding to the specified parameters
        for param in hide_params:
            if param in self.main_widget.ROWS.keys():
                self.main_widget.ROWS[param].hide()
        
        # Get the current parameter values as a dictionary
        self.PARAMS_DICT, _ = self.main_widget.get_param_dict_from_gui()

        # Copy the "original values"
        self.PARAMS_DICT_ORIG = dict(self.PARAMS_DICT)

        # Set the window title
        self.setWindowTitle('Parameters')

        # Set icon for upper left corner and toolbar
        self.setWindowIcon(QtGui.QIcon(':/resources/logo.png'))

        # Remove the "?" help button
        gi.remove_help_button(self)

        # Add main widget to QScrollArea, add buttons, etc.
        self.gen_layout()

        # Connect signals to slots
        self.connect_signals()

    def make_icon_button(self, icon):
        """
        Return a QPushButton button with an icon on it
        """
        button = QtWidgets.QPushButton()           # Create button
        button.setFocusPolicy(QtCore.Qt.NoFocus)   # Remove responsiveness to keyboard input (tab/enter)
        button.setFixedSize(40,40)                 # Set button size to 40px
        button.setIcon(icon)                       # Add icon
        button.setIconSize(QtCore.QSize(30,30))    # Set icon size to 30px

        # Set white background via stylesheet
        button_ss = """
            QPushButton {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #f0f0f0; /* light gray on hover */
            }
            QPushButton:pressed {
                background-color: #e0e0e0; /* darker gray when pressed */
            }
        """

        button.setStyleSheet(button_ss)

        return button
    
    def gen_layout(self):
        """
        Set up layout
        """
        # Get the current parameter file path
        self.current_param_file_path = ephys.base_dirs()[3]
        
        # Define stylesheet for QLineEdit
        le_ss = """
        QLineEdit {
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 5px;
        }
        """
        # Create a QLineEdit for displaying the current parameter file path 
        # Note: We use QLineEdit instead of QLabel because QLineEdit automatically displays the end of the string when the width of the window is less than the string, while QLabel shows the beginning.
        self.path_display_le = QtWidgets.QLineEdit(self.current_param_file_path)
        self.path_display_le.setTextMargins(0,4,0,4) # Margins for left, top, right, bottom
        self.path_display_le.setReadOnly(True)
        self.path_display_le.setStyleSheet(le_ss)

        # Create title and button to go around QLineEdit
        qlabel = QtWidgets.QLabel(f'<b>Parameter file</b>')
        self.paramfile_btn = self.make_icon_button(QtGui.QIcon(f":/resources/load-file.png"))
        self.paramfile_autofill_btn = self.make_icon_button(QtGui.QIcon(f":/resources/magic-wand.png"))
        self.paramfile_btn.setToolTip("Select a file")  # Add a hint to the "load file" buttons
        self.paramfile_autofill_btn.setToolTip("Generate a new parameter file with Toothy's default values") # Add a hint to the "autofill" parameter button
        row_elements = [self.path_display_le, self.paramfile_btn, self.paramfile_autofill_btn]
        
        # Create a horizontal row with the QLineEdit and icon buttons
        row_layout = pyfx.get_widget_container('h', *row_elements, spacing = 10)

        # Get a widget with the QLabel positioned above the QLineEdit/QPushButtons row from above
        paramfile_row_widget = pyfx.get_widget_container('v', qlabel, row_layout, spacing = 5, widget='widget')

        # Embed main parameter widget in scroll area
        self.main_widget.setContentsMargins(0,0,15,0)   # Set content margins (left, top, right, bottom)
        self.qscroll = QtWidgets.QScrollArea()          # Initialize QScrollArea
        self.qscroll.horizontalScrollBar().hide()       # Hide horizonal scroll
        self.qscroll.setWidgetResizable(True)           # Ensures the main widget fills the scroll area when it is resized
        self.qscroll.setWidget(self.main_widget)        # Embed the main widget (ParamObject)

        # Create a QHBoxLayout for bottom row of buttons
        button_layout = QtWidgets.QHBoxLayout()

        # Save button
        self.save_btn = QtWidgets.QPushButton('Save changes')
        self.save_btn.setAutoDefault(False)
        self.save_btn.setEnabled(False)

        # Reset changes button
        self.reset_btn = QtWidgets.QPushButton('Reset values')
        self.reset_btn.setAutoDefault(False)
        self.reset_btn.setEnabled(False)

        # Exit button
        self.exit_btn = QtWidgets.QPushButton('Confirm file')
        self.exit_btn.setAutoDefault(False)
        self.exit_btn.setEnabled(True)

        # Style the buttons    
        button_ss = """
            QPushButton {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 10px 0px; /* top and bottom padding, then left/right padding */
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #f0f0f0; /* light gray on hover */
            }
            QPushButton:pressed {
                background-color: #e0e0e0; /* darker gray when pressed */
            }

            QPushButton:enabled {
                border: 1px solid #ADADAD;
            }

            QPushButton:disabled {
                background-color: #F5F5F5;
            }
        """

        for button in [self.save_btn, self.reset_btn, self.exit_btn]:
            button.setStyleSheet(button_ss)

        # Add buttons to button layout
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.exit_btn)
        
        # Create a QVBoxLayout to stack widgets: file display/selection/generation, parameter scroll widget, and button widgets
        self.layout = QtWidgets.QVBoxLayout(self)

        # Add widgets
        self.layout.addWidget(paramfile_row_widget)
        self.layout.addWidget(QtWidgets.QLabel("<b>File values</b>"))
        self.layout.addWidget(QtWidgets.QLabel("<i>Hover over parameter name for a description</i>"))
        self.layout.addWidget(self.qscroll)
        self.layout.addLayout(button_layout, stretch=0)
    
    def connect_signals(self):
        """
        Connect widget signals to functions
        """
        # Signal from parent class; emits dictionary of parameters and list of valid/invalid values
        # Note: Emitted whenever a change is made, without needing any buttons being clicked
        self.main_widget.update_signal.connect(self.update_slot)
        
        # Parameters file buttons
        self.paramfile_btn.clicked.connect(self.choose_param_file)
        self.paramfile_autofill_btn.clicked.connect(self.generate_default_param_file)

        # Bottom row buttons
        self.save_btn.clicked.connect(self.save_param_file)
        self.reset_btn.clicked.connect(self.reset_params)
        self.exit_btn.clicked.connect(self.accept) # Exits QDialog
        
    def file_selection_update(self):
        """
        Called when a file selection is made (either by loading a different parameter file or by generating a default file)
        """
        # Update the path displayed in the QLineEdit widget
        self.path_display_le.setText(self.current_param_file_path)

        # Update "base dirs" (subsequent updates depend on this change)
        current_path_list = ephys.base_dirs()
        updated_path_list = current_path_list[0:3] + [self.current_param_file_path]
        ephys.write_base_dirs(updated_path_list)

        # Read in the selected file's parameter values as a dict
        new_param_dict = ephys.read_params()

        # Display the updated values in the GUI
        self.main_widget.update_gui_from_param_dict(new_param_dict)

        # Get the current parameter values as a dictionary
        self.PARAMS_DICT, _ = self.main_widget.get_param_dict_from_gui()
        
        # Update the "original values" dictionary
        self.PARAMS_DICT_ORIG = dict(self.PARAMS_DICT)

        # Disable the "save" and "reset" buttons, enable the "exit" button
        self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.exit_btn.setEnabled(True)

    def choose_param_file(self):
        """
        Select parameter file.
        Slot for parameter file selection button.
        """
        # Get the current path
        current_path = self.current_param_file_path 

        # Load the file
        param_dict, file_path = ephys.select_load_param_file(init_ppath=current_path, parent=self)

        # If the parameter file is valid, update
        if param_dict is not None:
            self.current_param_file_path = str(file_path) # Update the file path internally
            self.file_selection_update()
    
    def generate_default_param_file(self):
        """
        Save new parameter file with default values.
        Slot for default parameter file generation button.
        """
        # Get a dictionary of default parameters
        default_parameter_dict = qparam.get_original_defaults()

        # Save the dictionary as a .txt file; open a file dialog over the current QDialog with the provided title; returns path to saved file
        file_path = ephys.select_save_param_file(default_parameter_dict, title='Save default parameter file', parent=self)

        # If the file was saved, update
        if file_path:
            self.current_param_file_path = str(file_path) # Update internally
            self.file_selection_update()
    
    def update_slot(self, PARAMS_DICT_FROM_GUI):
        """
        Update parameter dictionary based on any user input
        """
        # Update the dictionary with the values from the GUI
        self.PARAMS_DICT.update(PARAMS_DICT_FROM_GUI)

        # If there were any changes made to the parameters, enable the save and reset buttons
        all_params_equal = all([self.PARAMS_DICT[k] == self.PARAMS_DICT_ORIG[k] for k in PARAMS_DICT_FROM_GUI.keys()])
        change_made = not all_params_equal
        self.save_btn.setEnabled(change_made)
        self.reset_btn.setEnabled(change_made)
        self.exit_btn.setEnabled(all_params_equal) # Only enable when there are no unsaved changes

    def save_param_file(self):
        """
        Save the current parameter dictionary into a file
        """
        # Get the path to the directory containing the currently selected file
        file_dialog_initial_dir = os.path.dirname(self.current_param_file_path)

        # Get the file name of the currently selected file
        file_dialog_initial_name = os.path.basename(self.current_param_file_path)

        # Open file dialog to the currently selected file (default will be to overwrite, with confirmation)
        file_path = ephys.select_save_param_file(self.PARAMS_DICT, init_ddir = file_dialog_initial_dir, init_fname = file_dialog_initial_name, parent=self)

        if file_path:
            self.current_param_file_path = str(file_path) # Update internally
            self.file_selection_update() # Run larger update

    def reset_params(self):
        """
        Reset parameters to original values (not default values; the original values in the file before user-made changes in GUI)
        """
        # Send the original values back to the GUI
        self.main_widget.update_gui_from_param_dict(self.PARAMS_DICT_ORIG)

        # Disable the "save" and "reset" buttons, enable the "exit" button
        self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.exit_btn.setEnabled(True)