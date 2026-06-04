"""
GUI classes for parameter-setting QDialog (opens from main menu button "Set parameters").

Defines class "SetParametersPopup" (QDialog) which embeds a "ParameterWidget" (QWidget, defined externally).

Allows users to create and edit parameter files.

Authors: Amanda Schott, Kathleen Esfahany

Last updated: 11-11-2025

# TODO (minor) edit module names for qparam, resources_v2 when complete
"""
# Import standard libraries
import os

# Import packages
from PyQt5 import QtWidgets, QtCore, QtGui

# Import custom modules
from . import qparam
from . import helpers_io as h_io
from . import pyfx
from . import gui_items as gi
from . import resources_v2     # For icons (":/resources/...") see resources.qrc

class SetParametersPopup(QtWidgets.QDialog):
    """
    Widget for setting parameters for analysis.
    Convenience widget; users can modify parameters in later parts of the workflow as well.
    """
        
    def __init__(self):
        # Initialize the QDialog
        super().__init__()

        # Set the window title
        self.setWindowTitle('Parameters')

        # Set icon for upper left corner and toolbar
        self.setWindowIcon(QtGui.QIcon(':/resources/logo.png'))

        # Remove the "?" help button
        gi.remove_help_button(self)

        # Create file selection buttons, QScrollArea, save/revert/confirm buttons, etc.
        self.gen_layout()

        # Connect signals to slots
        self.connect_signals()
       
    def initialize_parameter_widget(self):
        """
        Load a parameter dictionary and use it to instantiate a ParameterWidget. Add the ParameterWidget to the QDialog in the QScroll widget.
        """
        # Get the parameter dictionary
        param_dict = qparam.read_param_file(self.current_param_file_path)[0]
            
        # Initialize parameter input widget (ParameterWidget)
        self.main_widget = qparam.ParameterWidget(param_dict)

        self.main_widget.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)

        self.main_widget.setContentsMargins(0,0,15,0)

        # Hide some widgets corresponding to parameters used only in probe-making
        hide_params=['el_shape','el_area','el_h']
        for param_key in hide_params:
            if param_key in self.main_widget.ROWS.keys():
                self.main_widget.ROWS[param_key].setVisible(False)
        
        # Get the current parameter values as a dictionary
        self.PARAMS_DICT, _ = self.main_widget.get_param_dict_from_gui()

        # Copy the "original values"
        self.PARAMS_DICT_ORIG = dict(self.PARAMS_DICT)

        # Connect signal to slot
        self.main_widget.update_signal.connect(self.update_slot)

        # Replace the filler widget in the scroll area with this widget
        self.qscroll.setWidget(self.main_widget)

        # Expand the scroll area
        self.qscroll.setMinimumHeight(500)

        # Show instructions for interacting with parameters
        self.instructions_qlabel.setVisible(True)

        # Make the button for confirming a file active
        self.exit_btn.setEnabled(True)

    def gen_layout(self):
        """
        Set up layout
        """
        # Get the current parameter file path
        self.current_param_file_path = h_io.get_toothy_paths()[3]
            
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
        self.paramfile_btn = gi.make_icon_button(QtGui.QIcon(f":/resources/load-file.png"))
        self.paramfile_autofill_btn = gi.make_icon_button(QtGui.QIcon(f":/resources/magic-wand.png"))
        self.paramfile_btn.setToolTip("Select a file")  # Add a hint to the "load file" buttons
        self.paramfile_autofill_btn.setToolTip("Generate a new parameter file with Toothy's default values") # Add a hint to the "autofill" parameter button
        row_elements = [self.path_display_le, self.paramfile_btn, self.paramfile_autofill_btn]
        
        # Create a horizontal row with the QLineEdit and icon buttons
        row_layout = pyfx.get_widget_container('h', *row_elements, spacing = 10)

        # Get a widget with the QLabel positioned above the QLineEdit/QPushButtons row from above
        paramfile_row_widget = pyfx.get_widget_container('v', qlabel, row_layout, spacing = 5, widget='widget')

        self.qscroll = QtWidgets.QScrollArea()          # Initialize QScrollArea
        self.qscroll.horizontalScrollBar().hide()       # Hide horizonal scroll
        self.qscroll.setWidgetResizable(True)           # Ensures the main widget fills the scroll area when it is resized
        
        # Create a QHBoxLayout for bottom row of buttons
        button_layout = QtWidgets.QHBoxLayout()

        # Save button (allows for values to be saved to a file)
        self.save_btn = QtWidgets.QPushButton('Save changes to file')
        self.save_btn.setAutoDefault(False)
        self.save_btn.setEnabled(False)

        # Reset changes button (resets displayed values to those within the saved file)
        self.reset_btn = QtWidgets.QPushButton('Revert changes')
        self.reset_btn.setAutoDefault(False)
        self.reset_btn.setEnabled(False)

        # Exit button (sets the selected file to be used within Toothy)
        self.exit_btn = QtWidgets.QPushButton('Use file for analysis')
        self.exit_btn.setAutoDefault(False)

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
        self.instructions_qlabel = QtWidgets.QLabel("<i>Hover over parameter name for a description</i>")
        self.layout.addWidget(self.instructions_qlabel)
        self.layout.addWidget(self.qscroll)
        self.layout.addLayout(button_layout, stretch=0)

        # If there is a parameter file, create a ParameterWidget and add it to the qscroll area
        if os.path.isfile(self.current_param_file_path):
            self.initialize_parameter_widget()
        
        # Otherwise, add a "filler widget" to the qscroll area
        # Note that this filler widget gets overwritten later when qscroll.setWidget() is called again in "initialize_parameter_widget" to add the ParameterWidget to the GUI
        else:
            self.filler_widget = QtWidgets.QWidget()
            self.filler_widget.setStyleSheet("QWidget {background-color: white;}")
            layout = QtWidgets.QVBoxLayout(self.filler_widget)
            layout.addWidget(QtWidgets.QLabel("<i>Select a file to view and edit parameter values.</i>"))
            self.filler_widget.setLayout(layout)
            self.qscroll.setWidget(self.filler_widget)

            # Hide instructions for interacting with parameters since none are shown yet
            self.instructions_qlabel.setVisible(False)

            # Disable button for confirming file since one is not selected yet
            self.exit_btn.setEnabled(False)
    
    def connect_signals(self):
        """
        Connect widget signals to functions
        """
        # Parameters file buttons
        self.paramfile_btn.clicked.connect(self.choose_param_file)
        self.paramfile_autofill_btn.clicked.connect(self.generate_default_param_file)

        # Bottom row buttons
        self.save_btn.clicked.connect(self.save_param_file)
        self.reset_btn.clicked.connect(self.reset_params)
        self.exit_btn.clicked.connect(self.confirm_file) # Exits QDialog
        
    def file_selection_update(self):
        """
        Called when a file selection is made. Cases:
        1. Saving changes to an open file
        2. Loading a file from the "Select a file" button
        3. Generating (and saving) a new file with default parameters

        Updates the displayed filepath, displayed GUI values, internal variables representing the parameters, and toggles the appropriate buttons.

        Does not update the "toothy paths"; this is achieved with the "confirm/exit" button.
        """
        # Update the path displayed in the QLineEdit widget
        self.path_display_le.setText(self.current_param_file_path)

        # Read the values in the file
        new_param_dict = qparam.read_param_file(self.current_param_file_path)[0]

        # Display the updated values in the GUI
        self.main_widget.update_gui_from_param_dict(new_param_dict)

        # Set the current parameter values
        self.PARAMS_DICT = new_param_dict
        
        # Update the "original values" dictionary
        self.PARAMS_DICT_ORIG = dict(self.PARAMS_DICT)

        # Disable the "save" and "reset" buttons, enable the "confirm/exit" button
        self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.exit_btn.setEnabled(True)

    def choose_param_file(self):
        """
        Slot for parameter file selection button.
        Select parameter file from file explorer and trigger function to update the GUI.
        """
        # Get the current path
        current_path = self.current_param_file_path 

        # Open a file dialog and allow the user to load a file
        # Note: param_dict is None if invalid; otherwise, can be used without further checking of valid types
        param_dict, file_path = h_io.select_load_param_file(init_path=current_path, parent=self)

        # If the parameter file is valid (invalid files are returned as None), update 
        if param_dict is not None:
            # Make note of the current file path; if it is empty, we will need to initialize a parameter widget
            previous_file_path = self.current_param_file_path

            # Update the file path internally
            self.current_param_file_path = str(file_path)

            # Create a parameter widget if needed
            if previous_file_path == "":
                self.initialize_parameter_widget()

            # Propagate widger changes using the updated parameter file path
            self.file_selection_update()

        # If param_dict is None, the dictionary was invalid. We do not make any updates.
    
    def generate_default_param_file(self):
        """
        Slot for default parameter file generation button.
        Opens a file dialog for user to select a path to which Toothy will save a new parameter file with default values.
        """
        # Get a dictionary of default parameters
        toothy_default_parameter_dict = qparam.get_toothy_default_parameter_dict()

        # Save the dictionary as a JSON file; open a file dialog over the current QDialog with the provided title; returns path to saved file
        file_path = h_io.select_save_param_file(
            parameter_dictionary = toothy_default_parameter_dict, 
            initial_dir = os.getcwd(),
            initial_file_name="toothy_default_parameters.json",
            title = 'Save default parameter file', 
            parent = self
        )

        # If the file was saved, update
        if file_path:
            # Make note of the current file path; if it is empty, we will need to initialize a parameter widget
            previous_file_path = self.current_param_file_path

            # Update the file path internally
            self.current_param_file_path = str(file_path)

            # Create a parameter widget if needed
            if previous_file_path == "":
                self.initialize_parameter_widget()

            # Propagate widger changes using the updated parameter file path
            self.file_selection_update()
    
    def update_slot(self, param_dict_from_gui):
        """
        Slot connected to the "update_signal" signal from the ParameterWidget object.
        Update internal parameter dictionary based on any user input in the GUI widgets.
        (Also receives a list of invalid keys from signal; ignored by slot.)
        """
        # Update the internal dictionary with the values from the GUI
        self.PARAMS_DICT.update(param_dict_from_gui)

        # Compare the updated values from the GUI with the original values in the file.
        # If there were any changes made to the parameters, enable the save and reset buttons.
        all_params_equal = all([self.PARAMS_DICT[k] == self.PARAMS_DICT_ORIG[k] for k in param_dict_from_gui.keys()])
        change_made = not all_params_equal
        self.save_btn.setEnabled(change_made)
        self.reset_btn.setEnabled(change_made)
        self.exit_btn.setEnabled(all_params_equal) # Only enable when there are no unsaved changes

    def save_param_file(self):
        """
        Slot for "Save changes to file" button.
        Save the current parameter dictionary into a file.
        """
        # Get the path to the directory containing the currently selected file
        file_dialog_initial_dir = os.path.dirname(self.current_param_file_path)

        # Get the file name of the currently selected file
        file_dialog_initial_name = os.path.basename(self.current_param_file_path)

        # Open a file dialog to the currently selected file (default will be to overwrite, with confirmation); after user selects a path, save JSON to path
        file_path = h_io.select_save_param_file(
            parameter_dictionary = self.PARAMS_DICT,
            initial_dir = file_dialog_initial_dir,
            initial_file_name = file_dialog_initial_name,
            parent=self
        )

        # If a file was saved, update the selected file path
        if file_path:
            self.current_param_file_path = str(file_path)   # Update internal variable for current file path. Note that this does not update the "toothy" paths; a separate button for this becomes active after the file is saved.
            self.file_selection_update()                    # Run larger update

    def reset_params(self):
        """
        Slot for "Reset display to saved values" button.
        Reset parameters to original values (not default values; the original values in the file before user-made changes in GUI)
        """
        # Send the original values back to the GUI
        self.main_widget.update_gui_from_param_dict(self.PARAMS_DICT_ORIG)

        # Disable the "save" and "reset" buttons, enable the "exit" button
        self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.exit_btn.setEnabled(True)

    def confirm_file(self):
        """
        Slot for "Use file for analysis" button. Updates the "toothy paths" file to point to the selected file.
        """
        # Update "base dirs"
        current_path_list = h_io.get_toothy_paths()
        updated_path_list = current_path_list[0:3] + [self.current_param_file_path]
        h_io.write_toothy_paths(updated_path_list)

        # Close the window
        self.accept()