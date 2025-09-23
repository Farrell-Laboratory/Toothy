"""
GUI classes for path-setting QDialog (opens from main menu button "Set paths").

Defines class "SetPathsPopup" (QDialog) which embeds a "SetPathsWidget" (QWidget). 

Authors: Amanda Schott, Kathleen Esfahany

Last updated: 09-09-2025

"""
# Import standard libraries
import os

# Import packages
from PyQt5 import QtWidgets, QtCore, QtGui

# Import custom modules
import pyfx
import qparam
import ephys
import gui_items as gi
import resources_v2     # For icons (":/resources/...") see resources.qrc


class SetPathsWidget(QtWidgets.QWidget):
    """
    Widget for setting default locations for folders/files.
    Convenience widget; users can directly select any necessary folders/files in later steps as well.
    """

    # Define two custom signals
    path_updated_signal = QtCore.pyqtSignal(int)
    saved_signal = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)

        # Define the strings that will label each field
        self.folder_labels = [
            'Input data directory', 
            'Probe configurations directory',
            'Preferred probe configuration file',
            'Parameter file'
            ]
        
        # Get a list of initial paths (elements of self.path_list is later updated by various class functions when file paths are selected)
        self.path_list = [str(x) for x in ephys.base_dirs()]

        # Set up the widget
        self.gen_layout()

        # Connect the widget signals to slots
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
    
    def make_row(self, row_idx, second_icon = None):
        """
        Return widgets for a given row. Each row includes a QLineEdit (for displaying the path) and 1-2 icons.
        """
        # Define stylesheet for QLineEdit
        le_ss = """
        QLineEdit {
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 5px;
        }
        """
        # Create a QLineEdit for displaying the paths. 
        # Note: We use QLineEdit instead of QLabel because QLineEdit automatically displays the end of the string when the width of the window is less than the string, while QLabel shows the beginning.
        path_display_le = QtWidgets.QLineEdit(f'{self.path_list[row_idx]}')
        path_display_le.setTextMargins(0,4,0,4) # Margins for left, top, right, bottom
        path_display_le.setReadOnly(True)
        path_display_le.setStyleSheet(le_ss)
        
        # Create title and button
        qlabel = QtWidgets.QLabel(f'<b>{self.folder_labels[row_idx]}</b>')
        selection_type = self.folder_labels[row_idx].split(' ')[-1].lower() # Get the last word ("directory" or "file") to determine icon
        selection_to_icon_dict = dict(directory="folder", file="load-file")
        icon_resource_path = f":/resources/{selection_to_icon_dict[selection_type]}.png"
        row_elements = [path_display_le, self.make_icon_button(QtGui.QIcon(icon_resource_path))]
        
        # Add a second button if one is requested
        if second_icon is not None:
            row_elements.append(self.make_icon_button(QtGui.QIcon(f':/resources/{second_icon}.png')))
        
        # Create a horizontal row with the QLineEdit and button(s) with icon(s)
        row_layout = pyfx.get_widget_container('h', *row_elements, spacing = 10)

        # Get a widget with the QLabel positioned above the QLineEdit/QPushButton row from above
        row_widget = pyfx.get_widget_container('v', qlabel, row_layout, spacing = 5, widget='widget')

        return row_elements, row_widget
    
    def gen_layout(self):
        """
        Set up layout
        """
        # Create rows of labels (QLabel), path displays (QLineEdit), and buttons (QPushButton with icon)
        # Note: "make_row" returns a list of individual elements (QLineEdit, QPushButton) and the combined widget
        (self.inputdatadir_le, self.inputdatadir_btn), self.inputdatadir_w = self.make_row(0)
        (self.probeconfigdir_le, self.probeconfigdir_btn), self.probeconfigdir_w = self.make_row(1)
        (self.probefile_le, self.probefile_btn, self.probefile_clear_btn), self.probefile_w = self.make_row(2, 'clear-selection')   # Add a second icon for clearing the selected file
        (self.paramfile_le, self.paramfile_btn, self.paramfile_generate_btn), self.paramfile_w = self.make_row(3, 'magic-wand')     # Add a second icon for auto-generating a default parameters file

        # Add a hint to the "folder" buttons
        for button in [self.inputdatadir_btn, self.probeconfigdir_btn]:
            button.setToolTip("Select directory")

        # Add a hint to the "load file" buttons
        for button in [self.probefile_btn, self.paramfile_btn]:
            button.setToolTip("Select a file")
        
        # Add a hint to the "clear" probe configuration file button
        self.probefile_clear_btn.setToolTip('Clear selection')
        
        # Add a hint to the "autofill" parameter button
        self.paramfile_generate_btn.setToolTip("Generate a new parameter file with Toothy's default values")
        
        # Collect all the QLineEdit widgets (class variable used later)
        self.path_qlineedit_widgets = [
            self.inputdatadir_le,
            self.probeconfigdir_le,
            self.probefile_le,
            self.paramfile_le
        ]

        # Put all the widgets into a vertical layout
        layout = pyfx.get_widget_container('v', self.inputdatadir_w, self.probeconfigdir_w, self.probefile_w, self.paramfile_w, spacing = 30)
        
        # Set the layout of the QWidget
        self.setLayout(layout)
        
        # Create a "save" button, disable it, and add it to the layout
        self.save_btn = QtWidgets.QPushButton('Save changes')
        self.save_btn.setEnabled(False)
        self.layout().addWidget(self.save_btn)
    
    def connect_signals(self):
        """ Connect GUI inputs """
        # Connect icon QPushButtons to slots
        self.inputdatadir_btn.clicked.connect(lambda: self.choose_dir(row_idx=0))
        self.probeconfigdir_btn.clicked.connect(lambda: self.choose_dir(row_idx=1))
        self.probefile_btn.clicked.connect(self.choose_probe_file)
        self.probefile_clear_btn.clicked.connect(self.clear_probe_file)
        self.paramfile_btn.clicked.connect(self.choose_param_file)
        self.paramfile_generate_btn.clicked.connect(self.generate_default_param_file)

        # Connect custom signals to slots
        self.path_updated_signal.connect(lambda row_idx: self.update_path(row_idx))
        self.save_btn.clicked.connect(self.save_selections)
    
    def choose_dir(self, row_idx):
        """
        Select base folder for input data or probe configuration files.
        Slot for directory selection buttons.
        """
        # Get the current directory
        current_directory = str(self.path_list[row_idx])
        
        # Get the window heading
        window_title_base = 'Select directory for %s'
        window_titles = [window_title_base % x for x in ['input data', 'probe configuration files']]
        window_title = window_titles[row_idx]

        # When clicked, initialize a file explorer at "current_directory"; save a new selection at index "row_idx"
        selected_dir_path = ephys.select_directory(current_directory, title = window_title, parent = self)
        if selected_dir_path: # Non-empty strings evaluate to True
            self.path_list[row_idx] = str(selected_dir_path) # Update the path internally
            self.path_updated_signal.emit(row_idx) # Broadcast update for QLineEdit to be updated
    
    def choose_probe_file(self):
        """
        Select preferred probe file. 
        Slot for probe file selection button.
        """
        # Get the current file
        current_path = str(self.path_list[2])

        # If no file exists, set the initial path for the file explorer to the "probe configurations" directory
        if not os.path.isfile(current_path):
            current_path = str(self.path_list[1])
        
        # Select a probe file (probe_object == None if probe file is not valid)
        probe_object, file_path = ephys.select_load_probe_file(init_ppath = current_path, parent=self)
        
        # If the probe is valid, update
        if probe_object is not None:
            self.path_list[2] = str(file_path) # Update the file path internally
            self.path_updated_signal.emit(2) # Broadcast updates for QLineEdit to be updated
        
    def clear_probe_file(self):
        """
        Clear preferred probe file. 
        Slot for probe file clear button.
        """
        self.path_list[2] = '' # Clear the file name
        self.path_updated_signal.emit(2) # Broadcast updates
    
    def choose_param_file(self):
        """
        Select parameter file.
        Slot for parameter file selection button.
        """
        # Get the current path
        current_path = str(self.path_list[3])

        # Load the file
        param_dict, file_path = ephys.select_load_param_file(init_ppath=current_path, parent=self)

        # If the parameter file is valid, update
        if param_dict is not None:
            self.path_list[3] = str(file_path) # Update the file path internally
            self.path_updated_signal.emit(3) # Broadcast updates for QLineEdit to be updated
            
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
            self.path_list[3] = str(file_path) # Update internally
            self.path_updated_signal.emit(3) # Broadcast updates for QLineEdit to be updated
            
    def update_path(self, row_idx):
        """
        Update text to the selected location
        """
        # Get the updated text from the class variable
        updated_text = self.path_list[row_idx]

        # Set the text in the QLineEdit to the updated text
        self.path_qlineedit_widgets[row_idx].setText(updated_text)

        # Now that changes have been made, enable the save button
        self.save_btn.setEnabled(True) 
    
    def save_selections(self):
        """
        Save filepaths to "default_folders.txt"
        """
        # Write to the file
        ephys.write_base_dirs(self.path_list)

        # Emit custom "saved_signal" signal
        self.saved_signal.emit()

     
class SetPathsPopup(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set the window title
        self.setWindowTitle('Set paths')

        # Set icon for upper left corner and toolbar
        self.setWindowIcon(QtGui.QIcon(':/resources/logo.png'))

        # Remove the "?" help button
        gi.remove_help_button(self)

        # Make a SetPathsWidget and connect its "saved_signal" signal to this QDialog class' "accept" slot (closes the QDialog)
        self.widget = SetPathsWidget()
        self.widget.saved_signal.connect(self.accept)
        
        # Assign a vertical box layout to this QDialog and add the SetPathsWidget
        layout = QtWidgets.QVBoxLayout(self) 
        layout.addWidget(self.widget)