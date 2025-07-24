#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StyleSheets

@author: amandaschott
"""
from copy import deepcopy

##############################################################################
################                    BUTTONS                   ################
##############################################################################

TOGGLE_BTN = {'QPushButton' : {
                              'background-color' : 'whitesmoke',
                              'border-width'  : '1px',
                              'border-style'  : 'solid',
                              'border-color'  : 'gray',
                              'border-radius' : '3px',
                              'color' : 'black',
                              'font-weight' : 'normal',
                              'padding' : '4px',
                              }, 
              
              'QPushButton:pressed' : {
                                      'background-color' : 'dimgray',
                                      'border-color' : 'gray',
                                      'color' : 'white',
                                      },
            
              'QPushButton:checked' : {
                                      'background-color' : 'darkgray',
                                      'border-color' : 'gray',
                                      'color' : 'black',
                                      },
            
              'QPushButton:disabled' : {
                                       'background-color' : 'gainsboro',
                                       'border-color'  : 'darkgray',
                                       'color' : 'gray',
                                       },
               
              'QPushButton:disabled:checked' : {
                                               'background-color' : 'darkgray',
                                               'border-color'  : 'darkgray',
                                               'color' : 'dimgray',
                                               }
              }

###   "sunken" toggle buttons (e.g. on Toothy's home screen)

INSET_BTN = deepcopy(TOGGLE_BTN)
INSET_BTN['QPushButton'].update({
                                'border-width'  : '2px',
                                'border-style'  : 'outset',
                                })
INSET_BTN['QPushButton:pressed']['border-style'] = 'inset'
INSET_BTN['QPushButton:checked']['border-style'] = 'inset'
INSET_BTN['QPushButton:disabled']['border-style'] = 'outset'
INSET_BTN['QPushButton:disabled:checked']['border-style'] = 'inset'

BOLD_INSET_BTN = deepcopy(INSET_BTN)
BOLD_INSET_BTN['QPushButton']['border-width'] = '3px'
BOLD_INSET_BTN['QPushButton']['font-weight'] = 'bold'

HOME_BTN = deepcopy(BOLD_INSET_BTN)
HOME_BTN['QPushButton']['background-color'] = 'gainsboro'

###   expand/collapse buttons for settings panes, plots, etc

EXPAND_LEFT_BTN = deepcopy(INSET_BTN)  # frequency plots
EXPAND_LEFT_BTN['QPushButton'].update({
                                      'image' : 'url(:/icons/double_chevron_left.png)',
                                      'image-position' : 'left',
                                      })
EXPAND_LEFT_BTN['QPushButton:checked']['image'] = 'url(:/icons/double_chevron_right.png)'

EXPAND_DOWN_BTN = deepcopy(INSET_BTN) # settings pane
EXPAND_DOWN_BTN['QPushButton'].update({
                                      'image' : 'url(:/icons/double_chevron_down.png)',
                                      'image-position' : 'right',
                                      })
EXPAND_DOWN_BTN['QPushButton:checked']['image'] = 'url(:/icons/double_chevron_up.png)'

EXPAND_PARAMS_BTN = deepcopy(EXPAND_DOWN_BTN) # parameter inputs
EXPAND_PARAMS_BTN['QPushButton']['icon'] = 'url(:/icons/settings.png)'

EXPAND_AUX_BTN = deepcopy(EXPAND_DOWN_BTN)
EXPAND_AUX_BTN['QPushButton']['icon'] = 'url(:/icons/view.png)'

###   circular colored buttons in the main processed data hub

ANALYSIS_BTN = {'QPushButton' : {
                                'background-color' : 'white',
                                'border-width' : '4px',
                                'border-style' : 'outset',
                                'border-color' : 'rgb(128,128,128)',
                                'border-radius' : '11px',
                                'min-width'  : '15px',
                                'max-width'  : '15px',
                                'min-height' : '15px',
                                'max-height' : '15px',
                                },
    
                'QPushButton:disabled' : {
                                         'background-color' : 'rgb(220,220,220)',
                                         },
          
                'QPushButton:pressed' : {
                                        'background-color' : 'dimgray',
                                        }
                }

###   convert between relative and absolute timepoints

TRANGE_TOGGLE_BTN = deepcopy(TOGGLE_BTN)
TRANGE_TOGGLE_BTN['QPushButton'].update({
                                        'border' : 'none',
                                        'outline' : 'none',
                                        #'image' : 'url(:/icons/warning_yellow.png)',
                                        'image' : 'url(:/icons/swap_arrows_vert.png)',
                                        #'background-image':'url(:/icons/swap_arrows_vert.png)',
                                         'padding':'1px',
                                         })

###   show/hide raw recording metadata

META_TOGGLE_BTN = deepcopy(TOGGLE_BTN)
META_TOGGLE_BTN['QPushButton'].update({
                                      'image' : 'url(:/icons/info_blue.png)',
                                      'padding' : '4px',
                                      })

##############################################################################
################                     ICONS                    ################
##############################################################################

ICON_BTN = {'QPushButton' : {
                            'border' : 'none',
                            'outline' : 'none',
                            'padding' : '0px',
                              },
            
            'QPushButton#params_warning' : {
                                           'image' : 'url(:/icons/warning_yellow.png)',
                                           },
            }


##############################################################################
################                 INPUT WIDGETS                ################
##############################################################################


###   numeric inputs and dropdown menus

PARAM_INPUTS = {'QAbstractSpinBox' : {
                                     'background-color':'white',
                                     'color':'black',
                                     },
                'QAbstractSpinBox:disabled' : {'color':'gainsboro',},
       
                'QComboBox'      : {'color':'black',},
                'QComboBox:open' : {'background-color':'white',
                                    'color' : 'black',},
                }

###   missing/invalid input parameters

PARAM_INPUTS_OFF = deepcopy(PARAM_INPUTS)
PARAM_INPUTS_OFF['QAbstractSpinBox'].update({'background-color' : 'red', 
                                             'color' : 'transparent'})
PARAM_INPUTS_OFF['QAbstractSpinBox:disabled']['color'] = 'transparent'

###   list interfaces

QLIST = {'QListWidget' : {
                         'background-color' : 'rgba(255,255,255,50)',
                         'border-width'  : '2px',
                         'border-style'  : 'groove',
                         'border-color'  : 'rgb(150,150,150)',
                         'border-radius' : '2px',
                         'padding' : '0px',
                         },
              
         'QListWidget::item' : {
                               'background-color' : 'rgb(255,255,255)',
                               'color' : 'black',
                               'border-width'  : '1px',
                               'border-style'  : 'solid',
                               'border-color'  : 'rgb(200,200,200)',
                               'border-radius' : '1px',
                               'padding' : '4px',
                               },
              
         'QListWidget::item:selected' : {
                                        'background-color' : 'rgba(85,70,160,200)',
                                        'color' : 'white',
                                        }
         }

###   context menu

QMENU = {'QMenu' : {
                   'border-width'  : '4px',
                   'border-style'  : 'ridge',
                   'border-color'  : 'gray',
                   },
        
        'QMenu::item' : {
                        'padding' : '2px 5px 2px 20px',
                       },

        'QMenu::item:selected' : {
                              'background-color' : 'rgba(85,70,160,200)',
                              'color' : 'white',
                              },
        'QMenu::item:disabled' : {
                              'color' : 'lightgray',
                              },
        
        'QLabel' : {
                   'background-color' : 'rgb(200,200,200)',
                   'border-bottom' : '2px solid dimgray',
                   'border-top' : '2px solid dimgray',
                   'padding' : '2px',
                   },
        'QLabel#top_header' : {'border-top':'none'},
        }


##############################################################################
################             CHANNEL SELECTION GUI            ################
##############################################################################


###   centralized tools for event detection/curation

EVENT_GBOX = {'QGroupBox' : {
                           'background-color' : 'rgba(220,220,220,100)',  # gainsboro
                           'border' : '2px solid darkgray',
                           'border-top' : '5px double black',
                           'border-radius' : '6px',
                           'border-top-left-radius'  : '1px',
                           'border-top-right-radius' : '1px',
                           'font-size'   : '16pt',
                           'font-weight' : 'bold',
                           'margin-top' : '10px',
                           'padding' : '2px',
                           'padding-bottom' : '10px',
                           },
             
             'QGroupBox::title' : {
                                  'background-color'    : 'palette(button)',
                                  'subcontrol-origin'   : 'margin',
                                  'subcontrol-position' : 'top center',
                                  'padding' : '1px 4px', # top, right, bottom, left
                                  }
             }

###   color-coded line (bottom of event boxes)

EVENT_GBOX_LINE = {'QLabel' : {
                              'border' : '1px solid transparent',
                              'border-bottom-width' : '3px',
                              'border-bottom-color' : 'black',
                              'max-height' : '2px',
                              }
                  }

###   event show/hide button (eye icon)

EVENT_SHOW_BTN = {'QPushButton' : {
                                  'background-color' : 'whitesmoke',
                                  'border' : '2px outset gray',
                                  'image' : 'url(:/icons/show_outline.png)',
                                  },
                       
                  'QPushButton:checked' : {
                                          'background-color' : 'gainsboro',
                                          'border' : '2px inset gray',
                                          'image' : 'url(:/icons/hide_outline.png)',
                                          }
                  }


##############################################################################
################               EVENT ANALYSIS GUI             ################
##############################################################################


EVENT_SETTINGS_GBOX = {'QGroupBox' : {
                                     'border' : '1px solid gray',
                                     'border-radius' : '8px',
                                     'font-size' : '16pt',
                                     'font-weight' : 'bold',
                                     'margin-top' : '10px',
                                     'padding' : '10px 5px 10px 5px',
                                     },
           
                       'QGroupBox::title' : {
                                            'subcontrol-origin' : 'margin',
                                            'subcontrol-position' : 'top left',
                                            'padding' : '2px 5px',
                                            }
                       }
