#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StyleSheets

@author: amandaschott
"""

##############################################################################
################                 HOME BUTTONS                 ################
##############################################################################

INSET_BTN = {'QPushButton' : {
                            'background-color' : 'gainsboro',
                            'border-width'  : '3px',
                            'border-style'  : 'outset',
                            'border-color'  : 'gray',
                            'border-radius' : '2px',
                            'color' : 'black',
                            'padding' : '4px',
                            'font-weight' : 'bold',
                            }, 
    
            'QPushButton:pressed' : {
                                    'background-color' : 'dimgray',
                                    'border-style' : 'inset',
                                    'border-color'  : 'gray',
                                    'color' : 'white',
                                    },
            
            'QPushButton:checked' : {
                                    'background-color' : 'darkgray',
                                    'border-style' : 'inset',
                                    'border-color'  : 'gray',
                                    'color' : 'black',
                                    },
            
            'QPushButton:disabled' : {
                                     'background-color' : 'gainsboro',
                                     'border-style' : 'outset',
                                     'border-color'  : 'darkgray',
                                     'color' : 'gray',
                                     },
               
            'QPushButton:disabled:checked' : {
                                             'background-color' : 'darkgray',
                                             'border-style' : 'inset',
                                             'border-color'  : 'darkgray',
                                             'color' : 'dimgray',
                                             }
            }


##############################################################################
################               ANALYSIS BUTTONS               ################
##############################################################################


ANALYSIS_BTN = {'QPushButton' : {
                                'background-color' : 'white',
                                'border' : '4px outset rgb(128,128,128)',
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


##############################################################################
################                 LIST WIDGETS                 ################
##############################################################################


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


##############################################################################
################             CHANNEL SELECTION GUI            ################
##############################################################################


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

###   freq. band plot toggle button

FREQ_TOGGLE_BTN = {'QPushButton' : {
                                   'background-color' : 'whitesmoke',
                                   'border' : '2px outset gray',
                                   'color' : 'black',
                                   'image' : 'url(:/icons/double_chevron_left.png)',
                                   'image-position' : 'left',
                                   'padding' : '10px 5px',
                                   },
                                  
                   'QPushButton:checked' : {
                                           'background-color' : 'gainsboro',
                                           'border' : '2px inset gray',
                                           'image' : 'url(:/icons/double_chevron_right.png)',
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
