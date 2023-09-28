"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING


from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from magicgui.widgets import ComboBox, Container, PushButton, SpinBox, FileEdit, FloatSpinBox, Label, TextEdit, CheckBox, FloatSlider
from magicgui.widgets import create_widget, Widget
import napari_plot
from napari_plot._qt.qt_viewer import QtViewer
import os
import pandas as pd
from dask_image.imread import imread
import dask.dataframe as dd
import dask.array as da
import numpy as np
import cupy as cp
import caiman as cm
import ipyparallel as ipp
from ._reader import caiman_reader
from tifffile import natural_sorted
if cp.cuda.Device():
    
    from cudipy.align.imaffine import AffineMap        
else:
        
    from dipy.align.imaffine import AffineMap

if TYPE_CHECKING:
    import napari

class PreprocessWidget(Container):
    def __init__(self, napari_viewer):
        
        super().__init__()
        
        # multifile select widget for selecting tiff images
        
        # simple push button for rotate
        # crop widget
        # align to bonsai widget
        # parellelise log file reads and return the volume df for each, and info to subset the image

        self.fnames_wid = FileEdit(value='./Select files to process', tooltip = "Select files to process", mode ="rm")
        
        self.image_width_wid = SpinBox(value = 0, tooltip = "Select image width for center crop")
        self.image_heigh_wid = SpinBox(value = 0, tooltip = "Select image height for center crop")
        
        self.rotate_wid = SpinBox(value = 0, tooltip = "Select rotation angle", step = 90.0, max = 270.0)
        self.align = ComboBox(value = "None", choices = ["Bonsai", "Wavesurfer"])
        self.run_preprocess_wid = PushButton(text = "Run Preprocess", tooltip = "Run Preprocess")
        
        self.run_preprocess_wid.changed.connect(self.preprocess)
        # add widgets to self with extend
        self.extend([self.fnames_wid, self.image_width_wid, self.image_heigh_wid, self.rotate_wid, self.align])
        

    def get_params(self):
        self.fnames = self.fnames_wid.value
        # sort by time made to get acquisition order
        self.fnames.sort(key=os.path.getmtime)     

        self.image_width = self.image_width_wid.value
        self.image_height = self.image_heigh_wid.value
        self.rotate = self.rotate_wid.value
        self.align = self.align.value
        

    def preprocess(self):
        self.get_params()
        
        # read fnames using reader - maybe add argument to sort or not?
        # this should convert each file into a dask array
        # operate on each file separately to align to bonsai or wavesurfer
        # then when everything has been aligned it can be concatenated and then preprocessed simultaneously
        layer_data = caiman_reader(self.fnames)
        
        # align each volume to associated behavioural and log files
        

        # insert preprocess code
        if (self.image_width > 0) & (self.image_height > 0):
            # perform center crop
            pass
        if self.rotate > 0:
            # perform rotation
            if self.rotate/90 == 1:
                # rotate 90 using np.rot90 - avoids interpolation risk
                pass
            elif self.rotate/90 == 2:
                # rotate 180 using np.rot90 - avoids interpolation risk
                pass
            elif self.rotate/90 == 3:
                # rotate 270 using np.rot90 - avoids interpolation risk
                pass
            else:
                # rotate using interpolation
                pass
            pass
        
        
        pass
        
        

    