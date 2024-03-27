"""
This module is an example of a barebones QWidget plugin for napari
It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from codecs import raw_unicode_escape_encode
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
import caiman as cm


import dask.array as da
import numpy as np
#import cupy as cp
from tifffile import natural_sorted

from ._reader import caiman_reader, read_stim_zarr
from ._utils import sklearn_phase_corr



if TYPE_CHECKING:
    import napari

class d1Widget(Container):
    def __init__(self, napari_viewer):
        
        super().__init__()
        
        # 
        self.viewer = napari_viewer
        self.viewer1d = napari_plot.Viewer()
        

        # give access to ephys data and var data - load from xarray as dask so we have channel names
        # give each channel a specific offset
        # with each frame update assume fr is 60 Hz if video present or take the imaging volume rate

        
        # add file widget
        self.file_wid = FileEdit(value='./Select files to process', tooltip = "Select files to process", mode ="rm") # this will be to the xarray file
        self.load_wid = PushButton(text = "Load data", tooltip = "Load data")
        

        self.extend([self.file_wid, self.load_wid])
    
    def get_params(self):
        self.paths = self.file_wid.value
        self.sr = 6e3
        self.window_duration = 60 # seconds

    def load_xarray(self):
        # load xarray 
        # create channel indices 
        
        self.data = None
    
    def load_numpy(self):
        self.data = np.load(self.path)

    def initialise_data_layers(self):
        for channel in range(self.data.shape[1]):
            self.viewer1d.add_line((np.c_[[0], [0]]), color = np.random.random((1, 3)))
        # get the smallest time scale
        ts = []
        for layer in self.viewer.layers:
            ts.append(layer.scale[0])
        min_ts = np.min(np.array(ts))
        self.fastest_fr = 1/min_ts

    def update_layer_data(self):
        for nlayer, layer in enumerate(self.viewer1d.layers):
            data = self.data[self.start:self.stop, nlayer] + (nlayer*5) # add offset
            x = np.arange(len(data)) / self.sr
            layer.data = np.c_[x, data]
            
    def update_ephys_start_stop(self, frame):
        imaging_time = frame / self.fastest_fr # this refences to the timebase of the fastest image in the viewer
        self.stop = int(imaging_time * self.sr)
        ephys_duration_datapoints = int(self.window_duration * self.sr)
        
        self.start = self.stop - ephys_duration_datapoints
        if self.start < 0:
            self.start = 0
        
    def update_slider_callback(self, event):
        self.update_ephys_start_stop(event.value[0])
        self.update_layer_data()
        

## To DO-get ephys and var data saved and load up in gui
"""test

from zwartlab_viewer._1d_widget import d1Widget
import dask.array as da
import numpy as np

wid = d1Widget(viewer)
duration = 1000 #seconds
fake_ephys = da.random.random((1000 * 6e3, 5), chunks = 6000)
fake_vid = da.random.random((1000*60, 1, 3, 200, 200))
fake_data = da.random.random((1000 * 3.07, 1, 1, 200, 200))
viewer.add_image(fake_data, name = "images")
viewer.add_image(fake_vid, name = "stim")

viewer.layers[0].scale = np.array([1/3.07, 1, 1, 1, 1])
viewer.layers[1].scale = np.array([1/60, 1, 1, 1, 1])

wid.data = fake_ephys
wid.get_params()
wid.initialise_data_layers()
viewer.dims.events.current_step.connect(wid.update_slider_callback)"""
