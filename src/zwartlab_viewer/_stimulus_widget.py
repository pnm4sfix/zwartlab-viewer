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



import dask.array as da
import numpy as np
#import cupy as cp


from ._reader import caiman_reader, read_stim_zarr
from ._utils import sklearn_phase_corr



if TYPE_CHECKING:
    import napari

class StimulusWidget(Container):
    def __init__(self, napari_viewer):
        
        super().__init__()
        
        # separate viewer
        # load stimulus videos associated with fnames
        # crop in time and concateante them to seamlessly match the volume
        # load stimulus lazily maybe with napari video reader
        self.viewer = napari_viewer
        self.file_wid = FileEdit(value='./Select files to process', tooltip = "Select files to process", mode ="d") # mode for folder?
        self.load_wid = PushButton(text = "Load Video", tooltip = "Load Video")
        self.motion_wid = PushButton(text = "Calculate Motion", tooltip = "Calculate Motion")

        self.load_wid.clicked.connect(self.load_stimulus)
        self.motion_wid.clicked.connect(self.fast_motion_estimation)
        
        self.extend([self.file_wid, self.load_wid, self.motion_wid])


    def get_params(self):
        self.stim_zarr = self.file_wid.value
        

    def load_stimulus(self):
        self.get_params()
        self.stim_video = read_stim_zarr(self.stim_zarr)
        self.add_stimulus_to_viewer()

    def add_stimulus_to_viewer(self):
        y = 1 # change to px/mm coordinates
        x = 1 # change to px.mm coordinates
        scale = (1/60, 1, 1, y, x) # TZCYX
        metadata = {
            "name": "Stimulus",
            "scale": scale,
            "opacity": 0.5
        }
        self.viewer.add_image(data = self.stim_video, **metadata)


    # useful functions
    # image statistics
    # motion
    def fast_motion_estimation(self):#, log_df, skip = 8):
        overlap = da.overlap.overlap(self.stim_video, depth = 1, boundary = {0:0, 1: "none", 2:"none", 3:"none"})
        self.shifts = overlap.map_blocks(sklearn_phase_corr, chunks = (1, 1, 1, 2), dtype = np.uint8).compute()
        
        self.magnitudes = np.sqrt(self.shifts[:, :, :, 1]**2 + self.shifts[:, :, :, 0]**2)

        # get angles from shifts
        rads = np.arctan2(self.shifts[:, :, :, 1], self.shifts[:, :, :, 0]).flatten() *-1
        rads[rads < 0] = (2*np.pi) + rads[rads<0]
        
        self.degrees = np.degrees(rads)
        
    def virtual_position(self):
        self.position = np.cumsum(self.shifts)
       

        
