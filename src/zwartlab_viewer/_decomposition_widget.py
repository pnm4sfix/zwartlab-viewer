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

class DecompositionWidget(Container):
    def __init__(self, napari_viewer):
        
        super().__init__()
        # takes dff_tidy dataframe and either performs decomposition
        # decomposition can be either tensor decomposition or cebra
        