
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._imaging_widget import ImagingWidget
from ._registration_widget import RegistrationWidget
from ._reader import napari_get_reader
from ._caiman_widget import CaimanWidget
from ._preprocess_widget import PreprocessWidget
from ._stimulus_widget import StimulusWidget
from ._decomposition_widget import DecompositionWidget
from ._1d_widget import d1Widget

from ._utils import *  #this is slow because everything is imported

__all__ = (
    "ImagingWidget",
    "RegistrationWidget",
    "CaimanWidget",
    "PreprocessWidget",
    "StimulusWidget",
    "DecompositionWidget",
    "d1Widget",
    "napari_get_reader"
)


