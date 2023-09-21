
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._imaging_widget import ImagingWidget
from ._registration_widget import RegistrationWidget
from ._reader import napari_get_reader

__all__ = (
    "ImagingWidget",
    "RegistrationWidget"
    "napari_get_reader"
)


