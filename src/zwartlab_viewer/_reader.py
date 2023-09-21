"""
This module is an example of a barebones numpy reader plugin for napari.
It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/guides.html?#readers
"""
import glob
import os
import re
import warnings
from pathlib import Path
import numpy as np
import dask
import dask.array as da
import tifffile
from tifffile.tifffile import TiffFile
from napari.utils import progress
from napari_czifile2._reader import reader_function as czi_reader
from ScanImageTiffReader import ScanImageTiffReader
import json
import numbers
from tifffile import natural_sorted

try: 
    import cupy as cp
    if cp.cuda.Device():
        #device = "gpu" - gpu not working yet
        device = "cpu"
except:
    print("No cupy")
    device = "cpu"
    

def napari_get_reader(path):
    path = os.path.abspath(path)
    print("Path for napari reader is {}".format(path))
    # check if path is tif, check if tif was scanimage acquired
    if path.endswith(".tif"):
        # return tif reader
        return tif_reader

    # check if path is czi and return czi reader
    elif path.endswith(".czi"):
        return czi_reader



def extract_metadata(im_path):
    """Arguments:
            im_path: 
                    file path to .tif image stack for extracting pixel size"""
    
    with TiffFile(im_path) as tif:
        tags = tif.pages[0].tags
        axes = tif.series[0].axes
        kind = tif.series[0].kind
        imagej_metadata = tif.imagej_metadata

    return kind, tags, axes, imagej_metadata

def extract_im(im_path):
    """Arguments:
            im_path: 
                    file path to .tif image stack for extracting as array"""
    
    with TiffFile(im_path) as tif:
        vol = tif.asarray()

    return vol
    

def preprocess_imagej_metadata(imagej_metadata):
    """Arguments:
        imagej_metadata: str. Image J metadata from tifffile
        
       Returns:
        imagej_metadata: dict. ImageJ metadata converted to easy to use dictionary"""
    split_meta = imagej_metadata["Info"].split("\n")
    imagej_metadata = {meta.split("=")[0]: meta.split("=")[1] for meta in split_meta if len(meta.split("=")) ==2}
    
    return imagej_metadata

def get_xyz(tags, imagej_metadata):
    """Arguments:
        imagej_metadata: dict. ImageJ metadata in dictionary form.
       Return:
        x, y, z: pixel sizes in micron for x, y, z dim"""

    x_um_px = tags["XResolution"].value[1]/tags["XResolution"].value[0] 
    y_um_px = tags["XResolution"].value[1]/tags["YResolution"].value[0] 
    z_um_px = imagej_metadata["spacing"]


    #this was for czi but now using czi reader
    #x = imagej_metadata["Scaling|Distance|Value #1 "]
    #y = imagej_metadata["Scaling|Distance|Value #2 "]
    #z = imagej_metadata["Scaling|Distance|Value #3 "]

    return np.array([z_um_px, y_um_px, x_um_px])


def tif_reader(paths, lazy = True, gpu = False):
    layer_data = []
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        kind, tags, axes, imagej_metadata = extract_metadata(path)


        # this was for zeiss tif - now using zeiss reader for those czi
        #imagej_metadata = preprocess_imagej_metadata(imagej_metadata)

        # check if scanimage in tif tags
        if kind == "scanimage":
            
            print("Reading ScanImage File")
            
            return scanimage_reader(path, lazy, device)

        elif kind == "imagej":
            print("Reading ImageJ File")
            im = extract_im(path)

            z, y, x = get_xyz(tags, imagej_metadata) # does this work for zbrain

            if axes == "ZYX":
                # reshape to 5 dims of shape TZCYX
                im = im.reshape(1, 1, *im.shape)
                scale = (1, 1, z, y, x)
            elif axes == "CZYX":
                # reshape to 5 dims of shape TZCYX
                im = im.reshape(1, *im.shape)
                scale = (1,1, z, y, x)
                

            if im.dtype == "uint16":
                contrast_limits = (0, 2 ** 16 - 1)
                contrast_limits = (np.percentile(im, 10), np.percentile(im, 95))
            elif im.dtype == "uint8":
                contrast_limits = (0, 2 ** 8 - 1)
                contrast_limits = (np.percentile(im, 10), np.percentile(im, 95))

            elif im.dtype == "int16":
                contrast_limits = (-(2 ** 16 - 1)/2, (2 ** 16 - 1)/2)
                contrast_limits = (np.percentile(im, 10), np.percentile(im, 95))
            else:
                contrast_limits = None

            metadata = {
                "name": "image",
                "scale": scale,
                "contrast_limits": contrast_limits,
                "opacity": 0.5
            }

            if "H2B" in path:
                # flip image along z axis with numpy flip
                im = np.flip(im, axis=0)

        
        layer_data.append((im, metadata, "image"))

    return layer_data



def scanimage_reader(paths, lazy= True, device = "cpu"):
    # for now ignore axes as seems incorrect
    print("Reading ScanImage File")
    layer_data = []
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        scanimage_file = ScanImageTiffReader(path)
        block_size = int(scanimage_file.shape()[0] / 5)
        if lazy:
            # dask image
            if device == "gpu":
                im = lazy_scanimage_read(path, nframes = block_size, arraytype = "cupy")
            elif device == "cpu":
                im = lazy_scanimage_read(path, nframes = block_size)
        else:
            im = scanimage_file.data()
            
        metadata = scanimage_file.metadata()
        main_metadata, roi_metadata = parse_scanimage_metadata(metadata)
        z, y, x = get_scanimage_resolution(main_metadata, roi_metadata)
        numSlices = get_Z(main_metadata)
        C = get_numchannels(main_metadata)
        obj_res = get_objective_resolution(main_metadata)
        print("objective resolution was {}".format(obj_res))

        if obj_res == 15: # maybe bemore specific
            correction_factor = 4.62866667
            x, y = x*correction_factor, y * correction_factor
            #pass # apply correction to y and x
        
        nframes, height, width = im.shape
        remainder = nframes % numSlices
        full_length = nframes-remainder

        if full_length == nframes:
            print("No incomplete volumes")
        else:
            print("Incomplete volume")
            im = im[:full_length]
            
        nVols = int(full_length/numSlices)
        
        T, Z, Y, X = nVols, numSlices, height, width
        
        print("T {}, Z {}, C {}, Y {}, X {}".format(T, Z, C, Y, X))

        #if len(im.shape) == 3:
        # reshape to 5 dims of shape TZCYX
        im = im.reshape(T, Z, C, Y, X)
        scale = (1, z, 1, y, x)
            
        #elif len(im.shape) == 4:
            # reshape to 5 dims of shape TZCYX
            # if im is dask array:
        #    im = im.reshape((T, Z, C, Y, X))
            
        #    scale = (1,1, z, y, x)

        #if im.dtype == "uint16":
            #contrast_limits = (0, 2 ** 16 - 1)
        if type(im) == dask.array.core.Array:
            # use first 100 for speed
            in_mem_im_sub = im[:100].compute()
            if type(in_mem_im_sub) != np.ndarray:
                in_mem_im_sub = cp.asnumpy(in_mem_im_sub)
            contrast_limits = (np.percentile(in_mem_im_sub, 10), np.percentile(in_mem_im_sub, 95))
            #contrast_limits = (da.percentile(flat_im, 10).compute(), da.percentile(flat_im, 95).compute())
        else:
            contrast_limits = (np.percentile(im, 10), np.percentile(im, 95))

        #elif im.dtype == "int16":
        #    contrast_limits = -(2 ** 16 - 1)/2, (2 ** 16 - 1)/2
        #    contrast_limits = (np.percentile(im, 10), np.percentile(im, 95))

        #elif im.dtype == "uint8":
        #    contrast_limits = (0, 2 ** 8 - 1)
        #    contrast_limits = (np.percentile(im, 10), np.percentile(im, 95))
        #else:
        #    contrast_limits = None
        print("contrast limit is {}".format(contrast_limits))

        metadata = {
            "name": "image",
            "scale": scale,
            "contrast_limits": contrast_limits,
            "opacity": 0.5
        }
        
        layer_data.append((im, metadata, "image"))

    return layer_data




def parse_scanimage_metadata(metadata):


    # extract main metadata
    main_metadata = {}
    for string in metadata[:metadata.find("RoiGroups")-5].replace("=", ":").split("\n")[:-5]:

        try:
            key, value = string.split(":")
            main_metadata[key] = value
        except:
            print(key, value)
    roi_metadata = json.loads("{" +metadata[metadata.find("RoiGroups")-1:])

    return main_metadata, roi_metadata  

def get_scanimage_resolution(main_metadata, roi_metadata, xy_correction = 4.62866667):

    #xmin, ymin, xmax, ymax, tl, tr, bl, br
    tl, tr, bl, br  = main_metadata["SI.hRoiManager.imagingFovUm "].replace(" [", "").replace("]", "").split(";")
    xmin, ymin = tl.split(" ")
    xmin, ymin = float(xmin), float(ymin)

    xmax, ymax = bl.split(" ")
    xmax, ymax = float(xmax), float(ymax)

    x_range, y_range = xmax-xmin, ymax-ymin


    x_px, y_px = roi_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]["scanfields"]["pixelResolutionXY"]

    x_px_size = x_range / x_px
    y_px_size = y_range / y_px


    z_px_size = float(main_metadata["SI.hStackManager.actualStackZStepSize "])

    return np.array([z_px_size, y_px_size * xy_correction, x_px_size * xy_correction])

def get_Z(main_metadata):
    Z = int(main_metadata["SI.hStackManager.actualNumSlices "].replace(" [", "").replace("]", "").split(";")[0])
    #T = int(main_metadata["SI.hStackManager.actualNumVolumes "].replace(" [", "").replace("]", "").split(";")[0])
    return Z

def get_objective_resolution(main_metadata):
    # Useful for correcting images with wrong calibration
    obj_res = int(float(main_metadata["SI.objectiveResolution "].replace(" [", "").replace("]", "").split(";")[0]))
    return obj_res

def get_numchannels(main_metadata):
    # this is untested for dual color
    C = main_metadata['SI.hChannels.channelsActive '].replace(" [", "").replace("]", "").split(";")[0]
    try:
        # if C can be converted to integer then just one channel
        C = int(C)
        C = 1
    except:
        try:
            # if C cant be converted to integer then it is a list of channels
            len(C)
        except:
            C = 1
    return C
            



def lazy_scanimage_read(fname, nframes=1000, *, arraytype="numpy"): # maybe give option of using gpu - maybe a check box
    """
    Read image data into a Dask Array.

    Provides a simple, fast mechanism to ingest image data into a
    Dask Array.

    Parameters
    ----------
    fname : str or pathlib.Path
        A glob like string that may match one or multiple filenames.
        Where multiple filenames match, they are sorted using
        natural (as opposed to alphabetical) sort.
    nframes : int, optional
        Number of the frames to include in each chunk (default: 1).
    arraytype : str, optional
        Array type for dask chunks. Available options: "numpy", "cupy".

    Returns
    -------
    array : dask.array.Array
        A Dask Array representing the contents of all image files.
    """

    sfname = str(fname)
    if not isinstance(nframes, numbers.Integral):
        raise ValueError("`nframes` must be an integer.")
    if (nframes != -1) and not (nframes > 0):
        raise ValueError("`nframes` must be greater than zero.")

    if arraytype == "numpy":
        arrayfunc = np.asanyarray
    elif arraytype == "cupy":   # pragma: no cover
        import cupy
        arrayfunc = cupy.asanyarray

    with ScanImageTiffReader(sfname) as imgs:
        shape = tuple(imgs.shape()) #(len(imgs),) + imgs.frame_shape
        dtype = imgs.dtype() #np.dtype(imgs.pixel_type)
        
        # check shape?

    if nframes == -1:
        nframes = shape[0]

    if nframes > shape[0]:
        warnings.warn(
            "`nframes` larger than number of frames in file."
            " Will truncate to number of frames in file.",
            RuntimeWarning
        )
    elif shape[0] % nframes != 0:
        warnings.warn(
            "`nframes` does not nicely divide number of frames in file."
            " Last chunk will contain the remainder.",
            RuntimeWarning
        )

    # place source filenames into dask array after sorting
    filenames = natural_sorted(glob.glob(sfname))
    if len(filenames) > 1:
        ar = da.from_array(filenames, chunks=(nframes,))
        multiple_files = True
    else:
        ar = da.from_array(filenames * shape[0], chunks=(nframes,))
        multiple_files = False

    # read in data using encoded filenames
    a = ar.map_blocks(
        _map_read_frame,
        chunks=da.core.normalize_chunks(
            (nframes,) + shape[1:], shape),
        multiple_files=multiple_files,
        new_axis=list(range(1, len(shape))),
        arrayfunc=arrayfunc,
        meta=arrayfunc([]).astype(dtype),  # meta overwrites `dtype` argument
    )
    return a


def _map_read_frame(x, multiple_files, block_info=None, **kwargs):

    fn = x[0]  # get filename from input chunk

    if multiple_files:
        i, j = 0, 1
    else:
        i, j = block_info[None]['array-location'][0]

    return _read_frame(fn=fn, i=(i, j), **kwargs)


def _read_frame(fn, i, *, arrayfunc=np.asanyarray):
    with ScanImageTiffReader(fn) as imgs:
        return arrayfunc(imgs.data(i[0], i[1]))






