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
from cupy import broadcast_arrays
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
import caiman as cm
from caiman.utils.visualization import get_contours
from napari_video.napari_video import VideoReaderNP

try: 
    import cupy as cp
    if cp.cuda.Device():
        #device = "gpu" - gpu not working yet
        device = "cpu"
except:
    print("No cupy")
    device = "cpu"

## TO DO-refactor scanimage reader to be quicker
        # get gpu working
        # extract frame rate, num frames averaged and more metadata from scanimage tif
        # find way of keeping image metadata in mmap and caiman hdf5 files so scaling is correct in napari

#try:
#    print("Starting caiman cluster")
#    if 'dview' in locals():
#        cm.stop_server(dview=dview)
#    c, dview, n_processes = cm.cluster.setup_cluster(
#        backend='local', n_processes=None, single_thread=False)
#except:
#    pass
    

def napari_get_reader(path):
    #path = os.path.abspath(path)
    if not isinstance(path, list):
        path = [path]    
    print("Path for napari reader is {}".format(path))
    # check if path is tif, check if tif was scanimage acquired
    if path[0].endswith(".tif"):
        # return tif reader
        return tif_reader

    # check if path is czi and return czi reader
    elif path[0].endswith(".czi"):
        return czi_reader_wrapper 
    
    elif (path[0].endswith(".hdf5")) | (path[0].endswith(".mmap")):
        return caiman_reader

def czi_reader_wrapper(paths):
    layer_data = czi_reader(paths)
    
    for nlayer, layer in enumerate(layer_data):
        layer = list(layer)
        
        # set opacity to 0.5
        layer[1]["opacity"] = 0.5
        translate = layer[1]["translate"]
        translate = tuple([translate[0], translate[1], 0.0, translate[2], translate[3]])
        
        scale = layer[1]["scale"]
        scale = tuple([scale[0], scale[1], 1.0, scale[2], scale[3]])
        
        
        layer[1]["translate"] = translate
        layer[1]["scale"] = scale
        layer[1]["name"] = layer[1]["name"][0]
        layer[1].pop("channel_axis", None)

        print(layer[1], layer[2])
        ## reshape image to TZCYX
        #T, Z, Y, X = layer[0].shape # assumes in format TZYX
       # 
        #layer[0] = layer[0].reshape(T, Z, 1, Y, X)
        layer_data[nlayer] = tuple(layer)
     
    return layer_data
        
        

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


def tif_reader(paths, lazy = True, device = "cpu"):
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
            print("ImageJ file shape is {}".format(im.shape))
            if axes == "ZYX":
                # reshape to 5 dims of shape TZCYX
                im = im.reshape(1, 1, *im.shape)
                scale = (1, 1, z, y, x)
            elif axes == "CZYX":
                # reshape to 5 dims of shape TZCYX
                im = im.reshape(1, *im.shape)
                scale = (1,1, z, y, x)
            print("ImageJ file shape is {}".format(im.shape))   

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
                print("Flipping image along z axis")
                #im = np.flip(im, axis=0)

        else:
            print("Not sure what type of tif")
            im = extract_im(path)
            metadata = {
                "name": "image",
                
                "opacity": 0.5
            }

        
        layer_data.append((im, metadata, "image"))

    return layer_data



def scanimage_reader(paths, volumes = 1, lazy= True, device = "cpu"):
    # for now ignore axes as seems incorrect
    # TO_ DOget fr, averaging info
    # Might have to import scanimage_read_frame here for parallel processing later
    from ._utils import read_scanimage_frame
    read_scanimage_frames = dask.delayed(read_scanimage_frame)
    print("Reading ScanImage File")
    layer_data = []
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        scanimage_file = ScanImageTiffReader(path)
        metadata = scanimage_file.metadata()
        main_metadata, roi_metadata = parse_scanimage_metadata(metadata)
        obj_res = get_objective_resolution(main_metadata)
        imaging_rois = roi_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
        n_imaging_rois = len(imaging_rois)
     
        numSlices = get_Z(main_metadata)
        C = get_numchannels(main_metadata)
        
        nframes, height, width = scanimage_file.shape()
        remainder = nframes % (numSlices * C)
        full_length = nframes-remainder

        if full_length == nframes:
            print("No incomplete volumes")
        else:
            print("Incomplete volume")
            #im = im[:full_length]
            
        nVols = int(full_length/numSlices)
        
        T, Z, Y, X = nVols, numSlices, height, width
        
        print("T {}, Z {}, C {}, Y {}, X {}".format(T, Z, C, Y, X))
        
        print("objective resolution was {}".format(obj_res))

        blocksize = volumes * Z * C
        nblocks = int(np.ceil(full_length/blocksize))

        blocks = [(n*blocksize, (n+1) * blocksize) for n in range(nblocks)]    
        
        if lazy:
            if device == "gpu":
                
                im = da.concatenate([da.from_delayed(read_scanimage_frames(scanimage_file, start, end, cp.asanyarray), 
                                                 shape = (blocksize, Y, X), dtype = "int16").reshape((1, Z, C, Y, X)) for start, end in blocks], axis=0)
            elif device == "cpu":
                im = da.concatenate([da.from_delayed(read_scanimage_frames(scanimage_file, start, end), 
                                                 shape = (blocksize, Y, X), dtype = "int16").reshape((1, Z, C, Y, X)) for start, end in blocks], axis=0)

        else:

            im = scanimage_file.data()

        # loop through each roi and create dask image, metadata and add as individual layer
        # each scanfield may need new separate layer if in a different x, y or apply the affine beforehand?
        

        #pass # apply correction to y and x
        #if obj_res == 15: # maybe bemore specific
        #    correction_factor = 4.62866667
        #else:
        #    correction_factor = None
        
        # reshape to 5 dims of shape TZCYX
        #im = im.reshape(T, Z, C, Y, X)
        if type(imaging_rois) == list:
            # get roi indices
            zs = []
            #zs = main_metadata['SI.hStackManager.zs '] # use this instead
            for nroi, roi in enumerate(imaging_rois):
                for zss in roi["zs"]:
                    zs.append(zss)
            
            # then we have multiple rois - loop through and add as seperate layers
            for nroi, roi in enumerate(imaging_rois):
                t, z, y, x = get_scanimage_resolution(main_metadata, roi_metadata, roi = nroi)
                contrast_limits = get_contrast_limits(im)
                scale = (t, z, 1, y, x)
                roi_zs = roi["zs"]
                z_idx = np.where(np.isin(zs, roi_zs))[0]
                
                # subset im to get just the corresponding slices
                roi_im = im[:, z_idx]
                
                # add affine here for each roi scanfield?
                
                print("contrast limit is {}".format(contrast_limits))

                metadata = {
                    "name": roi["name"],
                    "scale": scale,
                    "contrast_limits": contrast_limits,
                    "opacity": 0.5
                }
        
                layer_data.append((roi_im, metadata, "image"))
                

                

        elif type(imaging_rois) == dict:
            # then just one main imaging FOV



        
            #x, y = x*correction_factor, y * correction_factor
            t, z, y, x = get_scanimage_resolution(main_metadata, roi_metadata)
            contrast_limits = get_contrast_limits(im)
        
            scale = (t, z, 1, y, x)
            
            print("contrast limit is {}".format(contrast_limits))

            metadata = {
                "name": imaging_rois["name"],
                "scale": scale,
                "contrast_limits": contrast_limits,
                "opacity": 0.5
            }
        
            layer_data.append((im, metadata, "image"))

    return layer_data

def get_contrast_limits(im):        
    if type(im) == dask.array.core.Array:
        # use first 100 for speed
        in_mem_im_sub = im[:100].compute()
        if type(in_mem_im_sub) != np.ndarray:
            in_mem_im_sub = cp.asnumpy(in_mem_im_sub)
        contrast_limits = (np.percentile(in_mem_im_sub, 10), np.percentile(in_mem_im_sub, 95))
        #contrast_limits = (da.percentile(flat_im, 10).compute(), da.percentile(flat_im, 95).compute())
    else:
        contrast_limits = (np.percentile(im, 10), np.percentile(im, 95))

    return contrast_limits


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

def get_scanimage_resolution(main_metadata, roi_metadata, obj_res = 69.4311, xy_correction = None, roi = None):
    
    # THIS NEEDS A LOTS OF WORK FOR COMPATIBILITY WITH MROI IMAGING
    fr = float(main_metadata['SI.hRoiManager.scanVolumeRate '].split(' ')[-1])
    #'SI.hRoiManager.scanVolumeRate ': ' 4.55824', #  for volume rate
    #xmin, ymin, xmax, ymax, tl, tr, bl, br
    imaging_rois = roi_metadata["RoiGroups"]["imagingRoiGroup"]["rois"]
    if roi == None:
        size_degrees = np.array(imaging_rois["scanfields"]["sizeXY"])
        size_px = np.array(imaging_rois["scanfields"]["pixelResolutionXY"])
        
        try:
            z_px_size = float(main_metadata["SI.hStackManager.actualStackZStepSize "]) # this will fail for MROI
        except:
            z_px_size = 1
        
    else:
        # size / x_px or y_px
        # 
        size_degrees = np.array(imaging_rois[roi]["scanfields"][0]["sizeXY"])
        size_px = np.array(imaging_rois[roi]["scanfields"][0]["pixelResolutionXY"])
        zs = imaging_rois[roi]["zs"]
        z_px_size = np.round(np.mean(np.diff(zs)))
        print("Average step size for this roi is {}".format(z_px_size))
        
    degree_px = size_degrees/size_px
    micron_px = degree_px * obj_res
        
    x_px_size = micron_px[0]
    y_px_size = micron_px[1]
    

        

    return np.array([1/fr, z_px_size, y_px_size, x_px_size]) # sampling interval, z interval, y interval, x interval

def get_Z(main_metadata):
    Z = int(main_metadata["SI.hStackManager.actualNumSlices "].replace(" [", "").replace("]", "").split(";")[0])
    #T = int(main_metadata["SI.hStackManager.actualNumVolumes "].replace(" [", "").replace("]", "").split(";")[0])
    return Z

def get_objective_resolution(main_metadata):
    # Useful for correcting images with wrong calibration
    obj_res = float(float(main_metadata["SI.objectiveResolution "].replace(" [", "").replace("]", "").split(";")[0]))
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
            



def lazy_scanimage_read(fname, nframes=1, *, arraytype="numpy"): # maybe give option of using gpu - maybe a check box
    """
    This is not very performant
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
    
def read_cnm_data(cnms):
    print("Reading cnm data")
    # get length of cnms
    layer_data = []
    
   

    T = cnms[0].estimates.C.shape[-1]
    Z = len(cnms)
    C = 1
    try:
        Y = cnms[0].dims[0]
        X = cnms[0].dims[1]
        dims = cnms[0].dims
    except:
        # online cnmf doesnt store dims in self
        Y = cnms[0].estimates.dims[0]
        X = cnms[0].estimates.dims[1]
        dims = cnms[0].estimates.dims
    
    # get num blocks frmo blocksize
    blocksize = 100
    nblocks = int(np.ceil(T/blocksize))

    blocks = [(n*blocksize, (n+1) * blocksize) for n in range(nblocks)]
    
    # get denoised dask array in TZCYX format
    im = da.concatenate([da.stack([da.from_delayed(reconstruct_cnm(cnm, start, end), shape = (blocksize, 1, dims[0], dims[1]), dtype = np.float64) for cnm in cnms], axis=1) for start, end in blocks], axis=0)

    # add extra shapes layer here for good components and bad components - make dask to extend in all dimensions lazily
    shape_layers = add_contour_layers(cnms)
    
    scale = (1, 1, 1, 1, 1)
            
        
    if type(im) == dask.array.core.Array:
        # use first 100 for speed
        in_mem_im_sub = im[:100].compute()
        if type(in_mem_im_sub) != np.ndarray:
            in_mem_im_sub = cp.asnumpy(in_mem_im_sub)
        contrast_limits = (np.percentile(in_mem_im_sub, 10), np.percentile(in_mem_im_sub, 95))
           
    else:
        contrast_limits = (np.percentile(im, 10), np.percentile(im, 95))

    print("contrast limit is {}".format(contrast_limits))

    metadata = {
        "name": "image",
        "scale": scale,
        "contrast_limits": contrast_limits,
        "opacity": 0.5
    }
    
    layer_data.append((im, metadata, "image"))
    layer_data.extend(shape_layers)

    return layer_data

def read_mmap_data(mmaps):
    layer_data = []
    # get shape from mmap file - format is Yr, dims, T
    T = mmaps[0][2]
    Z = len(mmaps)
    C = 1
    Y = mmaps[0][1][0]
    X = mmaps[0][1][1]

    dims = (Y, X)
        
    
    # get num blocks frmo blocksize
    blocksize = 100
    nblocks = int(np.ceil(T/blocksize))

    blocks = [(n*blocksize, (n+1) * blocksize) for n in range(nblocks)]
    
    # get denoised dask array in TZCYX format
    im = da.concatenate([da.stack([da.from_delayed(reconstruct_mmap(mmap[0], dims, T, start, end), shape = (blocksize, 1, dims[0], dims[1]), dtype = np.float64) for mmap in mmaps], axis=1) for start, end in blocks], axis=0)

    scale = (1, 1, 1, 1, 1)
            
        
    if type(im) == dask.array.core.Array:
        # use first 100 for speed
        in_mem_im_sub = im[:100].compute()
        if type(in_mem_im_sub) != np.ndarray:
            in_mem_im_sub = cp.asnumpy(in_mem_im_sub)
        contrast_limits = (np.percentile(in_mem_im_sub, 10), np.percentile(in_mem_im_sub, 95))
           
    else:
        contrast_limits = (np.percentile(im, 10), np.percentile(im, 95))

    print("contrast limit is {}".format(contrast_limits))

    metadata = {
        "name": "image",
        "scale": scale,
        "contrast_limits": contrast_limits,
        "opacity": 0.5
    }
        
    layer_data.append((im, metadata, "image"))
    return layer_data


def caiman_reader(paths, lazy= True, device = "cpu"):
    # might be worth adding dask array for shape layer for good components and bad components

    #try:
    #    print("Starting caiman cluster")
    #    if 'dview' in locals():
    #        cm.stop_server(dview=dview)
    #    c, dview, n_processes = cm.cluster.setup_cluster(
    #        backend='local', n_processes=None, single_thread=False)
    #except:
    #    pass

    print("Paths to caiman reader are {}".format(paths))
    layer_data = []
    if not isinstance(paths, list):
        paths = [paths]    
    # path[0]  endswith
    if paths[0].endswith(".hdf5"):
        print("Loading hdf5 caiman file")
        cnms = []
        for fname in natural_sorted(paths):
            print("Loading {}".format(fname))
            cnms.append(cm.source_extraction.cnmf.cnmf.load_CNMF(fname))

        cnm_layers = read_cnm_data(cnms)
        layer_data.extend(cnm_layers)
        

    elif paths[0].endswith(".mmap"):
        print("Loading mmap caiman file")
        mmaps = []
        for fname in natural_sorted(paths):
            print("Loading {}".format(fname))
            mmaps.append(cm.load_memmap(fname))
            
        mmap_layers = read_mmap_data(mmaps)
        layer_data.extend(mmap_layers)
        
    #print("Stopping caiman cluster")
    #if 'dview' in locals():
    #    cm.stop_server(dview=dview)   

    return layer_data

@dask.delayed
def reconstruct_cnm(cnm, start, end):
    # this will index error if end > than C.shape[1]
    try:
        dims = cnm.dims
    except:
        dims = cnm.estimates.dims
        
    return (cnm.estimates.A.dot(cnm.estimates.C[:, start:end]) + \
                        cnm.estimates.b.dot(cnm.estimates.f[:, start:end])).reshape((dims[0], dims[1]) + (-1,), order='F').transpose([2, 0, 1]).reshape(-1, 1,  dims[0], dims[1])
@dask.delayed
def reconstruct_mmap(mmap, dims, T, start, end):
    if end > T:
        diff = end - T
        end = T
        # pad
        pad = np.zeros((diff, dims[0], dims[1], end-start))
        # concatenate pad - tbc
    return np.reshape(mmap.T[start:end], [int(end-start)] + list(dims), order='F').reshape(-1, 1, dims[0], dims[1]) #TZCYX


    

        

def get_contour_data(cnm, cnm_idx):
    try:
        dims = cnm.dims
    except:
        dims = cnm.estimates.dims

    contours = get_contours(cnm.estimates.A, (dims[0], dims[1]), thr=0.9)
    # check previous usage
    good_shapes= []
    bad_shapes = []
    for ncon, contour in enumerate(contours):
        con = np.flip(contour["coordinates"], axis=1) # to get to (point, (y, x))
        z_levels = np.expand_dims(np.repeat(cnm_idx, con.shape[0]), 0).T
        #t_levels = np.zeros((con.shape[0], 1))
        #t_levels[:] = is there a way to have contours show every time
        c_levels = np.zeros((con.shape[0], 1))
        shape = np.concatenate([z_levels, c_levels, con], axis=1)
        #shape = np.round_(shape, 2)
        nan_filter = np.isnan(shape).any(axis=1)
        shape = shape[~nan_filter]

        #remove duplicates
        _, unique_index = np.unique(shape, axis=0, return_index=True)
        sorted_index = np.sort(unique_index)
        shape = shape[sorted_index]
        
        if ncon in cnm.estimates.idx_components:
            good_shapes.append(shape)
        elif ncon in cnm.estimates.idx_components_bad:
            bad_shapes.append(shape)
        
    #viewer.add_shapes(shapes,shape_type = "polygon", edge_width=0.4,
    #                              edge_color='white', face_color='transparent', name = "ROIs", opacity = 0.3)
    
    
    
    return good_shapes, bad_shapes

def add_contour_layers(cnms):
    layer_data = []
    good_shapes_all_z = []
    bad_shapes_all_z = []
    
    for cnm_idx, cnm in enumerate(cnms):
        good_shapes, bad_shapes = get_contour_data(cnm, cnm_idx)
        good_shapes_all_z.extend(good_shapes)
        bad_shapes_all_z.extend(bad_shapes)
        
    good_metadata = {
        "name": "good components",
        "shape_type" : "polygon",
        "edge_width" : 0.4,
        "edge_color" : 'white',
        "face_color" : 'transparent',
        "opacity": 0.3
    }
    
    bad_metadata = good_metadata.copy()
    bad_metadata["edge_color"] = "red"
    bad_metadata["name"] = "bad components"

    layer_data.append((good_shapes_all_z, good_metadata, "shapes"))    
    layer_data.append((bad_shapes_all_z, bad_metadata, "shapes"))
    
    return layer_data
# add scan image z, x, y position
# create reader for memmap files
# add delayed reader of contours

# create video reader
#@dask.delayed
def read_vid_frame(fname, start, end):
    vid = VideoReaderNP(fname)
    vid = vid[start:end]
    duration = int(end - start)
    if len(vid.shape) == 3: # check if just 1 frame -add pad of 99
        # pad
        difference = duration - 1
        pad = np.zeros((difference, *vid.shape))
        vid = np.concatenate([vid.reshape((1, *vid.shape)), pad])
        
    if vid.shape[0] != duration: # check if less than 100 frames and add pad
        difference = duration - vid.shape[0]
        pad = np.zeros((difference, *vid.shape[1:]))
        vid = np.concatenate([vid, pad])
    # return in TZCYX shape
    T, Y, X, C = vid.shape
    # TYXC -> TZCYX
    vid = vid.reshape((T, 1, Y, X, C))
    # TZYXC -> TZCXY
    vid = np.swapaxes(vid, 2, 4)
    # TZCXY -> TZCYX
    vid = np.swapaxes(vid, 3, 4)
    
    # probably convert to grayscale to save memory

    return vid

def read_video(fname, nframes):
    vid = VideoReaderNP(fname)
    # TYXC
    T,Y,X,C = vid.shape
    dtype = vid.dtype
    
    blocksize = nframes
    nblocks = int(np.ceil(len(vid)/blocksize))
    blocks = [(n*blocksize, (n+1) * blocksize) for n in range(nblocks)] # add trial here

    delayed_vid_reader = dask.delayed(read_vid_frame)
    
    vid_da = da.concatenate([da.from_delayed(delayed_vid_reader(fname, start, end), shape = (blocksize, 1, C, Y, X), dtype =
            dtype) for start, end in blocks])

    return vid_da

def read_stim_zarr(fname):
    return da.from_zarr(fname)


# create a reader for ephys.blk file
def read_ephys(fname, start, end):
    pass



