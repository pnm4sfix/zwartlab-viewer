"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from logging import raiseExceptions
import stat
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
from ._reader import caiman_reader, scanimage_reader
from tifffile import natural_sorted
from ._utils import *
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
        self.viewer = napari_viewer
        self.fnames_wid = FileEdit(value='./Select files to process', tooltip = "Select files to process", mode ="rm")
        self.crop_label = Label(value = "Center Crop")
        self.image_width_wid = SpinBox(value = 450, tooltip = "Select image width for center crop", label = "Image Width")
        self.image_height_wid = SpinBox(value = 380, tooltip = "Select image height for center crop", label = "Image Height")
        self.rotate_label = Label(value = "Rotate")
        self.rotate_wid = SpinBox(value = 180, tooltip = "Select rotation angle", step = 90.0, max = 270.0, label = "Rotation")
        self.align_wid = ComboBox(value = "Bonsai", choices = ["Bonsai", "Wavesurfer"], label = "Format to align to")
        self.fr_wid = FloatSpinBox(value = 3.07, tooltip = "Select frame rate", label = "Frame Rate")
        self.run_preprocess_wid = PushButton(text = "Run Preprocess", tooltip = "Run Preprocess")
        
        self.run_preprocess_wid.changed.connect(self.preprocess)
        # add widgets to self with extend
        self.extend([self.fnames_wid, self.crop_label, self.image_width_wid, self.image_height_wid, self.rotate_label, self.rotate_wid, self.align_wid, self.run_preprocess_wid])
        
        # self.fr_wid,
    def get_params(self):
        self.fnames = list(self.fnames_wid.value)
        # sort by time made to get acquisition order
        self.fnames.sort(key=os.path.getmtime)     
        print(self.fnames)
        self.fnames = [str(fname) for fname in self.fnames]
        print(self.fnames)
        self.image_width = self.image_width_wid.value
        self.image_height = self.image_height_wid.value
        self.rotate_value = self.rotate_wid.value
        self.align_type = self.align_wid.value
        #self.fr = self.fr_wid.value
        
    
    def align(self):
        
        
        # map async for all self.fnames
        with ipp.Cluster(n=len(self.fnames)) as rc:
                    
                        
                # get a view on the cluster
                view = rc.load_balanced_view()
                #args = [[v] * len(fnames) for k, v in params_dict.items()]
                #params_dicts = [params_dict] * len(fnames)  
                # ok pushing to cluster not working - try multiply all arguments by lenght of fnames
                # must be some issue with doing it within a class?
                # submit the tasks
                    
                # wait interactively for results
                

                if self.align_type == "Bonsai":
                    asyncresult = view.map_async(align_to_bonsai, self.fnames)
                    
                print("Waiting for results")
                asyncresult.wait_interactive()
                # retrieve actual results
                self.aligned_data = asyncresult.get()
        

    def parallel_load_images(self):
        # map async for all self.fnames
        with ipp.Cluster(n=len(self.fnames)) as rc:
                    
                        
                # get a view on the cluster
                view = rc.load_balanced_view()
               
                asyncresult = view.map_async(scanimage_reader, self.fnames)
                    
                print("Waiting for results")
                asyncresult.wait_interactive()
                # retrieve actual results
                self.raw_images = asyncresult.get() # these are in layer data format and can be added straight to the viewer
                
        print("Adding raw layers")
        try:
            for layer in self.raw_images:
                self.viewer.add_layer(layer[0])
        except:
            "print failed initial attempt to add layer - adding as image layer"
            for layer in self.raw_images:
                self.viewer.add_image(data = layer[0][0], **layer[0][1])
            
        self.fr = 1/layer[0][1]["scale"][0] # get frame rate from first image layer

    def preprocess(self):
        self.get_params()
        
        # align first to get volume df for each file - this is done in parallel for speed and returns aligned data into memory
        self.align()
        
        # read fnames using reader - maybe add argument to sort or not?
        # this should convert each file into a dask array
        # operate on each file separately to align to bonsai or wavesurfer

        # create function to load image stack for each file
        self.parallel_load_images() # returns raw images in layer data format and adds to viewer

        # then when everything has been aligned it can be concatenated and then preprocessed simultaneously
        #layer_data = caiman_reader(self.fnames)
        
        # align each volume to associated behavioural and log files
        # crop in time from first to last volume in volume_df to match volume df duration
        # if this is all done in dask then no need to parellise
        
        last_vols = [aligned_data[0].nVol.iloc[-1] for aligned_data in self.aligned_data]
        for nimage, image in enumerate(self.raw_images): #looping through raw images 
            # select raw_images da image and change in place
            last_vol = last_vols[nimage]
            self.raw_images[nimage][0] = list(self.raw_images[nimage][0]) # have to listify it first to do assignment
            self.raw_images[nimage][0][0] = self.raw_images[nimage][0][0][:last_vol]
            self.raw_images[nimage][0] = tuple(self.raw_images[nimage][0]) # convert back to tuple for napari viewer
        

        if (self.image_width > 0) & (self.image_height > 0):
            # perform center crop with
            for nimage, image in enumerate(self.raw_images): #looping through raw images 
            # select raw_images da image and change in place
                im_center = np.array(image[0][0].shape)[3:]/2 # should be y x
                crop_x0, crop_x1 = int(im_center[1] - (self.image_width/2)), int(im_center[1] +(self.image_width/2))
                crop_y0, crop_y1 = int(im_center[0] - (self.image_height/2)), int(im_center[0] + (self.image_height/2)) 
                
                self.raw_images[nimage][0] = list(self.raw_images[nimage][0]) # have to listify it first to do assignment
                self.raw_images[nimage][0][0] = self.raw_images[nimage][0][0][:, :, :, crop_y0:crop_y1, crop_x0:crop_x1] # TZCYX
                self.raw_images[nimage][0] = tuple(self.raw_images[nimage][0]) # convert back to tuple for napari viewer
            
        if self.rotate_value > 0:
            # perform rotation
            nrotations = int(self.rotate_value / 90)
            for nimage, image in enumerate(self.raw_images): #looping through raw images 

                if self.rotate_value % 90 == 0:
                    # rotate 90 using np.rot90 - avoids interpolation risk
                    self.raw_images[nimage][0] = list(self.raw_images[nimage][0]) # have to listify it first to do assignment
                    self.raw_images[nimage][0][0] = self.rotate(self.raw_images[nimage][0][0], nrotations)
                    self.raw_images[nimage][0] = tuple(self.raw_images[nimage][0]) # convert back to tuple for napari viewer
                    
        # amend layer data names and add to viewer
        try:
            print("adding processed layers")
            for layer in self.raw_images:
                self.viewer.add_layer(layer[0])
        except:
            "print failed initial attempt to add layer - adding as image layer"
            for layer in self.raw_images:
                metadata = layer[0][1]
                metadata["name"] = metadata["name"] + "preprocessed"
                self.viewer.add_image(data = layer[0][0], **metadata)
        
        # save layer data to memmap format for caiman 
        # create file dict for each z level containing dask arrays - bring these into memory and save using memmap which will save ind, then concatenate
        #file_dict[z] = [im1, im2, im3] 
        print("Starting caiman cluster") 
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)
        memmap_names = []

        # get image dims from first image
        T,Z,C,Y,X = self.raw_images[0][0][0].shape
        # get root folder
        root_folder = os.path.dirname(os.path.dirname(self.fnames[0]))
        denoised_folder = os.path.join(root_folder, "denoised")    
        
        # check folder exists
        if os.path.exists(denoised_folder) == False:
            os.mkdir(denoised_folder)

        for z in range(self.raw_images[0][0][0].shape[1]):
            z_ims = [im[0][0][:, z] for im in self.raw_images]
            in_mem_zims = [z_im.compute().reshape((z_im.shape[0], Y, X)) for z_im in z_ims]
            memmap_name = cm.save_memmap(in_mem_zims, base_name=os.path.join(denoised_folder, 'memmap_z{}'.format(z)), order='C', dview=dview)
            memmap_names.append(memmap_name)
            # this will save individual memmaps and then a large one - to save disk space remove ind memmaps

        # remove ind memmaps and keep concantenated one
        #folder = os.getcwd() # change for whatever working from
        # get all memmap names and remove any that don't involve 
        print(memmap_names)
        
        [os.remove(os.path.join(denoised_folder, file)) for file in os.listdir(denoised_folder) if ('.mmap' in file) and (os.path.join(denoised_folder, file) not in memmap_names) ]   

        print("Stopping caiman cluster") 
        if 'dview' in locals():
            cm.stop_server(dview=dview)
        
        print("concatenating ephys data")
        # concat matrix array into ephys
        
        ephys = np.concatenate([aligned_data[1] for aligned_data in self.aligned_data])
        print("correcting ephys length")
        # correct ephys range by ignoreing first imaging interval
        imaging_interval = int((1/self.fr) * 6e3) # can prbs get fr from imaging data
        ephys = ephys[imaging_interval:]
        
        
        print("concatenating volume_dfs")
        # map trials to volume - probably do this after 
        concat_volume_df = pd.concat([aligned_data[0] for aligned_data in self.aligned_data])
        print("reseting nVol")
        # reset vol number
        concat_volume_df.nVol = np.arange(concat_volume_df.shape[0])
        print("mapping trial")
        
        #concat_volume_df = map_trials(concat_volume_df)
        # concatenate stim
        if len(self.aligned_data[0]) == 3:
            full_stim = da.concatenate([aligned_data[2] for aligned_data in self.aligned_data]) 
            # rechunk
            full_stim = full_stim.rechunk(full_stim.chunksize)
        
        # save all to disk
        print("Saving stim video as zarr")
        full_stim.to_zarr(os.path.join(denoised_folder, "stim.zarr"))
        print("Saving ephys")
        np.save(os.path.join(denoised_folder, "ephys.npy"), ephys)
    
    def rotate(self, im, nrotations):
        # write if statement if im is numpy array or dask array
        # this could go in utils folder as general useful function
        if isinstance(im, np.ndarray):
            for nrotation in range(nrotations):
                im = np.rot90(im, axes = (3, 4))
        elif isinstance(im, da.Array):
            for nrotation in range(nrotations):
                im = da.rot90(im, axes=(3, 4))
        return im
        
      
    
    

    