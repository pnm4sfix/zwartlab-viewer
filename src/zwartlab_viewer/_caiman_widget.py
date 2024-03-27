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
from ._reader import read_cnm_data, napari_get_reader
from caiman.source_extraction.cnmf import params as params
from tifffile import natural_sorted
if cp.cuda.Device():
    
    from cudipy.align.imaffine import AffineMap        
else:
        
    from dipy.align.imaffine import AffineMap

if TYPE_CHECKING:
    import napari


class CaimanWidget(Container):
    
    def __init__(self, napari_viewer):
        
        super().__init__()

        if cp.cuda.Device():
            self.gpu = True
        else:
            self.gpu = False

        self.viewer = napari_viewer
        self.mode_wid = ComboBox(label='mode', choices = ["online", "batch"], tooltip = "mode")
        self.fr_wid = FloatSlider(label = "fr", tooltip = "framerate", value = 3.07)
        self.fnames_wid = FileEdit(value='./Select files to process', tooltip = "Select files to process", mode ="rm")
        self.decay_time_wid = FloatSlider(label = "decay_time", tooltip = "decay_time", value = 0.4)
        self.strides_wid = SpinBox(value = 48, label = "strides", tooltip = "strides")
        self.overlaps_wid = SpinBox(value = 24, label = "overlaps", tooltip = "overlaps")
        self.max_shifts_wid = SpinBox(value = 100, label = "max_shifts", tooltip = "max_shifts")
        self.max_deviation_rigid_wid = SpinBox(value = 3, label = "max_deviation_rigid", tooltip = "max_deviation_rigid")
        self.pw_rigid_wid = CheckBox(value = False, label = "pw_rigid", tooltip = "pw_rigid")
        self.num_frames_split_wid = SpinBox(value = 400, label = "num_frames_split", tooltip = "num_frames_split")
        
        self.p_wid = SpinBox(value = 1, label = "p", tooltip = "p")
        self.gnb_wid = SpinBox(value = 2, label = "gnb", tooltip = "gnb")
        self.merge_thr_wid = FloatSlider(label = "merge_thr", tooltip = "merge_thr", value = 0.95)
        self.rf_wid = SpinBox(value = 50, label = "rf", tooltip = "rf")
        self.stride_cnmf_wid = SpinBox(value = 20, label = "stride_cnmf", tooltip = "stride_cnmf")
        #self.K_wid = SpinBox(value = 20, label = "K", tooltip = "K")
        #self.gSig_wid = SpinBox(value = 3, label = "gSig", tooltip = "gSig")
        self.method_init_wid = ComboBox(label='method_init', choices = ["greedy_roi", "sparse_nmf"], tooltip = "method_init")
        self.ssub_wid = SpinBox(value = 1, label = "ssub", tooltip = "ssub")
        self.tsub_wid = SpinBox(value = 1, label = "tsub", tooltip = "tsub")
        self.min_SNR_wid = FloatSlider(label = "min_SNR", tooltip = "min_SNR", value = 2)
        self.SNR_lowest_wid = FloatSlider(label = "SNR_lowest", tooltip = "SNR_lowest", value = 1.2)
        self.rval_thr_wid = FloatSlider(label = "rval_thr", tooltip = "rval_thr", value = 0.85)
        self.rval_lowest_wid = FloatSlider(label = "rval_lowest", tooltip = "rval_lowest", value = 0.2)
        self.cnn_thr_wid = FloatSlider(label = "cnn_thr", tooltip = "cnn_thr", value = 0.95)
        self.cnn_lowest_wid = FloatSlider(label = "cnn_lowest", tooltip = "cnn_lowest", value = 0.2)
        self.ds_factor_wid = SpinBox(value = 1, label = "ds_factor", tooltip = "ds_factor")
        self.gSig_wid = SpinBox(value = 3, label = "gSig", tooltip = "gSig")
        self.mot_corr_wid = CheckBox(value = False, label = "mot_corr", tooltip = "mot_corr")
        self.max_shifts_online_wid = SpinBox(value = 100, label = "max_shifts", tooltip = "max_shifts_online")
        self.sniper_mode_wid = CheckBox(value = False, label = "sniper_mode", tooltip = "sniper_mode")
        self.init_batch_wid = SpinBox(value = 200, label = "init_batch", tooltip = "init_batch")
        #self.allow_overlap_wid = CheckBox(value = False, label = "allow_overlap", tooltip = "allow_overlap")
        self.K_wid = SpinBox(value = 20, label = "K", tooltip = "K")
        self.epochs_wid = SpinBox(value = 1, label = "epochs", tooltip = "epochs")
        self.show_movie_wid = CheckBox(value = False, label = "show_movie", tooltip = "show_movie")
        
        # add a widget to load saved hdf5 calling reader functions
        self.load_hdf5_wid = FileEdit(value='./Select hdf5 file', tooltip = "Select hdf5 file", mode ="rm")
        
        # add a button to run the analysis
        self.run_button = PushButton(text='Run', tooltip = "Run analysis")
        self.run_button.clicked.connect(self.run_analysis)
        
        # add all wid to self by extend
        
        self.extend([self.mode_wid, self.fr_wid, self.fnames_wid, self.decay_time_wid, self.strides_wid, self.overlaps_wid, self.max_shifts_wid, self.max_deviation_rigid_wid,
                     self.pw_rigid_wid, self.num_frames_split_wid, self.p_wid, self.gnb_wid, self.merge_thr_wid, self.rf_wid, self.stride_cnmf_wid, 
                     self.method_init_wid, self.ssub_wid, self.tsub_wid, self.min_SNR_wid, self.SNR_lowest_wid, self.rval_thr_wid,# self.rval_lowest_wid , self.cnn_thr_wid, 
                     self.cnn_lowest_wid, self.ds_factor_wid, self.gSig_wid, self.mot_corr_wid, self.max_shifts_online_wid, self.sniper_mode_wid, self.init_batch_wid,
                    self.K_wid, self.epochs_wid, self.show_movie_wid, self.load_hdf5_wid, self.run_button]) #self.allow_overlap_wid, 

    def get_params(self):
        print("Getting params")
        # gets params from all the widgets
        self.mode = self.mode_wid.value
        self.fr = self.fr_wid.value
        self.fnames = self.fnames_wid.value
        # sort fnames by time
        if not isinstance(self.fnames, list):
            self.fnames = list(self.fnames)
       
        self.fnames.sort(key=os.path.getmtime)


        self.decay_time = self.decay_time_wid.value
        self.strides = (self.strides_wid.value, self.strides_wid.value)
        self.overlaps = (self.overlaps_wid.value, self.overlaps_wid.value)
        self.max_shifts = (self.max_shifts_wid.value, self.max_shifts_wid.value)
        self.max_deviation_rigid = self.max_deviation_rigid_wid.value
        self.pw_rigid = self.pw_rigid_wid.value
        self.num_frames_split = self.num_frames_split_wid.value
        self.p = self.p_wid.value
        self.gnb = self.gnb_wid.value
        self.merge_thr = self.merge_thr_wid.value
        self.rf = self.rf_wid.value
        self.stride_cnmf = self.stride_cnmf_wid.value
        self.method_init = self.method_init_wid.value
        self.ssub = self.ssub_wid.value
        self.tsub = self.tsub_wid.value
        self.min_SNR = self.min_SNR_wid.value
        self.rval_thr = self.rval_thr_wid.value
        self.cnn_thr = self.cnn_thr_wid.value
        self.cnn_lowest = self.cnn_lowest_wid.value 
        self.rval_lowest = self.rval_lowest_wid.value
        self.SNR_lowest = self.SNR_lowest_wid.value
        self.ds_factor = self.ds_factor_wid.value
        self.gSig = (self.gSig_wid.value, self.gSig_wid.value)
        self.gSig = tuple(np.ceil(np.array(self.gSig) / self.ds_factor).astype('int')) 
        self.mot_corr = self.mot_corr_wid.value
        self.max_shifts_online = self.max_shifts_online_wid.value
        self.sniper_mode = self.sniper_mode_wid.value
        self.init_batch = self.init_batch_wid.value
        self.K = self.K_wid.value
        self.epochs = self.epochs_wid.value
        self.show_movie = self.show_movie_wid.value
        

        
    def start_cluster(self):
        try:
            print("Starting caiman cluster")
            if 'dview' in locals():
                cm.stop_server(dview=self.dview)
            c, self.dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=None, single_thread=False)
        except:
            pass
        
    def save_layer_as_memmap(self, layer, z):
        cm.save_memmap(layer.data, base_name='memmap_z{}'.format(z), order='C', dview=self.dview)
        
    def run_analysis(self):
        self.get_params()    
    
        if self.mode == "online":
            if len(self.fnames) > 1:
                self.parallel_caiman_online(self.fnames)
            elif len(self.fnames) == 1:
                self.run_single_test(self.fnames)
            
        elif self.mode == "batch":
            #self.caiman_batch
            pass
        
    @staticmethod
    def run_cnmf_online(fname, params_dict):
        # issue with running with sniper mode - tensorflow not playing ball with ipp
        import numpy as np
        
        from caiman.source_extraction.cnmf import cnmf as cnmf
        from caiman.source_extraction.cnmf import online_cnmf as o_cnmf
        from caiman.source_extraction.cnmf import params as params
        

        
        params_dict["fnames"] = [fname]
        opts = params.CNMFParams(params_dict=params_dict)
        # %% fit online
        
        cnm = o_cnmf.OnACID(params=opts)#, dview = dview)
        
        print("Fitting with OnACID")
        cnm.fit_online()
        print("Fitting finished")
        #UNCOMMENT AFTER TESTING
        #outfile = fname[:-5] + "_online.hdf5"
        #cnm.save(outfile)
        return cnm
        
    def run_single_test(self, fnames):
        # defunct
        print("Caiman unit test")
        # This assumes files are saved to disk as tif or mmap
        fnames = [str(fname) for fname in fnames]
        #define parameter dict here

        params_dict = {'fnames': [],
                       'fr': self.fr,
                       'decay_time': self.decay_time,
                       'gSig': self.gSig,
                       'p': self.p,
                       'min_SNR': self.min_SNR,
                       'rval_thr': self.rval_thr,
                       'ds_factor': self.ds_factor,
                       'nb': self.gnb,
                       'motion_correct': bool(self.mot_corr),
                       'init_batch': self.init_batch,
                       'init_method': 'bare',
                       'normalize': True,
                       'sniper_mode': bool(self.sniper_mode),
                       'K': self.K,
                       'epochs': self.epochs,
                       'max_shifts_online': self.max_shifts_online,
                       'pw_rigid': self.pw_rigid,
                       'dist_shape_update': True,
                       'min_num_trial': 10,
                       'show_movie': self.show_movie,
                       'save_online_movie': False}
        
        print(params_dict)
        
            
        cnm = self.run_cnmf_online(fnames[0], params_dict)
        self.cnms = [cnm]
        self.evaluate_new_cnms(fnames, params_dict)
        self.add_cnms_to_viewer()
        

    def parallel_caiman_online(self, fnames):
        print("Caiman online parallel")
        # So I can run this in parallel on the VM but not laptop
        # I can run a single unit test on laptop        

        # This assumes files are saved to disk as tif or mmap
        fnames = [str(fname) for fname in fnames]
        #define parameter dict here

        params_dict = {'fnames': [fnames],
                       'fr': self.fr,
                       'decay_time': self.decay_time,
                       'gSig': self.gSig,
                       'p': self.p,
                       'min_SNR': self.min_SNR,
                       'rval_thr': self.rval_thr,
                       'ds_factor': self.ds_factor,
                       'nb': self.gnb,
                       'motion_correct': bool(self.mot_corr),
                       'init_batch': self.init_batch,
                       'init_method': 'bare',
                       'normalize': True,
                       'sniper_mode': bool(self.sniper_mode),
                       'K': self.K,
                       'epochs': self.epochs,
                       'max_shifts_online': self.max_shifts_online,
                       'pw_rigid': self.pw_rigid,
                       'dist_shape_update': True,
                       'min_num_trial': 10,
                       'show_movie': self.show_movie,
                       'save_online_movie': False}
        
        print(params_dict)
        

        with ipp.Cluster(n=len(fnames)) as rc:
            #push param dict
            rc[:].push(params_dict)          
            # get a view on the cluster
            view = rc.load_balanced_view()
            #args = [[v] * len(fnames) for k, v in params_dict.items()]
            params_dicts = [params_dict] * len(fnames)  
            # ok pushing to cluster not working - try multiply all arguments by lenght of fnames
            # must be some issue with doing it within a class?
            # submit the tasks
            asyncresult = view.map_async(self.run_cnmf_online, fnames, params_dicts)
            # wait interactively for results
            print("Waiting for results")
            asyncresult.wait_interactive()
            # retrieve actual results
            self.cnms = asyncresult.get()
        print("Online analysis finished")
        print(self.cnms)
        
        # evaluate components in cnms to use cnn as sniper mode doesnt work
        # load memmap
        
        # reshape
        self.evaluate_new_cnms(fnames, params_dict)
        self.add_cnms_to_viewer()
        
        # save cnms
        
        #opts = cnms[0].estimates.CNMFParams(params_dict=params_dict)
    def evaluate_new_cnms(self, fnames, params_dict):
        for cnm_idx, cnm in enumerate(self.cnms):
            fname = fnames[cnm_idx]
            params_dict["fnames"] = [fname] #this assumes fname is mmap but sometime this is raw tiff
            opts = params.CNMFParams(params_dict=params_dict)
            Yr, dims, T = cm.load_memmap(fname) 
            images = Yr.T.reshape((T,) + dims, order='F')
            # loop through cnms.estimates and do cnm.estimates.evaluate_components(images, opts)
            # should use DNN
            self.cnms[cnm_idx].estimates.evaluate_components(images, opts)
            
    def add_cnms_to_viewer(self):
        # result should contain cnm objects
        layer_data = read_cnm_data(self.cnms)

        # loop through and add layers to viewer
        print("Adding analysis layers")
        for layer in layer_data:
            self.viewer.add_layer(layer)
            
        # add original fnames to viewer - issue is it may sort by natural order not by order submitted
        #napari_get_reader(self.names)


        
        # modify caiman reader to take cnm objects directly too


#def run_online(CaimanWidget, fnames):
    #return CaimanWidget.run_cnmf_online(fnames)    
        

    
        

        """fr = 3.07                       # imaging rate in frames per second
        decay_time = 0.4                    # length of a typical transient in seconds

        # motion correction parameters
        strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
        overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
        max_shifts = (100,100)          # maximum allowed rigid shifts (in pixels)
        max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
        pw_rigid = False             # flag for performing non-rigid motion correction
        num_frames_split = 400


        # parameters for source extraction and deconvolution
        p = 1                       # order of the autoregressive system
        gnb = 2                     # number of global background components
        merge_thr = 0.95           # merging threshold, max correlation allowed
        rf = 50                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
        stride_cnmf = 20             # amount of overlap between the patches in pixels
        K = 20               # number of components per patch
        gSig = [3, 3]               # expected half size of neurons in pixels
        method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        ssub = 1                    # spatial subsampling during initialization
        tsub = 1                    # temporal subsampling during intialization

        # parameters for component evaluation
        #min_SNR = 3               # signal to noise ratio for accepting a component
        #rval_thr = 0.85             # space correlation threshold for accepting a component
        #cnn_thr = 0.95              # threshold for CNN based classifier
        #cnn_lowest = 0.2 # neurons with cnn probability lower than this value are rejected
        min_SNR = 2               # signal to noise ratio for accepting a component
        rval_thr = 0.85             # space correlation threshold for accepting a component
        cnn_thr = 0.95              # threshold for CNN based classifier
        cnn_lowest = 1e-4 # neurons with cnn probability lower than this value are rejected
        rval_lowest = -10
        SNR_lowest = 0.1



        ds_factor = 1  # spatial downsampling factor (increases speed but may lose some fine structure)

        gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int')) # recompute gSig if downsampling is involved
        mot_corr = True  # flag for online motion correction
        pw_rigid = False  # flag for pw-rigid motion correction (slower but potentially more accurate)
        max_shifts_online = np.ceil(10.).astype('int')  # maximum allowed shift during motion correction
        sniper_mode = True  # use a CNN to detect new neurons (o/w space correlation)
        #rval_thr = 0.9  # soace correlation threshold for candidate components
        # set up some additional supporting parameters needed for the algorithm
        # (these are default values but can change depending on dataset properties)
        init_batch = 300  # number of frames for initialization (presumably from the first file)
        K = 20  # initial number of components
        epochs = 2  # number of passes over the data
        show_movie = False # show the movie as the data gets processed
    
        # might have to push this dict to the cluster
        params_dict = {'fnames': [fname],
                       'fr': self.fr,
                       'decay_time': self.decay_time,
                       'gSig': self.gSig,
                       'p': self.p,
                       'min_SNR': self.min_SNR,
                       'rval_thr': self.rval_thr,
                       'ds_factor': self.ds_factor,
                       'nb': self.gnb,
                       'motion_correct': self.mot_corr,
                       'init_batch': self.init_batch,
                       'init_method': 'bare',
                       'normalize': True,
                       'sniper_mode': self.sniper_mode,
                       'K': self.K,
                       'epochs': self.epochs,
                       'max_shifts_online': self.max_shifts_online,
                       'pw_rigid': self.pw_rigid,
                       'dist_shape_update': True,
                       'min_num_trial': 10,
                       'show_movie': self.show_movie,
                       'save_online_movie': False,
                        'movie_name_online': fname[:-5]}"""        
    