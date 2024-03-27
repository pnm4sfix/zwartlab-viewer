"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

from calendar import c
from re import S
from typing import TYPE_CHECKING
from unittest import result


from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from magicgui.widgets import ComboBox, Container, PushButton, SpinBox, FileEdit, FloatSpinBox, Label, TextEdit, CheckBox, FloatSlider, Select, QuantityEdit, LineEdit
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
from yaml import safe_dump_all
from ._reader import caiman_reader
from tifffile import natural_sorted
import cebra
from cebra import CEBRA
from napari.utils import colormaps
import tensorly as tl
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, non_negative_parafac_hals
from tensorly.decomposition._cp import initialize_cp
from tensorly.cp_tensor import CPTensor
import time
from copy import deepcopy
from tensorly.metrics.regression import RMSE
import cv2
import dask
from typing import List
from magicgui import magic_factory


if cp.cuda.Device():
    
    from cudipy.align.imaffine import AffineMap        
else:
        
    from dipy.align.imaffine import AffineMap

if TYPE_CHECKING:
    import napari

class DecompositionWidget(Container):
    def __init__(self, napari_viewer):
        # create separate viewer for this
        
        super().__init__()
        # takes dff_tidy dataframe and either performs decomposition
        # decomposition can be either tensor decomposition or cebra
        self.viewer = napari_viewer
        self.cebra_label_wid = Label(value = "CEBRA")
        
        # load neural data and behavioural data
        self.neural_data_file_wid = FileEdit(value='./Select neural data .npz', tooltip = "Select neural data .npz", mode ="r")
        self.auxillary_vars_file_wid = FileEdit(value='./Select auxillary .h5', tooltip = "Select auxillary .h5", mode ="r")
        self.auxillary_file = None
        # multiselect auxillary columns
        self.auxillary_columns_wid = Select(choices = [], tooltip = "Select auxillary columns", label = "Auxillary Columns")

        self.model_architecture_wid = ComboBox(choices = ["offset10-model"], tooltip = "Select model architecture", label = "Model Architecture")
        self.batch_size_wid = SpinBox(value = 512, tooltip = "Batch size for training", label = "Batch Size", max = 2048)
        self.learning_rate_wid = LineEdit(value = "3e-4", tooltip = "Learning rate for training", label = "Learning Rate")#, step = 0.0001, min = 0.000001, max = 1.0)
        self.temperature_wid = FloatSpinBox(value = 1, tooltip = "Temperature for training", label = "Temperature")
        self.output_dimension_wid = SpinBox(value = 3, tooltip = "Output dimension for training", label = "Output Dimension")
        self.max_iterations_wid = SpinBox(value = 5000, tooltip = "Maximum number of iterations for training", label = "Maximum Iterations", max = 100000)
        self.distance_wid = ComboBox(choices = ["cosine"], tooltip = "Distance metric for training", label = "Distance Metric")
        self.conditional_wid = ComboBox(choices = ["time_delta", "time"], tooltip = "Conditional for training", label = "Conditional")
        self.device_wid = ComboBox(choices = ["cuda_if_available"], tooltip = "Device for training", label = "Device")
        self.verbose_wid = CheckBox(value = True, tooltip = "Verbose for training", label = "Verbose")
        self.time_offsets_wid = SpinBox(value = 10, tooltip = "Time offsets for training", label = "Time Offsets")
        self.hybrid_wid = CheckBox(value = True, tooltip = "Hybrid for training", label = "Hybrid")
        self.colormap_column_wid = ComboBox(choices =[], tooltip = "Select column to map color to", label = "Colormap column")
        self.colormap_wid = ComboBox(choices = list(colormaps.AVAILABLE_COLORMAPS), tooltip = "Select colormap", label = "Colormap")
        self.do_cebra_wid = PushButton(tooltip = "Do CEBRA", label = "Do CEBRA")
        self.map_points_wid = PushButton(tooltip = "Map points", label = "Map points")
        
        self.neural_data_file_wid.changed.connect(self.load_neural_data)
        self.auxillary_vars_file_wid.changed.connect(self.load_auxillary_vars)
        
        self.do_cebra_wid.changed.connect(self.do_cebra)
        self.map_points_wid.changed.connect(self.map_points)
        
        # extend self with list of all wids
        self.extend([self.cebra_label_wid, self.neural_data_file_wid, self.auxillary_vars_file_wid, self.auxillary_columns_wid,
                     self.model_architecture_wid, self.batch_size_wid, self.learning_rate_wid, self.temperature_wid, 
                     self.output_dimension_wid, self.max_iterations_wid, self.distance_wid, self.conditional_wid, self.device_wid,
                    self.verbose_wid, self.time_offsets_wid, self.hybrid_wid,
                   self.colormap_column_wid, self.colormap_wid, self.do_cebra_wid, self.map_points_wid])


        # tensor decompososition
        self.tca_label_wid = Label(value = "Tensor Decomposition")
        self.ntk_file_wid = FileEdit(value='./Select ntk data .npz', tooltip = "Select ntk data .npz", mode ="r")
        self.rank_wid = SpinBox(value = 3, tooltip = "Rank for tensor decomposition", label = "Rank")
        self.backend_wid = ComboBox(choices = ["numpy", "cupy", "pytorch"], tooltip = "Backend for tensor decomposition", label = "Backend")
        self.sort_by_wid = ComboBox(choices = ["x", "y", "z"], tooltip = "Sort heatmap by", label = "Sort heatmap", value =  "z")
        self.do_tca_wid = PushButton(tooltip = "Do tensor decomposition", label = "Do tensor decomposition")
        self.use_existing_tca_wid = CheckBox(value = True, tooltip = "Use existing tensor decomposition", label = "Use existing tensor decomposition")
        self.plot_style_wid = ComboBox(choices = ["heatmap", "image"], tooltip = "Plot style", label = "Plot style", value = "image")
        self.do_tca_wid.changed.connect(self.do_tca)
        self.current_trial = 0
        self.current_t = 0
        # extend self with list of tensor decomposition widgets
        self.extend([self.tca_label_wid, self.ntk_file_wid, self.rank_wid, self.backend_wid, self.use_existing_tca_wid, self.do_tca_wid])

    def load_neural_data(self, event):
        print(event)
        
        self.neural_file = event
        self.neural_data = cebra.load_data(file=self.neural_file, key="neural")
        
    def load_auxillary_vars(self, event):
        
        
        print(event)
       
        self.auxillary_file = event
        # get columns of h5 file
        self.all_auxillary_cols = self.get_auxillary_cols()
        self.auxillary_columns_wid.choices = self.colormap_column_wid.choices = self.all_auxillary_cols
        # populate auxillary columns multiselect
        
    def get_auxillary_cols(self):
        print("Getting auxillary cols")
        if self.auxillary_file is None:
            cols = []
            
        else:
            cols = pd.read_hdf(self.auxillary_file, key="auxiliary_variables", start = 0, stop= 1).columns.tolist()
            
        return cols
        
    def get_cebra_params(self):
        
        # cebra params
        # assign wid values to new self objects of same name
        self.model_architecture = self.model_architecture_wid.value
        self.batch_size = self.batch_size_wid.value
        self.learning_rate = float(self.learning_rate_wid.value)
        print(self.learning_rate)
        self.temperature = self.temperature_wid.value
        self.output_dimension = self.output_dimension_wid.value
        self.max_iterations = self.max_iterations_wid.value
        self.distance = self.distance_wid.value
        self.conditional = self.conditional_wid.value
        self.device = self.device_wid.value
        self.verbose = self.verbose_wid.value
        self.time_offsets = self.time_offsets_wid.value
        self.hybrid = self.hybrid_wid.value
        self.colormap_column = self.colormap_column_wid.value
        self.cmap = self.colormap_wid.value
        
        # load auxillary data from h5 file using cols selected in multiselect
        self.select_auxillary_cols = self.auxillary_columns_wid.value
        print(self.select_auxillary_cols)
        self.auxillary_vars = cebra.load_data(file=self.auxillary_file, key="auxiliary_variables", columns = self.select_auxillary_cols)
        
        

    def init_model(self):
        print("initiating cebra model")
        
        self.cebra_model = CEBRA(model_architecture=self.model_architecture,
                            batch_size=self.batch_size,
                            learning_rate=self.learning_rate,
                            temperature=self.temperature,
                            output_dimension=self.output_dimension,
                            max_iterations=self.max_iterations,
                            distance=self.distance,
                            conditional=self.conditional,
                            device=self.device,
                            verbose=self.verbose,
                            time_offsets=self.time_offsets,
                            hybrid = self.hybrid)
        
    def do_cebra(self):
        self.get_cebra_params()
    
        self.init_model()
        if self.conditional == "time_delta":
            self.cebra_model.fit(self.neural_data, self.auxillary_vars)
        elif self.conditional == "time":
            self.cebra_model.fit(self.neural_data)
        
        # assuming this N, D shape in which case viewer can add points straight away
        self.embedding = self.cebra_model.transform(self.neural_data)
        print(self.embedding.max(axis=0))
        print(self.embedding.shape)
        print(self.colormap_column)
        self.map_points(None)
        
    def map_points(self, event):    
        # map a color map to data and use face_color argument of add points
        self.colormap_var = cebra.load_data(file=self.auxillary_file, key="auxiliary_variables", columns = [self.colormap_column])
        point_properties = {
            'feature': self.colormap_var.flatten() # this could be a data column 
        }
        
        
        self.viewer.add_points(self.embedding, name = "".join(self.select_auxillary_cols), size = 5e-2,
                               face_color = "feature", face_colormap = self.cmap, properties = point_properties )
        # add time dimension- TZYX points layer
        t = np.arange(self.embedding.shape[0]).reshape(self.embedding.shape[0], 1)
        time_embed = np.concatenate([t, self.embedding], axis=1)
        self.viewer.add_points(time_embed, name = "timecourse".join(self.select_auxillary_cols), size = 6e-2, face_color = "white") # make white and slighly large so can track through time
        #self.viewer.add_points(self.embedding, name = "".join(self.select_auxillary_cols), size = 1e-1)
    #@staticmethod    
    #@magic_factory(x={'widget_type': 'Select', "choices": self.get_auxillary_cols})
    #def multi_choice_vars(x: List):
       # print(f"you have selected {x}")

    def get_tca_params(self):
        self.rank = self.rank_wid.value
        self.backend = self.backend_wid.value
        self.ntk_file = self.ntk_file_wid.value
        self.order = self.sort_by_wid.value
        self.use_existing_tca = self.use_existing_tca_wid.value
        self.cmap = self.colormap_wid.value
        self.plot_style = self.plot_style_wid.value

    def do_tca(self):
        self.get_tca_params()
        
        tl.set_backend(self.backend)
        
        # create ntk/load
        with np.load(self.ntk_file, allow_pickle= True) as data:
            self.contours = data["contours"]
            self.centers = data["centers"]
            self.ntk = data["ntk"]
            
            
            self.regressors = data["regressors"]
            self.regressor_names = data["var_names"]
            self.decomp_results = data["results"].item()
            self.dims = data["dims"]
            
        if self.use_existing_tca == False:
            N, T, K = self.ntk.shape
            nonneg_ntk = (self.ntk - self.ntk.min(axis = (1, 2)).reshape(N, 1, 1))
            if self.backend == "numpy":
                ntk_tensor = tl.tensor(nonneg_ntk, dtype = tl.float32)
            elif (self.backend == "pytorch") | (self.backend == "cupy"):
                ntk_tensor = tl.tensor(nonneg_ntk, dtype = tl.float32, device = "cuda")
        
            # this is quicker on cpu than gpu
            recon, (weights, factors) = self.tensorly_cp(ntk_tensor, self.rank)
            
        elif self.use_existing_tca == True:
            print("Loading existing tca")
            nonneg_ntk = self.ntk
            weights, factors = self.decomp_results[self.rank][0]
            recon = self.recon_all_tc(self.decomp_results[self.rank][0])
            # create viewer1d
            self.add_regressor_widget()
            
        
        # add image layer to viewer for the original ntk and the reconstructed ntk
        # reshape to knt
        
        #comment out for now
        #self.add_ntk(nonneg_ntk, "original", "bop orange")
        #self.add_ntk(recon, "recon", "magenta")
        self.plot_tc_image(nonneg_ntk)
        self.plot_tc_image(recon)

        for component in range(self.rank):
            tc_cp_tensor, tc_tensor_recon = self.recon_single_tc(weights, factors, component)
        #    #self.add_ntk(tc_tensor_recon, "TC {}".format(component), "inferno")
            
            if self.plot_style == "heatmap":
                self.add_ntk(tc_tensor_recon, "TC {}".format(component), "inferno")

            elif self.plot_style == "image":
                self.plot_tc_image(tc_tensor_recon, "TC {}".format(component), "inferno")
        
        #for component in range(self.rank):
        #    self.plot_tc(weights, factors, component, self.contours)

    def preprocess_contours(self, contours):
        print("preprocessing contours")
        tc_contours = []
        z_level = []
        for ncon, contour in enumerate(contours):
            con = np.flip(contour, axis=1) # to get to (point, (cell_id, z, y, x))
            shape = con[:, 2:] # remove cell_id and z
            z = con[0, 1]
            #shape = np.round_(shape, 2)
            nan_filter = np.isnan(shape).any(axis=1)
            shape = shape[~nan_filter]

            #remove duplicates
            _, unique_index = np.unique(shape, axis=0, return_index=True)
            sorted_index = np.sort(unique_index)
            shape = shape[sorted_index]
            tc_contours.append(shape)
            z_level.append(z)
        
        return tc_contours, z_level
            
    def plot_tc(self, weights, factors, component, contours):
        # plot map
        # construct shapes layer from factors and contous  
        shape_properties = {"feature": factors[0][:, component], # neuron factor
                            "opacity" : 0.3,
                            "edge_color" : "transparent"
                            }
        tc_contours, z_level = self.preprocess_contours(contours)
            
        self.viewer.add_shapes(tc_contours, name = "TC {}".format(component),
                               face_color = "feature", face_colormap = self.cmap, properties = shape_properties,
                               shape_type = "polygon")
        
    # ideally we want to plot image like original cnm image
    # and maybe plot viewer1d regressors evolving through time
            
        
          

        ## TO DO - save com, regressors
        # plot tensor component maps
        # plot regressors
        # sort by center of mass  -add center of mass to saved file as well as all regressors      
        # add threshold to plot top neuron contributing to a certain factor
        


    def tensorly_cp(self, ntk, rank):
        
        weights_init, factors_init = initialize_cp(ntk, non_negative=True, init='random', rank=rank)
    
        cp_init = CPTensor((weights_init, factors_init))
        tic = time.time()
        tensor_hals, errors_hals = non_negative_parafac_hals(ntk, rank=rank, init=deepcopy(cp_init), return_errors=True)
        cp_reconstruction_hals = tl.cp_to_tensor(tensor_hals)
        time_hals = time.time()-tic
        rmse_error =RMSE(ntk, cp_reconstruction_hals)
        
        print("Rank-{} models: min obj {}, max obj {}  rmse error is {};  time to fit, {}".format(rank, min(errors_hals), max(errors_hals),  rmse_error, time_hals))
       
        return cp_reconstruction_hals, tensor_hals
    
    def add_ntk(self, ntk, name, colormap):
        #reshape 

        ntk = np.swapaxes(np.swapaxes(ntk, 1, 2), 0, 1)
        
        # sort by 
        self.viewer.add_image(ntk, name = name, colormap = colormap, contrast_limits = [0, 1])
        
    def recon_single_tc(self, weights, factors, component):
    
        
        tc_factors = []
        tc_weights = []
        
        tc_weights.append(weights[component])
    
        for dim in range(len(factors)):
            factor = factors[dim]
            tc_factors.append(factor[:, component:component+1])
        
        tc_cp_tensor = CPTensor((tc_weights, tc_factors))
        tc_tensor_recon = tl.cp_to_tensor(tc_cp_tensor)
        return tc_cp_tensor, tc_tensor_recon
        
    def recon_all_tc(self, cp_tensor):
        return tl.cp_to_tensor(cp_tensor)

    def plot_tc_image(self, ntk, name, colormap):
        # heatmap here is ntk
        # contour data is list of Y, X contours
        # shape is n pixels x N neurons
        # add z
        # add as shape TZCYX
        print("Tensordot of contours with ntk to produce images")
        N, T, K = ntk.shape
        Y, X = self.dims
        npixels = X*Y
        Z = 5  # replace with code to get z levels 
        con_im = np.zeros((npixels, N)) #y, x, N
        contour_data, z_level = self.preprocess_contours(self.contours)
      
        #points is list of yx points
        print("creating contour images")
        for npoint, point in enumerate(contour_data):
            
            empty_point_im = np.zeros((X,Y), dtype = np.int32) # xy image
            point = np.flip(point, axis=1) # to get x y
            filled = cv2.fillPoly(empty_point_im, [point.astype(np.int32)], color=1).T # fillpoly takes xy image
            con_im[:, npoint] = filled.flatten()
       

        # too big for my gpu
        #da_im = da.stack(da_im, axis=1)   
        da_im = self.read_ntk_as_image(con_im, ntk, z_level)
        print(da_im)  
        
        # release cupy cache
        #cp._default_memory_pool.free_all_blocks()

        self.viewer.add_image(da_im, name = name, colormap = colormap)

    @dask.delayed
    def reconstruct_ntk_as_image(self, con, ntk, dims, start, end, trial):
        # this will index error if end > than C.shape[1]
        result = np.tensordot(con, ntk[:, start:end, trial:trial+1], axes = ([1], [0]))
        result = np.reshape(result, (*dims, -1, 1))
        result = np.swapaxes(result, 0, 2) # T, X, Y, K
        result = np.swapaxes(result, 1, 3) # T, K, Y, X
        return result
    
    def read_ntk_as_image(self, con_im, ntk, z_level):
        N, T, K = ntk.shape
        Y, X = self.dims
        Z = len(np.unique(z_level))
        dims = self.dims
        print("dims are {}".format(dims))
        
        # get num blocks frmo blocksize
        blocksize = T#int(T/2)#1
        nblocks = int(np.ceil(T/blocksize))

        blocks = [(n*blocksize, (n+1) * blocksize) for n in range(nblocks)] # add trial here
        # logic is to reconstruct for every trial and time blockk for each separate z and then to stack in z dimension
        con_im = dask.delayed(con_im)
        ntk = dask.delayed(ntk)        

        time_stack = []
        # get denoised dask array in TZCYX format
        # loop through block time slices and trials and return da array of shape t1:t2, k, Y, X 
        for start, end in blocks:
            z_stack = []
            for z in np.sort(np.unique(z_level)):
                trial_stack = []
                for k in range(K):
                    #print(k)
                    trial_stack.append(da.from_delayed(self.reconstruct_ntk_as_image(con_im[:, np.where(np.array(z_level) ==z)[0]], 
                                                                                   ntk[np.where(np.array(z_level) ==z)[0]], dims,  
                                                                                   start, end, k), dtype = "float64", shape = ( blocksize, 1, self.dims[0], self.dims[1]))) #tkyx
                z_stack.append(da.concatenate(trial_stack, axis=1))
            time_stack.append(da.stack(z_stack, axis=1))

        im = da.swapaxes(da.concatenate(time_stack, axis=0), 3, 4) # have to swap last 2 dims for some reason
        return im
    
    def add_regressor_widget(self):
        
        self.viewer1d = napari_plot.ViewerModel1D()
        widget = QtViewer(self.viewer1d)
        self.viewer.window.add_dock_widget(widget, area="bottom", name="Regressor Widget")
        self.viewer1d.axis.x_label = "Time"
        self.viewer1d.axis.y_label = "Regressor"
        self.viewer1d.reset_view()
        
        self.viewer.dims.events.current_step.connect(self.plot_rtk)
        
        self.viewer1d_time_marker = self.viewer1d.add_line(np.c_[
                                                    [self.current_t, self.current_t],
                                                    [-1, 1]], name = "Frame", color = "gray")
        
    def plot_rtk(self, event):
        current_step = event.value
        old_trial = self.current_trial
        self.current_trial = current_step[2]
        if old_trial!=self.current_trial:
            # clear canvas
            #self.viewer1d.clear_canvas() # probably better to update existing layers than clear canvas completely
            current_layer_names = [layer.name for layer in self.viewer1d.layers]
            trial = self.current_trial
            for regressor in range(self.regressors.shape[0]):
                subset = self.regressors[regressor, :, trial]
                t = np.arange(self.regressors.shape[1])
                regressor_name = self.regressor_names[regressor]
                if regressor_name in current_layer_names:
                    layer = self.viewer1d.layers[current_layer_names.index(regressor_name)]
                    layer.data = np.c_[t, subset]
                else:
                    self.viewer1d.add_line(np.c_[t, subset], color = "magenta", name =self.regressor_names[regressor], visible = False)
                    self.viewer1d.set_y_view(-1, 1.)
        
        old_t = self.current_t
        self.current_t = current_step[0]
        if old_t != self.current_t:
            self.viewer1d_time_marker.data = np.c_[
                                                    [self.current_t, self.current_t],
                                                    [-1, 1]]
            
        
    