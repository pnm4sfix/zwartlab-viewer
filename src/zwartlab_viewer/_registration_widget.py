"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from base64 import a85encode
from typing import TYPE_CHECKING


from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from magicgui.widgets import ComboBox, Container, PushButton, SpinBox, FileEdit, FloatSpinBox, Label, TextEdit, CheckBox
from magicgui.widgets import create_widget, Widget
import napari_plot
from napari_plot._qt.qt_viewer import QtViewer
import os
import pandas as pd
from dask_image.imread import imread
import dask.dataframe as dd
import dask.array as da
import dask
import numpy as np
import cupy as cp
from napari.layers import Image
from scipy.ndimage import rotate
if cp.cuda.Device():
    free, total =cp.cuda.Device().mem_info
    from cupyx.scipy.ndimage import affine_transform
    if free > 4e9:
        from cudipy.align.imaffine import AffineMap  
        
    else:
        from dipy.align.imaffine import AffineMap
        
else:
    from scipy.ndimage import affine_transform
    from dipy.align.imaffine import AffineMap

if TYPE_CHECKING:
    import napari
    
### To Do 
### Fix rotation
### Add Dipy functions 

class RegistrationWidget(Container):
    
    # This should be the registration widget
    # Show napari images in micron world coordinates
    # Create zbrain tiff reader that loads zbrain images and stores microscopy metadata
    # issue with high resolution image requiring rotation - implement basic cupy.ndimage or scipy image processing functions - or add rotation functions to affine matrix -issue would be that rotation would not be centered
    # Get accurate scanimage micron coordinates
    # Create a scanimage tiff reader plugin for napari
    # Create a zbrain tiff reader plugin for napari that loads metadata in appropriate way

    


    def __init__(self, napari_viewer):
        
        super().__init__()

        if cp.cuda.Device():
            # check large enough gpu
            free, total =cp.cuda.Device().mem_info

            if free > 4e9:

                self.gpu = True
            else:
                self.gpu = False
        else:
            self.gpu = False

        self.viewer = napari_viewer
        

        # Registration widgets
        registration_label = Label(name = "Registration", label = "Registration")
        # Add check box - grey out if self io_affine etc is None - if checkbox is ticked all images and contours will be transformed
        self.registration_checkbox = CheckBox(name = "Live registration enabled")
        #self.registration_checkbox.changed.connect(self.affine_map_all)

        # Add dropdown to select appropriate channel for registration
        self.channel_to_register_dropdown = ComboBox(label='Channel to Register', choices = [], tooltip =  "Select channel to register")
        self.channel_to_register_to_dropdown = ComboBox(label='Channel to Register to', choices = [], tooltip =  "Select channel to register")

        # populate the dropdown with the channels in the viewer
        self.update_channel_menu(None)

        #self.channel_to_register_dropdown.changed.connect(self.update_registration_channels)
        #self.channel_to_register_to_dropdown.changed.connect(self.update_registration_channels)


        self.translate_x = FloatSpinBox(label = "translate x", tooltip = "change x translation", min = -1000, max = 1000, value = 0)
        self.translate_y = FloatSpinBox(label = "translate y", tooltip = "change y translation", min = -1000, max = 1000, value = 0)
        self.translate_z = FloatSpinBox(label = "translate z", tooltip = "change z translation", min = -1000, max = 1000, value = 0)
        self.scale_xy = FloatSpinBox(label = "scale xy", tooltip = "change xy scale", min = -1000, max = 1000, value = 1)
        self.rotate_x = FloatSpinBox(label = "rotate x", tooltip = "change x rotation", min = -180, max = 180, value = 0)
        self.rotate_y = FloatSpinBox(label = "rotate y", tooltip = "change y rotation", min = -180, max = 180, value = 0)
        self.rotate_z = FloatSpinBox(label = "rotate z", tooltip = "change z rotation", min = -180, max = 180, value = 0)
        self.rotate90_button = PushButton(label = "Rotate 90")
        
        # add button for numpy flip along z axis
        self.flip_z_button = PushButton(label = "Flip Z")
            
        #self.affine_name = TextEdit(label = "affine filename", value = "affine.npy")
        self.create_registered = PushButton(label = "Affine Map")
        
        self.translate_x.changed.connect(self.update_affine)
        self.translate_y.changed.connect(self.update_affine)
        self.translate_z.changed.connect(self.update_affine)
        self.scale_xy.changed.connect(self.update_affine)
        self.rotate_x.changed.connect(self.update_affine)
        self.rotate_y.changed.connect(self.update_affine)
        self.rotate_z.changed.connect(self.update_affine)
        self.rotate90_button.changed.connect(self.rotate90)
        self.flip_z_button.changed.connect(self.flip_z)


        self.create_registered.clicked.connect(self.update_registration_channels)
        self.viewer.layers.events.changed.connect(self.update_channel_menu)
        self.viewer.layers.events.inserted.connect(self.update_channel_menu)

        # connect viewer layers events renamed to update channel menu
        #self.viewer.layers.events.renamed.connect(self.update_channel_menu)    

        
        
        self.extend([ registration_label, self.registration_checkbox, self.channel_to_register_dropdown, self.channel_to_register_to_dropdown,
                    self.translate_x, self.translate_y, self.translate_z, self.scale_xy, self.rotate90_button, self.flip_z_button, self.rotate_x, self.rotate_y, self.rotate_z, self.create_registered])

    def flip_z(self):
         self.viewer.layers[self.channel_to_register_dropdown.value].data = np.flip(self.viewer.layers[self.channel_to_register_dropdown.value].data,  axis=1)
                    
    def rotate90(self):
        # use np.rot to rotate the image in the registered channel
        self.viewer.layers[self.channel_to_register_dropdown.value].data = rotate(self.viewer.layers[self.channel_to_register_dropdown.value].data, 90, axes = (3, 2), reshape = False)

    
    def update_channel_menu(self, event):
        print("layers changed")
        # update the channel to register dropdown and channel to register to dropdown with the image layers in the viewer
        self.channel_to_register_dropdown.choices = [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        self.channel_to_register_to_dropdown.choices = [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]

        # connect all layers to name change
        [layer.events.name.connect(self.update_channel_menu) for layer in self.viewer.layers if isinstance(layer, Image)]

    def update_affine(self, event):
        self.x = self.translate_x.value
        self.y = self.translate_y.value
        self.z = self.translate_z.value
        self.scale = self.scale_xy.value
        

        #self.grid_copy = np.eye(4) # assume 3 dims ZYX

        
        #self.grid_copy[2, -1] += self.x # x
        #self.grid_copy[1, -1] += self.y # y
        #self.grid_copy[0, -1] += self.z # z
        #t, z, y, x, 1

        
        #self.grid_copy[2, 2] = self.scale
        #self.grid_copy[1, 1] = self.scale
        # set translation of affine matrix
        channel_to_apply_affine = self.viewer.layers[self.channel_to_register_dropdown.value]
        registered_affine = np.array(channel_to_apply_affine.affine)
        registered_affine[1, -1] = self.z
        registered_affine[2, -1] = self.y
        registered_affine[3, -1] = self.x 

        # set scale of affine matrix
        registered_affine[2, 2] = self.scale
        registered_affine[3, 3] = self.scale

        #set rotation of affine matrix


        # if rotation could add a center shift, rotate then unshift?
        angle = np.radians(self.rotate_z.value)
        

        if angle > 0:

            # create a translation matrix
            move_to_center = np.eye(registered_affine.shape[0])
            move_to_center[3, -1] -= channel_to_apply_affine.data.shape[3] / 2
            move_to_center[4, -1] -= channel_to_apply_affine.data.shape[4] / 2

            # create a rotation matrix
            rotate = np.eye(registered_affine.shape[0])
            rotate[4, 4] = np.cos(angle)
            rotate[3, 3] = np.cos(angle)
            rotate[3, 4] = np.sin(angle)
            rotate[4, 3] = -np.sin(angle)
            
            # np dot registered_affine_copy with a move back matrix
            move_back = np.eye(registered_affine.shape[0])
            move_back[3, -1] += channel_to_apply_affine.data.shape[3] / 2
            move_back[4, -1] += channel_to_apply_affine.data.shape[4] / 2
            

            r = np.dot(move_to_center, rotate)
            b = np.dot(r, move_back)
            registered_affine = np.dot(registered_affine, b)


        

        channel_to_apply_affine.affine = registered_affine

    def download_extract_zbrain(self, marker):

        
        import requests, zipfile, io

        if marker == "elavl3-H2BRFP":
            zip_file_url = r"https://zebrafishmodel.zib.de/fishexplorer/lm/download/datasets/Elavl3-H2BRFP.zip?AWSAccessKeyId=workerUser&Expires=1688478515&Signature=glSbO8vnwZKHb%2BUhYY9FddHvFiA%3D"
        r = requests.get(zip_file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("/downloads")
    
    def apply_tranform(self, ref_grid2world, im_grid2world, affine, ref, im):
        
        
        affine_map = AffineMap(affine,
                               ref.shape, ref_grid2world,
                               im.shape, im_grid2world)
        
        if self.gpu:
            resampled = cp.asnumpy(affine_map.transform(cp.asarray((im/256).astype("int8")))) # why change in bit depth
            
            
        else:
            resampled = affine_map.transform(im)
        
        return resampled

    def affine_map_moving2ref(self, event):
        print("affine mapping")
        if self.moving_grid2world is not None:
            # manually estimate translation


            
            
            self.grid_copy = np.eye(4)
            # get dimensions of each channel and only select the last 3 as dipy only takes dims 3
            if len(self.channel_to_register.data.shape) == 5:
                #maybe loop through each dim greater than 3 and do registration
                for c in range(self.channel_to_register.data.shape[0]):
                    for t in range(self.channel_to_register.data.shape[1]):
                        self.registered_channel.data[c, t, :, :, :] = self.apply_tranform(self.ref_grid2world, self.moving_grid2world, self.grid_copy,
                                                                            self.channel_to_register_to.data[c, t, :, :, :], self.channel_to_register.data[c, t, :, :, :])

            elif len(self.channel_to_register.data.shape) == 4:
                #maybe loop through each dim greater than 3 and do registration
                
                for t in range(self.channel_to_register.data.shape[0]):
                    self.registered_channel.data[t, :, :, :] = self.apply_tranform(self.ref_grid2world, self.moving_grid2world, self.grid_copy,
                                                                        self.channel_to_register_to.data[t, :, :, :], self.channel_to_register.data[t, :, :, :])

            elif len(self.channel_to_register.data.shape) == 3:
                
                self.registered_channel.data[ :, :, :] = self.apply_tranform(self.ref_grid2world, self.moving_grid2world, self.grid_copy,
                                                                            self.channel_to_register_to.data[:, :, :], self.channel_to_register.data[ :, :, :])

            
            #if len(self.channel_to_register.data.shape) > 2: # if not still empty array
            #if (self.moving_grid2world is not None) & (self.ref_grid2world is not None):
            #    self.registered_channel.data = self.apply_tranform(self.ref_grid2world, self.moving_grid2world, self.grid_copy,
            #                                        self.channel_to_register_to.data, self.channel_to_register.data)
                
            #else:
            #    self.template.data = self.apply_tranform(self.ref_grid2world, self.moving_grid2world, self.grid_copy, self.reference_brain.data, self.template_orig)

    def reset_registered_channels(self):
        # assign this function to a button to reset any affine mapping performed on channel
        pass

    def convert_scale_to_grid(self, scale: np.array) -> np.array:
        
        """Parameters:
            scale: np.array(z, y, x) - stored in image layer metadata

           Returns:
            affine: np.array(5, 5) where row order is z, y, x, 1
            """
        affine = np.eye(4)
        affine[0, 0] = scale[0]
        affine[1, 1] = scale[1]
        affine[2, 2] = scale[2]
        

        return affine


    
    def update_registration_channels(self, event):

        assert len(self.viewer.layers) >= 2

        self.channel_to_register = [layer for layer in self.viewer.layers if layer.name == self.channel_to_register_dropdown.value][0]
        self.channel_to_register_to = [layer for layer in self.viewer.layers if layer.name == self.channel_to_register_to_dropdown.value][0]
        

        # grid2world should be able to be created from layer metadata
        # assumes last 3 dims are z, y, x
        self.moving_grid2world = self.convert_scale_to_grid(self.channel_to_register.scale[-3:])
        self.ref_grid2world = self.convert_scale_to_grid(self.channel_to_register_to.scale[-3:])

        # affine transform grid2world for moving and ref using the affine associated with each channel
        self.moving_grid2world = np.matmul(self.moving_grid2world, self.channel_to_register.affine)
        self.ref_grid2world = np.matmul(self.ref_grid2world, self.channel_to_register_to.affine)
        
        print("Moving grid to world is {} and ref grid to world is {}".format(self.moving_grid2world,
                                                                             self.ref_grid2world))

        # check new channel name doesn't already exist
        new_channel_name = self.channel_to_register.name+"-registered"
        if self.channel_to_register.name+"-registered" not in [layer.name for
                                                          layer in self.viewer.layers]:
            # create new image layer to store registered channel 
            self.registered_channel = self.viewer.add_image(data = np.zeros(
                                            self.channel_to_register_to.data.shape, dtype = self.channel_to_register_to.dtype),
                                           name = self.channel_to_register.name+"-registered", scale = self.channel_to_register_to.scale,
                                           translate = self.channel_to_register_to.translate, opacity = 0.5)
        else:
            self.registered_channel = [layer for layer in self.viewer.layers 
                                       if layer.name == new_channel_name][0]
        self.registered_channel.affine = np.array(self.registered_channel.affine)
        self.affine_map_moving2ref(None)


    #def save_starting_affine(self):
    #    #save affine transform
    #    filename = self.affine_name.value
    #    np.save(os.path.join(self.denoised_dir, filename), self.grid_copy)

    """def affine_map_all(self, event):
        # check value
        print("checkbox {}".format(event))
        # check for io_affine.npy and affine_map.npy
        # apply both to IO_template, denoised and contours
        # apply affine_map to overview only

        if event:

            if (self.io_affine is not None) & (self.overview_affine is not None):
                print("registering all")
                io_to_overview = self.apply_transform(self.overview_grid_to_world, self.io_grid2world, self.io_affine, self.overview_im, self.denoised_layer.data)
                self.denoised_layer.data = self.apply_transform(self.reference_grid2world, self.overview_grid_to_world, self.overview_affine, self.reference_brain, io_to_overview)

                # map contours
                self.transform_contours(self.io_affine, self.io_grid2world, self.overview_grid_to_world)
                self.transform_contours(self.overview_affine, self.overview_grid_to_world, self.reference_grid2world)

            if (self.io_affine is not None):
                print("registering IO to overview")
        else:
            self.reset_affine()"""

    def reset_affine(self):
        # to reset 
       
        pass
           
    
    


    def transform_coordinates(self, coordinates, affine_transform, domain_transform, codomain_transform):
        # in shape z, y, x, 1
        new_coords = np.zeros(coordinates.shape)
        for coord_idx, coord in enumerate(coordinates):

            new_coord = np.dot(np.linalg.inv(codomain_transform), np.dot(affine_transform, np.dot(domain_transform, coord)))
            new_coords[coord_idx] = new_coord

        return new_coords

    def transform_contours(self, affine, domain_grid2world, codomain_grid2world): 
        
        contours = self.con_df_copy.loc[:, ["z", "y", "x"]].to_numpy()
        contour_coords = np.concatenate([contours, np.ones((contours.shape[0], 1))], axis= 1)
        new_coords = self.transform_coordinates(contour_coords, affine, domain_grid2world, codomain_grid2world)
        self.con_df_copy.loc[:, ["z", "y", "x"]] = new_coords[:, :3]

    @dask.delayed
    def dask_affine_transform(im, affine, arraytype = "numpy"):
        
        if arraytype == "numpy":
            # use scipy.ndimage affin transform
            return affine_transform(im, affine)
        
        elif arraytype ==  "cupy":
            return affine_transform(im, cp.asarray(affine))
               
        
        
        """LEGACY
        # Load fluorescence data - returns self.im
        self.lazy_load_images()

        # load overview, overview_grid_to_world
        overview_dir = os.path.join(os.path.dirname(self.denoised_dir), "overview")
        try:
            overview_file = [os.path.join(overview_dir, file) for file in os.listdir(overview_dir) if "overview" in file][0]
        
            self.overview_im  = da.rot90(da.rot90(imread(overview_file), axes = (1, 2)), axes = (1, 2))
            self.overview_im = da.reshape(self.overview_im, (self.overview_im.shape[0], 1, *self.overview_im.shape[1:]))
            overview_affine_path = os.path.join(overview_dir, "affine_map.npy")

            if os.path.exists(overview_affine_path):
                self.overview_affine = np.load(overview_affine_path)
            else:
                self.overview_affine = None


        except:
            print("No overview file")
            self.overview_im = None

        self.overview_grid_to_world = np.array([[ 1.,    0.,         0.,          0.],
                                         [0.,   0.01171,   0.,            0.],
                                         [0.,   0.,         0.01171,      0.],
                                         [0.,   0.,         0.,           1.]])

        # load io template, io_grid_to_world, io_to_overview 
        io_template_path = os.path.join(self.denoised_dir, "io_template.tif")
        io_grid_path = os.path.join(self.denoised_dir, "io_grid2world.npy")
        io_affine_path = os.path.join(self.denoised_dir, "io_affine.npy")

        if os.path.exists(io_template_path):
            self.io_template =  imread(io_template_path)
            self.io_template = self.io_template.reshape((self.io_template.shape[0], 1, *self.io_template.shape[1:]))
        else:
            self.io_template = None

        if os.path.exists(io_grid_path):
            self.io_grid2world = np.load(io_grid_path)
        else:
            self.io_grid2world = None

        if os.path.exists(io_affine_path):  
            self.io_affine = np.load(io_affine_path)
        else:
            self.io_affine = None


        # load zbrain reference
        self.reference_brain = da.rot90(
                                    da.rot90(
                                    da.flip(imread(r"Z:\Pierce\Elavl3-H2BRFP\Elavl3-H2BRFP_rotated.tif"), 
                                            axis=0), axes = (1, 2)), axes = (1, 2))
        
        self.reference_brain = da.reshape(self.reference_brain, (self.reference_brain.shape[0], 1, *self.reference_brain.shape[1:]))
        self.reference_grid2world = np.eye(4)

        # maybe need an affine map button to select whether images should be mapped to zbrain reference
        self.set_default_filters()
        self.update_menu_choices()
        self.add_layers()
        self.load_subset()

    def add_layers(self):

        
        if isinstance(self.im, (np.ndarray, da.core.Array)):
            print(self.im.shape)
            self.denoised_layer = self.viewer.add_image(np.zeros((self.im.shape[0], *self.im.shape[2:])), 
                                                        name = "Denoised Recording", opacity = 0.5, colormap = "inferno")

        if isinstance(self.rois, (np.ndarray, list)):
            self.roi_layer = self.viewer.add_shapes(self.rois,shape_type = "polygon", edge_width=0.4,
                                  edge_color='white', face_color='transparent', name = "ROIs", opacity = 0.3)
            self.roi_layer.events.highlight.connect(self.roi_selected)

        if isinstance(self.io_template, (np.ndarray, da.core.Array)):
            self.template_layer = self.viewer.add_image(self.io_template, name = "IO Template", opacity = 0.5, visible = False, colormap = "magenta")

        if isinstance(self.overview_im, (np.ndarray, da.core.Array)):
            self.overview_im_layer = self.viewer.add_image(self.overview_im, name = "Overview", opacity = 0.5, visible = False)

        if isinstance(self.reference_brain, (np.ndarray, da.core.Array)):
            self.reference_brain_layer = self.viewer.add_image(self.reference_brain, name = "H2B:Elavl3", opacity = 0.5, visible = False)

        # set registrataion options as layers
        
        self.channel_to_register_dropdown.choices = [layer.name for layer in self.viewer.layers]
        self.channel_to_register_to_dropdown.choices  = [layer.name for layer in self.viewer.layers]

        #print(self.template_layer, self.denoised_layer)
        print(self.viewer.layers)
        print(type(self.viewer.layers))

    def plot_all_contours(self): # slow
        
        #try:
        #if self.io_affine is not None:
        #    io_affine = self.io_affine

        #elif self.grid_copy is not None:
        #    io_affine = self.grid_copy

        #else:
        #    io_affine = np.eye(4)

        #self.transform_contours(io_affine, self.io_grid2world, self.ref_grid2world)
            
        #except:
        #    print("error registering contours")
        
        z_index = [] 
        shapes= []
        for cell in self.con_df_copy.cell_id.sort_values().unique():
            subset = self.con_df_copy[self.con_df_copy.cell_id == cell].dropna().copy()
            contours = subset[["z","cell_id", "y", "x"]]

            contours.cell_id = np.zeros(contours.shape[0])


            z_index.append(subset.iloc[0, 2])
            
            shapes.append(np.round_(contours.to_numpy(), 2))
            
        self.rois = shapes
        #print(np.array(self.rois).shape)
        self.roi_z = z_index
        #self.roi_layer.selected_data = set(range(self.select_roi.nshapes))
        #self.roi_layer.remove_selected()

        #self.roi_layer.add(shapes,shape_type = "polygon", edge_width=0.4,
        #                          edge_color='white', face_color='transparent')
        
    
    def roi_selected(self, event):
        if len(self.roi_layer.selected_data) > 0:
            self.cell_ids.value = list(self.roi_layer.selected_data)[0]

    def set_default_filters(self):
        self.cell_id = 0 
        self.cell_subset = dd.read_parquet(self.dff_path, filters = [("cell_id", "==", self.cell_id)]).compute()

        stims = self.volume_df.stim.dropna().unique().tolist()
        #[print(type(stim), stim) for stim in stims]
        #self.stim_menu.choices = stims
        print(self.stim_menu.choices)
        self.stim = stims[0]
        self.stim_subset = self.volume_df[self.volume_df.stim == self.stim]

        self.visual_angle = None
        

    def stim_changed(self, event):
        print(self.volume_df.trial.unique())
        if self.stim != event:
            self.stim = event
            self.stim_subset  = self.volume_df[self.volume_df.stim == self.stim]
            self.update_menu_choices()
            self.load_subset()
            print(event)

    def trial_changed(self, event):
        if self.trial != event:
            self.trial = event
            self.load_subset()
            print("new trial {}".format(event))
        
    def angle_changed(self, event):
        if self.angle != event:
            self.angle = event
            self.load_subset()
        
        print("new angle {}".format(event))
        
    def velocity_changed(self, event):
        print("velocity before {}".format(self.velocity))
        if self.velocity != event:
            self.velocity = event
            print("velocity after {}".format(self.velocity))
            self.load_subset()
            print("new velocity {}".format(event))
        
        
    def load_subset(self):
        # load subset data
        
        self.get_volume_subset()
        
        if self.volume_subset.shape[0] > 0:
            
            self.load_image()
            self.plot_cell(self.cell_id)
        

    def update_menu_choices(self):
        # should only be called if new stim
        trials = self.stim_subset.trial.sort_values().dropna().unique().tolist()
        angles = self.stim_subset.angle.sort_values().dropna().unique().tolist()
        velocities = self.stim_subset.velocity.sort_values().dropna().unique().tolist()
        
        self.angle = angles[0]
        self.velocity = velocities[1]
        self.trial = trials[0]
        
        # initiates change loop
        self.trial_menu.choices = trials
        self.angle_menu.choices = angles
        self.velocity_menu.choices = velocities

        if "visual_angle" in self.stim_subset.columns:
            visual_angles = self.stim_subset.visual_angle.dropna().unique().tolist()
            self.visual_angle = visual_angles[0]

            visual_velocities = self.stim_subset.visual_vel.dropna().unique().tolist()
            self.visual_velocity = visual_velocities[0]

            self.visual_angle_menu.choices = visual_angles
            self.visual_velocity_menu.choices = visual_velocities

        else:
            self.visual_angle = None
            self.visual_velocity = None

    def update_filters(self, df):
        print(self.stim, self.angle, self.velocity, self.trial)
        stim_filter = df.stim == self.stim
        angle_filter = df.angle == self.angle
        velocity_filter = df.velocity == self.velocity
        trial_filter = df.trial == self.trial

        if self.visual_angle != None:
            visual_angle_filter = df.angle == self.visual_angle
            visual_velocity_filter = df.velocity == self.visual_velocity
        else:
            visual_angle_filter = None
            visual_velocity_filter = None

        filter_dict = {}

        filter_dict["stim"] = stim_filter
        filter_dict["angle"] = angle_filter
        filter_dict["velocity"] = velocity_filter
        filter_dict["trial"] = trial_filter
        filter_dict["visual_angle"] = visual_angle_filter
        filter_dict["visual_velocity"] = visual_velocity_filter

        return filter_dict

    def get_volume_subset(self):
        filters = self.update_filters(self.volume_df)

        if filters["visual_angle"] != None:
            self.volume_subset = self.volume_df[(filters["stim"]) & (filters["angle"]) & (filters["velocity"])
                                                & (filters["trial"]) & (filters["visual_angle"]) & (filters["visual_velocity"])]
        else:
            self.volume_subset = self.volume_df[(filters["stim"]) & (filters["angle"]) & (filters["velocity"])
                                                & (filters["trial"])]
        
        if self.volume_subset.shape[0] > 0:
         # 5 seconds before
            vols_before = int(5 * self.fr)
            
            
            self.cond_start  = self.volume_subset.nVol.iloc[0] 
            self.cond_end = self.volume_subset.nVol.iloc[-1] 
            self.first_vol = self.cond_start - vols_before
            self.end_vol = self.cond_end + vols_before
            self.vols = np.arange(self.first_vol, self.end_vol)
            
        

    



    def lazy_load_images(self):
        
        print("loading images")
        # load all images as one dask array - dask.array.stack(data = [],  axis = 0)
        #files = os.listdir(self.denoised_dir)
        #ims = [os.path.join(self.denoised_dir,file) for file in files if 'denoised' in file]
        
        #da_ims = []

        #for im in ims:
        #    da_im = imread(im, nframes = 1)
        #    da_ims.append(da_im)
        self.im  = da.from_zarr(os.path.join(self.denoised_dir, "8bit.zarr"), chunks= (5, 1000, -1, -1))
        
        #self.im = da.stack(da_ims)
        
        print("loading finished")
        
   


    #### Registration functions   #############################################

    #### Exploration functions    #############################################
    def load_image(self):
        #"Load image"
        if self.volume_subset.shape[0] > 0:
            print("loading dask images")
            self.denoised_layer.data = self.im[:, self.first_vol:self.end_vol ].compute()
            self.denoised_layer.contrast_limits_range = (0, 255)
            self.denoised_layer.contrast_limits = [0, 25]
            #self.denoised_layer.reset_contrast_limits_range()
            #self.denoised_layer.reset_contrast_limits()
            
            print("images loaded")
            
        else:
            print("no volumes")
        pass
    def plot_cell(self, event):
        print(self.stim)
        if type(self.stim) == str:
            print("Changing cell")
            self.clear_plot()
            # define cell event
            # get dff info for cell
            # get contour of cell
            print(type(event))
            self.cell_id = int(event)
            print("New cell is {}".format(self.cell_id))

            
            self.cell_subset = dd.read_parquet(self.dff_path, filters = [("cell_id", "==", str(self.cell_id))]).compute()
            filters = self.update_filters(self.cell_subset)
            print(self.cell_subset.cell_id.unique())
            if filters["visual_angle"] != None:

                self.dff_subset  = self.cell_subset[(filters["stim"]) & (filters["angle"]) & 
                                    (filters["velocity"]) & (filters["visual_angle"]) &
                                    (filters["visual_velocity"])]
            else:
                self.dff_subset  = self.cell_subset[(filters["stim"]) & (filters["angle"]) & 
                                    (filters["velocity"])]

            # plot every trial but highlight current one with thicker line

            for trial in self.dff_subset.trial.unique():
                trial_subset = self.dff_subset[self.dff_subset.trial == trial]
                print(trial_subset.shape)
                trial_start = trial_subset.nVol.iloc[0]
                trial_end = trial_subset.nVol.iloc[-1]
                vols_before = int(5 * self.fr)
                first_vol = trial_start - vols_before
                last_vol = trial_end + vols_before
                vols = np.arange(first_vol, last_vol)
                vol_filter = self.cell_subset.nVol.isin(vols)

                plotting_subset = self.cell_subset[vol_filter]
                t = np.arange(0, plotting_subset.shape[0]/self.fr, 1/self.fr)
                

                if trial == self.trial:

                    self.viewer1d.add_line(np.c_[t, plotting_subset.smooth_dff.to_numpy()], color = "magenta")
                    regions = [
                            ([t[trial_start-first_vol], t[trial_end-first_vol]], "vertical"),
                        ]

                    layer = self.viewer1d.add_region(
                            regions,
                            color=["green"],
                            opacity = 0.4,
                            name = "Stim",
                        )

                else:
                    self.viewer1d.add_line(np.c_[t, plotting_subset.smooth_dff.to_numpy()], name="trial {}".format(trial), color="gray")



            self.viewer1d.reset_view()
            self.viewer1d.set_y_view(-0.05, 1.)
        
    def add_dff_widget(self):
        
        self.viewer1d = napari_plot.ViewerModel1D()
        widget = QtViewer(self.viewer1d)
        self.viewer.window.add_dock_widget(widget, area="bottom", name="DF/F Widget")
        self.viewer1d.axis.x_label = "Time"
        self.viewer1d.axis.y_label = "DeltaF/F"
        self.viewer1d.reset_view()
    
    def clear_plot(self):
        self.viewer1d.clear_canvas()

        """

