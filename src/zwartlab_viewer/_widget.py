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
from magicgui.widgets import ComboBox, Container, PushButton, SpinBox, FileEdit, FloatSpinBox, Label, TextEdit, CheckBox
from magicgui.widgets import create_widget, Widget
import napari_plot
from napari_plot._qt.qt_viewer import QtViewer
import os
import pandas as pd
from dask_image.imread import imread
import dask.dataframe as dd
import dask.array as da
import numpy as np

if TYPE_CHECKING:
    import napari


class ExampleQWidget(Container):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    # This should be the registration widget
    


    def __init__(self, napari_viewer):
        
        super().__init__()
        self.viewer = napari_viewer
        self.fr = 3.07
        # Fish data
        self.fish = FileEdit(value='./Select reference image', tooltip = "Select fish folder", mode ="d")
        
        # Exploration widgets
        self.stim_menu = ComboBox(label='Stim', choices = ["visual", "flow", "motion", "visualmotion", "visualflow"],
                                 tooltip = "Select stim")
        self.trial_menu = ComboBox(label='Trial', choices = [], tooltip = "Select trial")
        self.angle_menu = ComboBox(label='Angle', choices = [], tooltip = "Select angle")
        self.velocity_menu = ComboBox(label='Velocity', choices = [], tooltip = "Select velocity")
        self.visual_angle_menu = ComboBox(label='Visual Angle', choices = [], tooltip = "Select visual angle (combo exps only)")
        self.visual_velocity_menu = ComboBox(label='Visual Velocity', choices = [], tooltip =  "Select visual velocity (combo exps only)")
        self.fish.changed.connect(lambda x: self.load_fish_data(x))
        self.cell_ids = SpinBox(label = "cell_id", tooltip = "change cell")
        self.cell_ids.changed.connect(self.plot_cell)
        self.extend([self.fish, self.stim_menu, self.trial_menu, self.angle_menu, self.velocity_menu,
                             self.visual_angle_menu, self.visual_velocity_menu, self.cell_ids])

        self.stim_menu.changed.connect(self.stim_changed)
        self.trial_menu.changed.connect(self.trial_changed)
        self.angle_menu.changed.connect(self.angle_changed)
        self.velocity_menu.changed.connect(self.velocity_changed)

        

        # Registration widgets
        registration_label = Label(name = "Registration", label = "Registration")
        # Add check box - grey out if self io_affine etc is None - if checkbox is ticked all images and contours will be transformed
        self.registration_checkbox = CheckBox(name = "Live registration enabled")

        # Add dropdown to select appropriate channel for registration
        self.channel_to_register_dropdown = ComboBox(label='Channel to Register', choices = [], tooltip =  "Select channel to register")
        self.channel_to_register_to_dropdown = ComboBox(label='Channel to Register to', choices = [], tooltip =  "Select channel to register")



        self.translate_x = FloatSpinBox(label = "translate x", tooltip = "change x translation", min = -500, max = 500, value = 0)
        self.translate_y = FloatSpinBox(label = "translate y", tooltip = "change y translation", min = -500, max = 500, value = 0)
        self.translate_z = FloatSpinBox(label = "translate z", tooltip = "change z translation", min = -500, max = 500, value = 0)
        self.affine_name = TextEdit(label = "affine filename", value = "affine.npy")
        self.save_affine_button = PushButton(label = "save affine")
        
        self.translate_x.changed.connect(self.translate)
        self.translate_y.changed.connect(self.translate)
        self.translate_z.changed.connect(self.translate)
        self.save_affine_button.clicked.connect(self.save_starting_affine)
        
        
        self.extend([ registration_label, self.registration_checkbox, self.channel_to_register_dropdown, self.channel_to_register_to_dropdown,
                    self.translate_x, self.translate_y, self.translate_z, 
                    self.affine_name, self.save_affine_button])
        
        
        # Create napari1d instances
        self.add_dff_widget() # overlay ephys data on top
        
    #### Load all dataframes, images, transform matrices #############################################

    def load_fish_data(self, event):
        
        self.root_folder = event
        # define denoised folder - where all data is
        self.denoised_dir = os.path.join(self.root_folder, "denoised")

        # define overview folder - where wholebrain overview is
        self.overview_dir = os.path.join(self.root_folder, "overview")

        # Load volume df 
        vol_path = os.path.join(self.denoised_dir, "volume_df.h5")
        self.volume_df = pd.read_hdf(vol_path)

        # Load coms df 
        coms_path = os.path.join(self.denoised_dir, "all_neuron_centers.h5")
        self.coms_df = pd.read_hdf(coms_path)

        # Load con_df
        con_path = os.path.join(self.denoised_dir, "all_neuron_contours.h5")
        self.con_df = pd.read_hdf(con_path)
        self.plot_all_contours()

        # Load dff_df
        self.dff_path = os.path.join(self.denoised_dir, "df.parquet.gzip")
        
        # read this dff on fly using cell_id as filter
        # default cell id is 0
        

        # Load fluorescence data - returns self.im
        self.lazy_load_images()

        # load overview, overview_grid_to_world
        overview_dir = os.path.join(os.path.dirname(self.denoised_dir), "overview")
        try:
            overview_file = [os.path.join(overview_dir, file) for file in os.listdir(overview_dir) if "overview" in file][0]
        
            self.overview_im  = da.rot90(da.rot90(imread(overview_file), axes = (1, 2)), axes = (1, 2))
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
        # maybe need an affine map button to select whether images should be mapped to zbrain reference
        self.set_default_filters()
        self.update_menu_choices()
        self.add_layers()
        self.load_subset()

    def add_layers(self):

        
        if isinstance(self.im, (np.ndarray, da.core.Array)):
            print(self.im.shape)
            self.denoised_layer = self.viewer.add_image(np.zeros((self.im.shape[0], *self.im.shape[2:])), 
                                                        name = "Denoised Recording", opacity = 0.5)

        if isinstance(self.rois, (np.ndarray, list)):
            self.roi_layer = self.viewer.add_shapes(self.rois,shape_type = "polygon", edge_width=0.4,
                                  edge_color='white', face_color='transparent', name = "ROIs")
            self.roi_layer.events.highlight.connect(self.roi_selected)

        if isinstance(self.io_template, (np.ndarray, da.core.Array)):
            self.template_layer = self.viewer.add_image(self.io_template, name = "IO Template", opacity = 0.5, visible = False)

        if isinstance(self.overview_im, (np.ndarray, da.core.Array)):
            self.overview_im_layer = self.viewer.add_image(self.overview_im, name = "Overview", opacity = 0.5, visible = False)

        if isinstance(self.reference_brain, (np.ndarray, da.core.Array)):
            self.reference_brain_layer = self.viewer.add_image(self.reference_brain, name = "H2B:Elavl3", opacity = 0.5, visible = False)



        #print(self.template_layer, self.denoised_layer)
        print(self.viewer.layers)

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
        for cell in self.con_df.cell_id.sort_values().unique():
            subset = self.con_df[self.con_df.cell_id == cell].dropna().copy()
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
        """Load image"""
        if self.volume_subset.shape[0] > 0:
            print("loading dask images")
            self.denoised_layer.data = self.im[:, self.first_vol:self.end_vol ].compute()
            self.denoised_layer.contrast_limits_range = (0, 255)
            self.denoised_layer.contrast_limits = [0, 40]
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

    def translate(self, event):
        
        if self.moving_grid2world is not None:
            # manually estimate translation
            self.x = self.translate_x.value
            self.y = self.translate_y.value
            self.z = self.translate_z.value

            self.grid_copy = np.eye(4)
            self.grid_copy[2, -1] += self.x # x
            self.grid_copy[1, -1] += self.y # y
            self.grid_copy[0, -1] += self.z # z
            
            if len(self.overview_im.data.shape) > 2: # if not still empty array
                self.template.data = self.apply_tranform(self.ref_grid2world, self.moving_grid2world, self.grid_copy, self.overview_im.data, self.template_orig)
                
            else:
                self.template.data = self.apply_tranform(self.ref_grid2world, self.moving_grid2world, self.grid_copy, self.reference_brain.data, self.template_orig)
            
    def save_starting_affine(self):
        #save affine transform
        filename = self.affine_name.value
        np.save(os.path.join(self.denoised_dir, filename), self.grid_copy)
               
