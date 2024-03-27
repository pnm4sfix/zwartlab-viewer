import seaborn as sns
from scipy.signal import resample, convolve
#from regression import LinearRegression, CustomRegression
import warnings
warnings.filterwarnings("ignore")
import cv2
from caiman.utils.visualization import get_contours
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import neo
import quantities as pq
from scipy import ndimage
from cv2 import warpAffine
import os
import time
from skimage.io import imread
import dask.dataframe as dd
#from numba import jit
from dask_image.imread import imread as da_imread
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.ndimage import gaussian_filter1d
import tifffile as tf
from skimage.registration import phase_cross_correlation
import dask.array as da
from scipy.ndimage.interpolation import shift

try:
    from scipy.stats import median_absolute_deviation as mad
except:
    from scipy.stats import median_abs_deviation as mad

from dipy.align.imaffine import transform_centers_of_mass, AffineMap
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray, gray2rgb
#from colormap import rgb2hsv as r2h
#from colormap import hsv2rgb as h2r
from skimage.color import hsv2rgb
from scipy.interpolate import interp1d
from tkinter import filedialog
from zwartlab_viewer._reader import read_video, read_vid_frame
import dask




def update_rois(t, dffs, ims, theta= None, vel = None):
    """ Update function to be processed by func animation - updates each subplot based on t, dff and im."""
    
    for idx, im in enumerate(ims[:-1]):
        dff = dffs[idx]
        im.axes.collections[0].set_array(dff[t, :])
        im.axes.set_ylim((0, 512))
        im.axes.set_xlim((512, 0))
        im.axes.axis('off')
        time = str(t*0.4)[:4]
        im.axes.set_title('T = {}'.format(time))
        im.axes.text(480, 480, "z = " + str(idx * -5) +" microns", c="white")

        im.axes.arrow(50, 55, 0, 25, width = 3, color = "white")
        im.axes.text(35, 80, "Rostral", c = "white", rotation = 90)
        im.axes.arrow(50, 55, 25, 0, width = 3, color = "white")
        im.axes.text(105, 20, "Left", c = "white")
    
    # grating im
    gain = int(vel[t])
    #print(gain, vel[t])
    ims[-1].set_array(create_grating(t, theta, gain))
                

def create_grating(i, theta, gain):
    
    # create grid
    grid = np.zeros((512, 512))

    for n in range(20):
        grid[n+(i*gain)::40] =1
        grid[n+(i*gain)::-40] =1
    
    grid = rotate(grid, theta)
    
    return grid

def dff_plot(contours, dffs, vidName, cmap = None, fps = 2.5, bitrate = 1800, theta = 0, velocity = None):

    """Parameters: 
            contours: list of contours
            dffs: list of dffs
            cmap: plt.get_cmap('inferno')"""
    
    n_ax = len(contours) +1

    # set up figure and initial roi collections
    fig, axes = plt.subplots(ncols = n_ax, figsize=(40, 20))
    
    collections = []
    # loop through contours, convert to matplotlib polygons and create patch collection
    for contour in contours:
        patches = convert_contours_to_polygons(contour)
        collection = PatchCollection(patches, cmap = cmap, alpha = 0.9)
        collections.append(collection)
        
    
    cmin = np.array([float(dff.min()) for dff in dffs]).min()
    cmax = np.array([float(dff.max()) for dff in dffs]).max()
    
    ims = []
    
    # loop through subplots and setup initial plots, adding patch collection
    for idx, ax in enumerate(axes[:-1]):
        rois = collections[idx]
        im = ax.imshow(np.zeros((512, 512)), cmap='inferno', interpolation = 'None')
        rois.set_clim(cmin, cmax)
        im.axes.add_collection(rois)
        
        ims.append(im)
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax.axis('off')
    fig.colorbar(collections[-1], ax =cbar_ax)
    
    # get number of frames
    nFrames = dffs[0].shape[0]
    
    # add grating subplot
    grating_im = axes[-1].imshow(create_grating(0, theta, 0), "gray")
    axes[-1].set_ylim((0, 512))
    axes[-1].set_xlim((0, 512))
    axes[-1].axis('off')
    ims.append(grating_im)
    
    vel = 1000 * np.abs(velocity.magnitude.flatten())
    vel = 10 * (resample(vel, nFrames).astype("int64"))
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps= fps, metadata=dict(artist='Me'), bitrate=bitrate)

    # define the animation
    ani = animation.FuncAnimation(fig, update_rois, frames = nFrames, interval = (1./fps) * 1000, fargs = (dffs, ims, theta, vel), repeat=False)

    # save the animation
    ani.save(vidName, writer = writer, dpi = dpi)
    Video(vidName, embed=True)




def convert_contours_to_polygons(contours, closed = True, fill = True):
    """ Converts contours from cnm.estimates to matplotlib polygons"""
    patches = []

    # loop through all contours and get polygon values
    for contour in contours:
            p = Polygon(contour['coordinates'], closed = closed, fill = fill)
            patches.append(p)
    return patches

def convert_contours_to_mask(contours):
    patches = []

    for n, contour in enumerate(contours):
            #p = Polygon(contour['coordinates'], closed=True)
            p = contour["coordinates"]
            patch = p[~np.isnan(p)]
            patch = patch.reshape(-1, 2).astype("int32")
            patches.append(patch)
    mask = np.zeros((512, 512))
    cv2.fillPoly(mask, patches, 1)
    mask = mask.astype(bool)
    
    return mask



def create_motor_regressors(fish, electrodes, nsamples):
    """Takes electrode recordings and turns them into regressors
        
        Parameters: 
            
            fish: neo.Block object representing all fish data
            electrodes: list of neo.AnalogSignal representing
            nsamples: number of samples electrode recordings need to be resampled to
            
            
        Returns:
            
            motor_regressors: np.array of motor regressors
        """
    
    motor_regressors = []
    # create gcamp6f kernel with sampling rate of electrodes
    # time base is 3 seconds at sampling rate of 6KhZ
    x = np.linspace(0, 3, 18000)
    gcamp6f_kernel = biexp(x)
    plot_kernel(x, gcamp6f_kernel)
    
    
    # loop  through electrodes
    for electrode in electrodes:
        # flatten array
        asig = electrode.magnitude.flatten()
        # convolve electrode recording with gcamp6f kernel
        regressor = convolution(asig, gcamp6f_kernel, padding = 40000)
        #resample convolved trace
        resampled = resample(regressor, nsamples)
        # append to list
        motor_regressors.append(resampled)
    motor_regressors = np.array(motor_regressors)
    
    return motor_regressors

def create_visual_regressors(fish, epoch, sr, nFrames):
    """Takes epoch labels at creates regressors of them at sampling rate of image acquisition.
        
        Parameters: 
            
            fish: neo.Block object representing all fish data
            epoch: neo.Epoch object containing exp trials
            sr: sampling_rate of image aquisition
            nFrames: number of frames from image acquisition
            
            
        Returns:
            
            visual_regressors: np.array of visual regressors
        """
    
    # create gcamp6f kernel at sampling rate of image acquisition
    x = np.linspace(0, 4, int(4 * sr))
    gcamp6f_kernel = biexp(x)
    plot_kernel(x, gcamp6f_kernel)


    visual_regressors = []
    
    # loop through labels in order
    for label in np.sort(np.unique(epoch.labels).astype("int64")):
       
        if label != -57:
            
            # get starts and ends for that label
            starts, ends = fish.get_starts_by_label(epoch, str(label))
            empty = np.zeros(nFrames)

            # loop through starts and ends, get idx 
            for start, end in zip(starts, ends):
                idx = np.where((asig.times > start) & (asig.times< end))[0]

                # assign these indices in empty array to 1
                empty[idx] = 1
            
            # convolve the binary trace
            convolved = convolution(empty, gcamp6f_kernel, padding = 200)

            visual_regressors.append(convolved)
    visual_regressors = np.array(visual_regressors)
    
    return visual_regressors

def plot_regressors(regressors, labels):
    """Plots regressors for inspection
        
        Parameters: 
            
            regressors: np.array of regressors
            labels: list of labels
            
            
        Returns:
            
            fig: regressor figure
        """
    
    n_regressors = regressors.shape[0]
    fig, axes = plt.subplots(nrows = n_regressors, figsize = (10, 15))
    for n, regressor in enumerate(regressors):
        
        if n_regressors >1 :
            
            axes[n].plot(regressor)
            
            axes[n].set_title(labels[n])
        else:
            axes.plot(regressor)
    fig.tight_layout()
    sns.despine()
    return fig

def plot_trials(trials):
    """Plots trials
        
        Parameters: 
            
            trials: list of neo.AnalogSignals
            
            
        Returns:
            
            fig: trial figure
        """
    
    fig, axes  = plt.subplots(ncols = 3, sharex = True, sharey = True, figsize= (20, 5))
    max_f = np.array([trial.max() for trial in trials]).max()
    min_f = np.array([trial.min() for trial in trials]).min()
    for n, trial in enumerate(trials[:3]):
        
        _ = axes[n].plot(trial)
        axes[n].set_title("Trial {}".format(n))
        axes[n].set_ylim((0, max_f))
    sns.despine()   
    fig.tight_layout()
    return fig

def plot_asig_trials_for_all_labels(asig, epoch):
    """Plots trials for all labels
        
        Parameters: 
            
            asig: neo.AnalogSignal of interest
            epoch: neo.Epoch of interest
            
            
        Returns:
            
            figs: list of trial figures
        """
    figs = []
    for label in np.sort(np.unique(epoch.labels).astype("int64")):
        if label != -57:
            starts, ends = get_starts_by_label(epoch, str(label))
            trials = get_label_trials(asig, starts, ends)
            fig= plot_trials(trials)
            figs.append(fig)
            
    return figs

"""DUPLICATE def regression(regressors, F_dff, fit_intercept = False):
    Computes mass univariate regression
        
        Parameters: 
            
            regressors: np.array of regressors
            F_dff: F_dffs from cnm.estimates
            
            
        Returns:
            
            weights:beta * rsq
            betas: coefficients from regression
            rsq: explained variance
            
        
    try:
        algorithm = LinearRegression(fit_intercept=fit_intercept)
    except:
        from sklearn.linear_model import LinearRegression as LR
        algorithm = CustomRegression(LR(fit_intercept=fit_intercept))
    model = algorithm.fit(regressors.T, F_dff)
    betas = model.betas.toarray()
    rsq = model.score(regressors.T, F_dff).toarray()
    #calculate weights, defined as the product of the Rsquared * Betas
    weights = betas.T * rsq.T
    
    return weights, betas, rsq"""
               
    
def multi_z_regression(cnms):
    """Computes regression for all z levels.
        
        Parameters: 
            
            cnms: list of cnm objects
            
        Returns: 
        
            regression_results: list of regression results for each z level
        """

    regression_results = []

    for z_idx, cnm in enumerate(cnms):
        est = cnm.estimates
        results = regression(regressors, est.F_dff)
        regression_results.append(results)
        
    return regression_results




#def biexp(x):
#    # kernel has rise time of 50ms and fall of 490 ms therefore
#    # b is rise and a is decay
#    a = 1/0.49
#    b = 1/0.05
#    
#    return (a*b/(a-b)) * (np.exp(-b*x) - np.exp(-a*x))

#def plot_kernel(x, kernel):
#    fig, ax = plt.subplots()
#    ax.plot(x, kernel, c='m')
#    ax.set_title("GCaMP6f kernel")
#    ax.set_xlabel("Time (s)")
#    ax.set_ylabel("Relative Intensity")
#    sns.despine()
    

#def convolution(array, kernel, padding):
#    """A utility function to include padding in signal convolution to avoid edge artifacts."""
#    
#    half_padding = int(padding/2)
#    padded = np.zeros(array.shape[0] + padding)
#    padded[half_padding:-half_padding] = array
#    padded[:half_padding] = array[:half_padding]
#    padded[-half_padding:] = array[-half_padding:]
#    
#    convolved = np.convolve(padded, kernel, 'same')
#    
#    return convolved[half_padding:-half_padding]

def plotting_palette():
    """ Defines a custom colorblind seaborn palette. 
    
    Returns:
        palette : Seaborn palette"""
    
    
    custom = [(0, 0, 0), (255, 0, 102), (16, 127, 128), (64, 0, 127),
              (107, 102, 255), (102, 204, 254)]
    custom = [tuple(np.array(list(rgb))/255) for rgb in custom]
    custom2 = [sns.light_palette(color, n_colors = 3, reverse=True) for color in custom]
    custom3 = np.array(custom2).reshape([18, 3]).tolist()
    custom4 = custom3[0::3]+custom3[1::3]+custom3[2::3]
    palette = sns.color_palette(custom4)
    #sns.palplot(palette)
    return palette



def read_bonsai_log_file(file):
    log_df = pd.read_csv(file)
    return log_df
    
def read_bonsai_matrix_file(file, channels = 5):
    matrix_data = np.fromfile(file).reshape(-1, channels)
    return matrix_data

def get_volume_clock(matrix_data):
    volume_clock = matrix_data[:,3]
    return volume_clock

def get_frame_clock(matrix_data):
    frame_clock = matrix_data[:,2]
    return frame_clock

def get_clock_starts_ends(clock):
    # indices where clock is high
    high = np.where(clock >2.5)[0]
    # indices where high 
    starts = np.where(np.diff(high) > 1)[0] +1
    
    # append the first high val
    acq_starts = np.append(high[0],  high[starts])
    
    
    low = np.where(clock <2.5)[0]
    ends = np.where(np.diff(low) > 1)[0] +1
    
    acq_ends = low[ends]
    
    if acq_starts.shape != acq_ends.shape:
        print("starts does not equal ends, diff is {}".format(acq_starts.shape[0] - acq_ends.shape[0]))
        
    #false_starts = np.where(np.diff(acq_starts) < 1000)[0]
    
    #if false_starts.shape[0] > 0 :
    #    print("false start detected - analyse volume clock")
    #    # look at acq_starts from the last false start + 2 volumes  -this doesnt make any sense to me
    #    acq_starts = acq_starts[false_starts[-1]+2 : ]
    #    acq_ends = acq_ends[false_starts[-1]+2 : ]
                                
        
    return acq_starts, acq_ends


def get_clock_timestamps(clock, acq_starts, acq_ends, t_start):
    clock_asig = neo.AnalogSignal(signal = clock, sampling_rate = 6000 * pq.Hz, units = pq.V, t_start = t_start)
    acq_starts_t = clock_asig.times[acq_starts]
    acq_ends_t = clock_asig.times[acq_ends]
    
    return acq_starts_t, acq_ends_t
    

def create_volume_df(volume_acq_starts_t, volume_acq_ends_t):
    
    volume_df = pd.DataFrame()
    volume_df["starts"]  = pd.Series(volume_acq_starts_t, dtype = "float64")
    volume_df["ends"]  = pd.Series(volume_acq_ends_t, dtype = "float64")
    volume_df["nVol"] = np.arange(volume_df.shape[0])
    
    return volume_df
    

def get_moving_times(log_df):
    stage_move_away = log_df[(log_df.Value.str.contains("XStage"))].Timestamp
    
    stage_move_back = log_df[(log_df.Value.str.contains("Move"))].Timestamp
    return stage_move_away, stage_move_back
    
def get_moving_vols(volume_df, stage_move_away, stage_move_back, samp_int):
    print("getting moving vols at time {}".format(time.time()))
    move_away_vols = []
    move_back_vols = []
    
    # loop through times when the stage was moving
    for move_time in stage_move_away:
        
        # subset volumes that start before and end after movement- might be worth setting a 300 ms window
        subset = volume_df[((volume_df.starts > move_time-0.08) & (volume_df.starts <move_time + (samp_int-0.08) ))]#|
                              #((volume_df.ends > (move_time +0.2)) & (volume_df.ends <move_time + 0.300))]
        if subset.shape[0]>0:
            
            for row in range(subset.shape[0]):
                move_away_vols.append(subset.iloc[row, 2])
            
    for move_time in stage_move_back:
        
        subset = volume_df[((volume_df.starts > move_time-0.08) & (volume_df.starts <move_time + (samp_int-0.08)))]#|
                              #((volume_df.ends > (move_time +0.2)) & (volume_df.ends <move_time + 0.3300))]
        if subset.shape[0]>0:
            
            for row in range(subset.shape[0]):
                move_back_vols.append(subset.iloc[row, 2])
            
    move_away_vols = np.array(move_away_vols)
    move_back_vols = np.array(move_back_vols)
    
    print("finished getting moving vols at time {}".format(time.time()))
    return move_away_vols, move_back_vols

def fill_moving_vols(im, move_away_vols, move_back_vols):
    print("filling moving vols at time {}".format(time.time()))
    # set moving vols to NaN and fillna - other options could be interpolate?
    # this might need to be more dynamic for situations where stage needs correcting
    #im_df = pd.DataFrame(im.reshape(im.shape[0], -1).T)
    
    
    #im_df.loc[:,move_away_vols] = np.nan
    #im_dd = dd.from_pandas(im_df, 44)
    #im_df = im_dd.fillna(method = "ffill", axis=1).compute()
    #
    #im_df.loc[:,move_back_vols] = np.nan
    #im_dd = dd.from_pandas(im_df, 44)
    #im_df = im_dd.fillna(method = "bfill", axis=1).compute()
    
    filled_im = fast_fill_moving_frames(im, move_away_vols, move_back_vols)
    print("finished filling moving vols at time {}".format(time.time()))
    return filled_im
    

def return_microsteps(series):
    split = series.split(" ")
    microsteps = int(split[0])
    if microsteps < 0 :
        microsteps = -1
    elif microsteps > 0 :
        microsteps = 1
    return microsteps

def create_displacement_df(log_df, volume_df, samp_int):
    print("creating displacement_df at time {}".format(time.time()))
    x = log_df[(log_df.Value.str.contains("XStage"))]
    y = log_df[(log_df.Value.str.contains("YStage"))]
    x_steps = x.Value.apply(return_microsteps).to_numpy()
    y_steps  = y.Value.apply(return_microsteps).to_numpy()
    x_start = x.Timestamp.to_numpy()
    x_end = x_start + 5


    displace_df = pd.DataFrame([x_steps, y_steps, x_start, x_end]).T
    displace_df.columns = ["x", "y", "start", "end"]


    # which volumes are within displaced period- and what is displacement of each 
    displaced_vols = np.array([])
    volume_df["displacement_x"] = [np.nan] * volume_df.shape[0]
    for idx, start_time in enumerate(displace_df.start):
        end = displace_df.end.iloc[idx]
        displacement_x, displacement_y = (displace_df.x.iloc[idx], displace_df.y.iloc[idx])

        # check if volume occurs within period
        start_filter = (volume_df.starts > (start_time + (samp_int-0.08)))
        end_filter = (volume_df.ends <end)
        volume_df.loc[start_filter & end_filter, "displacement_x"] = displacement_x
        volume_df.loc[start_filter & end_filter, "displacement_y"] = displacement_y
        
    
   
    
    print("finished displacement_df at time {}".format(time.time()))
        
    return volume_df
        
def calc_correction(volume_df, move_back_vols):
    # specify a correction map that maps the movement in cartesian coordinates to pixel coords after rotating the image to represent the correct orientation

    cart_px_map_y = {1 : -1,
                     -1 : 1,
                     0  : 0}
    volume_df["correction_x"] = -volume_df.displacement_x
    volume_df["correction_y"]  = -(volume_df.displacement_y.map(cart_px_map_y))
    
    
    
    # check no null rows just before move back
    previous_frames  = volume_df.loc[(volume_df.nVol.isin(move_back_vols-1)) &(volume_df.displacement_x.isnull()), ["displacement_x",
                                                     "displacement_y" , 
                                                     "correction_x", 
                                                     "correction_y"]]
    previous_two_frames = volume_df.loc[previous_frames.index-1, ["displacement_x",
                                                         "displacement_y" , 
                                                         "correction_x", 
                                                         "correction_y"]].values

    if previous_frames.shape == previous_two_frames.shape:
         volume_df.loc[previous_frames.index, ["displacement_x",
                                                         "displacement_y" , 
                                                         "correction_x", 
                                                         "correction_y"]]= previous_two_frames
    else:
        print("shapes don't match")
    return volume_df


def correct(im, rotate = True, crop = True, width = 300, height = 250, motion_correct = True, volume_df = None, scale = 100):
    # shape is y, x
    nFrames = im.shape[0]
    transformed_im = np.zeros((nFrames, height, width))
    
    
    if rotate:
        print("fast rotating {} frames".format(nFrames))
        transformed_im = fast_rotate(im)

    if motion_correct:
        print("motion correct")
        transformed_im = translate(transformed_im, volume_df, scale)

    if crop:
        im_center = np.array(im[0].shape)/2
        crop_x0, crop_x1 = int(im_center[1] - (width/2)), int(im_center[1] +(width/2))
        crop_y0, crop_y1 = int(im_center[0] - (height/2)), int(im_center[0] + (height/2))    

        transformed_im = fast_crop(transformed_im, crop_x0, crop_x1, crop_y0, crop_y1)

        #transformed_frame = transformed_frame[crop_y0:crop_y1, crop_x0:crop_x1]

        #transformed_im[nVol] = transformed_frame
            
    
            
    return transformed_im   

def get_any_associated_files(file_to_associate, filetype):
    

    original_time = os.path.getmtime(file_to_associate)
    
    
    hour, mins, = time.gmtime(original_time).tm_hour, time.gmtime(original_time).tm_min

    # loop through all files in root dir
    # get original im and time
    # get all other files that have the same timestamp
    root_dir = os.path.dirname(file_to_associate)
    root_files = os.listdir(root_dir)
    
    corresponding_files = []


    for root_file in root_files:

        # get time
        creation_time = os.path.getmtime(os.path.join(root_dir, root_file))
        root_hour, root_min = time.gmtime(creation_time).tm_hour, time.gmtime(creation_time).tm_min

        if (root_hour == hour) & (np.abs(root_min -mins)<=1 ):
            if os.path.join(root_dir, root_file) != file_to_associate:
                corresponding_files.append(root_file)
                
    log_files = [os.path.join(root_dir, file) for file in corresponding_files if filetype in file]
    
    if len(log_files) > 1:
        print("multiple file matches")
        # choose the biggest one 
        #file_sizes = [os.path.getsize(log_file) for log_file in log_files]
        #biggest_file = np.argmax(np.array(file_sizes))
        #log_file = log_files[biggest_file]
        log_file = filedialog.askopenfilename(initialdir = root_dir, title = "Custom select file associated with {}".format(file_to_associate))
        
    elif len(log_files) == 1:
        log_file = log_files[0]
        
    else:
        print("No log matches -custom select")
        log_file = filedialog.askopenfilename(initialdir = root_dir, title = "Custom select file associated with {}".format(file_to_associate))
                         
    return log_file


def get_associated_files(folder):
    original_im = folder[:-13] +".tif"

    original_time = os.path.getmtime(original_im)
    
    
    hour, mins, = time.gmtime(original_time).tm_hour, time.gmtime(original_time).tm_min

    # loop through all files in root dir
    # get original im and time
    # get all other files that have the same timestamp
    root_dir = os.path.dirname(original_im)
    root_files = os.listdir(root_dir)
    
    corresponding_files = []


    for root_file in root_files:

        # get time
        creation_time = os.path.getmtime(os.path.join(root_dir, root_file))
        root_hour, root_min = time.gmtime(creation_time).tm_hour, time.gmtime(creation_time).tm_min

        if (root_hour == hour) & (np.abs(root_min -mins)<=1 ):
            if os.path.join(root_dir, root_file) != original_im:
                corresponding_files.append(root_file)
                
    log_files = [os.path.join(root_dir, file) for file in corresponding_files if "log" in file]
    
    if len(log_files) > 1:
        print("multiple log file matches")
        # choose the biggest one 
        #file_sizes = [os.path.getsize(log_file) for log_file in log_files]
        #biggest_file = np.argmax(np.array(file_sizes))
        #log_file = log_files[biggest_file]
        log_file = filedialog.askopenfilename(initialdir = root_dir, title = "Custom select log file associated with {}".format(original_im))
        
    elif len(log_files) == 1:
        log_file = log_files[0]
        
    else:
        print("No log matches -custom select")
        log_file = filedialog.askopenfilename(initialdir = root_dir, title = "Custom select log file associated with {}".format(original_im))
                                      
    
    matrix_files =[os.path.join(root_dir, file) for file in corresponding_files if "NI" in file]
    
    if len(matrix_files) > 1:
        print("multiple matrix file matches")
        # choose the biggest one 
        #file_sizes = [os.path.getsize(matrix_file) for matrix_file in matrix_files]
        #biggest_file = np.argmax(np.array(file_sizes))
        #matrix_file = matrix_files[biggest_file]
        matrix_file = filedialog.askopenfilename(initialdir = root_dir, title = "Custom select matrix file associated with {}".format(original_im))
        
    elif len(matrix_files) == 1:
        matrix_file = matrix_files[0]
        
    else:
        print("No matrix matches -custom select")
        matrix_file = filedialog.askopenfilename(initialdir = root_dir, title = "Custom select matrix file associated with {}".format(root_file))
    
    return log_file, matrix_file

#@jit
def fast_rotate(im, volume_df):
    h, w = im.shape[1:]
    nFrames = volume_df.nVol.unique().shape[0]
    rotated = np.zeros((nFrames, h, w))
    for nVol in range(nFrames):
        frame = im[nVol]
        
        frame = np.rot90(frame)
        frame = np.rot90(frame)
        rotated[nVol] = frame
    
    return rotated


def fast_crop(im, x0, x1, y0, y1, volume_df):
    h, w = im.shape[1:]
    nFrames = volume_df.nVol.unique().shape[0]
    cropped = np.zeros((nFrames, y1-y0, x1-x0))
    for nVol in range(nFrames):
        frame = im[nVol]
        
        cropped[nVol] = frame[y0:y1, x0:x1]
    return cropped

#@jit
def fast_fill_moving_frames(im, moving_vols, volume_df):
    nFrames = volume_df.nVol.unique().shape[0]
    for frame in range(nFrames):
        #if frame in move_back_vols:
        #    # back fill
        #    # check boundary
        #    if frame+1 < nFrames:
        #        im[frame] = im[frame +1]
        #    else:
        #        im[frame] = im[frame -20]

        if frame in moving_vols:
            # forward fill
            im[frame] = im[frame -1]
            
    return im


def translate(im, volume_df, scale):

    transformed_im  = np.zeros(im.shape)
    nFrames = volume_df.nVol.unique().shape[0]
    for nVol in range(nFrames):
            frame = im[nVol]
             # check if any displacment required
            if np.isnan(volume_df.correction_x.iloc[nVol]):

                #no correction required
                transformed_frame  = frame

            else:
                # correct displacement    
                # make a translation matrix for each 

                x_correct = float(volume_df.correction_x.iloc[nVol]) *scale
                y_correct = float(volume_df.correction_y.iloc[nVol]) *scale

                translation_matrix = np.array([[1, 0, x_correct],
                                               [0, 1, y_correct]])


                transformed_frame =  warpAffine(frame, translation_matrix, (frame.shape[1], frame.shape[0]))
                
            transformed_im[nVol] = transformed_frame
    return transformed_im

def read_pump_log(log_df, matrix_data, direction, flow_angle_map):
    
    
    # get first analogread and subtract by delay in reading the first buffer of data
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (1/6) # buffer is 1000, sr is 6000 therefore subtract 1/6 seconds
    
    # read the pump clock
    pump_clock = matrix_data[:,-1]
    
    # create an asig of the pump clock
    pump_asig = neo.AnalogSignal(pump_clock, units= pq.V, sampling_rate = 6000 * pq.Hz, t_start = asig_t_start * pq.s)
    
    # subset log_df using to get when the rate is changed
    rate = log_df[log_df.Value.str.contains("MM")]
    
    # get flow rate, pump times, state and make a dataframe
    pump_rates = get_flow_rate(rate)
    pump_times = rate.Timestamp
    pump_states = get_pump_states(pump_times, pump_asig)
    
    pump_df = pd.DataFrame([pump_times.to_numpy(), pump_states, pump_rates]).T
    pump_df.columns = ["start", "pump_state", "velocity"]
    pump_df["mode"] = [np.nan] * pump_df.shape[0]
    
    
    # if pump is high >2.5 then it is infusing
    pump_df.loc[pump_df.pump_state >2.5, "mode"] = "infuse"
    pump_df.loc[pump_df.pump_state <2.5, "mode"] = "withdraw"
    
    
    # use a dictionary to map the direction and mode to an angle of flow
    pump_df["angle"] = pump_df["mode"].map(flow_angle_map[direction])
    pump_df["end"] = pump_df.start + 5
    
    # get volume_clock - use this create a df with the timestamp of each volume
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)
    
    first_vol = volume_acq_starts[0]
    
    return pump_df, volume_df, first_vol

def read_flow_log(log_df, matrix_data, flow_angle_map):
    
    
    # get first analogread and subtract by delay in reading the first buffer of data
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (1/6) # buffer is 1000, sr is 6000 therefore subtract 1/6 seconds
    
    # subset log_df using to get when the rate is changed
    rate = log_df[log_df.Value.str.contains("MM")]

    # get flow rate, pump times, state and make a dataframe
    pump_rates = get_flow_rate(rate)
    pump_times = rate.Timestamp
    
    # get pump 1 direction
    all_dir = log_df[log_df.Value.str.contains("DIR")]
    direction = all_dir[all_dir.Value.str.startswith("DIR")]
    pump_dir = direction.Value.apply(get_pump_dir)
    
    # get open valves
    valve_df = all_dir[~all_dir.Value.str.startswith("DIR")]
    valve_df.drop_duplicates(inplace = True)
    valve_idx = valve_df.Value.apply(get_pin_number)
    
    flow_df = pd.DataFrame([pump_times.to_numpy(), valve_idx, pump_rates, pump_dir]).T
    flow_df.columns = ["start", "valve", "velocity", "dir"]
    flow_df["angle"] = flow_df.set_index(["dir", "valve"]).index.map(flow_angle_map)
    flow_df["end"] = flow_df.start + 5
    
    
    # get volume_clock - use this create a df with the timestamp of each volume
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)
    
    first_vol = volume_acq_starts[0]
    
    return flow_df, volume_df, first_vol


def map_angle_vel(volume_df, stim_df, angle_col = "angle", vel_col = "velocity"):
    
    vol_angle_map =  {k:np.nan for k in range(volume_df.shape[0])}
    vol_velocity_map = {k:np.nan for k in range(volume_df.shape[0])}

    for nstart, start in enumerate(stim_df.start):
        angle = stim_df.loc[nstart, angle_col]
        velocity = stim_df.loc[nstart, vel_col]

        end = end = stim_df.loc[nstart, "end"]
        start_filter = volume_df.starts >= start
        end_filter = volume_df.ends <= end

        subset = volume_df[start_filter & end_filter]

        # get volume idx
        vols = subset.nVol.to_numpy()

        for vol in vols:
            vol_angle_map[vol] = angle
            vol_velocity_map[vol] = velocity
    
    
    volume_df[angle_col] = volume_df.nVol.map(vol_angle_map)
    volume_df[vel_col] = volume_df.nVol.map(vol_velocity_map)
    
    return volume_df

def map_angle_vel2(volume_df, stim_df, angle_col="angle", vel_col="velocity"):
    vol_angle_map = {k: np.nan for k in range(volume_df.shape[0])}
    vol_velocity_map = {k: np.nan for k in range(volume_df.shape[0])}

    for nstart, start in enumerate(stim_df.start):
        angle = stim_df.iloc[nstart][angle_col]
        velocity = stim_df.iloc[nstart][vel_col]
        end = stim_df.iloc[nstart]["end"]
        start_filter = volume_df.starts >= start
        end_filter = volume_df.ends <= end
        subset = volume_df.loc[start_filter & end_filter]

        # get volume idx
        vols = subset.nVol.to_numpy()

        for vol in vols:
            vol_angle_map[vol] = angle
            vol_velocity_map[vol] = velocity

    volume_df[angle_col] = volume_df.nVol.map(vol_angle_map)
    volume_df[vel_col] = volume_df.nVol.map(vol_velocity_map)

    return volume_df


def map_trials(volume_df):
    
    # maps combi trials - visual +flow etc
    if ("visual_angle" in volume_df.columns) & ("visual_vel" in volume_df.columns):
        core_volume = pd.DataFrame()
        for stim in volume_df.stim.unique():
            if stim != "visual":
                vol_df = map_combi_trials(volume_df[volume_df.stim == stim])
            elif stim == "visual":
                vol_df = map_uni_trials_v2(volume_df[volume_df.stim ==  stim])
            core_volume = pd.concat([core_volume, vol_df])
        volume_df = core_volume.copy()
    
    # map OMR, sensory stim trials
    else:
        core_volume = pd.DataFrame()
        # loop through stim map uniform stim presented in open loop - maybe set gain to zero in future
        for stim in volume_df.stim.unique():
            print(stim)
            if stim != "OMR":
                if stim=="dots":
                    vol_df = map_dots_trials_v2(volume_df[volume_df.stim ==  stim])
                else:
                    vol_df = map_uni_trials_v2(volume_df[volume_df.stim ==  stim])
                core_volume = pd.concat([core_volume, vol_df])
            elif stim == "OMR":
                vol_df, templates = map_OMR_trials_v2(volume_df[volume_df.stim ==  stim])
                core_volume = pd.concat([core_volume, vol_df])
                
        volume_df = core_volume.copy()
        
        # resort based on nVol
        volume_df.sort_values("nVol", inplace = True)
    
        
    return volume_df


def map_dots_trials_v2(volume_df):
    print("Mapping dot trials")      
    volume_df.velocity = volume_df.velocity.fillna(0)
    
    # changed 08/11/23
    volume_df.angle = volume_df.angle.fillna(0)
    volume_df.coherence = volume_df.coherence.fillna(0)
    volume_df.dot_size = volume_df.dot_size.fillna(0)

    #-------add possible experimental workflows here-------
    unique_vels = volume_df[volume_df.velocity!=0].velocity.dropna().unique()
    unique_angles = volume_df.angle.dropna().unique()
    unique_coherences = volume_df.coherence.dropna().unique()
    unique_dotsizes = volume_df.dot_size.dropna().unique()
    

    possible_combinations = []

    for vel in unique_vels:
        for angle in unique_angles:
            for coh in unique_coherences:
                for dotsize in unique_dotsizes:
                    possible_combinations.append((vel, angle, coh, dotsize))


    vol_trial_map =  {k:np.nan for k in volume_df.nVol}
    vol_paradigm_map = {k:np.nan for k in volume_df.nVol}
    vol_order_map = {k:np.nan for k in volume_df.nVol}


    for vel, angle, coh, dotsize in possible_combinations:
        #print(vel, angle, coh, dotsize)
        template = [(vel, angle, coh, dotsize), (0, 0, 0, 0)]
 
        vel_diff_filter = volume_df.velocity.diff() != 0
        #gain_diff_filter = volume_df.gain.diff() != 0

        vel_diff_filter[0] = True # because first diff is set to nan
        #gain_diff_filter[0] = True # because first diff is set to nan

        # use vel and gain filters and drop any na values 
        # this logic specifies that if the vel or gain has changed then its a new condition
        # this should work for all current cases
        diff_df = volume_df[vel_diff_filter ].dropna(subset = ["angle", "velocity"])

        # order in time
        diff_df.sort_values("nVol", inplace  = True)



        # loop throuh templates
        #for paradigm, template in templates.items():
        trial_counter = 0
        for trial in range(diff_df.shape[0]-len(template) -1):
            # does stim condition order match the whole template
            vel_comparison = (diff_df.velocity.iloc[trial : trial +len(template)].to_numpy() == np.array(template)[:, 0]).all() # all rows, first col should give vel from template
            angle_comparison = (diff_df.angle.iloc[trial : trial +len(template)].to_numpy() == np.array(template)[:, 1]).all() # all rows, second col should give angle from template
            coh_comparison = (diff_df.coherence.iloc[trial : trial +len(template)].to_numpy() == np.array(template)[:, 2]).all() # all rows, first col should give vel from template
            dotsize_comparison = (diff_df.dot_size.iloc[trial : trial +len(template)].to_numpy() == np.array(template)[:, 3]).all() # all rows, second col should give angle from template
            
            comparison = vel_comparison & angle_comparison & coh_comparison & dotsize_comparison
            #print("Vel comp is  {} angle comp is {}, therefore comp is {}".format(vel_comparison, angle_comparison, comparison))
            if trial > 0:
                # check the previous vel was zero -should be the case for all trial structures
                check_new_trial = diff_df.velocity.iloc[trial-1] == 0
            elif trial == 0:
                check_new_trial = True
            # if matches
            if comparison & check_new_trial:#(comparison == np.zeros().all():
                #print("Paradigm {} - Trial {}".format(paradigm, trial_counter))
                trial_start = diff_df.iloc[trial].nVol
                trial_end = diff_df.iloc[trial+len(template)].nVol

                # map volumes 
                for vol in np.arange(trial_start, trial_end):
                    vol_trial_map[vol] = trial_counter
                    paradigm = "direction"
                    #if coh < 1:
                    #    paradigm = "coherence"
                    #else:
                    #    paradigm = "direction"
                    vol_paradigm_map[vol] = paradigm

                # map gain order
                for n_stim, stim in enumerate(template):

                    stim_start = diff_df.iloc[trial+n_stim].nVol
                    stim_end = diff_df.iloc[trial+1+n_stim].nVol
                    for vol in np.arange(stim_start, stim_end):
                        vol_order_map[vol] = n_stim


                trial_counter += 1



    volume_df["trial"] = volume_df.nVol.map(vol_trial_map)
    #volume_df["trial"] = volume_df.OMR_trial.copy()
    volume_df["dot_paradigm"] = volume_df.nVol.map(vol_paradigm_map)
    volume_df["gain_order"] = volume_df.nVol.map(vol_order_map)
    return volume_df

def map_uni_trials_v2(volume_df):
    print("Mapping uni trials")      
    volume_df.velocity = volume_df.velocity.fillna(0)
    
    # changed 08/11/23
    volume_df.angle = volume_df.angle.fillna(0)

    #-------add possible experimental workflows here-------
    unique_vels = volume_df[volume_df.velocity!=0].velocity.dropna().unique()
    unique_angles = volume_df.angle.dropna().unique()
    

    possible_combinations = []

    for vel in unique_vels:
        for angle in unique_angles:
            possible_combinations.append((vel, angle))


    vol_trial_map =  {k:np.nan for k in volume_df.nVol}
    #vol_paradigm_map = {k:np.nan for k in volume_df.nVol}
    vol_order_map = {k:np.nan for k in volume_df.nVol}


    for vel, angle in possible_combinations:
        #print(vel, angle)
        template = [(vel, angle), (0, 0)]
 
        vel_diff_filter = volume_df.velocity.diff() != 0
        #gain_diff_filter = volume_df.gain.diff() != 0

        vel_diff_filter[0] = True # because first diff is set to nan
        #gain_diff_filter[0] = True # because first diff is set to nan

        # use vel and gain filters and drop any na values 
        # this logic specifies that if the vel or gain has changed then its a new condition
        # this should work for all current cases
        diff_df = volume_df[vel_diff_filter ].dropna(subset = ["angle", "velocity"])

        # order in time
        diff_df.sort_values("nVol", inplace  = True)



        # loop throuh templates
        #for paradigm, template in templates.items():
        trial_counter = 0
        for trial in range(diff_df.shape[0]-len(template) -1):
            # does stim condition order match the whole template
            vel_comparison = (diff_df.velocity.iloc[trial : trial +len(template)].to_numpy() == np.array(template)[:, 0]).all() # all rows, first col should give vel from template
            angle_comparison = (diff_df.angle.iloc[trial : trial +len(template)].to_numpy() == np.array(template)[:, 1]).all() # all rows, second col should give angle from template
            
            comparison = vel_comparison & angle_comparison
            #print("Vel comp is  {} angle comp is {}, therefore comp is {}".format(vel_comparison, angle_comparison, comparison))
            if trial > 0:
                # check the previous vel was zero -should be the case for all trial structures
                check_new_trial = diff_df.velocity.iloc[trial-1] == 0
            elif trial == 0:
                check_new_trial = True
            # if matches
            if comparison & check_new_trial:#(comparison == np.zeros().all():
                #print("Paradigm {} - Trial {}".format(paradigm, trial_counter))
                trial_start = diff_df.iloc[trial].nVol
                trial_end = diff_df.iloc[trial+len(template)].nVol

                # map volumes 
                for vol in np.arange(trial_start, trial_end):
                    vol_trial_map[vol] = trial_counter
                    #vol_paradigm_map[vol] = paradigm

                # map gain order
                for n_stim, stim in enumerate(template):

                    stim_start = diff_df.iloc[trial+n_stim].nVol
                    stim_end = diff_df.iloc[trial+1+n_stim].nVol
                    for vol in np.arange(stim_start, stim_end):
                        vol_order_map[vol] = n_stim


                trial_counter += 1



    volume_df["trial"] = volume_df.nVol.map(vol_trial_map)
    #volume_df["trial"] = volume_df.OMR_trial.copy()
   # volume_df["OMR_paradigm"] = volume_df.nVol.map(vol_paradigm_map)
    volume_df["gain_order"] = volume_df.nVol.map(vol_order_map)
    return volume_df
          
def map_OMR_trials_v2(volume_df):
          
    print("Mapping OMR trials")      
    ## Warning - for OMR experiments 2023 when gain is 0 velocity is 0 but this isnt mapping correctly-therefore
    
    

    #assert volume_df.loc[(volume_df.gain == 0) & (volume_df.velocity >0)].shape[0] == 0
    #assert volume_df.loc[(volume_df.gain > 0) & (volume_df.velocity ==0)].shape[0] == 0#, "velocity"] =0

    #-------add possible experimental workflows here-------
    unique_gains = volume_df.gain.dropna().unique()
    low_gains = unique_gains[(unique_gains < 1) & (unique_gains !=0)]
    high_gains = unique_gains[(unique_gains > 1) & (unique_gains !=0)]

    possible_gain_combinations = []

    for low_gain in low_gains:
        for high_gain in high_gains:
            possible_gain_combinations.append((low_gain, high_gain))


    vol_trial_map =  {k:np.nan for k in volume_df.nVol}
    vol_paradigm_map = {k:np.nan for k in volume_df.nVol}
    vol_gain_order_map = {k:np.nan for k in volume_df.nVol}



    for low, high in possible_gain_combinations:


        low_high_template = [low, high, 0]
        high_low_template = [high, low, 0]
        high_low_high_template = [high, low, high, 0]
        low_high_low_template = [low, high, low, 0]
        low_high_low_base_template = [low, high, low, 1,  0]
        high_low_high_low_template = [high, low, high, low, 0]
        low_high_low_high_template = [low, high, low, high, 0]
        acclim_template = [1, 0]

        templates = {
                "gain_up" : low_high_template,
                "gain_down": high_low_template, 
                "gain_down_up": high_low_high_template,
                "gain_up_down": low_high_low_template,
                "gain_up_down_base": low_high_low_base_template,
                "gain_up_down_up": low_high_low_high_template,
                "gain_down_up_down": high_low_high_low_template,
                "acclimatisation": acclim_template

        }
        vel_diff_filter = volume_df.velocity.diff() != 0
        gain_diff_filter = volume_df.gain.diff() != 0

        vel_diff_filter[0] = True # because first diff is set to nan
        gain_diff_filter[0] = True # because first diff is set to nan

        # use vel and gain filters and drop any na values 
        # this logic specifies that if the vel or gain has changed then its a new condition
        # this should work for all current cases
        diff_df = volume_df[vel_diff_filter | gain_diff_filter].dropna(subset = ["gain", "velocity"])

        # order in time
        diff_df.sort_values("nVol", inplace  = True)



        # loop throuh templates
        for paradigm, template in templates.items():
            trial_counter = 0
            for trial in range(diff_df.shape[0]-len(template) -1):
                # does stim condition order match the whole template
                comparison = (diff_df.gain.iloc[trial : trial +len(template)].to_numpy() == np.array(template)).all()
                if trial > 0:
                    # check the previous gain was zero -should be the case for all trial structures
                    check_new_trial = diff_df.gain.iloc[trial-1] == 0
                elif trial == 0:
                    check_new_trial = True
                # if matches
                if comparison & check_new_trial:#(comparison == np.zeros().all():
                    #print("Paradigm {} - Trial {}".format(paradigm, trial_counter))
                    trial_start = diff_df.iloc[trial].nVol
                    trial_end = diff_df.iloc[trial+len(template)].nVol

                    # map volumes 
                    for vol in np.arange(trial_start, trial_end):
                        vol_trial_map[vol] = trial_counter
                        vol_paradigm_map[vol] = paradigm

                    # map gain order
                    for n_gain, gain in enumerate(template):

                        gain_start = diff_df.iloc[trial+n_gain].nVol
                        gain_end = diff_df.iloc[trial+1+n_gain].nVol
                        for vol in np.arange(gain_start, gain_end):
                            vol_gain_order_map[vol] = n_gain


                    trial_counter += 1



    volume_df["OMR_trial"] = volume_df.nVol.map(vol_trial_map)
    volume_df["trial"] = volume_df.OMR_trial.copy()
    volume_df["OMR_paradigm"] = volume_df.nVol.map(vol_paradigm_map)
    volume_df["gain_order"] = volume_df.nVol.map(vol_gain_order_map)
    return volume_df, templates

def load_templates(low, high):
    low_high_template = [low, high, 0]
    high_low_template = [high, low, 0]
    high_low_high_template = [high, low, high, 0]
    low_high_low_template = [low, high, low, 0]
    low_high_low_base_template = [low, high, low, 1,  0]
    high_low_high_low_template = [high, low, high, low, 0]
    low_high_low_high_template = [low, high, low, high, 0]
    acclim_template = [1, 0]

    templates = {
            "gain_up" : low_high_template,
            "gain_down": high_low_template, 
            "gain_down_up": high_low_high_template,
            "gain_up_down": low_high_low_template,
            "gain_up_down_base": low_high_low_base_template,
            "gain_up_down_up": low_high_low_high_template,
            "gain_down_up_down": high_low_high_low_template,
            "acclimatisation": acclim_template

    }
    
    return templates

def map_OMR_trials(volume_df):
    # create empty dict with every vol as a key
    vol_trial_map =  {k:np.nan for k in range(volume_df.shape[0])}
    volume_df.reset_index(drop = True, inplace = True)
    # create vel and gain filters where diff !=
    vel_diff_filter = volume_df.velocity.diff() != 0
    gain_diff_filter = volume_df.gain.diff() != 0
    
    # use vel and gain filters and drop any na values 
    diff_df = volume_df[vel_diff_filter | gain_diff_filter].dropna()
    
    # order in time
    diff_df.sort_values("nVol", inplace  = True)
    
    # get trials by grouping by gain and velocity and apply function return series
    trials = diff_df.groupby(["adaptation", "gain", "velocity"]).nVol.apply(lambda x: return_series(x)).reset_index().nVol
    
    # order in gain and velocity as groupby does the same and trials has been ordered by gain and velocity
    diff_df.sort_values(["adaptation", "gain", "velocity"], ascending = True, inplace = True)
    
    # assign trials to trial column
    diff_df["trial"] = trials.to_numpy()
    
    # re order in time
    diff_df.sort_values("nVol", inplace  = True)
    
    # loop through diff_df nVols and add to dictionary
    for nVol, vol in enumerate(diff_df.nVol):
        vol_trial_map[vol] = diff_df.iloc[nVol, -1]
        
    # as no breaks in protocol just fill forward into na with previous value and assign to trial column in volume df
    vol_trial_map = pd.Series(vol_trial_map).fillna(method = "ffill", axis = 0).to_dict()
    volume_df["trial"] = volume_df.nVol.map(vol_trial_map)
    
    # fill any other nas in
    volume_df.fillna(method = "ffill", axis = 0, inplace = True)
    
    return volume_df
        
def map_uni_trials(volume_df):
    print("mapping uni trials")
    vol_trial_map =  {k:np.nan for k in volume_df.nVol}
    
    # check for trials
    # fill na velocity to 0
    volume_df.velocity = volume_df.velocity.fillna(0)
    
    # changed 08/11/23
    volume_df.angle = volume_df.angle.fillna(0)
    
    # create a diff column to see when velocity changes
    #vel_diff_filter = volume_df.velocity.diff() != 0
    #gain_diff_filter = volume_df.gain.diff() != 0
    volume_df["diff"] = volume_df.velocity.diff()
    
    # subset where difference is greater than 0 indicating an increase in velocity
    diff_df = volume_df[volume_df["diff"] !=0]
    #diff_df = volume_df[vel_diff_filter | gain_diff_filter].dropna()
    
    # get all combinations
    combinations = []

    for angle in diff_df.angle.unique():
        
        for velocity in diff_df.velocity.unique():
            #print("angle {} vel {} filter{}".format(angle, velocity, ((angle != 0) & (velocity != 0)).any()))
            if velocity != 0: # why is this restricting angle 0 and velocity 0.2
                print(angle, velocity)
                combinations.append((angle, velocity))
    # add rest 0 vel 0 angle condition to end
    combinations.append((0, 0))
    # loop through combinations 
    for combination in combinations:
        angle, velocity = combination
        print(combination)
        subset = diff_df[(diff_df.angle== angle) & (diff_df.velocity == velocity)] # this could fail we're assuming combinations are unique between stim-could create another subset and loop through unique stim
        
        # loop througgh possible trial starts in unique subset
        for nvol, vol in enumerate(subset.nVol):

            vol_trial_map[vol] = nvol

            # get the row number of the vol in the diff df
            vol_idx = int(np.where(diff_df.nVol == vol)[0])

            # get the vol of the next change
            try:
                trial_end_vol = diff_df.iloc[vol_idx+1].nVol+1
            except:
                print("possible the last condition, end vol is last vol")
                trial_end_vol = volume_df.shape[0]
            trial_range = np.arange(vol, trial_end_vol -1)
            
            if (angle == 0) & (velocity == 0):
                # make nvol the same trial number as preceding trial where movement had occurred
                if vol != list(vol_trial_map)[0]: # make sure vol isnt first vol
                    preceding_condition_start = diff_df.iloc[vol_idx-1].nVol
                    nvol = vol_trial_map[preceding_condition_start]
                    #print("New trial for angle 0 and vel 0 is {}".format(nvol))
                else:
                    nvol = 0 
                    
            # loop through vols in range
            for trial_vol in trial_range:
                vol_trial_map[trial_vol] = nvol
                
    volume_df["trial"] = volume_df.nVol.map(vol_trial_map)
    volume_df.drop("diff", axis =1, inplace = True)
    
    return volume_df



def map_combi_trials(volume_df):
    print("mapping combi trials")
             
    vol_trial_map =  {k:np.nan for k in range(volume_df.shape[0])}
    
    # check for trials
    # fill na velocity to 0
    volume_df.velocity = volume_df.velocity.fillna(0)
    volume_df.visual_vel = volume_df.visual_vel.fillna(0)
    
    # create a diff column to see when velocity changes
    volume_df["diff"] = volume_df.velocity.diff()
    volume_df["vis_diff"] = volume_df.visual_vel.diff()
    
    # subset where difference is greater than 0 indicating that change in velocity
    diff_df = volume_df[(volume_df["diff"] >0) | (volume_df["vis_diff"] >0) | (volume_df.angle.diff() > 0)]
    
    
    # get all combinations
    combinations = []

    for angle in diff_df.angle.unique():
        for velocity in diff_df.velocity.unique():
            for visual_angle in diff_df.visual_angle.unique():
                for visual_vel in diff_df.visual_vel.unique():
                    combinations.append((angle, velocity, visual_angle, visual_vel))
   
    # loop through combinations 
    for combination in combinations:
        angle, velocity, visual_angle, visual_vel = combination
        subset = diff_df[(diff_df.angle== angle) & (diff_df.velocity == velocity) & 
                        (diff_df.visual_angle == visual_angle) & (diff_df.visual_vel == visual_vel)] # this could fail we're assuming combinations are unique between stim-could create another subset and loop through unique stim
        for nvol, vol in enumerate(subset.nVol):

            vol_trial_map[vol] = nvol

            # get the row number of the vol in the diff df
            vol_idx = int(np.where(diff_df.nVol == vol)[0])

            # get the vol of the next change
            try:
                trial_end_vol = diff_df.iloc[vol_idx+1].nVol
            except:
                print("possible the last condition, end vol is last vol")
                trial_end_vol = volume_df.shape[0]
            trial_range = np.arange(vol, trial_end_vol -1)

            # loop through vols in range
            for trial_vol in trial_range:
                vol_trial_map[trial_vol] = nvol
                
    volume_df["trial"] = volume_df.nVol.map(vol_trial_map)
    volume_df.drop("diff", axis =1, inplace = True)
    
    return volume_df


def get_flow_rate(series):
    
    rate_list = series.Value.tolist()
    flow_rate = [float(rate.split(" ")[1][:-2]) for rate in rate_list]
    
    return pd.Series(flow_rate)

def get_pump_states(pump_times, pump_asig):
    pump_states = []
    for pump_time in pump_times.to_numpy():

        pump_state =  pump_asig.magnitude.flatten()[(pump_asig.times > pump_time+0.5) & (pump_asig.times < pump_time +1)][0]
        pump_states.append(pump_state)
    pump_states = np.array(pump_states)
    
    return pump_states

def get_pump_dir(series):
    direction = series.split(" ")[1]
    return direction

def get_pin_number(series):
    pin = int(series.split(" ")[0])
    return pin
    
    
def get_start_end_times(starts_ends, asig):
    
    starts_ends_t = []
    
    for start, end in starts_ends:
        start_t = asig.times[start]
        end_t = asig.times[end]
        
        starts_ends_t.append((start_t, end_t))
        
    return starts_ends_t


def map_correction(volume_df):
    cart_px_map_y = {1 : -1,
                     -1 : 1,
                     0  : 0}
    volume_df["correction_x"] = -volume_df.x_move
    volume_df["correction_y"]  = -(volume_df.y_move.map(cart_px_map_y))
    return volume_df



def map_motion_angle(x_series, y_series, motion_angle_map):
    angles = []
    for x_m, y_m in zip(x_series, y_series):
        
        angle =  motion_angle_map[(x_m, y_m)]
        angles.append(angle)
            
    return pd.Series(angles)

def get_stage_move(series):
    
    move_list = series.tolist()
    move_val = [float(move.split(" ")[0]) for move in move_list]
    
    return pd.Series(move_val)

def get_OMR_data(series):
    move_list = series.tolist()
    move_val = [float(move.split(" ")[1]) for move in move_list]
    
    return pd.Series(move_val)

def read_motion_log(log_df, matrix_data, motion_angle_map):
    
    
    # get first analogread and subtract by delay in reading the first buffer of data
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (1/6) # buffer is 1000, sr is 6000 therefore subtract 1/6 seconds
    
    motion_starts = log_df[log_df.Value.str.contains("XStage")].Timestamp
    x_move = log_df[log_df.Value.str.contains("XStage")].Value
    x_move = x_move.apply(return_microsteps)

    y_move = log_df[log_df.Value.str.contains("YStage")].Value
    y_move = y_move.apply(return_microsteps)
    accel = log_df[log_df.Value.str.contains("Accel")].Value
    accel = get_stage_move(accel)
    
    motion_angles = map_motion_angle(x_move, y_move, motion_angle_map)
    
    motion_df = pd.DataFrame([motion_starts.to_numpy(), 
                          accel,
                          x_move,
                          y_move,
                          motion_angles]).T
    motion_df.columns = ["start", "velocity", "x_move", "y_move", "angle"]
    motion_df["end"] = motion_df.start +5
    
    # get volume_clock - use this create a df with the timestamp of each volume
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)
    first_vol = volume_acq_starts[0]
    if first_vol == 0:
        print(" WARNING! Imaging started before bonsai")
        
    
    return motion_df, volume_df, first_vol



def map_motion(volume_df, stim_df):
    moving_vol_map  = {k:np.nan for k in range(volume_df.shape[0])}
    vol_motion_x_map =  {k:np.nan for k in range(volume_df.shape[0])}
    vol_motion_y_map =  {k:np.nan for k in range(volume_df.shape[0])}

    for nstart, start in enumerate(stim_df.start):
        motion_x = stim_df.loc[nstart, "x_move"]
        motion_y = stim_df.loc[nstart, "y_move"]

        end = start + 5
        start_filter = volume_df.starts > start
        end_filter = volume_df.ends < end

        subset = volume_df[start_filter & end_filter]

        # get volume idx
        vols = subset.nVol.to_numpy()

        for vol in vols:
            vol_motion_x_map[vol] = motion_x
            vol_motion_y_map[vol] = motion_y
        
        if vols.shape[0] >0:
            # set first vol in the movement to true in moving vol map
            moving_vol_map[vols[0]] = True
            # set last vol +1 in the movement to true in moving vol map
            moving_vol_map[vols[-1] +1] = True
        
        
    
    
    volume_df["x_move"] = volume_df.nVol.map(vol_motion_x_map)
    volume_df["y_move"] = volume_df.nVol.map(vol_motion_y_map)
    volume_df["movingVol"] = volume_df.nVol.map(moving_vol_map)
    
    return volume_df

def motion(log_file, log_df,  matrix_data):
    print("reading motion log")
    
    # def read_motion_log
    motion_angle_map = {(0, 1) : 0,
                        (1, 1) : 45,
                        (1, 0) : 90,
                        (1, -1) : 135,
                        (0, -1) : 180,
                        (-1, -1) : 225,
                        (-1, 0) : 270,
                        (-1, 1) : 315 }
    
    motion_df, volume_df, first_vol = read_motion_log(log_df, matrix_data, motion_angle_map)
    
    volume_df = map_angle_vel(volume_df, motion_df)
    volume_df = map_motion(volume_df, motion_df)
    volume_df = map_correction(volume_df)
    
    volume_df["stim"] = ["motion"] * volume_df.shape[0]
    
    return volume_df, motion_df, first_vol

def flow(log_file, log_df, matrix_data):
    print("reading flow log")
    
    if "AP" in log_file:
            direction = "AP"

    elif "lateral" in log_file:
        direction = "lateral"

    # angle of flow is direction + infuse - high is infuse, infuse is rostral caudal for AP and right-left for lateral
    flow_angle_map =  {"AP": {"infuse": 180,
                              "withdraw": 0},

                       "lateral": {"infuse": 270,
                                    "withdraw": 90}
    }
    
    
    

    # read pump log file
    pump_df, volume_df, first_vol = read_pump_log(log_df, matrix_data, direction, flow_angle_map)

    # map angle and vel to volume
    volume_df = map_angle_vel(volume_df, pump_df)

    
    volume_df["stim"] = ["flow"] * volume_df.shape[0]
    
    return volume_df, pump_df, first_vol

def flow_8way(log_file, log_df, matrix_data):
    print("reading flow log")
    
    
    flow_angle_map = {("WDR", 10):  0  ,
                  ("WDR", 11):  45 ,
                  ("WDR", 5):   90,
                  ("WDR", 6):   135 ,
                  ("INF", 10):  180 ,
                  ("INF", 11):  225 ,
                  ("INF", 5):   270,
                  ("INF", 6):   315}
    
    

    # read flow log file
    flow_df, volume_df, first_vol = read_flow_log(log_df, matrix_data, flow_angle_map)

    # map angle and vel to volume
    volume_df = map_angle_vel(volume_df, flow_df)

    
    volume_df["stim"] = ["flow"] * volume_df.shape[0]
    
    return volume_df, flow_df, first_vol

def flow_laminar(log_file, log_df, matrix_data):
    print("reading laminar flow log")
    
    
    flow_angle_map = {"D1" : 180,
                      "D0": 0}
    
    

    # read flow log file
    flow_df, volume_df, first_vol = read_flow_log(log_df, matrix_data, flow_angle_map)

    # map angle and vel to volume
    volume_df = map_angle_vel(volume_df, flow_df)

    
    volume_df["stim"] = ["flow"] * volume_df.shape[0]
    
    return volume_df, flow_df, first_vol

def read_visual_log(log_df, matrix_data, visual_angle_map, vis_duration = 10):
    
    
    # get first analogread and subtract by delay in reading the first buffer of data
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (1/6) # buffer is 1000, sr is 6000 therefore subtract 1/6 seconds
    
    radians = get_stage_move(log_df[log_df.Value.str.contains("radians")].Value)
    vis_times  = log_df[log_df.Value.str.contains("radians")].Timestamp
    velocity = get_stage_move(log_df[log_df.Value.str.contains("freq")].Value)
    vis_df = pd.DataFrame([vis_times.to_numpy(), radians, velocity]).T
    vis_df.columns = ["start", "radians", "velocity"]
    vis_df["angle"] = vis_df.radians.map(visual_angle_map)
    
    
    # to add
    # if VisStopped not in ...:
    vis_df["end"] = vis_df.start + vis_duration # this is a bit arbitrary - better to get from log file when stim stops
    #else: vis_df['end'] =  log_df[log_df.Value.str.contains("VisStopped")].Timestamp
    
    
    # get volume_clock - use this create a df with the timestamp of each volume
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)
    
    first_vol = volume_acq_starts[0]
    
    
    
    return vis_df, volume_df, first_vol

def visual(log_file, log_df, matrix_data, duration = 10):
    print("reading visual log")
    #feedback = input("Preset duration is 10 seconds, press y to continue with 10 or enter a new duration")
    #if feedback.lower() == "y":
    #    duration = duration
    #else:
    #    duration = float(feedback)
    # angle of visual motion
    visual_angle_map = {0 : 270, 
                    0.7853982 : 225,
                    1.570796 : 180,
                    2.356194 : 135,
                    3.141593 : 90,
                    3.926991 : 45,
                    4.712389 : 0,
                    5.497787 : 315
                    }
    
    
    

    # read visual log file
    vis_df, volume_df, first_vol = read_visual_log(log_df, matrix_data, visual_angle_map, duration)

    # map angle and vel to volume
    volume_df = map_angle_vel(volume_df, vis_df)

    
    volume_df["stim"] = ["visual"] * volume_df.shape[0]
    
    return volume_df, vis_df, first_vol


def read_visual_motion_log(log_df, matrix_data, motion_angle_map, visual_angle_map):
    
    # get first analogread and subtract by delay in reading the first buffer of data
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (1/6) # buffer is 1000, sr is 6000 therefore subtract 1/6 seconds
    radians = get_stage_move(log_df[log_df.Value.str.contains("radians")].Value)
    vis_times  = log_df[log_df.Value.str.contains("radians")].Timestamp
    velocity = get_stage_move(log_df[log_df.Value.str.contains("freq")].Value)
    vis_df = pd.DataFrame([vis_times.to_numpy(), radians, velocity]).T
    vis_df.columns = ["start", "radians", "velocity"]
    vis_df["angle"] = vis_df.radians.map(visual_angle_map)
    vis_df["end"] = vis_df.start + 5
    
    motion_starts = log_df[log_df.Value.str.contains("XStage")].Timestamp
    x_move = log_df[log_df.Value.str.contains("XStage")].Value
    x_move = x_move.apply(return_microsteps)

    y_move = log_df[log_df.Value.str.contains("YStage")].Value
    y_move = y_move.apply(return_microsteps)
    accel = log_df[log_df.Value.str.contains("Accel")].Value
    accel = get_stage_move(accel)
    
    motion_angles = map_motion_angle(x_move, y_move, motion_angle_map)
    
    motion_df = pd.DataFrame([motion_starts.to_numpy(), 
                          accel,
                          x_move,
                          y_move,
                          motion_angles]).T
    motion_df.columns = ["start", "velocity", "x_move", "y_move", "angle"]
    motion_df["end"] = motion_df.start +5
    
    motion_df["visual_angle"] = vis_df.angle
    
    # get volume_clock - use this create a df with the timestamp of each volume
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)
    
    first_vol = volume_acq_starts[0]
    
    return motion_df, volume_df, first_vol


def visual_motion(log_file, log_df, matrix_data):
    print("reading visual motion log")

    # angle of visual motion
    visual_angle_map = {0 : 270, 
                    0.7853982 : 225,
                    1.570796 : 180,
                    2.356194 : 135,
                    3.141593 : 90,
                    3.926991 : 45,
                    4.712389 : 0,
                    5.497787 : 315
                    }
    
    # def read_motion_log
    motion_angle_map = {(0, 1) : 0,
                        (1, 1) : 45,
                        (1, 0) : 90,
                        (1, -1) : 135,
                        (0, -1) : 180,
                        (-1, -1) : 225,
                        (-1, 0) : 270,
                        (-1, 1) : 315 }
    
    motion_df, volume_df, first_vol = read_visual_motion_log(log_df, matrix_data, motion_angle_map, visual_angle_map)
    
    volume_df = map_angle_vel(volume_df, motion_df)
    volume_df = map_motion(volume_df, motion_df)
    volume_df = map_correction(volume_df)
    
    volume_df["stim"] = ["visualmotion"] * volume_df.shape[0]
    
    
    return volume_df, motion_df, first_vol

def get_rauc(series, fr):
    # ignore first frame
    rauc = np.trapz(series.iloc[1:], dx = 1/fr)
    
    return rauc

def get_mean_derivative(series, fr, start_frame = 1, end_frame = None):
    # ignore first frame
    dF = series.iloc[start_frame: end_frame].diff()
    dx = 1/fr
    dFdx = dF / dx
    
    mean_dFdx = dFdx.rolling(3).mean().max()
    
    return mean_dFdx

def constant_timebase(series, fr):
    nVols = series.shape[0]
    new_time = np.arange(nVols) / fr
    
    return pd.Series(new_time, index = series.index)


def biexp(x):
    # kernel has rise time of 50ms and fall of 490 ms therefore
    # b is rise and a is decay
    a = 1/0.49
    b = 1/0.05
    
    return (a*b/(a-b)) * (np.exp(-b*x) - np.exp(-a*x))



def plot_kernel(x, kernel):
    fig, ax = plt.subplots()
    ax.plot(x, kernel, c='m')
    ax.set_title("GCaMP6f kernel")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Intensity")
    sns.despine()
    
def gcamp6f_kernel(x):
    kernel = biexp(x)
    scaler = MinMaxScaler()
    norm_g6f = scaler.fit_transform(kernel.reshape((-1, 1)))
    
    gcamp6f_kernel = norm_g6f.flatten()
    plot_kernel(x, gcamp6f_kernel)
    return gcamp6f_kernel
    
def convolution(array, kernel, padding):
    """A utility function to include padding in signal convolution to avoid edge artifacts."""
    
    half_padding = int(padding/2)
    padded = np.zeros(array.shape[0] + padding)
    padded[half_padding:-half_padding] = array
    padded[:half_padding] = array[:half_padding]
    padded[-half_padding:] = array[-half_padding:]
    
    convolved = convolve(padded, kernel, 'same', method = "fft")
    
    left_pad = int(kernel.shape[0] / 2)
    right_pad = left_pad - 1
    
    return convolved[half_padding-left_pad:-half_padding-right_pad]

    
def smooth(series, width):
    return pd.Series(gaussian_filter1d(series, width), index = series.index)

def get_px2ref(filename):
        with tf.TiffFile(filename) as tif:
            roi_artist = tif.pages[0].tags["Artist"]
            metadata = roi_artist.value.split("\n")
            px2ref_row = [datum for datum in metadata if "pixelToRef" in datum][0]
            row_idx = metadata.index(px2ref_row)
            x = metadata[row_idx +1]
            y = metadata[row_idx +2]
            x_scale = float(x.replace(" ", "").replace("[", "").replace("]", "").split(",")[0])
            y_scale = float(y.replace(" ", "").replace("[", "").replace("]", "").split(",")[1])
            
            return x_scale, y_scale
        


def save_image_to_h5(im, h5_file, name, stim, regressor, method, formats, registered, types):
    
    if name not in dict(h5_file.items()):
        
        
        # create new dataset
        d = h5_file.create_dataset(name, dtype='f', shape=im.shape,
                         compression='gzip', compression_opts=1, data = im, chunks = True)
        d.attrs["stim"] = stim
        d.attrs["regressor"] = regressor
        d.attrs["format"] = formats
        d.attrs["registered"] = registered
        d.attrs["type"] = types
        d.attrs["method"] = method
        
    else:
        #overwrite
        print("overwriting")
        d = h5_file[name]
        d[:] = im
        d.attrs["stim"] = stim
        d.attrs["regressor"] = regressor
        d.attrs["format"] = formats
        d.attrs["registered"] = registered
        d.attrs["type"] = types
        d.attrs["method"] = method
        
        


def fill_blurred_frames(im, ref, volume_df):
    print("filling blurred_frames")
    im_da = da.from_array(im, (1, im.shape[1], im.shape[2]))
    calc_shifts = da.map_blocks(lambda x, y: get_shifts(x, y), im_da, ref, dtype=np.int16)
    shifts = calc_shifts.compute()    
    shifts_rs = shifts.reshape(int(shifts.shape[0]/2), 2)
    # fill blurred frames
    mad_shift = mad(np.abs(shifts_rs), axis=0)
    median = np.median(np.abs(shifts_rs), axis= 0)
    x_thresh = median[0] + (10*mad_shift[0])
    y_thresh = median[1] + (10*mad_shift[1])
    x_idx = np.where(np.abs(shifts_rs[:, 0]) > x_thresh)[0]
    y_idx = np.where(np.abs(shifts_rs[:, 1]) > y_thresh)[0]
    bad_frames =np.unique(np.concatenate([x_idx, y_idx]))
    fill_blurred = fast_fill_moving_frames(im, bad_frames, volume_df)
    return fill_blurred

def registration(im, ref):
    print("pre-registering images")
    im_da = da.from_array(im, (1, im.shape[1], im.shape[2]))
    register_images = da.map_blocks(lambda x, y: register(x, y), im_da, ref, dtype=np.int16)
    registered_images = register_images.compute()
    return registered_images
    
def fill_zero_pixels(im):
    print("filling zero pixels")
    # to speed this up precompute the mean std of the pixels before hand not on each loop
    px_mean = im.mean(axis=0)
    px_std = im.std(axis=0)
    filled = np.zeros(im.shape)
    for nVol in range(im.shape[0]):
        frame = im[nVol]
        nzero_pixels = frame[frame == 0].shape[0]
        nzero_idx = np.where(frame == 0)
        #nzero_px_mean = im[:, nzero_idx[0], nzero_idx[1]].mean(axis = 0)
        #nzero_px_std = im[:, nzero_idx[0], nzero_idx[1]].std(axis = 0)
        nzero_px_mean = px_mean[nzero_idx[0], nzero_idx[1]]
        nzero_px_std = px_std[nzero_idx[0], nzero_idx[1]]

        normal_std = np.sqrt(np.log(1 + (nzero_px_std/nzero_px_mean)**2)) 
        normal_mean = np.log(nzero_px_mean) - normal_std**2 / 2
        # generate random values using lognnormal
        new_vals = np.random.lognormal(normal_mean, normal_std)
        # set np.nan to mean px val
        nan_idx = np.where(np.isnan(new_vals))
        new_vals[nan_idx] = nzero_px_mean[nan_idx]
        frame[frame == 0] = new_vals#.astype("int16")
        filled[nVol] = frame

    return filled
        
def get_shifts(image, ref):
    image = image.reshape(ref.shape)
    shift, error, diffphase = phase_cross_correlation(ref, image)
    
    return shift.reshape((*shift.shape, 1, 1))

def register(image, ref):
    """
    function to register each frame (= time point) to its reference  (=image
    at the beginning of the experiment)

    image : array
        image to be registered

    ref : array
        image to register to

    returns registered image
    """

    # print('mov shape= ' + str(mov.shape))
    image_rs = image.reshape(ref.shape)
    translation, error, diff_phase = phase_cross_correlation(ref, image_rs)
    shifted = shift(image.reshape(ref.shape), translation)

    return shifted.reshape(image.shape)


def read_visual_flow_log(log_df, matrix_data, flow_angle_map, visual_angle_map):
    
    # get first analogread and subtract by delay in reading the first buffer of data
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (1/6) # buffer is 1000, sr is 6000 therefore subtract 1/6 seconds
    
    
    radians = get_stage_move(log_df[log_df.Value.str.contains("radians")].Value)
    
    
    # change to vis started - loose definition - assert than vis_times length == radian length
    vis_times  = log_df[log_df.Value.str.contains("VisStarted")].Timestamp
    velocity = get_stage_move(log_df[log_df.Value.str.contains("freq")].Value)
    
    assert vis_times.shape[0] == radians.shape[0]
    
    vis_df = pd.DataFrame([vis_times.to_numpy(), radians, velocity]).T
    vis_df.columns = ["start", "radians", "velocity"]
    vis_df["angle"] = vis_df.radians.map(visual_angle_map)
    vis_df["end"] = vis_df.start + 5
    
    
    # change this to flow specific things
    flow_starts = log_df[log_df.Value.str.contains("FlowStarted")].Timestamp
    flow_dir = get_uniflow_params(log_df[log_df.Value.str.contains("direction")].Value)
    flow_vel = get_uniflow_params(log_df[log_df.Value.str.contains("velocity")].Value)

    assert flow_starts.shape[0] == flow_vel.shape[0]

    
    flow_df = pd.DataFrame([flow_starts.to_numpy(), flow_dir, flow_vel]).T
    flow_df.columns = ["start", "dir", "velocity"]
    flow_df["angle"] = flow_df.dir.map(flow_angle_map)
    flow_df["end"] = flow_df.start + 5
    
   
    flow_df["visual_angle"] = vis_df.angle
    flow_df["visual_vel"] = vis_df.velocity
    
    
    # get volume_clock - use this create a df with the timestamp of each volume
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)
    
    first_vol = volume_acq_starts[0]
    
    return flow_df, vis_df, volume_df, first_vol


def get_uniflow_params(series):
    return (series.str.split(" ").str[0].str[1:]).astype("float64")
    
    

def visual_flow(log_file, log_df, matrix_data):
    print("reading visual flow log")

    # angle of visual flow
    visual_angle_map = {0 : 270, 
                    0.7853982 : 225,
                    1.570796 : 180,
                    2.356194 : 135,
                    3.141593 : 90,
                    3.926991 : 45,
                    4.712389 : 0,
                    5.497787 : 315
                    }
    
    # def read_motion_log
    flow_angle_map = {0 :  0,
                      1:   180}
    
    flow_df, visual_df, volume_df, first_vol = read_visual_flow_log(log_df, matrix_data, flow_angle_map, visual_angle_map)
    
    volume_df = map_angle_vel(volume_df, flow_df)
    volume_df = map_angle_vel(volume_df, flow_df, "visual_angle", "visual_vel")
    
    volume_df["stim"] = ["visualflow"] * volume_df.shape[0]
    
    
    return volume_df, flow_df, first_vol


def do_regression(dff_tidy, core_regressor_df, intercept = False, min_max = (0, 1.8)):
    print("Regressing data against regressors")
    core_regression_df = pd.DataFrame()
    
    
    for stim in core_regressor_df.stim.unique():

        dff_df = dff_tidy[dff_tidy.stim == stim].pivot(columns = "cell_id", index = "nVol", values = "smooth_dff")
        regressors = core_regressor_df[core_regressor_df.stim == stim].pivot(columns = "regressor", values = "value")
        if regressors.shape[0] > 0:
            if stim  == "visual":
                # normalised motor
                motor_cols = [col for col in regressors.columns if "motor" in col]
                non_motor_cols = [col for col in regressors.columns if "motor" not in col]

                # get max value
                #max_val = regressors.loc[:, non_motor_cols].max().max()

                scaler = MinMaxScaler(feature_range  = min_max)

                # normalise the motor to the stimuli range
                regressors.loc[:, motor_cols] = scaler.fit_transform(regressors.loc[:, motor_cols].to_numpy())

                # for some reason the ouwith and within motor are giving weird results
                
            if stim == "OMR":
                scaler = RobustScaler(with_centering = False, quantile_range = min_max)

                # normalise the motor to the stimuli range
                #regressors.loc[:] = scaler.fit_transform(regressors.to_numpy())
                #minmax_scaler = MinMaxScaler()
                #regressors.loc[:] = minmax_scaler.fit_transform(regressors.to_numpy())


            # perform regression
            dff_subset = dff_df.T.to_numpy()
            weights, betas, intercepts, model = regression(regressors.to_numpy().T, dff_subset, intercept = intercept)

            regression_df = pd.DataFrame(weights[1:]).T
            regression_df.columns = regressors.columns

            # melt dataframe
            tidy_regression = regression_df.reset_index().melt(var_name = "regressor", value_name = "weight", id_vars = "index")

            # add coefficients
            coefficient_df = pd.DataFrame(betas.T[1:]).T
            coefficient_df.columns = regressors.columns
            tidy_coefficient = coefficient_df.reset_index().melt(var_name = "regressor", value_name = "beta", id_vars = "index")
            tidy_regression["beta"] = tidy_coefficient.beta


            # get t scores
            t_score_df = get_tscores(regressors, dff_subset, betas, regression_df)

            # get selective cells - 95 th percentile in weight and t score
            #selective_df = get_selective_cells(regression_df, t_score_df)



            # melt and concat to core df
            tidy_regression["tscore"] = t_score_df.melt(ignore_index = False).value.to_numpy()

            # create empty selective column for later
            tidy_regression["selective"]  = [False] * tidy_regression.shape[0]

            tidy_regression["stim"] = [stim] * tidy_regression.shape[0]

            core_regression_df = pd.concat([core_regression_df, tidy_regression])
            
            
            
    core_regression_df.columns = ["cell_id", "regressor", "weight", "beta",  "tscore", "selective", "stim"]

    return core_regression_df

def fx(x, beta, alpha):
    
    y = alpha + beta*x
    
    return y

def get_tscores(regressors, dff_subset, betas, regression_df):
    
    t_scores = np.zeros((regressors.shape[1], dff_subset.shape[0]))
    
    for regressor in range(regressors.shape[1]):
       
        # loop through each cell
        for cell in range(dff_subset.shape[0]):

            # use beta to get estimated y
            y_hat = fx(dff_subset[cell], betas[cell, regressor+1], 0)


            # calculate standard error for each regression
            n = dff_subset.shape[1]
            SE = np.sqrt((((y_hat - dff_subset[cell])**2).sum()/(n-2)))


            # use standard error to get t statistic - null hypothesis is B0 as 0 - slope of 0.
            t = (betas[cell, regressor+1] - 0)/ SE

            t_scores[regressor, cell] = t
            
    t_score_df = pd.DataFrame(t_scores).T
    t_score_df.columns = regression_df.columns
    return t_score_df

def regression(regressors, F_dff, intercept = False):
    try:
        algorithm = LinearRegression(fit_intercept=intercept)
    except:
        print("Defaulting to custom regression model from sklearn")
        from sklearn.linear_model import LinearRegression as LR
        algorithm = CustomRegression(LR(fit_intercept=intercept))
    model = algorithm.fit(regressors.T, F_dff)
    betas = model.betas.toarray()
    rsq = model.score(regressors.T, F_dff).toarray()
    intercepts = model.intercept_
    #calculate weights, defined as the product of the Rsquared * Betas
    weights = betas.T * rsq.T
    #weights = weights[1:]
    
    return weights, betas, intercepts, model


def plot_multisensory_regression_maps(regression_sub, con_df, coms_df, dims, h5_file = None, log = False,  cmap = plt.get_cmap("inferno"), cmin = None, cmax = None, save = False):
    
    contours_maps = {}
    centers_maps = {}
    stim = regression_sub.stim.unique()[0]
    for ax_idx, regressor in enumerate(regression_sub.regressor.unique()):
        plot_regressor_map(regressor = regressor, regression_sub = regression_sub, con_df = con_df, coms_df = coms_df, dims = dims, h5_file = h5_file, log = log,  cmap = cmap, cmin = cmin, cmax = cmax, save = save)
        
def plot_regressor_map(regressor, regression_sub, con_df, coms_df, dims, h5_file = None, log = False,  cmap = plt.get_cmap("inferno"), cmin = None, cmax = None, save = False):
        fig, ax = plt.subplots( figsize = (20, 10), sharex = True, sharey = True)

        # get regressors weights
        regressor_sub = regression_sub[regression_sub.regressor == regressor].weight

        # create contour image mask
        contour_mask = np.zeros((5, dims[1], dims[0]))
        center_mask =  np.zeros((5, dims[1], dims[0]))
        # define colors in 8 bit range - normalise to -1 and 1
        #colors = ((((regressor_sub-cmin)/(cmax - cmin)) * 2)-1).to_numpy()
        colors = regressor_sub.to_numpy() # leave raw

        if log == True:
            regressor_subset = np.log10(regressor_subset)
        patches = []

        # loop through and get cell contours
        for cell in con_df.cell_id.unique():
                contour = con_df.loc[con_df.cell_id == cell, ["x", "y"]].to_numpy()
                p = Polygon(contour, closed=True, edgecolor = "white", linewidth = 0.2)
                # append patch to collection
                patches.append(p)
                
                color = colors[cell]
                
                # get z, color and fill polygon in image mask
                z, y, x  = coms_df.loc[coms_df.cell_id == cell, ["z", "y", "x"]].to_numpy().astype("int64")[0]

                #  fill center mask pixels with color
                center_mask[z, y-2:y+2, x-2: x+2] = color

                #uset z, color and fill polygon in image mask
                
                patch = contour[~np.isnan(contour)]
                patch = patch.reshape(-1, 2).astype("int32")
                if patch.shape[0] > 0:
                    cv2.fillPoly(contour_mask[z], [patch], color)

        
       

        ax.imshow(np.zeros((dims[1], dims[0])), cmap = "gray")
        collection = PatchCollection(patches, cmap =cmap, alpha=0.3)
        collection.set_array(colors) # sets color
        collection.set_clim(cmin, cmax)
        collection.set_edgecolors("gray")
        collection.set_linewidth(1)


        ax.add_collection(collection)


        ax.set_title(str(regressor))
        sns.despine()
        
        #save figure
        if save:
            out_dir = os.path.join(denoised_folder, "regression_figures")
            if os.path.exists(out_dir):
                pass
            else:
                os.mkdir(out_dir)

                # save mask array to dict or something
                utils.save_image_to_h5(contour_mask, h5_file, "{}_{}_mono_contour_False".format(stim, regressor), stim, regressor, "mono", False, "contours", "regression")
                utils.save_image_to_h5(center_mask, h5_file, "{}_{}_mono_center_False".format(stim, regressor), stim, regressor, "mono", False, "centers", "regression")


                fig.savefig(os.path.join(out_dir, "{} stim, {} regressor.svg".format(stim, regressor)))
                
        return contour_mask, center_mask
                
                

def plot_multisensory_regression_selectivity(regression_df, con_df, coms_df, dims, colors, h5_file=None, save = False):
    # average directionality map for top 5%

    # loop through regressors
    # add contour to 
    contour_maps = {}
    center_maps = {}

    # loop through and get cell contours
    # plot to selective regression map
    for stim in regression_df.stim.unique():

        regressor_sub = regression_df[(regression_df.stim ==stim) & (~regression_df.regressor.str.contains("motor")) & (~regression_df.regressor.str.contains("all"))& (~regression_df.regressor.str.contains("any"))]
        regressor_select = regressor_sub[regressor_sub.selective == True]
        non_selective_cells = regressor_sub[~regressor_sub.cell_id.isin(regressor_select.cell_id.unique())].cell_id.unique()
        if regressor_sub.shape[0] > 0:

            # loop through regressors - assign circular color to each direction regressor
            contour_map = []
            center_map = []
            # define colors in 8 bit range 
            for nregressor, regressor in enumerate(regressor_sub.regressor.astype("float64").sort_values().unique()):
                contour_mask = np.zeros((5, dims[1], dims[0], 3))
                center_mask = np.zeros((5, dims[1], dims[0], 3))

                selective_cells = regressor_select[regressor_select.regressor == str(regressor)].cell_id

                if selective_cells.shape[0] > 0:
                    color = colors[nregressor]
                    # loop through and get cell contours
                    for cell in selective_cells:
                            contour = con_df.loc[con_df.cell_id == cell, ["x", "y"]].to_numpy()

                            # get z, color and fill polygon in image mask
                            z, y, x  = coms_df.loc[coms_df.cell_id == cell, ["z", "y", "x"]].to_numpy().astype("int64")[0]

                            #  fill center mask pixels with color
                            center_mask[z, y-2:y+2, x-2: x+2] = color


                            patch = contour[~np.isnan(contour)]
                            patch = patch.reshape(-1, 2).astype("int32")
                            if patch.shape[0] > 0:
                                cv2.fillPoly(contour_mask[z], [patch], color)

                contour_map.append(contour_mask)
                center_map.append(center_mask)


            # add a grayscale image mask for non-selective cells
            grayscale_mask = np.zeros((5, dims[1], dims[0], 3))
            for cell in non_selective_cells:

                contour = con_df.loc[con_df.cell_id == cell, ["x", "y"]].to_numpy()
                color = (0.5, 0.5, 0.5)
                 # get z, color and fill polygon in image mask
                z = int(coms_df[coms_df.cell_id == cell].z)
                patch = contour[~np.isnan(contour)]
                patch = patch.reshape(-1, 2).astype("int32")
                if patch.shape[0] > 0:
                    cv2.fillPoly(grayscale_mask[z], [patch], color)
            contour_map.append(grayscale_mask)

            # stack 
            contour_map = np.stack(contour_map)
            center_map = np.stack(center_map)
            
            if save:
                utils.save_image_to_h5(contour_map, h5_file, "{}_{}_rgb_contour_False".format(stim, regressor), stim, regressor, "rgb", False, "contours", "regression")
                utils.save_image_to_h5(center_map, h5_file, "{}_{}_rgb_center_False".format(stim, regressor), stim, regressor, "rgb", False, "centers", "regression")
            #contour_maps[stim] = contour_map
            #center_maps[stim]  = center_map
        
    return contour_map, center_map

def plot_rgb_overlay(z_mip, zbrain_ref_max, zbrain_ref_max_x, x_mip, alpha = 0.5):
    fig = plt.figure(figsize = (15, 10))
    gs = fig.add_gridspec(2,2, width_ratios = [1, 2], height_ratios = [1, 2])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    im = ax1.imshow(z_mip)
    
    ax1.imshow(zbrain_ref_max, alpha = alpha)
    ax1.axis('off')


    ax2.imshow(x_mip, aspect = 30, extent=[0,100,0,1])
    ax2.imshow(zbrain_ref_max_x, alpha = alpha, aspect = 30, extent=[0,100,0,1])
    ax2.axis('off')


    xmin, xmax = 100, 500
    ymin, ymax = 800, 1100
    ax3.imshow(z_mip[ymin:ymax, xmin:xmax])
    ax3.axis('off')

    ax3.imshow(zbrain_ref_max[ymin:ymax, xmin:xmax], alpha = alpha)

    fig.tight_layout()
    return fig

def plot_overlay(z_mip, zbrain_ref_max, zbrain_ref_max_x, x_mip, vmax = 0.0005, alpha = 0.5):
    fig = plt.figure(figsize = (15, 10))
    gs = fig.add_gridspec(2,2, width_ratios = [1, 2], height_ratios = [1, 2])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    im = ax1.imshow(z_mip, plt.get_cmap("inferno"), vmax = vmax)
    ax1.axis('off')
    plt.colorbar(im)
    ax1.imshow(zbrain_ref_max, alpha = alpha)



    ax2.imshow(x_mip, plt.get_cmap("inferno"), vmax = vmax, aspect = 30, extent=[0,100,0,1])
    ax2.imshow(zbrain_ref_max_x, alpha = alpha, aspect = 30, extent=[0,100,0,1])
    ax2.axis('off')


    xmin, xmax = 100, 500
    ymin, ymax = 800, 1100
    ax3.imshow(z_mip[ymin:ymax, xmin:xmax], plt.get_cmap("inferno"), vmax = vmax)
    ax3.axis('off')

    ax3.imshow(zbrain_ref_max[ymin:ymax, xmin:xmax], alpha = alpha)

    fig.tight_layout()
    return fig

        
def plot_rgb_direction(dir_pref_sub, con_df, coms_df, dims, h5_file = None, save = False):
    
    colors = sns.husl_palette(8)
    bins = np.arange(0, 360, 45)
    bins = [(b, b+45) for b in bins]
    bins
    
    stim = dir_pref_sub.stim.unique()[0]
    dynamic = dir_pref_sub.dynamic.unique()[0]
    velocity = dir_pref_sub.velocity.unique()[0]
    
    # create image mask
    contour_map = []
    center_map = [] 
    # define colors in 8 bit range 
    for b_idx, b in enumerate(bins):
        contour_mask = np.zeros((5, dims[1], dims[0], 3))
        center_mask = np.zeros((5, dims[1], dims[0], 3))
        selective_cells = dir_pref_sub[(dir_pref_sub.preferred_dir >= b[0]) & 
                                       (dir_pref_sub.preferred_dir <= b[1]) &
                                      (dir_pref_sub.selective == True)].cell_id.sort_values().unique()

        if selective_cells.shape[0] > 0:
            
            color = colors[b_idx]
            # loop through and get cell contours
            for cell in selective_cells:
                
                
                    contour = con_df.loc[con_df.cell_id == cell, ["x", "y"]].to_numpy()
                    

                    # get z, color and fill polygon in image mask
                    try:
                         # get z, color and fill polygon in image mask
                        z, y, x  = coms_df.loc[coms_df.cell_id == cell, ["z", "y", "x"]].to_numpy().astype("int64")[0]

                        #  fill center mask pixels with color
                        center_mask[z, y-2:y+2, x-2: x+2] = color
                    except:
                        print("z info missing for cell {}".format(cell))
                        z, y, x = 0., 0., 0.
                    patch = contour[~np.isnan(contour)]
                    patch = patch.reshape(-1, 2).astype("int32")
                    if patch.shape[0] > 0:
                        cv2.fillPoly(contour_mask[z], [patch], color)
                        
            contour_map.append(contour_mask)
            center_map.append(center_mask)
        

    
            # add a grayscale image mask for non-selective cells
            non_selective_cells = con_df[~con_df.cell_id.isin(selective_cells)].cell_id.sort_values().unique()
            grayscale_mask = np.zeros((5, dims[1], dims[0], 3))
            for cell in non_selective_cells:

                contour = con_df.loc[con_df.cell_id == cell, ["x", "y"]].to_numpy()
                color = (0.5, 0.5, 0.5)
                 # get z, color and fill polygon in image mask
                try:
                    z = int(coms_df[coms_df.cell_id == cell].z)
                except:
                    z = 0
                    print("missing coms info for non selective cell {}".format(cell))
                patch = contour[~np.isnan(contour)]
                patch = patch.reshape(-1, 2).astype("int32")
                if patch.shape[0] > 0:
                    cv2.fillPoly(grayscale_mask[z], [patch], color)
            contour_map.append(grayscale_mask)
        else:
            contour_map.append(contour_mask)
            center_map.append(center_mask)
            print("no selective cells")
            
    # stack 
    contour_map = np.stack(contour_map)
    center_map = np.stack(center_map)
    

    if save:
        utils.save_image_to_h5(contour_map, h5_file, "{}_{}_{}_{}_rgb_contour_vector_False".format(stim, b[0], dynamic, velocity), stim, str(bins[0]), "rgb", False, "contours", "vector")
        utils.save_image_to_h5(center_map, h5_file, "{}_{}_{}_{}_rgb_center_vector_False".format(stim, b[0], dynamic, velocity), stim, str(bins[0]), "rgb", False, "centers", "vector")
        
    return contour_map, center_map



def blend_rgb_stacks(fish_sum, kernel = (5, 3, 3)):
    colors = sns.husl_palette(8, s = 0.99, l = 0.6, h =0)
    reg_stack = [] 
    reg_stack_x = []
    for channel in range(fish_sum.shape[0]):

        reg_smooth = gaussian_filter(fish_sum[channel], kernel)
        reg_max = reg_smooth.max(axis = 0)
        reg_max_x = reg_smooth.max(axis = 2)

        # reg max
        reg_max_rgb = reg_max.reshape(*reg_max.shape, 1)

        # add color channels
        r, g, b = colors[channel]
        h, s, v = list(r2h(*colors[channel]))

        # red channel 
        h_channel = np.zeros(reg_max_rgb.shape)
        reg_max_rgb_norm = (((reg_max_rgb - reg_max_rgb.min()) / (reg_max_rgb.max() - reg_max_rgb.min())) )# sacle between 0 and 0.3 then add 0.69

        h_channel[:]= h # hue

        # green channel
        s_channel = np.zeros(reg_max_rgb.shape)
        s_channel[:] = s #  saturation by 

        # blue channel
        v_channel = np.zeros(reg_max_rgb.shape)
        v_channel[:] = reg_max_rgb_norm # value

        reg_max_final = hsv2rgb(np.concatenate([h_channel, s_channel, v_channel], axis=2)) # convert back to rgb
        reg_stack.append(reg_max_final)


        # reg max x
        reg_max_rgb_x = reg_max_x.reshape(*reg_max_x.shape, 1)

        # red channel 
        h_channel = np.zeros(reg_max_rgb_x.shape)
        reg_max_rgb_norm_x = (((reg_max_rgb_x - reg_max_rgb_x.min()) / (reg_max_rgb_x.max() - reg_max_rgb_x.min())) )# sacle between 0 and 0.3 then add 0.69

        h_channel[:]= h # hue

        # green channel
        s_channel = np.zeros(reg_max_rgb_x.shape)
        s_channel[:] = s #  saturation by 

        # blue channel
        v_channel = np.zeros(reg_max_rgb_x.shape)
        v_channel[:] = reg_max_rgb_norm_x # value

        reg_max_final_x = hsv2rgb(np.concatenate([h_channel, s_channel, v_channel], axis=2)) # convert back to rgb
        reg_stack_x.append(reg_max_final_x)


        #reg_stack_mip_x.append(reg_max_final_x)

    final_stack = np.stack(reg_stack)
    final_stack_blended = np.sum(final_stack * (1/8), axis=0)
    # brighten                 
    normalised = (final_stack_blended/(final_stack_blended.max()))

    final_stack_x = np.stack(reg_stack_x)
    final_stack_blended_x = np.sum(final_stack_x * (1/8), axis=0)
    # brighten                 
    normalised_x = (final_stack_blended_x/(final_stack_blended_x.max()))
    
    return normalised, normalised_x

def register_image(root_dir, image, zbrain_ref):
    # lazy load zbrain image to get shape
    # lazy load overview image to get shape
    # load io_affine.npy
    # load affine_map.npy
    # load io_grid2world.npy
    # map io to overview
    # map transformed io to zbrain
    # save as 16 bit
    
    
    overview_dir = os.path.join(root_dir, "overview")
    denoised_dir = os.path.join(root_dir, "denoised")
    
    
    overview_files = os.listdir(overview_dir)
    overview_im_file = [os.path.join(overview_dir, file) for file in overview_files if "overview" in file][0]

    ref = np.rot90(np.rot90(imread(overview_im_file), axes = (1, 2)), axes = (1, 2))


    io_affine = np.load(os.path.join(denoised_dir, "io_affine.npy"))
    io_grid2world = np.load(os.path.join(denoised_dir, "io_grid2world.npy"))

    ref_grid2world = np.array([[ 1.,    0.,         0.,          0.],
                                 [0.,   0.01171,   0.,          0.],
                                 [0.,   0.,         0.01171,   0.],
                                 [0.,   0.,         0.,          1.]])



    print("Affine mapping to wholebrain overview")
    affine_map = AffineMap(io_affine,
                       ref.shape, ref_grid2world,
                       image.shape, io_grid2world)
    transformed = affine_map.transform(image)


    zbrain_grid2world = np.array([[2.,    0.,     0.,    0.],
                             [0.,     1.,    0.,    0.],
                             [0.,     0.,     1.,   0.],
                             [0.,     0.,     0.,   1.]])


    ref_grid2zbrain = np.array([[1.,   0.,      0.,    0.],
                                 [0.,     1.,     0.,    0.],
                                 [0.,     0.,     1.,    0.],
                                 [0.,     0.,     0.,     1.]]) 

    ref_affine = np.load(os.path.join(overview_dir, "affine_map.npy"))


    print("Affine mapping to zbrain reference")
    affine_map = AffineMap(ref_affine,
                           zbrain_ref.shape, zbrain_grid2world,
                           transformed.shape, ref_grid2zbrain)
    fully_registered = affine_map.transform(transformed)
    #except:
    #    print("Folder {} failed".format(root_dir))
    #    fully_registered = np.zeros(shape = zbrain_ref.shape, dtype = "float64")
    
        
    return fully_registered


def register_rgb_stack(root_dir, rgb_stack, zbrain_ref):
    #rgb stack shape should be C x Z x Y x X x(RBG)
    registered_rgb_stack = []
    for channel in range(rgb_stack.shape[0]):
        reg_channel = register_image(root_dir, rgb_stack[channel, :, :, :, 0], zbrain_ref)
        registered_rgb_stack.append(reg_channel)

    registered_rgb_stack = np.stack(registered_rgb_stack)
    
    return registered_rgb_stack

def parallel_register_rgb_stack(root_dir, image, zbrain_ref):
    # format image shape before entry into main function
    if len(image.shape) > 3:
        image = image.reshape(*image.shape[1:])
        
    
    fully_registered = register_image(root_dir, image, zbrain_ref)
     #format image shape on return of image function
    
    return fully_registered.reshape(1, *fully_registered.shape)

def parallel_register_mono_stack(root_dir, image, zbrain_ref):
    # format image shape before entry into main function
    if len(image.shape) > 3:
        image = image.reshape(*image.shape[1:])
        
    
    fully_registered = register_image(root_dir, image, zbrain_ref)
     #format image shape on return of image function
    
    return fully_registered.reshape(1, *fully_registered.shape)
    
def create_unique_cell_id(df):
    
    unique_entries = df.drop_duplicates(["cell_id", "fish"])
    unique_entries["unique_cellid"] = np.arange(unique_entries.shape[0])
    unique_cell_map = unique_entries.set_index(["cell_id", "fish"]).unique_cellid.to_dict()
    df.set_index(["cell_id", "fish"], inplace = True)
    df["unique_cellid"] = df.index.map(unique_cell_map)
    df.reset_index(inplace=True)
    return df, unique_cell_map


def map_unique_id(df, unique_cell_map):
    df.set_index(["cell_id", "fish"], inplace = True)
    df["unique_cellid"] = df.index.map(unique_cell_map)
    df.reset_index(inplace=True)
    return df


def zero_trace(series):
    zeroed_trace = series-series.iloc[0]
    return zeroed_trace


def interpolate(old_length, new_length, start, end, data_to_interp):
    
    old_x = np.linspace(start, end, old_length)
    new_x = np.linspace(start, end, new_length)
    
    f = interp1d(old_x, data_to_interp, kind = "previous")(new_x)
    return f, old_x, new_x

def read_OMR_log(log_df, matrix_data):

    # get first analogread and subtract by delay in reading the first buffer of data
    #print(type(log_df))
    #print(log_df['Value'].unique())

    #since adding a new colum called replay, Value is no longer last column so use this to extract buffersize.
    buffer_size = int(log_df[log_df.Value.str.contains("AnalogRead")]['Value'].iloc[0].split(", ")[1].split("=")[-1])
    
    #buffer_size = int(log_df[log_df.Value.str.contains("AnalogRead")].iloc[0, -1].split(", ")[1].split("=")[-1])
    print("buffer size is {}".format(buffer_size))
    
    # get asig start and end
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (buffer_size/6000) # buffer is {}, sr is 6000 therefore subtract buffer_size/sample_rate
    asig_t_end = asig_t_start + (matrix_data.shape[0]/6000) # use matrix data shape - more accurate as initial OMR experiments buffered 1 second before saving which caused loss of last second when file ended

    
    gain = get_OMR_data(log_df[log_df.Value.str.contains("Gain") & 
                               (~log_df.Value.str.contains("Power"))].drop_duplicates(
                                "Timestamp",keep="last").Value)


    # change to vis started - loose definition - assert than vis_times length == radian length
    vis_times  = log_df[log_df.Value.str.contains("VisStarted")].Timestamp

    if vis_times.shape[0] ==0: # no log event for start of vis - use gain change timestamp instead as vis stim continuously moving
        vis_times = log_df[log_df.Value.str.contains("Gain")].Timestamp
    

    velocity = get_OMR_data(log_df[log_df.Value.str.contains("SceneVel")].drop_duplicates(keep = "last").Value)
    
    

    if vis_times.shape[0] != gain.shape[0]:
        # use gain changes as vis times
        vis_times = log_df[log_df.Value.str.contains("Gain") & 
                               (~log_df.Value.str.contains("Power"))].drop_duplicates(
                                "Timestamp",keep="last").Timestamp


    vis_df = pd.DataFrame([vis_times.to_numpy(), gain, velocity]).T
    vis_df.columns = ["start", "gain", "velocity"]
    #vis_df["angle"] = vis_df.radians.map(visual_angle_map)
    
    if (vis_df.isnull().any()).all():
        print("null values present")
    vis_df.dropna(inplace  =True)

    # get volume_clock - use this create a df with the timestamp of each volume
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)
    
    
    # define end of stim
    
    ends = vis_df.start[1:].tolist()
    ends.append(volume_df.dropna().ends.iloc[-1])
    ends = pd.Series(ends)
    vis_df["end"] = ends
    

    first_vol = volume_acq_starts[0]
    first_vol_t = volume_acq_starts_t[0]
    
    feedback_vel = log_df[log_df.Value.str.contains("Velocity")]
    
    #OMR replay
   # if feedback_vel.shape[0] ==0:
    #    feedback_vel = log_df[log_df.Value.str.contains("Bonsai")
        #feedback_vel.Value == 1 #maybe include a value for each one. so .Value = 1, 2, 3 etc.
                              
    #feedback_vel.Value = get_OMR_data(feedback_vel.Value).to_numpy()
    #better to use this:
    feedback_vel.loc[:, 'Value'] = get_OMR_data(feedback_vel['Value']).to_numpy()

    #OMR replay
    #else:
     #   feedback_vel.Value == 1                          

    
    
    # align feedback vel with matrix data and concat after function
    
   
    # get sampling interval and create timebase for ephys
    sampling_interval  = 1/6000
    asig_x = np.arange(asig_t_start, asig_t_end, sampling_interval)

    # get datapoint where vel starts and stops in ephys timebase
    vel_start_t = feedback_vel.Timestamp.iloc[0]
    vel_end_t = feedback_vel.Timestamp.iloc[-1]

    vel_start = np.where(asig_x > vel_start_t)[0][0] # get first asig read after vel start - precision will change if buffer size increases
    vel_stop = np.where(asig_x <= vel_end_t)[0][-1]
    
    # add correction - trace was ~34ms out
    correct_dp = int((1/30) * 6000)
    vel_start = vel_start + correct_dp
    vel_stop = vel_stop + correct_dp

    # create empty array as same shape as matrix data
    vel_matrix = np.zeros(matrix_data.shape[0])

    # define new length
    new_length = asig_x[vel_start:vel_stop].shape[0]
    old_length = feedback_vel.Value.shape[0]

    interp_vel, old_x, new_x = interpolate(old_length, new_length, vel_start_t, vel_end_t, feedback_vel.Value)
    try:
        vel_matrix[vel_start:vel_stop] = interp_vel
    except:
        vel_matrix[vel_start:vel_stop] = interp_vel[:-1]
    # add some extra columns
    vis_df["duration"] = vis_df.end - vis_df.start
    vis_df["adaptation"] = ["short"] * vis_df.shape[0]
    
    vis_df.loc[vis_df.duration > 30, "adaptation"] = "long"                         
    #vis_df.loc[vis_df.duration > 16, "adaptation"] = "long"
    
    
    return vis_df, volume_df, first_vol, vel_matrix

                              

def read_OMR_replay_log(log_df, matrix_data):

    vis_df = pd.DataFrame(columns=["start", "gain", "velocity", "replay"]).T
    volume_df = pd.DataFrame(columns=["starts", "ends", "nVol"])
    first_vol = None
    vel_matrix = np.zeros(matrix_data.shape[0])

    # get first analogread and subtract by delay in reading the first buffer of data
    buffer_size = int(log_df[log_df.Value.str.contains("AnalogRead")].iloc[0, -1].split(", ")[1].split("=")[-1])
    print("buffer size is {}".format(buffer_size))
    
    # get asig start and end
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (buffer_size/6000) # buffer is {}, sr is 6000 therefore subtract buffer_size/sample_rate
    asig_t_end = asig_t_start + (matrix_data.shape[0]/6000) # use matrix data shape - more accurate as initial OMR experiments buffered 1 second before saving which caused loss of last second when file ended

    print("Unique values in log_df.Value:", log_df.Value.unique())
    print("Condition 1: 'Gain' in log_df.Value.unique():", "Gain" in log_df.Value.unique())
    print("Condition 2: 'SceneVel' in log_df.Value.unique():", "SceneVel" in log_df.Value.unique()) 
    print("Number of missing values in log_df.Value:", log_df.Value.isnull().sum())

      #  velocity = np.zeros(matrix_data.shape[0]) * np.nan
    #if ("Gain" in log_df.Value.any()) and ("SceneVel" in log_df.Value.any()):
    if log_df.Value.str.contains("Gain").any() and log_df.Value.str.contains("SceneVel").any():

        gain = get_OMR_data(log_df[log_df.Value.str.contains("Gain") & (~log_df.Value.str.contains("Power"))].drop_duplicates("Timestamp", keep="last").Value)
        velocity = get_OMR_data(log_df[log_df.Value.str.contains("SceneVel")].drop_duplicates(keep="last").Value)
    else:
        print("Using np.nan for gain and velocity")
        gain = np.zeros(matrix_data.shape[0]) * np.nan
        velocity = np.zeros(matrix_data.shape[0]) * np.nan
        

    
    vis_times = log_df[log_df.Value.str.contains("VisStarted")].Timestamp
    if vis_times.shape[0] == 0:
        vis_df["replay"] = True
    else:
        vis_df["replay"] = False


    if log_df.Value.str.contains("VisStarted").any():
        vis_df["start"] = vis_times
    elif log_df.Value.str.contains("Gain").any():
        vis_df["start"] = log_df[log_df.Value.str.contains("Gain") & (~log_df.Value.str.contains("Power"))].drop_duplicates("Timestamp", keep="last").Timestamp
    else:
        print("vis started is np.nan")
        vis_df["start"] = np.nan

    vis_df["gain"] = gain
    vis_df["velocity"] = velocity
    
    #vis_df = pd.DataFrame([vis_times.to_numpy(), gain, velocity]).T
    #vis_df.columns = ["start", "gain", "velocity"]
    
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)

    ends = vis_df.start[1:].tolist()
    ends.append(volume_df.dropna().ends.iloc[-1])
    ends = pd.Series(ends)
    vis_df["end"] = ends

    first_vol = volume_acq_starts[0] if len(volume_acq_starts) > 0 else None

    feedback_vel = None

    if log_df.Value.str.contains("Velocity").any():
        feedback_vel = log_df.loc[log_df.Value.str.contains("Velocity")].copy()

    if feedback_vel is not None and feedback_vel.shape[0] > 0:
        feedback_vel.Value = get_OMR_data(feedback_vel.Value).to_numpy()
        vel_start_t = feedback_vel.Timestamp.iloc[0]
        vel_end_t = feedback_vel.Timestamp.iloc[-1]
        sampling_interval = 1 / 6000
        asig_x = np.arange(asig_t_start, asig_t_end, sampling_interval)
        vel_start = np.where(asig_x > vel_start_t)[0][0]
        vel_stop = np.where(asig_x <= vel_end_t)[0][-1]
        correct_dp = int((1 / 30) * 6000)
        vel_start = vel_start + correct_dp
        vel_stop = vel_stop + correct_dp
        new_length = asig_x[vel_start:vel_stop].shape[0]
        old_length = feedback_vel.Value.shape[0]
        interp_vel, old_x, new_x = interpolate(old_length, new_length, vel_start_t, vel_end_t, feedback_vel.Value)
        try:
            vel_matrix[vel_start:vel_stop] = interp_vel
        except:
            vel_matrix[vel_start:vel_stop] = interp_vel[:-1]
    else:
        vel_start_t = np.nan
        vel_end_t = np.nan
        vel_start = 0
        vel_stop = 0
        interp_vel = np.zeros(matrix_data.shape[0]) * np.nan
        old_x = np.zeros(0)
        new_x = np.zeros(0)

    vis_df["duration"] = vis_df.end - vis_df.start
    vis_df["adaptation"] = "short"
    # vis_df.loc[vis_df.duration > 16, "adaptation"] = "long"

    return vis_df, volume_df, first_vol, vel_matrix


                              
def OMR(log_file, log_df, matrix_data):
    print("reading OMR log")

    
    
    visual_df, volume_df, first_vol, vel_matrix = read_OMR_log(log_df, matrix_data)
    
    volume_df = map_angle_vel(volume_df, visual_df, "gain", "velocity")
    volume_df = map_angle_vel(volume_df, visual_df, "duration", "adaptation")
    volume_df["stim"] = ["OMR"] * volume_df.shape[0]
    
    
    
    return volume_df, visual_df, first_vol, vel_matrix



def OMR_replay(log_file, log_df, matrix_data):
    print("reading OMR log")

    # If 'replay' column doesn't exist, create it and fill with False
    if 'replay' not in log_df.columns:
        log_df['replay'] = False

    visual_df, volume_df, first_vol, vel_matrix = read_OMR_log(log_df, matrix_data)

    # Ensure 'replay' column is included in the new dataframes
    volume_df['replay'] = log_df['replay']
    visual_df['replay'] = log_df['replay']

    volume_df = map_angle_vel(volume_df, visual_df, "gain", "velocity")
    volume_df = map_angle_vel(volume_df, visual_df, "duration", "adaptation")
    volume_df["stim"] = ["OMR"] * volume_df.shape[0]

    return volume_df, visual_df, first_vol, vel_matrix


#def OMR_replay(log_file, log_df, matrix_data):
 #   print("reading OMR log")

    
    
  #  visual_df, volume_df, first_vol, vel_matrix = read_OMR_replay_log(log_df, matrix_data)
    
   # volume_df = map_angle_vel2(volume_df, visual_df, "gain", "velocity")
    #volume_df = map_angle_vel2(volume_df, visual_df, "duration", "adaptation")
    #volume_df["stim"] = ["OMR"] * volume_df.shape[0]
    
    
    
    #return volume_df, visual_df, first_vol, vel_matrix

def return_series(series):
    return pd.Series(np.arange(series.shape[0])) 


from skimage.registration import phase_cross_correlation
from skimage.color import rgb2gray
from skimage.transform import rescale


def fast_motion_estimation(vid_da, log_df, skip = 8):
    overlap = da.overlap.overlap(vid_da[::skip], depth = 1, boundary = {0:0, 1: "none", 2:"none", 3:"none"})
    shifts = overlap.map_blocks(sklearn_phase_corr, chunks = (1, 1, 1, 2), dtype = np.uint8).compute()
    
    # get angles from shifts
    rads = np.arctan2(shifts[:, :, :, 1], shifts[:, :, :, 0]).flatten() *-1
    rads[rads < 0] = (2*np.pi) + rads[rads<0]
    
    degrees = np.degrees(rads)
    
    nframes = vid_da.shape[0]
    full_arr = np.zeros(nframes)
    full_arr[:] = np.nan
    full_arr[::skip] = degrees
    
    # forward fill na
    full_arr = pd.Series(full_arr).fillna(method = "ffill")
    
    shader_frame_clock = log_df[log_df.Value.str.contains("Shaders")].Timestamp
    # sometimes frames don't fully match for some reason - is videowriter saving extra frames or 
    full_arr = full_arr[:len(shader_frame_clock)]
    full_arr.index = shader_frame_clock
    
    return full_arr
    

def sklearn_phase_corr(overlap):
    # get motion shifts on grayscale, downscaled frames for speed
    shift, _, _  = phase_cross_correlation(rescale(rgb2gray(overlap[1]), 0.25, anti_aliasing = True), rescale(rgb2gray(overlap[2]), 0.25, anti_aliasing = True))

    return shift.reshape((1, 1, 1, 2))



def dots(log_file, log_df, matrix_data, duration = 10):
    # read visual log file
    vis_df, volume_df, first_vol = read_dot_log(log_df, matrix_data, duration)

    # map angle and vel to volume
    volume_df = map_angle_vel(volume_df, vis_df)
    volume_df = map_angle_vel(volume_df, vis_df, "dot_size", "coherence")

    
    volume_df["stim"] = ["dots"] * volume_df.shape[0]
    
    return volume_df, vis_df, first_vol
    


def read_dot_log(log_df, matrix_data, vis_duration = 10):
    
    print("reading dot log")
    
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (1/6) # buffer is 1000, sr is 6000 therefore subtract 1/6 seconds
    
    radians = np.radians(get_stage_move(log_df[log_df.Value.str.contains("degrees")].Value))
    vis_times  = log_df[log_df.Value.str.contains("degrees")].Timestamp
    velocity = get_OMR_data(log_df[log_df.Value.str.contains("SceneVel")].Value)
    dot_size = get_OMR_data(log_df[log_df.Value.str.contains("DotSize")].Value)
    coherence = get_OMR_data(log_df[log_df.Value.str.contains("Coherence")].Value)
    vis_df = pd.DataFrame([vis_times.to_numpy(), radians, velocity, dot_size, coherence]).T
    vis_df.columns = ["start", "radians", "velocity", "dot_size", "coherence"]
    vis_df["angle"] = np.degrees(vis_df.radians)
    
    
    # to add
    # if VisStopped not in ...:
    vis_df["end"] = vis_df.start + vis_duration # this is a bit arbitrary - better to get from log file when stim stops
    #else: vis_df['end'] =  log_df[log_df.Value.str.contains("VisStopped")].Timestamp
    
    
    # get volume_clock - use this create a df with the timestamp of each volume
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)
    
    first_vol = volume_acq_starts[0]
    
    
    
    return vis_df, volume_df, first_vol
    

def natural_scene(log_file, log_df, matrix_data, duration = 10):
    # initial experiments missing angle info but maybe we could extract image statistics and flow from video?
    
    vis_df, volume_df, first_vol = read_natural_scene_log(log_file, log_df, matrix_data, duration)
    
    # map angle and vel to volume
    volume_df = map_angle_vel(volume_df, vis_df)
    
    volume_df["stim"] = ["natural"] * volume_df.shape[0]
    
    return volume_df, vis_df, first_vol
    

def read_natural_scene_log(log_file, log_df, matrix_data, vis_duration = 10):
    print("reading natural scene log")
    
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (1/6) # buffer is 1000, sr is 6000 therefore subtract 1/6 seconds

    vis_times  = log_df[log_df.Value.str.contains("VisStarted")].Timestamp

    if "degrees" in log_df.Value.unique():
        radians = np.radians(get_stage_move(log_df[log_df.Value.str.contains("degrees")].Value))
    else:
        # estimated with motion
        # get video file
        vid_file = get_any_associated_files(log_file, 'avi')
        # load into dask
        from dask_image import imread
        vid_da = imread.imread(vid_file)

        # calcl motion direction
        degrees = fast_motion_estimation(vid_da, log_df, skip = 8)
        
        # get direction at time during vis moving
        radians = vis_times.copy()
        for ntime, vis_time in enumerate(vis_times.to_numpy()):
            end = vis_time + vis_duration
            # subset stable part of motion
            motion_subset = degrees[(degrees.index >= vis_time+2) & (degrees.index <end-2)]
            mean_direction  = motion_subset.mean()

            radians.iloc[ntime] = np.radians(mean_direction)
            
        radians = radians.to_numpy()
        
        
    if "SceneVel" in log_df.Value.unique():
        velocity = get_OMR_data(log_df[log_df.Value.str.contains("SceneVel")].Value)
    else:
        velocity = vis_times.copy()
        velocity.iloc[:] = 0.2
        velocity = velocity.to_numpy()
        
    
    vis_df = pd.DataFrame([vis_times.to_numpy(), radians, velocity]).T
    vis_df.columns = ["start", "radians", "velocity"]
    vis_df["angle"] = np.degrees(vis_df.radians)
    
    
    # to add
    # if VisStopped not in ...:
    vis_df["end"] = vis_df.start + vis_duration # this is a bit arbitrary - better to get from log file when stim stops
    #else: vis_df['end'] =  log_df[log_df.Value.str.contains("VisStopped")].Timestamp
    
    
    # get volume_clock - use this create a df with the timestamp of each volume
    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)
    
    first_vol = volume_acq_starts[0]
    
    
    
    return vis_df, volume_df, first_vol
    

def align_to_bonsai(im_fname, fr = 3.07, sr  = 6e3):
   
        
    folder = os.path.dirname(im_fname)
        
    # get log file and matrix file
    log_file = get_any_associated_files(im_fname, "log")
    matrix_file = get_any_associated_files(im_fname, "NI")
        
        
        
    # read log file
    log_df = read_bonsai_log_file(log_file)
   
    # edge case where visual motion only saved 4 channels
    if ("motion" in folder) and ("visual" in folder):
        channels = 4
    else:
        channels = 5
        
    # read ephys data
    matrix_data = read_bonsai_matrix_file(matrix_file, channels)
        
        
        
    volume_df, first_vol, var_data = universal_bonsai_log_parse(log_df, matrix_data, sr = sr)   
    
    if ("flow" in folder) & ("visual" not in folder):
        
        # uncomment if analysis data pre 25042022 for 4 way flow
        #volume_df, pump_df, first_vol = utils.flow(log_file, log_df, matrix_data)
        
        volume_df, flow_df, first_vol = flow_8way(log_file, log_df, matrix_data)

    elif ("motion" in folder) & ("visual" not in folder) :
        volume_df, motion_df, first_vol = motion(log_file, log_df, matrix_data)

    elif ("visual" in folder) & ("motion" not in folder) & ("flow" not in folder):
        volume_df, visual_df, first_vol = visual(log_file, log_df, matrix_data)
        

    elif ("visual" in folder) & ("motion" in folder):
        volume_df, motion_df, first_vol = visual_motion(log_file, log_df, matrix_data)
        
    elif ("visual" in folder) & ("flow" in folder):
        volume_df, motion_df, first_vol = visual_flow(log_file, log_df, matrix_data)
        
    elif ("OMR" in folder):
        volume_df, visual_df, first_vol, vel_matrix = OMR(log_file, log_df, matrix_data) 
        matrix_data[:, -1] = vel_matrix # replace empty pump 5th (last) channel with velocity
            
        
        
    ##---------------QUALITY CONTROL-----------------#
    # check volume_df has whole frames > 300 ms for fr 3.07
    durations = volume_df.ends - volume_df.starts
    print("volume_df shape is {}".format(volume_df.shape))
        
    volume_duration_cutoff  = (np.round(1e3/fr) - (0.1 * 1e3/fr))/1e3
    print("volume_cutoff is {}".format(volume_duration_cutoff))
    volume_df = volume_df[durations > volume_duration_cutoff]
    print("volume_df shape is {}".format(volume_df.shape))
        
    # get imaging interval in ephys datapoints to fill in gap between imaging
    imaging_interval  = int(np.round((1/fr) * sr))
        
    # subset ephys to begin from first volume
    subset = matrix_data[first_vol:, : ]
        

    # subset matrix data from first volume and last volume to timelock to imaging data
    imaging_duration = (volume_df.nVol / fr).iloc[-1]
    print("Imaging duration is {} ".format(imaging_duration))
    ephys_duration  = subset.shape[0] / sr # sampling_rate of NI data
    print("Ephys duration is {}".format(ephys_duration))
    
    # check to see if ephys and imaging duration is the same and truncate ephys
    difference = ephys_duration - imaging_duration
    print("Difference between ephys and imaging duration is".format(difference))
        
        

    # if there is a positive difference then truncate the ephys data
    if difference > 0 :
        
        diff_datapoints = int(np.round(difference * sr))
        #subset = subset[:-diff_datapoints]
        subset = matrix_data[(first_vol - imaging_interval) : -diff_datapoints, :]
        ephys_duration  = subset.shape[0] / sr
        print("Corrected imaging duration is {} and ephys duration is {}, first vol is {}, diff datapoint is {}".format(imaging_duration, ephys_duration, first_vol, diff_datapoints))
         
    # if difference is negative then there are more volumes than ephys data
    # this shouldnt happen as volume df is created from the volume clock in ephys data but added assertion error in case
    elif difference < 0:
        assert difference == 0
        print("caution! more volumes than ephys data")
        
    else:
        subset = matrix_data[(first_vol - imaging_interval) : , :]
        print("Non-corrected imaging duration is {} and ephys duration is {}, first vol is {}, diff datapoint is {}".format(imaging_duration, ephys_duration, first_vol, diff_datapoints))
        
    #try:
    print("Reading stimulus video")
    vid_da = read_stim(im_fname, log_df, volume_df)
        
    #except:
    #    print("Failed to find a stimulus video")
    #    vid_da = None

    return volume_df, subset, vid_da

def read_stim(im_fname, log_df, volume_df, fr = 60):
        
        # get stim file
        vid_file = get_any_associated_files(im_fname, ".avi")
        
        # read video as dask array
        #vid_da = da_imread(vid_file, nframes = 100) # 
        vid_da = read_video(vid_file, 100) # wrote a custom video reader that is faster
        
        # vid da has same time as log data - volume_df is in bonsai time
        first_vol_t = volume_df.iloc[0].starts
        last_vol_t = volume_df.iloc[-1].ends
        
        end_log = log_df.iloc[-1].Timestamp
        end_vid = vid_da.shape[0] / fr
        
        # get vid time
        vid_time = np.arange(0, end_vid, 1/fr)
        
        # check that video and log file have same duration
        assert np.allclose(end_log, end_vid, 1e-2)
        
        # subset vid_da by finding where vid time is between first_vol_t and last_vol_t
        vid_da = vid_da[(vid_time >= first_vol_t) & (vid_time <= last_vol_t)] # changed to see if stops error message of boolean being larger than vid_da
        
        return vid_da

def read_scanimage_frame(scanimage_im, start, end, arrayfunc = np.asanyarray):
    return arrayfunc(scanimage_im.data(start, end))


#from numba import njit, prange



#@njit(parallel=True)
def fast_nearest_match(raw_values, times, ephys_timebase, var_data, asig_t_start, nvar = 0, sr  =6e3):
    for nval in prange(raw_values.shape[0]): # val in prange(enumerate(raw_values)):
        val = raw_values[nval]
        time = times[nval]
        
        # nearest ephys point
        nearest_ephys_point = int(np.round((time - asig_t_start) * sr))
        
        var_data[nvar, nearest_ephys_point] = val
            #print(empty[min_x])
    return var_data

def universal_bonsai_log_parse(log_df, matrix_data, sr = 6e3, split = 1):
    
    # gets volume clock, loops through vars in log df, puts them on ephys base
    
    #since adding a new colum called replay, Value is no longer last column so use this to extract buffersize.
    buffer_size = int(log_df[log_df.Value.str.contains("AnalogRead")]['Value'].iloc[0].split(", ")[1].split("=")[-1])

    #buffer_size = int(log_df[log_df.Value.str.contains("AnalogRead")].iloc[0, -1].split(", ")[1].split("=")[-1])
    print("buffer size is {}".format(buffer_size))

    # get asig start and end
    asig_t_start = log_df[log_df.Value.str.contains("AnalogRead")].Timestamp.iloc[0] - (buffer_size/6000) # buffer is {}, sr is 6000 therefore subtract buffer_size/sample_rate
    asig_t_end = asig_t_start + (matrix_data.shape[0]/6000) # use matrix data shape - more accurate as initial OMR experiments buffered 1 second before saving which caused loss of last second when file ended


    volume_clock = get_volume_clock(matrix_data)
    volume_acq_starts, volume_acq_ends = get_clock_starts_ends(volume_clock)
    volume_acq_starts_t, volume_acq_ends_t = get_clock_timestamps(volume_clock, volume_acq_starts, volume_acq_ends, asig_t_start*pq.s)
    volume_df = create_volume_df(volume_acq_starts_t, volume_acq_ends_t)

    first_vol = volume_acq_starts[0]
    first_vol_t = volume_acq_starts_t[0]

    ephys_timebase = np.arange(matrix_data.shape[0]) / sr + asig_t_start
    
    unique_vars = log_df.Value.apply(lambda x: x.split()[split-1]).unique() # sample, get time relative to asig_t_start
    unique_vars = unique_vars[~np.isin(unique_vars, ['AnalogRead', 'Bonsai.Shaders.FrameEvent'])]
    var_data = np.full((len(unique_vars), matrix_data.shape[0]), np.nan)

    for nvar, var in enumerate(unique_vars):
        print(var)

        var_subset = log_df[log_df.Value.apply(lambda x: x.split()[split-1] == var)]


        if var == "Trial":
            # is it start or end
            var_subset["Raw"] = var_subset.Value.apply(lambda x: x.split()[split])
            var_subset.loc[var_subset.Raw == "End", "Raw"] = 2
            var_subset.loc[var_subset.Raw == "Start", "Raw"] = 1  
            var_subset["Raw"] = var_subset.Raw.astype("int64")

        elif var == "'VisStarted'":
            var_subset["Raw"] = [1] * var_subset.shape[0]

        else:

            var_subset["Raw"] = var_subset.Value.apply(lambda x: float(x.split()[split]))

        raw_values = var_subset.Raw.to_numpy()
        times =  var_subset.Timestamp.to_numpy()


        var_data = fast_nearest_match(raw_values, times, ephys_timebase, var_data, asig_t_start, nvar = nvar, sr=sr)

    #for var in unique_vars
    #empty2 = fast_nearest_match(raw_values, times, ephys_timebase, empty)
    var_data = pd.DataFrame(var_data.T).interpolate("nearest", limit_direction = "forward")
    var_data.fillna(0, inplace= True)
    #empty2 = np.nan_to_num(empty2)
    #empty2.max()

    # get var_df
    var_df = var_data.iloc[volume_acq_starts]
    var_df.columns = unique_vars
    var_df.reset_index(inplace = True, drop = True)

    volume_df = pd.concat([volume_df, var_df], axis=1)
    
    return volume_df, first_vol, var_data

