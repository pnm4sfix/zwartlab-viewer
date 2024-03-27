# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:11:39 2021

@author: zwartlab-users
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal as sig
import neo
from elephant.signal_processing import butter
import quantities as pq
import os
from tkinter import filedialog
import time
import math
import seaborn as sns
from scipy.signal import find_peaks
import elephant as el
from datetime import datetime

try:
    from scipy.stats import median_absolute_deviation as mad
except:
    from scipy.stats import median_abs_deviation as mad

# To Do - determine if threshold is too low
#- e.g. if burst is longer than 2 seconds because inter burst peaks are being picked up then raise threshold




class Fish(neo.Block):
    
    def __init__(self, description = None, load = False, folder = None):
        """ Initialises fish class and the inheritance class of neo.Block. 
        Gets all Flt files in a fish folder.
        
        Parameters: 
                    description : Default is None. Can be used to specify
                    further metadata of experiment."""
        
        #initialise block
        neo.Block.__init__(self)
        
        if load == False:
            
            self.get_directory(folder)
        
        else:
            self.load_block(load)
        self.description = description
        
        
    
    def get_directory(self, folder = None):
        """Creates a pop up asking for the directory containing data for one fish. 
        Then loops through data Flt files, returns all filepaths 
        and the time folder was created.
        
        Parameters: None
        
        Returns: 
            
            folder: directory path containing data for selected fish
            
            fish_id: the fish id. Assumes the folder name is in the format
                        "Fish[n] e.g. Fish1 or Fish21"
                        
            file_dict: a dictionary containing all Flt file paths. Keys in this
                        dict are the subfolder of the Flt file and the values
                        are the full Flt file path.
                        
            rec_time: the datetime the data folder was created."""
        
        if folder is None:
            folder = filedialog.askdirectory(initialdir = "./")
        fish_id = os.path.split(folder)[-1]
        
        files_dict = {}
    
        for dirpath, dirnames, files in os.walk(folder):
            
            flt_files = [os.path.join(dirpath, file) for file in files if "Flt" in file]
            if flt_files: # non empty list evaluates to True
                files_dict[os.path.split(dirpath)[-1]] = flt_files
        
        rec_time = time.strftime("%Y%m%d %H:%M:%S", time.gmtime(os.path.getctime(folder)))
        rec_time = datetime.strptime(rec_time, '%Y%m%d %H:%M:%S')
        print("segments is {}self.segments")        
        self.name = fish_id
        self.file_origin = folder
        self.rec_datetime = rec_time
        self.files = files_dict
    
    def read_file(self, file, channels = 10):
        """Opens and reads binary file 10chFlt.
        
        Parameters: 
            
            file: filepath of binary Flt file
            
            channels: number of channels contained in file. Default is 10.
            
        Return:
            
            A : 2D numpy array containing all channels
        """
        f = open(file, 'rb')
        A =  np.fromfile(f, np.float32).reshape((-1, channels)).T
        return A
    
    
    def get_experiments(self):
        """Loops through all flt files associated with a fish and 
        adds each experiment as a segment to the fish block.
        
        Parameters:
            
        Returns:
        """
        
        for protocol, files in self.files.items():
            name = protocol
            
            for n, file in enumerate(files):
                
                file_rec = time.strftime("%Y%m%d %H:%M:%S", time.gmtime(os.path.getmtime(file)))
                file_rec = datetime.strptime(file_rec, '%Y%m%d %H:%M:%S')
                A = self.read_file(file, 10)
                
                if np.shape(A)[1] > 0:
                    self.add_segment(A, name, file_rec, file.split("\\")[-1], n)
    
    def add_segment(self, A, name, file_rec, file_origin, index = 0, projection = "forward"): # file_origin
        """Each Flt file is a separate experiment and will be contained in 
            a neo.Segment. Neo.Analog signals and neo.Epochs are added to each 
            segment before the segment is finally added to the block.
        
        Parameters: 
            
            block: a neo.Block object containing metadata.
            
            A: a 2D numpy array containing the binary data of the Flt files. The
                structure of A is as follows:
                    A[0] = AnalogIn1 = left electrode
                    A[1] = AnalogIn2 = Right electrode
                    A[2] = AnalogIn3 
                    A[3] = stimparam4 - typically unused
                    A[4] = stimparam5 - typically unused
                    A[5] = stimparam3 - condition e.g. 0 is stationary, 1 is forward and 2 is back
                    A[6] = stimparam1 - velocity of visual stim
                    A[7] = stimparam2 - gain
                    A[8] = AnalogIn4 = Camera exposures
                    A[9] = AnalogIn5 = Pump activation
                    
            
            name: the subfolder name. Best for this to be the name of the protocol
                    used, e.g. Orientation Tuning
                    
            index: specify if function is being used in a loop, as it provides a 
                unique id for each segment. Default is 0.
                
            file_rec: the datetime the lt file was last modified.
            
        
        Returns:
            
            bl: the inputted block with
        
        """
        amp_gain = 1e5
        c1 = A[0]
        c2 = A[1]
        c1_asig = neo.AnalogSignal(signal = c1/amp_gain, 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz, 
                                   name = "Left Electrode " +name+str(index))
        
        c2_asig = neo.AnalogSignal(signal = c2/amp_gain, 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz,
                                   name = "Right Electrode " +name+str(index))
        
        """Create filtered channels"""
        
        fltc1 = self.smooth_channel(c1)
        fltc2 = self.smooth_channel(c2)
        
        fltc1_asig = neo.AnalogSignal(signal = fltc1/amp_gain, 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz,
                                      name = "Left Filtered " +name+str(index))
        
        fltc2_asig = neo.AnalogSignal(signal = fltc2/amp_gain, 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz, 
                                      name = "Right Filtered "+name+str(index))
        try:
            vel_asig = neo.AnalogSignal(signal = A[6], 
                                      units = pq.m/pq.s, sampling_rate = 6000 *pq.Hz, 
                                        name = "Velocity "+name+str(index))

            gain_asig = neo.AnalogSignal(signal = A[7], 
                                      units = pq.V, sampling_rate = 6000 *pq.Hz, 
                                         name = "Gain "+name+str(index))

       
            A2 = neo.AnalogSignal(signal = A[2], 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz,
                                  name = "A2 "+name+str(index))
            
            A3 = neo.AnalogSignal(signal = A[8], 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz,
                                  name = "A3 "+name+str(index))
            A4 = neo.AnalogSignal(signal = A[9], 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz, 
                                  name = "A4 "+name+str(index))
        except:
            print("no extra signals")
            
        seg = neo.Segment(name = name + " " +str(index), description = name, 
                          index = index, rec_datetime = file_rec, file_origin = file_origin)
        
        seg.analogsignals.append(c1_asig.rescale(pq.uV))
        seg.analogsignals.append(c2_asig.rescale(pq.uV))
        seg.analogsignals.append(fltc1_asig.rescale(pq.uV))
        seg.analogsignals.append(fltc2_asig.rescale(pq.uV))
       
        
        
        try:
            seg.analogsignals.append(vel_asig)
            seg.analogsignals.append(gain_asig)
            seg.analogsignals.append(A2)
            seg.analogsignals.append(A3)
            seg.analogsignals.append(A4)
            
        except:
            pass
        
        
        st1 = self.get_peaks(seg.analogsignals[2])
        st2 = self.get_peaks(seg.analogsignals[3])
        
        bout_epc = self.define_bouts(st1, st2)
        
           
        
        if name == "OMRgain":
            #print("Extracting protocol from gain values not stimparam3")
            starts, durations = self.extract_protocol(A[5])
            epc = self.create_epoch(c1_asig, starts, durations, name)
            
            #st3_start, st3_duration = self.extract_protocol(A[5])
            #st3_epc = self.create_epoch(c1_asig, starts, durations, name)
            #st3_epc.name = st3_epc.name+"st3"
            
        else:
        
            starts, durations = self.extract_protocol(A[5])
            epc = self.create_epoch(c1_asig, starts, durations, name, projection = projection)
        
        
        seg.epochs.append(epc)
        seg.epochs.append(bout_epc)
        
        try:
            seg.epochs.append(st3_epc)
        except:
            print("no st3 epc")
        
        seg.spiketrains.append(st1)
        seg.spiketrains.append(st2)
        
        self.segments.append(seg)
        
    def create_epoch(self, c1_asig, starts, durations, protocol, projection = "forward"):
        """Takes stim_param3 levels and turns into a dict, that can be used to
        create epochs for different levels of the protocol.
        
        Parameters: 
            
            c1_asig : left electrode signal.
            
            starts: the start of a change in stim_param3 level.
            
            durations: the durations of a protocol level.
            
            protocol: the name of the subfolder should be the protocol used.
                        Currently accepted protocols are Orientation Tuning,
                        IO Directional, AutoSwitch Simple, OMRgain.
        Returns:
            
            epc : neo.Epoch containing the times, durations and labels of
                    the different protocol levels."""
                    
        degree_sign= u'\N{DEGREE SIGN}'
        
        if protocol == "Orientation Tuning":
            self.protocol_dict = {k : (int(math.degrees(k))) for k in starts.keys()}
            
            if projection == "right":
                rotated_angles = [-57, 90, 120, 149, 180, 209, 240, 270, 299, 329, 0, 30, 60]
                self.protocol_dict = {k: rotated_angles[n] for n, k in enumerate(self.protocol_dict.keys())}
                
            print("Orientation Tuning")
        elif protocol == "IO Directional":
            print("IO Directional")
            self.protocol_dict = {0.0 : "Stationary", 1.0: "Forward", 2.0: "Backward"}
        elif protocol == "AutoSwitch Simple":
            print(protocol)
            self.protocol_dict = {0.0: "Misc", 1.0: "Low Gain", 2.0: "High Gain"}
        
        elif protocol == "OMRgain":
            print(protocol)
            #protocol_dict = {k: str(k) for k in starts.keys()}
            self.protocol_dict = {0.0: "Misc", 1.0: "Rest", 2.0: "Low Gain", 3.0: "High Gain"}
        else:
            print(protocol)
            print("protocol dict failed")
            
        
        
        labels = []
        times = []
        for k, v in starts.items():
            for value in v:
                labels.append(str(self.protocol_dict[k]))
                times.append((c1_asig.times[value]))    
    
        durations_list = []
    
        for k, v in durations.items():
            for value in v:
                durations_list.append(float((value / c1_asig.sampling_rate))*pq.s)
        times = np.array(times).flatten()*pq.s
        durations_list = np.array(durations_list).flatten()*pq.s
        epc = neo.Epoch(times=times, durations = durations_list, 
                        labels = labels, name = c1_asig.name[13:])
        
        return epc
    
    
    def smooth_channel(self, channel):
        """ Produces a filtered signal with high signal to noise.
        
        Parameters: 
            
            channel: a neo.AnalogSignal typically either of the left electrode
                    or right electrode.
                    
        Returns:
            
            fltc : a filtered analogsignal."""
        
        ker = np.exp(-np.arange(-60,61,1) **2/(2*20**2.))
        smch1 = np.convolve(channel,ker/ker.sum(),mode='same')
        pow1 = (channel - smch1)**2
        fltc = np.convolve(pow1,ker/ker.sum(),mode='same')
        
        return fltc
    
    def extract_protocol(self, channel):
        """Extracts the levels of each protocol based on stim_param3.
        
        Parameters:
            
            A : 2D numpy array containing data from an Flt file.
            
        Returns:
            
            level_starts : the times of each protocol level.
            
            level_durations : the durations of each protocol level."""
        level_starts = {}
        level_durations = {}
        protocol_levels = np.unique(channel)
        for n in protocol_levels:
            level = np.where(channel == n)[0]
            diffs = np.diff(level) != 1
            indexes = np.nonzero(diffs)[0]+1
            groups = np.split(level, indexes)
            
            starts = [x[0] for x in groups]
            durations = [x[-1]-x[0] for x in groups]
            level_starts[n] = starts
            level_durations[n] = durations
        
        return level_starts, level_durations
    
    
    def plot_experiment(self, seg, t1 = False, t2 = False, electrodes_only = True):
        """This will plot each channel in an experiement (segment). 
        Segment can be sliced by time.
        
        Parameters:
            
            seg: a neo.Segment associated with a fish block.
            
            t1: the start timepoint for a time slice. Default is False.
            
            t2: the end timepoint for a time slice. Default is False.
        
        Assigns:
            
            self.fig: experiment figure.
            
            self.axes: the axes in the experiment figure."""
        
        if (type(t1) != bool) & (type(t2) != bool):
            seg = seg.time_slice(t1, t2)
            print("time sliced")
            
        n_asigs = len(seg.analogsignals)
        
        if electrodes_only:
            # the number of asigs to plot up to
            n_asigs = 4
        else:
            n_asigs = len(seg.analogsignals)
            
        self.fig, self.axes = plt.subplots(nrows = n_asigs, figsize=(10, 10),
                                           sharex=True)
        self.fig.subplots_adjust(hspace=0)
        
        
        
        for n, asig in enumerate(seg.analogsignals):
            
            if n < n_asigs:
                
                self.axes[n].plot(asig.times, asig.magnitude.flatten(), 
                                  linewidth = 0.3, c = "k") #use own palette here
                self.axes[n].set_ylabel(str(asig.units).split(" ")[-1])


                if n == 2:
                    if (type(t1) != bool) & (type(t2) != bool):
                        st_times = seg.spiketrains[1].time_slice(t1, t2).times
                    else:
                        st_times = seg.spiketrains[1].times
                    st_idx = np.where(np.isin(asig.times, st_times))[0]
                    asig_std = asig.magnitude.std()
                    self.axes[n].set_ylim(0, 20*asig_std)
                    #print(np.shape(st_idx))
                    #print(np.shape(st_times))
                    self.axes[n].scatter(asig.times[st_idx], asig.magnitude.flatten()[st_idx],
                                         c="r", marker ="x")
                if n == 3:
                    if (type(t1) != bool) & (type(t2) != bool):
                        st_times = seg.spiketrains[0].time_slice(t1, t2).times
                    else:
                        st_times = seg.spiketrains[0].times
                    st_idx = np.where(np.isin(asig.times, st_times))[0]
                    
                    self.axes[n].scatter(asig.times[st_idx], asig.magnitude.flatten()[st_idx],
                                         c="r", marker ="x")
                    asig_std = asig.magnitude.std()
                    self.axes[n].set_ylim(0, 20*asig_std)
                
        sns.despine()
        
        self.fig.tight_layout()
        colors = plotting_palette()
        colors_dict = {}
        for n, label in enumerate(np.unique(seg.epochs[0].labels)):
            try:
                colors_dict[label] = colors[n]
            except:
                print("out of colors, defaulting to magenta")
                colors_dict[label] = colors[1]
         
        unique_label = {k:0 for k, v in colors_dict.items()}
        
        for ax in self.axes:
            for n, time in enumerate(seg.epochs[0].times):
                start = time 
                end = time + seg.epochs[0].durations[n]
                label = seg.epochs[0].labels[n]
                color = colors_dict[label]
                
                if unique_label[label] == 0:
                    ax.axvspan(start, end, color=color, alpha=0.5, lw=0, label = label)
                    ax.legend()
                    
                else:
                    ax.axvspan(start, end, color=color, alpha=0.5, lw=0)
                unique_label[label] = 1
            if len(seg.epochs) > 1:
                if len(seg.epochs[1].times) > 0:
                    for n, bout_start in enumerate(seg.epochs[1].times):

                        bout_end = bout_start + seg.epochs[1].durations[n]
                        ax.axvspan(bout_start, bout_end, color="r", alpha=0.5, lw=0)
                
    def remove_artifacts(self, asig):
        """ Removes transients or noise to improve peak detection"""
        pass
        
        
    
    
    def get_peaks(self, asig, invert = False, gamma = 2, suppress_output = False):
        """ Gets peaks specifically for the filtered channels. Method 
        calculates the histogram of the channel, gets the peak bin of the histogram,
        and the first bin of the histogram. The range between these two bins
        is multiplied by a factor, 1.4-2. The threshold is this new value added
        to the peak bin.
        
        Parameters: 
            
            asig: neo.Analogsignal.
            
            invert: if signal has negative deflections invert signal
                    to get peaks. Default is False.
        
        Returns:
            
            st : neo.SpikeTrain containing peak times"""
        
        #bins = int(np.shape(asig)[0]/500)
        #z_score = st.zscore(asig.magnitude.flatten(), ddof=1, nan_policy = "omit")
        #asig = asig.magnitude.flatten()[z_score<6]
        #asig_sub = asig.magnitude.flatten()[asig.magnitude.flatten()<0.1]
        asig_sub = asig.magnitude.flatten() - np.median(asig.magnitude.flatten())
        arr, b = np.histogram(asig_sub, bins = 2000)
        peak_s = b[np.argmax(arr)]

        min_s = b[0]
        if suppress_output == False:
            print("Gamma is {}".format(gamma))
        if gamma == 2:

            threshold = peak_s + gamma*(peak_s-min_s)
            print("initial threshold is {}".format(threshold))

            if (threshold < 0.05) & (threshold > 0.03):
                threshold = peak_s + 2.5*(peak_s-min_s)

            elif (threshold < 0.03) & (threshold > 0.02):
                threshold = peak_s + 2*(peak_s-min_s)

            elif (threshold < 0.02):
                threshold = peak_s + 2*(peak_s-min_s)
        else:
            threshold = peak_s + gamma*(peak_s-min_s)
            
        #print("min_s is {} and peak_s is {}".format(min_s, peak_s))
        if suppress_output == False:
            print("threshold is {}".format(threshold))
        
        
        if invert == True:
            sig = asig-asig-asig
            #sig = sig.magnitude.flatten()
            sig = asig.magnitude.flatten() - np.median(asig.magnitude.flatten())
            
        else:
            #sig = asig.magnitude.flatten()
            sig = asig.magnitude.flatten() - np.median(asig.magnitude.flatten())
            
        idx, wv_data = find_peaks(sig, height  = threshold)
        times = asig.times[idx]
        st = neo.SpikeTrain(times, sampling_rate = asig.sampling_rate, 
                            t_start = asig.t_start, t_stop = asig.t_stop,
                            name = asig.name + "peaks")
        
        
        # assign st to analogsignal
        
        return st
    
    def define_bouts(self, st1, st2, ignore = None, gap_between_bouts = 0.2, bursts_per_bout = None):
        """ Define start and end of each swim bout using the burst times for 
        both filtered channels and create an epoch to be added to segment.
        
        Parameters:
            
            st1: neo.SpikeTrain containing bursts in fltc1.
            
            st2: neo.SpikeTrain containing bursts in fltc2
            
        Returns:
            
            epc: neo.Epoch containing the starts, durations and count
                    of each swim bout."""
        if ignore == None:
        # combine sts
            combined_times = np.sort(np.append(st1.times, st2.times))
            combined_st = neo.SpikeTrain(combined_times, 
                                     sampling_rate = st1.sampling_rate, 
                                     t_start = st1.t_start, 
                                     t_stop = st1.t_stop)
            bout_length = 5
            
        elif ignore == 0:
            combined_st = st1
            bout_length = 4
            
        elif ignore == 1:
            combined_st = st2
            bout_length = 4
        
        if bursts_per_bout is not None:
            bout_length = bursts_per_bout
        
        st_isi = el.statistics.isi(combined_st)
        st_idx = np.where(st_isi > gap_between_bouts)[0] +1
        
        bouts = np.split(combined_st.times.magnitude, st_idx)
        #st_index = np.where(np.isin(asig.times, combined_st.times))[0]
        #bouts_idx = np.split(st_index, idx)
        
        bout_number = []
        bout_starts = []
        bout_durations = []
        count = 1
        for bout in bouts:
            
            if len(bout) >= bout_length:
                bout_starts.append((bout[0] - 0.01) * pq.s)
                bout_durations.append(((bout[-1]-bout[0]) + 0.05) * pq.s)
                bout_number.append(str(count))
                
                count += 1
                
        
        
        epc = neo.Epoch(times = bout_starts, durations = bout_durations, 
                        labels = bout_number, name = st1.name + "bursts",
                        units = pq.s)
        
        return epc
        
      
    
    def save_block(self, name = "ephys.blk", use=True):
        """This saves the block in NIXIO format, a standardised neuroscience format,
        at the same location as the fish main folder.
        
        Parameters:
            
            use: use block annotations. Default is True.
            
        
        """
        print("Saving block")
        #name = "ephys.blk"
        filepath = self.file_origin
        """Check file exists, if so delete it"""
        path = os.path.join(filepath, name)
        if os.path.exists(path):
            os.remove(path)
        nixfile = neo.NixIO(path, mode="rw")
        nixfile.write_block(self, use)
        nixfile.close()
        
    def load_block(self, nixfile=False):
        """Reads block and loads into fish class
        
        Parameters: 
            
            nixfile: full path for .blk nixfile
            
        Assigns:
            
            self.segments: neo.Segments are assigned to fish class
            
            self.rec_datetime: datetime object of file creation attached to
                                fish class.
            
            self.name: attaches fish name"""
        
        if nixfile == False:
            nixfile = filedialog.askopenfilename()
        reader = neo.NixIO(nixfile)
        bl = reader.read(lazy=False)[0]
        self.segments = bl.segments.copy()
        self.rec_datetime = bl.rec_datetime
        self.name = bl.name
        del bl
        
    
    def calculate_rauc(self, asig, bout_starts, bout_durations, baseline = "median"):
        """ Calculates the rectified area under the curve for each swim bout.
        
        Parameters: 
            
            asig: neo.Analogsignal.
            
            bout_starts: start time of each bout.
            
            bout_durations: duration of each bout.
        
        Returns:
            
            bout_wv: dictionary containing the waveforms of each bout.
            
            bout_rauc: dictionary contraining the rauc of each bout."""
        
        bout_wv = {}
        bout_rauc = {}
        #bout_rauc_asigs = {}
        
        bout_ends = bout_starts + bout_durations
        asig_median = np.median(asig.magnitude) * pq.uV
        if baseline is None:
            baseline = None
        elif baseline == "median":
            baseline = asig_median
        #print("Baseline is  {}".format(baseline))
        for n, start in enumerate(bout_starts):
            
            try:
                wv = asig.time_slice(start, bout_ends[n])
                bout_wv[n] = wv
                rauc = el.signal_processing.rauc(wv, baseline = baseline, bin_duration = 1 * pq.s)
                bout_rauc[n] = float(rauc.sum())
                #bout_rauc_asigs[n] = rauc
                
                
                # also get instantaneous firing rate of bursts and also swim freq
            except:
                print("bout_start and end is {} and {}."
                      "Recording start and end is {} and {}.".format(start,
                                                                     bout_ends[n],
                                                                     asig.times[0],
                                                                     asig.times[-1]))
                
                # get power for part of bout within seg
                if start < asig.times[0]:
                    new_start = asig.times[0]
                    new_end = bout_ends[n]
                elif bout_ends[n] > asig.times[-1]:
                    new_start = start
                    new_end = asig.times[-1]
                    
                
                wv = asig.time_slice(new_start, new_end)
                bout_wv[n] = wv
                rauc = el.signal_processing.rauc(wv, baseline = baseline, bin_duration = 1 * pq.s)
                bout_rauc[n] = float(rauc.sum())
                
        return bout_wv, bout_rauc#, bout_rauc_asigs


    
    def rauc_df(self, bout_rauc1, bout_rauc2, condition_epc, bout_epc):
        """ Transforms rauc calculation into convenient to use pandas dataframe.
        
        Parameters:
            
            bout_rauc1: the rauc dictionary for ch1.
            
            bout_rauc2: the rauc dictionary for ch2.
            
            condition_epc: the first epoch in the block. Contains
                            the times of different protocol levels.
                            
            bout_epc: the second epoch in the block. Contains the times
                    all the swim bouts.
                    
        Returns:
            
            df: pd.DataFrame bout related info."""
        
        df = pd.DataFrame([bout_rauc1, bout_rauc2]).transpose()
        df.columns = ["Left_RAUC", "Right_RAUC"]
        bout_times = pd.Series(bout_epc.times, bout_epc.labels.astype("int64"))
        df.index = pd.Series(bout_epc.labels.astype("int64"))[:df.shape[0]]
        df.index.name = "bout"
        df["bout_times"] = bout_times
        
        for n, time in enumerate(condition_epc.times):
                start = float(time)
                end = float(time + condition_epc.durations[n])
                label = condition_epc.labels[n]
                
                df.loc[(df.bout_times>=start)&(df.bout_times<end), "label"] = label
                df.loc[(df.bout_times>=start)&(df.bout_times<end), "epoch"] = n
                df.loc[(df.bout_times>=start)&(df.bout_times<end), "trial_start"] = start
                df.loc[(df.bout_times>=start)&(df.bout_times<end), "trial_end"] = end
        
        df.reset_index(inplace=True)
        wv_epoch = df.set_index(["epoch"]) #bout in epoch
        first_bouts = wv_epoch.groupby(level=[0])[['bout']].min()
        first_bouts_dict = first_bouts.to_dict()["bout"]
        df["first_bout"] = df.epoch.map(first_bouts_dict)
        df["boutbyepoch"] = (df.bout - df.first_bout) + 1
        df["mean_rauc"] = df[["Left_RAUC", "Right_RAUC"]].mean(axis=1)
        
        return df
    

    
    def plot_polar_bout(self, df):
        """ Plots swim bout counts and rauc by angle.
        
        Parameters:
            
            df: pd.DataFrame containing rauc data
            
        Returns:
            
            g: FacetGrid of 4 axes."""
        
        df["radians"] = df.label.astype("int64").apply(math.radians) 
        swim_bout_count = df.groupby("radians").mean_rauc.count()
        swim_bout_count.name ="swim_bout_count"
        mean_rauc = df.groupby("radians").mean_rauc.mean()
        mean_l_rauc = df.groupby("radians").Left_RAUC.mean()
        mean_r_rauc = df.groupby("radians").Right_RAUC.mean()
        
        # concat these series 
        concat_df = pd.concat([swim_bout_count, mean_rauc, 
                               mean_l_rauc, mean_r_rauc], axis=1)
        concat_melt = concat_df.reset_index().melt(id_vars="radians")
        
        
        g = sns.FacetGrid(concat_melt,col ="variable",
                      subplot_kws=dict(projection='polar'), size=4,
                      sharex=False, sharey=False, despine=False, 
                      palette = plotting_palette())
        for ax in g.fig.axes:
            ax.set_theta_zero_location('N', offset=0)
            ax.set_theta_direction(-1)
        #g.fig.axes[0].set_ylabel("")
        g.map(plt.bar, "radians", "value", color = list(plotting_palette()))
        
        g.fig.axes[0].set_xlabel("")
        #sns.despine()
        return g
    
    def get_rate_df(self, asig, st, condition_epc, bout_epc):
        """"""
        asig_rauc = el.signal_processing.rauc(asig, bin_duration = 1*pq.s)
        asig_rauc.t_start = asig.t_start
        
        burst_rate = el.statistics.instantaneous_rate(st, sampling_period = 1*pq.s)
        burst_rate.t_start = asig.t_start
        
        bout_st = neo.SpikeTrain(bout_epc.times[bout_epc.times>0], units ="s", t_stop = asig.t_stop)
        bout_asig_rate = el.statistics.instantaneous_rate(bout_st, sampling_period = 1*pq.s)
        bout_asig_rate.t_start = asig.t_start
        
        df = pd.DataFrame([bout_asig_rate.magnitude.flatten(), burst_rate.magnitude.flatten()/10, asig_rauc.magnitude.flatten()[:-1]])
        df = df.transpose().reset_index()
        df.columns = ["times", "Swim freq", "Burst freq/10", "Power"]
        
        for n, time in enumerate(condition_epc.times):
                start = float(time)
                end = float(time + condition_epc.durations[n])
                label = condition_epc.labels[n]
                df.loc[(df.times>start)&(df.times<end), "label"] = label
                df.loc[(df.times>start)&(df.times<end), "epoch"] = n
                
        first_dict = df.set_index("epoch").groupby(level=[0]).times.min().to_dict()
        df["first_times"] = df.epoch.map(first_dict)
        df["timesbyepoch"] = df.times-df.first_times
        
        df_melt = df.melt(id_vars = ["timesbyepoch", "label", "epoch"], value_vars = ["Swim freq", "Burst freq/10", "Power"])
        
        return df_melt
    

    def plot_rates(self, rate_df):
        """Plots swim rate, left and right burst rate and binned power. Make it a facet plot
        
        Parameters: 
        
            rate_df: pd.DataFrame in long form containing rate info as well as label
            
        Returns:
            
            f: Relplot
        """
        
        f = sns.relplot(data=rate_df, hue = "label", col = "variable", y= "value", x= "timesbyepoch",
                        kind = "line", facet_kws={'sharey': False, 'sharex': True},
                       palette = plotting_palette())
        
        return f
    
    def power_per_bout(self, df):
        """ Plots swim bout power by protocol level.
        
        Parameters: 
            
            df: pd.DataFrame containing rauc data.
            
        Returns:
            
            f: Relplot"""
       
        f = sns.relplot(kind="line", x="boutbyepoch", y="mean_rauc", 
                        hue="label", data=df, markers = True,
                       ci=68, 
                       palette=plotting_palette()[:df.label.unique().shape[0]])
        ax = f.fig.axes[0]
        ax.set_xticks(df.boutbyepoch.unique().tolist())
        ax.set_ylabel("Power")
        ax.set_xlabel("Bout")
        sns.despine(offset = 10, trim=True)
        
        return f
    
    def power_per_time(self, df):
        """ Plots swim bout power by protocol level.
        
        ##Doesn't work yet
        
        Parameters: 
            
            df: pd.DataFrame containing rauc data.
            
        Returns:
            
            f: Relplot"""
            
        data = df.sort_values("label").sort_values("bout_times")
        f = sns.relplot(kind="line", x="bout_times", y="mean_rauc",
                        hue="label", data=data, markers = True, ci=68,
                        palette=plotting_palette())
        ax = f.fig.axes[0]
        #ax.set_xticks(df.boutbyepoch.unique().tolist())
        return f
    
    def power_hist_per_bout(self, df):
        """Plots the power histogram for the first bout,
        second bout and fifth bout for each condition.
        
        Parameters: 
            
            df: pd.DataFrame containing rauc data.
            
        Returns:
            
            f: Displot
        """
        
        #fig, axes =plt.subplots(ncols=3, sharex=True, sharey=True)
        data = df[df.boutbyepoch.isin([1,2,5])]
        data = data.sort_values("label")
        f = sns.displot(data=data, x="mean_rauc", kde=True, hue="label",
                     col = "boutbyepoch", 
                     palette=plotting_palette()[:data.label.unique().shape[0]])
        f.fig.tight_layout()
        sns.despine(offset = 10, trim=True)
        for ax in f.axes:
            try:
                ax.set_xlabel("Power")
                ax.get_legend().get_frame().set_edgecolor('w')
                ax.get_legend().set_title("")
            except:
                pass
        return f
    
    
    def append_dff(self, cnms, fr):
        
        """Appends dff and deconvolved fluorescence signals from cnm objects to the block.
        
        Parameters: 
            
            cnms: list of cnm objects
            fr: frame rate
            
        Assigns:
            
            dff_asig: neo.AnalogSignal of df/F trace
            decon_asig: neo.AnalogSignal of deconvolved fluorescence trace
        """
        try:
            # get frame clock first and last frame
            frame_clock = self.segments[0].analogsignals[6]
            
        except:
            frame_clock = self.segments[0].analogsignals[5]
        
        # get indices where frame is captured - where voltage >2.5
        exposed_timepoints = np.where(frame_clock > 2.5)[0]
        
        # get time of first frame
        first_frame = exposed_timepoints[0]
        t_first_frame = frame_clock.times[first_frame]
        
        # get time of last frame
        last_frame = exposed_timepoints[-1]
        t_last_frame = frame_clock.times[last_frame]
        #print(t_first_frame, t_last_frame)

        # loop through cnms list and create asigs for dff and decon
        for z_idx, cnm in enumerate(cnms):
            est = cnm.estimates

            dff_asig = neo.AnalogSignal(est.F_dff.T, units = pq.F, sampling_rate = fr *pq.Hz, t_start = t_first_frame, t_stop = t_last_frame, name = "dff_df_z" + str(z_idx))
            decon_asig = neo.AnalogSignal(est.S.T, units = pq.V, sampling_rate = fr *pq.Hz, t_start = t_first_frame, t_stop = t_last_frame, name = "decon_z" + str(z_idx))

            # append dff and decon asig to segments
            self.segments[0].analogsignals.append(dff_asig)
            self.segments[0].analogsignals.append(decon_asig)
    
    
    def get_starts_by_label(self, epoch, label):
        """Get starts and ends of epochs of specific label
        
        Parameters: 
            
            epoch: neo.Epoch object
            label: the label of an epoch
            
        Returns:
            
            starts: start times of label specific epochs
            ends: end times of label specific epochs
        """
        
        starts = epoch.times[epoch.labels == label]

        durations = epoch.durations[epoch.labels == label]
        ends = starts + durations

        return starts, ends
    
    
    def get_label_trials(self, asig, starts, ends):
        """Get starts and ends of epochs of specific label
        
        Parameters: 
            
            asig: neo.AnalogSignal
            starts: start times of specific label
            ends: end times of specific label
            
        Returns:
            
            trials: list of sliced neo.Analogsignals that represent specific trials 
        """
        
        trials = []

        for trial, start in enumerate(starts):
            try:
                sliced_asig = asig.time_slice(starts[trial], ends[trial])

                # measure calcium events here - store somewhere

                trials.append(sliced_asig)



            except ValueError:
                print('t_start, t_stop have to be within the analogsignal duration')
        return trials
    
    def append_ims(self, da_ims):
        """Appends image sequences to block
        
        Parameters: 
            
            da_ims: list of dask_images
            
        Assigns:
            
            im_seq: neo.ImageSequence for all z levels
        """
        
        for z_idx, da_im in enumerate(da_ims):
            im_arr = da_im.compute()
            t_start = decon_asigs[z_idx].t_start
            sr = decon_asigs[z_idx].sampling_rate
            im_seq = neo.ImageSequence(im_arr, units='V', sampling_rate=sr, spatial_scale=1 * pq.micrometer, t_start = t_start, name = str(z_idx) + ".tif")
            self.segments[0].imagesequences.append(im_seq)
            
    def time_slice_im(self, start, end, im_seq, decon):
        """Time slices image sequences using decon asig time slice values
        
        Parameters: 
            
            start: start of time slice
            end: end of time slice
            im_seq: neo.ImageSequence to time slice
            decon: a decon neo.AnalogSignal
            
        Returns:
            
            sliced_im: time sliced image sequence
        """
        
        t_start = decon.time_slice(start, end).t_start
        t_stop = decon.time_slice(start, end).t_stop
        frame_idx =  np.where((decon.times >= t_start) & (decon.times<= t_stop))[0]
        sliced_im = im_seq.magnitude[frame_idx]
        
        return sliced_im
    
    def create_epoch_from_volumedf(self, volume_df, fr):
        # get first frame at start of each condition
        
        if ("visual_angle" in volume_df.columns) & ("visual_vel" in volume_df.columns):
            
            if "visual" in volume_df.stim.unique():
                # set visual angle and visual vel to zero if visual only trials
                volume_df.loc[volume_df.stim == "visual", ["visual_angle", "visual_vel"]] = 0 
            
            condition_starts = (volume_df.groupby(["stim", "angle", "velocity", "visual_angle", "visual_vel", "trial"]).nVol.first() / fr).reset_index()
            condition_ends = (volume_df.groupby(["stim", "angle", "velocity", "visual_angle", "visual_vel", "trial"]).nVol.last() / fr).reset_index()
            
        elif ("gain" in volume_df.columns):
            condition_starts = (volume_df.groupby(["stim", "gain", "velocity", "adaptation", "trial"]).nVol.first() / fr).reset_index()
            condition_ends = (volume_df.groupby(["stim", "gain", "velocity", "adaptation", "trial"]).nVol.last() / fr).reset_index()
        
        else:
            condition_starts = (volume_df.groupby(["stim", "angle", "velocity", "trial"]).nVol.first() / fr).reset_index()
            condition_ends = (volume_df.groupby(["stim", "angle", "velocity", "trial"]).nVol.last() / fr).reset_index()
        
        condition_starts["durations"] = condition_ends.nVol - condition_starts.nVol
        
        
        
        if ("gain" not in volume_df.columns):
            # subset only the durations that are less than 10 secs
            condition_starts = condition_starts[condition_starts.durations < 10]
            
        elif ("gain"  in volume_df.columns):
            # check no condition starts > 60
            # sets durations over 70 seconds to 60 seconds
            condition_starts.loc[condition_starts.durations > 70, "durations"] = 60
        
        if ("visual_angle" in volume_df.columns) & ("visual_vel" in volume_df.columns):
            condition_starts["labels"] = condition_starts.apply(self.concat_combi_strings, axis=1)
        
        elif ("gain" in volume_df.columns):
            condition_starts["labels"] = condition_starts.apply(self.concat_omr_strings, axis=1)
        
        else:
            condition_starts["labels"] = condition_starts.apply(self.concat_uni_strings, axis=1)
        
        epc = neo.Epoch(times = condition_starts.nVol, labels = condition_starts.labels, durations = condition_starts.durations, name = "condition_epoch", units = pq.s)

        return epc
    
    def concat_combi_strings(self, series):
        label = "{} angle {} velocity {} vis_angle {} vis_vel {}".format(series.stim, series.angle, series.velocity, series.visual_angle, series.visual_vel)
        return label

    def concat_uni_strings(self, series):
        label = "{} angle {} velocity {}".format(series.stim, series.angle, series.velocity)
        return label
    
    def concat_omr_strings(self, series):
        label = "{} gain {} velocity {}".format(series.stim, series.gain, series.velocity)
        return label
    
    def add_segment_from_npy(self, A, name, file_rec, file_origin, volume_df, fr=3.07,  index = 0, ignore= None, gammas = (2, 2), velocity = False): # file_origin

        #A = ephys_data.T
        #name = fish.name
        #file_rec = None
        #file_origin = fish.file_origin
        #index = 0
        """Each ephys file will be contained in 
            a neo.Segment. Neo.Analog signals and neo.Epochs are added to each 
            segment before the segment is finally added to the block.

        Parameters: 

            block: a neo.Block object containing metadata.

            A: a 2D numpy array containing the binary data of the Flt files. The
                structure of A is as follows:
                    A[0] = AnalogIn1 = right electrode
                    A[1] = AnalogIn2 = left electrode
                    A[2] = AnalogIn3 = frame clock
                    A[3] = AnalogIn3 = volume clock
                    
                    if velocity:
                    A[4] = reafferent velocity


            name: the subfolder name. Best for this to be the name of the protocol
                    used, e.g. Orientation Tuning

            index: specify if function is being used in a loop, as it provides a 
                unique id for each segment. Default is 0.

            file_rec: the datetime the lt file was last modified.


        Returns:

            bl: the inputted block with

        """
        amp_gain = 1e5
        c1 = A[0]
        c2 = A[1]
        
        
        ## clean channels
        c1 = remove_artifacts(c1)
        c2 = remove_artifacts(c2)
        
        c1_asig = neo.AnalogSignal(signal = c1/amp_gain, 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz, 
                                   name = "Right Electrode " +name+str(index))

        c2_asig = neo.AnalogSignal(signal = c2/amp_gain, 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz,
                                   name = "Left Electrode " +name+str(index))
        
        
        
        

        """Create filtered channels"""

        fltc1 = self.smooth_channel(c1)
        fltc2 = self.smooth_channel(c2)
        
        
        
        

        fltc1_asig = neo.AnalogSignal(signal = fltc1/amp_gain, 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz,
                                      name = "Right Filtered " +name+str(index))

        fltc2_asig = neo.AnalogSignal(signal = fltc2/amp_gain, 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz, 
                                      name = "Left Filtered "+name+str(index))

        frame_clock = neo.AnalogSignal(signal = A[2], 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz, 
                                      name = "Frame Clock "+name+str(index))

        volume_clock = neo.AnalogSignal(signal = A[3], 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz, 
                                      name = "Volume Clock "+name+str(index))

        seg = neo.Segment(name = name + " " +str(index), description = name, 
                          index = index, rec_datetime = file_rec, file_origin = file_origin)

        seg.analogsignals.append(c1_asig.rescale(pq.uV))
        seg.analogsignals.append(c2_asig.rescale(pq.uV))
        seg.analogsignals.append(fltc1_asig.rescale(pq.uV))
        seg.analogsignals.append(fltc2_asig.rescale(pq.uV))
        seg.analogsignals.append(frame_clock.rescale(pq.uV))
        seg.analogsignals.append(volume_clock.rescale(pq.uV))
        
        if velocity:
            
            reaff_vel = neo.AnalogSignal(signal = A[4], 
                                  units = pq.V, sampling_rate = 6000 *pq.Hz, 
                                      name = "Visual Velocity "+name+str(index))
            
            seg.analogsignals.append(reaff_vel.rescale(pq.uV))
            



        right_bursts = self.get_peaks(seg.analogsignals[2], gamma = gammas[0])
        left_bursts = self.get_peaks(seg.analogsignals[3], gamma = gammas[1])

        bout_epc = self.define_bouts(left_bursts, right_bursts, ignore)




        epc = self.create_epoch_from_volumedf(volume_df, fr)



        seg.epochs.append(epc)
        seg.epochs.append(bout_epc)

        seg.spiketrains.append(left_bursts)
        seg.spiketrains.append(right_bursts)

        self.segments.append(seg)
    
    def create_bout_data_from_segment(self, seg_idx =None):
        motor_raucs = []
        if seg_idx is None:
            segments_to_analyse = self.segments[1:]
        else:
            segments_to_analyse = [self.segments[seg_idx]]
        for seg in segments_to_analyse:
            #print(seg.annotations)
            left_bout_wv, left_bout_rauc = self.calculate_rauc(asig=seg.analogsignals[2],
                                                               bout_starts=seg.epochs[1].times,
                                                               bout_durations=seg.epochs[1].durations)

            right_bout_wv, right_bout_rauc = self.calculate_rauc(asig=seg.analogsignals[3],
                                                                 bout_starts=seg.epochs[1].times,
                                                                 bout_durations=seg.epochs[1].durations)
            
            

            if (len(left_bout_rauc) > 0) | (len(right_bout_rauc) >0):
                motor_rauc = self.rauc_df(bout_rauc1=left_bout_rauc,
                                          bout_rauc2=right_bout_rauc,
                                          condition_epc=seg.epochs[0],
                                          bout_epc=seg.epochs[1])
                
                # add a velocity and distance one

                if "Velocity" in self.segments[0].analogsignals[-1].name:
                    velocity_wv, displacement = self.calculate_rauc(asig=seg.analogsignals[-1],
                                                   bout_starts=seg.epochs[1].times,
                                                   bout_durations=seg.epochs[1].durations, baseline = None)

                    mean_vel = pd.Series({b : vel.rescale(pq.V).magnitude.flatten().mean() for b, vel in velocity_wv.items()}, name = "mean_vel")
                    max_backwards_vel = pd.Series({b : vel.rescale(pq.V).magnitude.flatten().min() for b, vel in velocity_wv.items()}, name = "max_backwards_vel")
                    dt = 1/6e3
                    max_backwards_accel = pd.Series({b : (np.diff(vel.rescale(pq.V).magnitude.flatten()) / dt).min() for b, vel in velocity_wv.items()}, name = "max_backwards_accel")

                    #motor_rauc.reset_index(drop=True, inplace=True)  # Reset the index

                    # Append motor_rauc to the list with bout identifier
                    #motor_rauc["bout_id"] = [bout_id] * motor_rauc.shape[0]
                    motor_rauc = pd.concat([motor_rauc, mean_vel, max_backwards_vel, max_backwards_accel], axis=1)
                motor_rauc.reset_index(inplace = True, drop = True)

                # check for duplicated columns - not sure how this happens but causes issues
                if motor_rauc.columns.duplicated().any():
                    print("bout data contains extra duplicated empty columns")
                    motor_rauc = motor_rauc.loc[:, ~motor_rauc.columns.duplicated()]

                motor_raucs.append(motor_rauc)
                
                
                
        try:

            # Concatenate the list of motor_rauc dataframes into a single dataframe
            bout_data = pd.concat(motor_raucs)

            # Create unique bout identifier column based on the shape of bout_data2
            bout_data["bout_identifier"] = np.arange(bout_data.shape[0])

            return bout_data
        except:
            return motor_raucs
    
    
    def reextract_bouts(self, gammas = (1.8,1.8),  ignore = None, bursts_per_bout = None, gap_between_bouts = 0.2,
                       suppress_output = True):
        
        
        #for seg in fish.segments[1:6]:
        all_bout_times = []
        all_bout_durations = []
        
        for seg in self.segments[1:]:
            #print(seg.annotations)
            #print(seg.trial_seg)
            right_bursts = self.get_peaks(seg.analogsignals[2], gamma = gammas[0], suppress_output = suppress_output)
            left_bursts = self.get_peaks(seg.analogsignals[3], gamma = gammas[1], suppress_output = suppress_output)

            bout_epc = self.define_bouts(left_bursts, right_bursts, ignore,
                                         bursts_per_bout = bursts_per_bout, 
                                         gap_between_bouts= gap_between_bouts)
            #fish.plot_experiment(seg)
            if len(seg.epochs) >1:
                seg.epochs[1] = bout_epc
                
            elif len(seg.epochs) == 1:
                seg.epochs.append(bout_epc)
            elif len(seg.epochs) == 0:
                print("No epochs")
                
            
            
            all_bout_times.extend(bout_epc.times)
            all_bout_durations.extend(bout_epc.durations)
            
        # create new bout_epc for seg[0]
        labels = np.arange(1, len(all_bout_times)+1).astype(str)
        new_epoch = neo.Epoch(times = all_bout_times, durations = all_bout_durations, labels = labels, units= pq.s)
        self.segments[0].epochs[1] = new_epoch
        #self.segments[1].epochs[1].durations = all_bout_durations
            #fish.plot_experiment(seg)
            #fish.plot_experiment(fish.segments[1: ])
        
        
    
    
       
    
    


def read_single():
    file = filedialog.askopenfilename()
    A = read_file(file, 10) 
    bl = neo.Block()
    bl = create_segment(bl, A, name, n, file_rec)
    return bl

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

def remove_artifacts(series, min_std = 10, max_std = 10):
    
    high_threshold = np.median(series) + mad(series)*max_std
    low_threshold = np.median(series) - mad(series)*min_std

    series[series>high_threshold] = np.median(series)
    series[series<low_threshold] = np.median(series)


    series -= np.median(series)
    return series


    
    
    
        
        
    
    