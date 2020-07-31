import os
import sys

import cv2
import glob
import numpy as np
import xmltodict
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas

from scipy.signal import butter, lfilter, resample_poly 
from sklearn.model_selection import train_test_split


try:
    from . import global_constants as gc
except ImportError:
    import global_constants as gc
try:
    from .DataPreprocessor import *
except ImportError:
    from DataPreprocessor import *


class KampffDataProcessor(DataPreprocessor):

    def __init__(self, *args, **kwargs):
        super(KampffDataProcessor, self).__init__(*args, **kwargs)
        self.preprocessor_name = "kampff"
        self.timespan=64
        self.fsize = os.path.getsize(os.path.join(self.data_dir, self.fname, "amplifier.bin"))
        self.fr = 30000

        # Basic config
        self.probe_dtype = np.uint16
        self.probe_voltage_step_size = 0.195e-6
        self.probe_y_digitization = 32768
        self.process_offset = 200 #for removing the artifacts caused by the butterworth filter. 

        # For convinience we overwrite the only_positives boolean, because in case of Kampff DB there is no reason to cluster from the same n
        self.only_positives = False
    
    def load_labels(self):
        if self.loaded_labels:
            return

        labels = np.genfromtxt(os.path.join(self.data_dir, self.fname, "label.csv"), delimiter='\n', dtype=np.int32)
        labels += 2000
        self.activations = labels 
        self.neurons = np.expand_dims(self.activations, axis=0)
        self.loaded_labels = True
        return

    def load_raw(self):
        fdata = np.fromfile(os.path.join(self.data_dir, self.fname, "amplifier.bin"), offset=self.offset, dtype=self.probe_dtype, count=(self.data_to_load_at_once+self.process_offset)*self.channels)
        numsamples = len(fdata) // self.channels
        data = np.reshape(fdata, (numsamples, self.channels))

        # Setting offset for the next cycle 
        self.offset = self.offset + self.channels*2*(self.data_to_load_at_once - int(self.timespan/2)-self.process_offset)
        return data

    def load_data_at(self):
        start_byte = self.start_from_sample * 128 *2 
        self.offset = max(start_byte-self.process_offset, 0)

        self.data_timestep = self.offset / (2*self.channels) #in samples
        self.data_timestep += self.process_offset

        data_array = self.load_raw()
        processed_data = (self.offset-start_byte)*100/self.fsize
        
        self.file_offset = self.offset / (2*self.channels) 
        
        print("\r --- Processed data: " + str(processed_data) + "%", end="")
        # Indexing by the electrode map
        data_array = data_array[:, gc.channel_map_kampff]
        
        return data_array


    def load_data(self): 

        start_byte = self.start_from_sample * self.channels *2 
        self.offset = max(start_byte-(2*self.channels*self.process_offset), 0)

        while True:
        
            self.data_timestep = self.offset / (2*self.channels) #in samples
            self.data_timestep += self.process_offset

            data_array = self.load_raw()
            processed_data = (self.offset-start_byte)*100/self.fsize
            

            if not data_array.size:
                break

            self.file_offset = self.offset / (2*self.channels) 
            
            print("\r --- Processed data: " + str(processed_data) + "%", end="")
            # Indexing by the electrode map
            data_array = data_array[:, gc.channel_map_kampff]
            
            yield data_array

            if processed_data > self.limit_to_percent:
                break
        


