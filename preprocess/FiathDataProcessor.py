import os
import sys

import cv2
import glob
import numpy as np
import xmltodict
import tensorflow as tf
import matplotlib.pyplot as plt 

from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split

try:
    from .DataPreprocessor import *
except ImportError:
    from DataPreprocessor import *

class FiathDataProcessor(DataPreprocessor):

    def __init__(self, *args, **kwargs):
        super(FiathDataProcessor, self).__init__(*args, **kwargs)
        self.preprocessor_name = "fiath"
        self.fsize = os.path.getsize(self.fname)
        self.timespan=64
        self.channels = 128
        self.data_to_load_at_once = self.timespan* 1000

    def load_labels(self):
        if self.loaded_labels:
            return
        
        self.neurons = np.array(())
        meta_files = glob.glob(os.path.join(self.data_dir, self.fname.strip(".dat"), '*.ev2'))
        for mfile in meta_files:
            with open(mfile, "r") as mf:
                
                spikes = []
                for line in mf:
                    spikes.append(np.int64((line.split(' ')[-1])))
                spikes = np.asarray(sorted(spikes))
                spikes = np.expand_dims(spikes, 0)

                if self.neurons.shape[0] == 0:
                    self.neurons = spikes
                else:
                    if self.neurons.shape[-1] > spikes.shape[-1]:
                        spikes = np.pad(spikes, ((0,0), (self.neurons.shape[-1]-spikes.shape[-1], 0)), mode="constant", constant_values=0)

                    elif self.neurons.shape[-1] < spikes.shape[-1]:
                        self.neurons = np.pad(self.neurons, ((0,0), (spikes.shape[-1]-self.neurons.shape[-1], 0)), mode="constant", constant_values=0)

                    self.neurons = np.concatenate((self.neurons, spikes), axis=0)


        # at this point we have the different detected neurons loaded 
        self.activations = self.neurons.reshape((self.neurons.shape[0]*self.neurons.shape[1], ))
        self.activations = np.sort(self.activations)
        self.loaded_labels = True
        self.neurons = np.asarray(self.neurons)

    def load_data(self):
        self.data_timestep = 0

        start_byte = self.start_from_sample*self.channels*2

        with open(self.fname, 'rb') as f:
            f.seek(start_byte, os.SEEK_SET)  # seek : 2 bytes for every datapoint

            self.data_timestep = f.tell()/(2*self.channels)
            self.file_offset = self.data_timestep
            while True:
                processed_data = ((f.tell() - start_byte)*100)/self.fsize
                print("\r --- Processed data: " + str(processed_data) + "%", end="")
                
                
                self.data_timestep = f.tell()/(2*self.channels)
                self.file_offset = self.data_timestep
                data_array = np.fromfile(f, dtype='int16', count=self.data_to_load_at_once*self.channels)

                if not data_array.size:
                    break


                if (data_array.size/self.channels) < self.data_to_load_at_once:
                    self.data_to_load_at_once = (int)(data_array.size/self.channels)
                    print(" %%% Getting to the end, the load_at_once is : "+str(self.data_to_load_at_once))

                data_array = np.reshape(data_array, (self.data_to_load_at_once, self.channels))

                yield data_array
                
                f.seek(f.tell()-2*self.channels*int(self.timespan/2))

                self.data_timestep = f.tell()/(2*self.channels)
                self.file_offset = self.data_timestep

                if processed_data > self.limit_to_percent:
                    break



