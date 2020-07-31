import os
import sys

import cv2
import glob
import numpy as np
import xmltodict
import tensorflow as tf
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MultiLabelBinarizer

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def butter_bandpass(lowcut, highcut, fs, order=5, btype="band"):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0, btype="band"):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    y = lfilter(b, a, data, axis=axis)
    return y

class DataPreprocessor(object):

    def __init__(self, _data_dir, _fname, test=False, limit = 2, threshold=3, for_meta_learning = False, only_positives = False, multi_label = False, start_from_percent=-1):
        self.channels = 128
        self.timespan = 64 # the length in time dimension of each sample
        self.fr = 20000 # Hz
        self.data_dir = _data_dir
        self.fname = os.path.join(_data_dir, _fname)
        self.fsize = 0
        self.data_to_load_at_once = self.timespan* 1000
        self.neurons = [] # here we store the activations per neuron
        self.activations = []
        self.test_mode = test
        self.limit_to_percent = limit#0.2 
        self.threshold_std_mult = threshold
        self.start_from_sample = 0#5000000
        self.start_from_percent = start_from_percent
        self.max_spike_length = 12
        self.max_samples = 800000 # max number of samples to process
        self.processed_samples = 0
        self.selected_labels = []
        self.file_offset = 0 # in samples 
        self.positive_samples = np.array(())
        self.data_timestep = 0
        self.process_offset = 0
        self.target_ratio = 0.2
        self.for_meta_learning = for_meta_learning
        self.loaded_labels = False
        self.only_positives = only_positives
        self.preprocessor_name = "raw"
        self.target_sample_to_split = 200 # and the meta-learning test batch size 
        self.multi_label = multi_label

        self.cluster_snr =[]
        self.cluster_sums = [] # cluster x channels
        self.overall_sums = [] # timestep x channels

    def load_labels(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def fill_to_target_ratio(self, x, y):

        if(self.positive_samples.shape[0] == 0): 
            return x, y
            
        target_pos_count = x.shape[0] * self.target_ratio
        positives_needed = target_pos_count - np.count_nonzero(y)

        unselected_positives = np.setdiff1d(self.activations, np.asarray(self.selected_labels), assume_unique = False)
        index_of_uns_pos = [ i for i,x in enumerate(self.activations) if x in unselected_positives]

        for i in range(positives_needed):

            if len(index_of_uns_pos) >= i:
                return x,y 

            x = np.append(x, self.positive_samples[index_of_uns_pos[i]], axis=0)
            y = np.append(y, np.ones(1,1))
            self.selected_labels.append(self.activations[index_of_uns_pos[i]])

        return x, y

    
    def get_positive_samples(self, regenerate=False):
        
        file_path = os.path.join(self.fname, "positive_samples.npy")
        
        if os.file.exists(file_path) and not regenerate:
            self.positive_samples = np.load(file_path)
            print("\nPositive sample have been loaded successfully, shape: "+str(self.positive_samples.shape))
            return
        self.load_labels()
        # Check whether we got activations to begin working with 
        if len(self.activations) < 1:
            raise ValueError

        original_start_sample = self.start_from_sample
        orig_data_to_load = self.data_to_load_at_once 
        orig_limit = self.limit_to_percent
        
        self.limit_to_percent = 100
        
        additional_load = self.fr *2
        self.data_to_load_at_once = self.timespan + additional_load ## the process offset is handled at the loading stage
        

        for i, activation in enumerate(self.activations):
            self.start_from_sample = activation-self.timespan//2 - additional_load//2
            
            data_array = next(self.load_data())
            
            data_array = butter_bandpass_filter(data_array, 300, 3000, self.fr, 5, axis=0)

            data_array = np.asarray(data_array[self.process_offset+additional_load//2:-additional_load//2])
            data_array = np.expand_dims(data_array, axis=0)

            if self.positive_samples.shape[0] == 0:
                self.positive_samples = data_array
            else:
                self.positive_samples = np.append(self.positive_samples, data_array, axis=0 )

            if(self.positive_samples.shape[0] % 100 == 0):
                print("\nAdded samples currently: "+str(self.positive_samples.shape))

        self.positive_samples = np.asarray(self.positive_samples)
        # restore to the original 
        self.limit_to_percent = orig_limit
        self.data_to_load_at_once = orig_data_to_load
        self.start_from_sample = original_start_sample
        np.save(os.path.join(self.fname, "positive_samples.npy"), self.positive_samples)
        print("\nPositive sample have been generated successfully, shape: "+str(self.positive_samples.shape))
    
    def get_sample(self):
        
        self.load_labels()

        self.cluster_sums = [[] for _ in range(self.neurons.shape[0])]

        if self.start_from_percent > 0:
            self.start_from_sample = int((self.start_from_percent * self.fsize) / (2*self.channels*100))

        for data_array in self.load_data():
            # Filtering by frequency

            data_array = butter_bandpass_filter(data_array, 300, 3000, self.fr, 5, axis=0)
            
            data_array = np.asarray(data_array[self.process_offset:])
            

            ## This is for SNR 

            self.overall_sums.extend(np.abs(data_array)) 
            #self.overall_samples += data_array.shape[0]

            ### 

            # Evaluate the standard deviation and the mean for each channel
            data_std = np.std(data_array, axis=0)
            data_median = np.median(np.abs(data_array), axis=0) / 0.6745

            # Evaluate the thresholds for amplitude filtering
            thresh_upper = data_median + self.threshold_std_mult * data_std

            abs_data_array = np.absolute(data_array)
            threshold_pass = abs_data_array >= thresh_upper

            # this is necesarry so that we can filter those channel which do not contain data, this can happen when we pad a smaller channel number to 128
            threshold_pass = threshold_pass * np.minimum(data_std, 1) 

            data_array = data_array.astype(np.int16)

            thresholding_indexes = np.any(threshold_pass, axis=1)

            thresholding_max = np.max(abs_data_array, axis=1)

            possible_candidates = thresholding_indexes * thresholding_max 

            possible_candidates_indexes = np.nonzero(possible_candidates)[0]

            i = np.argmax(possible_candidates_indexes >= self.timespan // 2)
            
            while i <= len(possible_candidates_indexes):
                
                current_index = possible_candidates_indexes[i]

                lower_idx = current_index - self.timespan // 2
                upper_idx = current_index + self.timespan // 2

                tmp_sample = possible_candidates[lower_idx:upper_idx]

                if(len(tmp_sample) < self.timespan):
                    break


                best_cand_idx = len(tmp_sample) - np.argmax(tmp_sample[::-1]) - 1

                best_cand_idx_abs = lower_idx + best_cand_idx

                if best_cand_idx_abs == current_index or best_cand_idx_abs < current_index:  

                    sample = data_array[lower_idx:upper_idx]

                    label = np.where((self.neurons >= self.data_timestep + lower_idx) & (self.neurons < self.data_timestep + upper_idx))

                    if(len(label[0]) > 0):

                        self.select_label(self.neurons[label[0], label[1]])

                        for l in label[0]:
                        
                            if len(self.cluster_sums) < l:
                                raise IndexError("IndexError. Please initialize the cluster_sums in the load_labels.")
                        
                            self.cluster_sums[l-1].append(sample)
                        
                        if not self.multi_label:
                            label = label[0][0]+1
                        else:
                            label = label[0] + 1 
                    
                    else:
                        label = 0

                    if sample.shape[0] < self.timespan:
                        continue

                    yield sample, label
                    
                    i = np.argmax(possible_candidates_indexes >= upper_idx)

                else:
                    
                    i = np.argmax(possible_candidates_indexes >= best_cand_idx_abs) 

                if i == 0:
                    break

    def select_label(self, label_list):

        for label_id in label_list:
            if label_id not in self.selected_labels:
                self.selected_labels.append(label_id)

    def check_labels(self):
        
        self.activations.sort()

        target_list = [x for x in self.activations if x < self.file_offset and x > self.start_from_sample]

        return len(self.selected_labels)*100 / len(target_list)

    def get_train_test(self):
        samples = []
        labels = []
        self.processed_samples = 0 

        import time
        print (" %%% Loading data from main .dat file and processing ... ")
        t1 = time.time()
        for sample, label in self.get_sample():
            samples.append(sample)
            labels.append(label)

            if len(samples) == self.target_sample_to_split : # >2 because we want at least one positive sample in each dataset

                # we totally ignore the sequences in which there are no positive samples
                if self.for_meta_learning and sum(labels) < 1:
                    samples = []
                    labels = []
                    continue 

                t2 = time.time()

                labels = np.asarray(labels)

                if self.only_positives:
                    samples = np.asarray(samples)[labels > 0]
                    labels = labels[labels > 0]

                self.processed_samples += self.target_sample_to_split

                if self.test_mode:
                    yield(None, np.asarray(samples), None, np.asarray(labels))
                    samples = []
                    labels = []
                    continue
                    
                if self.for_meta_learning:
                    yield(np.asarray(samples), None, np.asarray(labels), None)
                    samples = []
                    labels = []
                    continue 

                # Having chosen this method, the disadvantage is that we cannot control the distribution, so we cannot guarantee the same distribution over train/test datasets
                label_train = []
                label_test = []


                sample_train, sample_test, label_train, label_test = train_test_split(samples, labels, shuffle=True)

                samples = []
                labels = []
    
                yield(np.asarray(sample_train), np.asarray(sample_test), np.asarray(label_train), np.asarray(label_test))
                t1 = time.time()


        if len(samples) != 0 and not self.for_meta_learning:
            try:
                sample_train, sample_test, label_train, label_test = train_test_split(samples, labels, shuffle=True)
                yield(np.asarray(sample_train), np.asarray(sample_test), np.asarray(label_train), np.asarray(label_test))
            except ValueError as e:
                return

        try:
            self.get_snr_for_clusters()
        except MemoryError as me:
            print(me)
            
        print("The labels were processed in the target interval, with percent of :"+str(self.check_labels())+"%")

    def get_snr_for_clusters(self):

        self.overall_sums = np.asarray(self.overall_sums)
        final_overall_sum = np.median(self.overall_sums, axis=0)

        noise = final_overall_sum / 0.6745 

        cluster_snr = []

        for cluster_id in range(len(self.cluster_sums)):
            
            if len(self.cluster_sums[cluster_id]) == 0:
                cluster_snr.append(0.)
                continue

            cluster_waveforms = np.asarray(self.cluster_sums[cluster_id])

            template = np.median(cluster_waveforms, axis=0)

            maxp2p = np.max(template, axis=0) - np.min(template, axis=0)

            peak_channel = np.argmax(maxp2p)

            signal = np.max(np.abs(template[:, peak_channel]))


            cluster_snr.append(signal / noise[peak_channel])

        self.cluster_snr = np.asarray(cluster_snr)

    def write_tfrecord(self, f, data, labels):

        mlb = MultiLabelBinarizer(classes=list(range(self.neurons.shape[0]+1)))

        for sample, label in zip(data, labels):

            if  self.multi_label: 
                if isinstance(label, int):
                    label = [label]

                label = np.asarray(mlb.fit_transform([label]))

                feature = {'label': _bytes_feature(tf.compat.as_bytes(label.tostring())), 'image': _bytes_feature(tf.compat.as_bytes(sample.tostring()))}
                
            else:
                feature = {'label': _int64_feature(label), 'image': _bytes_feature(tf.compat.as_bytes(sample.tostring()))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            f.write(example.SerializeToString())

    def process_file(self, ftrain, ftest, incr_label=0):

        print("\n Processing file %s:", self.fname)
        train_nr = 0
        test_nr = 0
        spike_num_train = 0
        spike_num_test = 0

        for x_train, x_test, y_train, y_test in self.get_train_test():
            
            if not self.test_mode:
                ## Saving the train data

                train_nr += x_train.shape[0]

                self.write_tfrecord(ftrain, x_train, y_train)
                
                for k in range(len(y_train)):
                    if isinstance(y_train[k], (np.ndarray, list)):
                        y_train[k] = 1 # just for simplicity, the correct value was already saved 


            if x_test is None or y_test is None:
                yield [x_train.shape[0], 0, np.count_nonzero(y_train), 0]
                continue


            ## Saving the test/val data 
            self.write_tfrecord(ftest, x_test, y_test)

            test_nr += x_test.shape[0]
            
            for k in range(len(y_test)):
                if isinstance(y_test[k], (np.ndarray, list)):
                    y_test[k] = 1 


            if self.test_mode:
                yield (0, x_test.shape[0], 0, np.count_nonzero(y_test))
            else:
                yield [x_train.shape[0], x_test.shape[0], np.count_nonzero(y_train), np.count_nonzero(y_test)]

        print("\n --- File was processed succesfully train/test size: %d/%d" % (train_nr, test_nr))
        return train_nr, test_nr


