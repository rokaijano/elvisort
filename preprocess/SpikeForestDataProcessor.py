try:
    from .DataPreprocessor import *
except ImportError:
    from DataPreprocessor import *
    
from spikeforest2_utils import AutoRecordingExtractor, AutoSortingExtractor
import kachery as ka
import numpy as np

import itertools
import global_constants as gc
import matplotlib.pyplot as plt


def padding_for_diff_dataset(data_array):
    padded_data = []

    return padded_data


class SpikeForestDataProcessor(DataPreprocessor):
    def __init__(self, dataset_name, recording_name, recording_path, sorting_true_path, label_for_only_pos, *args, **kwargs):
        super(SpikeForestDataProcessor, self).__init__(*args, **kwargs)
        self.preprocessor_name = dataset_name
        self.recording_name = recording_name
        # Configure kachery to download data from the public database
        ka.set_config(fr='default_readonly')
        self.label_for_only_pos = label_for_only_pos
        self.data_to_load_at_once = self.timespan * 1000

        """
            Spikeforest datasets data shape: (channel, measures)
            True firing shape: (neuron, true firing location in the data)
        """
        self.recording = AutoRecordingExtractor(recording_path, download=True)
        self.sorting_true = AutoSortingExtractor(sorting_true_path)
        self.fr = self.recording.get_sampling_frequency()
        self.channels = self.recording.get_num_channels()
        self.timespan = 32

    def load_labels(self):
        self.neurons = np.array(())
        for unit in self.sorting_true.get_unit_ids():
            spikes = (self.sorting_true.get_unit_spike_train(unit))
            spikes = np.expand_dims(spikes, 0)

            if self.neurons.shape[0] == 0:
                self.neurons = spikes
            else:
                if self.neurons.shape[-1] > spikes.shape[-1]:
                    spikes = np.pad(spikes, ((0, 0), (self.neurons.shape[-1] - spikes.shape[-1], 0)), mode="constant",
                                    constant_values=0)
                elif self.neurons.shape[-1] < spikes.shape[-1]:
                    self.neurons = np.pad(self.neurons, ((0, 0), (spikes.shape[-1] - self.neurons.shape[-1], 0)),
                                          mode="constant", constant_values=0)

                self.neurons = np.concatenate((self.neurons, spikes), axis=0)
        self.activations = list(itertools.chain.from_iterable((self.neurons)))
        self.activations = np.array(self.activations)
        self.activations = np.sort(self.activations)
        self.loaded_labels = True
        self.neurons = np.asarray(self.neurons)

    def load_data(self):

        tmp_array = np.array(self.recording.get_traces())
        if self.preprocessor_name.lower() == 'hybrid_janelia':
            # Zero array for electrode geometry reasons
            tmp_array = np.insert(tmp_array, 0, np.zeros(tmp_array.shape[1]), axis=0)
            tmp_array = tmp_array[gc.channel_map_hybrid_janelia, :]

        iterator = self.start_from_sample

        while True:

            self.data_timestep = iterator

            processed_data = ((iterator - self.start_from_sample) * 100) / tmp_array.shape[1]
            print("\r --- Processed data: " + str(processed_data) + "%", end="")

            if tmp_array.shape[1] < iterator:
                break
            if tmp_array[0, iterator:].size < self.data_to_load_at_once:
                data_array = tmp_array[:, iterator:]
                self.data_to_load_at_once = data_array.shape[1]
                print(" %%% Getting to the end, the load_at_once is : " + str(self.data_to_load_at_once))

            else:
                data_array = tmp_array[:, iterator:iterator + self.data_to_load_at_once]
            if data_array.size != 0:
                yield np.array(data_array).transpose()

            iterator += self.data_to_load_at_once
            self.file_offset = iterator

            if processed_data >= self.limit_to_percent:
                break