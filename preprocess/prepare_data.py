import os
import sys

import cv2
import glob
import numpy as np
import xmltodict
import tensorflow as tf


import time 

if __name__ == "__main__":

    from FiathDataProcessor import *
    from KampffDataProcessor import *
    from SpikeForestDataProcessor import *

else:
    from .FiathDataProcessor import *
    from .KampffDataProcessor import *
    from .SpikeForestDataProcessor import *

def preprocess_and_filter(data_dir, out_dir, config_file="data_config.xml", data_processor = FiathDataProcessor, train_dict = None, test_dict = None, config={}):


    ## FILE LIMIT IN PERCENTAGE
    config.setdefault("train_limit",100)
    config.setdefault("test_limit", 100)

    config.setdefault("threshold_std", 2.5)
    config.setdefault("only_positives", False)
    config.setdefault("data_channels", 128)

    config.setdefault("multi_label", False)
    config.setdefault("for_meta_learning", False)
    config.setdefault("incremental_labeling", False)  # so that every file`s class label is different number 


    kampff_train_data = {}
    kampff_test_data = {}
    fiath_train_data = {}
    fiath_test_data = {}
    spikeforest_train_data = {}
    spikeforest_test_data = {}

    ## KAMPFF DATA INPUTS
    """
    kampff_train_data = {
        KampffDataProcessor: 
            [
                "2015_08_21_Cell.3.0_gt0"
            ]
    }
    
    kampff_test_data  = {
        KampffDataProcessor: 
            [
                #"2015_08_21_Cell.3.0_gt0"
            ]
    }
    #"""
    ## FIATH DATA INPUTS
    """
    fiath_train_data = {
        
        FiathDataProcessor: 
            [
                "ARat_2016_07_18__0_002.dat", 
                #"ARat_2017_05_31__0_002.dat", 
                #"ARat_2016_07_27__0_002.dat", 
                #"ARat_2016_10_19__0_002.dat"
            ]
        
    }

    fiath_test_data = {
        
        FiathDataProcessor: 
            [
                "ARat_2016_07_20__0_002.dat"
            ]
        
    }
    #"""
    

    ## SPIKEFOREST DATA INPUTS
    #"""
    spikeforest_train_data = {
        SpikeForestDataProcessor:
            [
                [
                    "hybrid_janelia",
                    "hybrid_drift_siprobe_64C_600_S11",
                    "sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/drift_siprobe/rec_64c_600s_11",
                    "sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/drift_siprobe/rec_64c_600s_11/firings_true.mda"
                ]
            ]
    }
    #"""
    spikeforest_test_data = {
        
        SpikeForestDataProcessor:
            [
                [
                    "hybrid_janelia",
                    "hybrid_drift_siprobe_64C_600_S12",
                    "sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/drift_siprobe/rec_64c_600s_12",
                    "sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/drift_siprobe/rec_64c_600s_12/firings_true.mda"
                ]
            ]

        
    }
    #"""
    all_train_files = []
    all_test_files = []

    if train_dict is None:
        train_dict = {**kampff_train_data, **fiath_train_data, **spikeforest_train_data}
    
    if test_dict is None:
        test_dict = {**kampff_test_data, **fiath_test_data, **spikeforest_test_data}    

    ## ATTENTION : LIST IS CHANGED WITH DICT

    train_tfrecord_fname = os.path.join(out_dir, "train.tfrecord")
    val_tfrecord_fname = os.path.join(out_dir, "val.tfrecord")
    test_tfrecord_fname = os.path.join(out_dir, "test.tfrecord")

    ftrain = tf.io.TFRecordWriter(train_tfrecord_fname)
    fval = tf.io.TFRecordWriter(val_tfrecord_fname)
    ftest = tf.io.TFRecordWriter(test_tfrecord_fname)

    data_type = ""

    clusters = []
    clusters_test = []

    label_for_only_pos = 1
    max_incr_label = 0
    target_to_split = 100
    data_timespan = 64
    train_sample =0
    val_sample = 0
    test_sample = 0
    train_spikes = 0
    val_spikes = 0
    test_spikes = 0

    cluster_snr = {}


    for data_processor, train_files in train_dict.items():
        all_train_files.extend(train_files)
        for f_id, fname in enumerate(train_files):
            if data_processor == SpikeForestDataProcessor:
                if config["only_positives"]:
                    label_for_only_pos = itr + 1

                dp = data_processor(fname[0], fname[1], fname[2], fname[3], label_for_only_pos, data_dir[data_processor], fname[1], limit=config["train_limit"],
                    threshold=config["threshold_std"], multi_label = config["multi_label"], only_positives=config["only_positives"],
                    for_meta_learning=config["for_meta_learning"])  

            else:
                dp = data_processor(data_dir[data_processor], fname, limit = config["train_limit"], threshold=config["threshold_std"], multi_label = config["multi_label"], only_positives = config["only_positives"], for_meta_learning = config["for_meta_learning"])
            
            if data_type == "":
                data_type = dp.preprocessor_name

            incr_label = 0
            if config["incremental_labeling"]:
                incr_label = f_id
                max_incr_label = max(max_incr_label, f_id)

            dp.start_from_sample = 1#20000*100
            data_timespan = dp.timespan
            target_to_split = dp.target_sample_to_split
            #dp.get_positive_samples()


            it = iter(dp.process_file(ftrain, fval, incr_label))
            dp.load_labels()
            while True:
                try:
                    train_nr, test_nr, train_spike, val_spike = next(it)
                    train_sample += train_nr
                    train_spikes += train_spike
                    val_sample += test_nr
                    val_spikes += val_spike

                    still_writing = True
                except StopIteration:
                    if data_processor == SpikeForestDataProcessor:
                        fname = fname[0]
                    cluster_snr["r"+fname+"_train"] = {"cluster": dp.cluster_snr}
                    cluster_snr["r"+fname+"_train"] = {"spike_filter_ratio": dp.check_labels()}
                    break

            clusters.append(dp.neurons.shape[0])


    for data_processor, test_files in test_dict.items():
        all_test_files.extend(test_files)
        for f_id, fname in enumerate(test_files):
            if data_processor == SpikeForestDataProcessor:

                if config["only_positives"]:
                    label_for_only_pos = itr + 1

                dp = data_processor(fname[0], fname[1], fname[2], fname[3], label_for_only_pos, data_dir[data_processor], fname[1], limit=config["test_limit"],
                    threshold=config["threshold_std"], multi_label = config["multi_label"], only_positives=config["only_positives"],
                    for_meta_learning=config["for_meta_learning"], test = True)        

            else:
                dp = data_processor(data_dir[data_processor], fname, test=True, limit = config["test_limit"], threshold=config["threshold_std"], multi_label = config["multi_label"], only_positives = config["only_positives"], for_meta_learning = config["for_meta_learning"])

            incr_label = 0
            if config["incremental_labeling"]:
                incr_label = f_id
                max_incr_label = max(max_incr_label, f_id)

            dp.start_from_sample = 0#20000*100
            dp.start_from_percent = 0#config["train_limit"]
            it = iter(dp.process_file(None, ftest, incr_label))
            
            while True:
                try:
                    _, test_nr,_, test_spike = next(it)
                    test_sample += test_nr
                    test_spikes += test_spike

                    still_writing = True
                except StopIteration:
                    if data_processor == SpikeForestDataProcessor:
                        fname = fname[0]
                    cluster_snr["r"+fname+"_test"] = {"cluster": dp.cluster_snr}
                    cluster_snr["r"+fname+"_test"] = {"spike_filter_ratio": dp.check_labels()}
                    break

            clusters_test.append(dp.neurons.shape[0])

    if train_sample == 0:
        train_ratio = 0
    else:              
        train_ratio = (float)(train_spikes)*100 / train_sample

    if val_sample == 0:
        val_ratio = 0
    else:
        val_ratio = (float)(val_spikes)*100 / val_sample

    if test_sample == 0:
        test_ratio = 0
    else:
        test_ratio = (float)(test_spikes)*100 / test_sample

    print("\n --- All datasets were processed succesfully, train/val/test size: %d/%d/%d" % (train_sample, val_sample, test_sample))
    print(" --- All datasets were processed succesfully, train/val/test spike ratios: %f/%f/%f" % (train_ratio, val_ratio, test_ratio))

    ftrain.close()
    ftest.close()

    max_clusters = 0
    max_clusters_test = 0

    if len(clusters) > 0:
        max_clusters = max(max(clusters),1)
       
    if len(clusters_test) > 0 :
        max_clusters_test = max(max(clusters_test),1)        

    max_clusters = max(max_clusters, max_clusters_test)

    if config["incremental_labeling"]:
        max_clusters = max_incr_label*100+max_clusters+1

    config_dict={"data":{
        "dir":out_dir, 
        "meta_learning_data":config["for_meta_learning"], 
        "target_to_split":target_to_split, 
        "max_clusters":max_clusters, 
        "train_limit":config["train_limit"],
        "test_limit":config["test_limit"], 
        "threshold":config["threshold_std"], 
        "dataset":data_type, 
        "timespan":data_timespan,
        "channels":config["data_channels"],
        "cluster_snr":cluster_snr,
        "incremental_labeling": config["incremental_labeling"],
        "multi_label": config["multi_label"]
        }}

    for t in ["train", "val", "test"]:
        config_dict["data"][t] = {"samples":0, "ratio":0.0}
   
    config_dict["data"]["train"]["samples"] = train_sample
    config_dict["data"]["train"]["ratio"] = train_ratio
    config_dict["data"]["train"]["files"] = all_train_files 

    config_dict["data"]["val"]["samples"] = val_sample
    config_dict["data"]["val"]["ratio"] = val_ratio
    config_dict["data"]["val"]["files"] = all_train_files 

    config_dict["data"]["test"]["samples"] = test_sample
    config_dict["data"]["test"]["ratio"] = test_ratio
    config_dict["data"]["test"]["files"] = all_test_files 
   
    with open(os.path.join(out_dir, config_file), "w") as fd:
        fd.write(xmltodict.unparse(config_dict, pretty=True))

    print(" --- Config file was written succesfully to "+ config_file) 

    sys.stdout.flush()
   

# Please use the appropiate files for generated tfrecords
def main_func():
    data_dir = {
        KampffDataProcessor : "",
        FiathDataProcessor : "",  
        SpikeForestDataProcessor : ""       
    }

    config = {}
    config.setdefault("train_limit",60)
    config.setdefault("test_limit", 100)


    ## THRESHOLDD!!!
    config.setdefault("threshold_std", 2.5)
    config.setdefault("only_positives", False)
    config.setdefault("data_channels", 128)

    config.setdefault("for_meta_learning", False)
    config.setdefault("incremental_labeling", False)
    config.setdefault("multi_label", True)

    addon = ""
    if not config["for_meta_learning"]:
        addon = "non_"

    output_dir = "generated_data\\spikeforest\\hybrid_janelia\\"

    t1 = time.time()

    preprocess_and_filter(data_dir, output_dir, config=config)

    print("Finished under >>> ", time.time()-t1)


if __name__ == "__main__":
    main_func()