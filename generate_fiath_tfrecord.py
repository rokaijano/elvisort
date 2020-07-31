import numpy as np
import os 
from pathlib import Path


from preprocess.prepare_data import *


## 
## Raw data dir of Fiath dataset.
##
data_dir = {
    FiathDataProcessor : ""   
}

train_data = { FiathDataProcessor: [] }
test_data  = { FiathDataProcessor: [] }

file_list = [
	
	"ARat_2016_07_18__0_002.dat", 
    "ARat_2016_07_20__0_002.dat",
    "ARat_2016_07_27__0_002.dat",
    "ARat_2016_10_12__0_002.dat",
    "ARat_2016_10_19__0_002.dat",
    "ARat_2017_04_05__0_002.dat",
    "ARat_2017_05_19__0_002.dat",
    "ARat_2017_05_31__0_002.dat",  
    "ARat_2017_06_01__0_002.dat"
]

output_dir = "data\\generated_data\\fiath\\"

config = {}
config.setdefault("train_limit",70)
config.setdefault("test_limit", 30)


## THRESHOLDD!!!
config.setdefault("threshold_std", 3)
config.setdefault("only_positives", False)
config.setdefault("data_channels", 128)

config.setdefault("for_meta_learning", False)
config.setdefault("incremental_labeling", False)

for file in file_list:

	train_data[FiathDataProcessor] = [file]
	test_data[FiathDataProcessor] = [file]

	unique_output_dir = os.path.join(output_dir, file)

	Path(unique_output_dir).mkdir(parents=True, exist_ok=True)

	preprocess_and_filter(data_dir, unique_output_dir, train_dict=train_data, test_dict=test_data, config=config)

