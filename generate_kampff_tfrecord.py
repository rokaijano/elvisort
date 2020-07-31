import numpy as np
import os 
from pathlib import Path


from preprocess.prepare_data import *

# Raw data dir of Kampff data.
# Please format every Recording of the Kampf dataset accordingly:
# The amplifier file, should be renamed simply to amplifier.bin
# The labels should be in a csv file under the name of labels.csv

data_dir = {
    KampffDataProcessor : ""     
}

kampff_train_data = { KampffDataProcessor: [] }
kampff_test_data  = { KampffDataProcessor: [] }

file_list = [
	
	#"2015_08_21_Cell.3.0_gt0",
	#"2015_08_21_Cell.3.1_gt0",
	#"2015_08_28_Cell.2.0_gt0",
	#"2015_08_28_Cell.2.1_gt0",
	#"2015_08_28_Cell.2.2_gt0",
	#"2015_08_28_Cell.9.0_gt0",
	#"2015_09_03_Cell.6.0_gt0",
	#"2015_09_03_Cell.6.1_gt0",
	"2015_09_03_Cell.9.0_gt0",
	#"2015_09_04_Cell.5.0_gt0",
	#"2015_09_04_Cell.6.0_gt0",
	#"2015_09_04_Cell.6.1_gt0",
	#"2015_09_09_Cell.4.0_gt0",
	#"2015_09_09_Cell.6.0_gt0",
	#"2015_09_09_Cell.7.0_gt0",
	#"2015_09_09_Cell.7.1_gt0"
]

output_dir = "data\\generated_data\\kampff\\"

config = {}
config.setdefault("train_limit",70)
config.setdefault("test_limit", 30)


## THRESHOLDD!!!
config.setdefault("threshold_std", 2)
config.setdefault("only_positives", False)
config.setdefault("data_channels", 128)

config.setdefault("for_meta_learning", False)
config.setdefault("incremental_labeling", False)

for file in file_list:

	kampff_train_data[KampffDataProcessor] = [file]
	kampff_test_data[KampffDataProcessor] = [file]

	unique_output_dir = os.path.join(output_dir, file)

	Path(unique_output_dir).mkdir(parents=True, exist_ok=True)

	preprocess_and_filter(data_dir, unique_output_dir, train_dict=kampff_train_data, test_dict=kampff_test_data, config=config)

