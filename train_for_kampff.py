#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf 
import csv

import time 

from tensorflow import TensorShape

from DatasetLoader import *
from models.vae_model import *

## Experimental
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


res_list = ["Log name", "Reconstruction loss", "Detection loss", "Val R", "Val P", "Test R", "Test P"]
with open('kampff_all_results.csv', 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(res_list)

file_list = [
	
	#"2015_08_21_Cell.3.0_gt0",
	#"2015_08_21_Cell.3.1_gt0",
	#"2015_08_28_Cell.2.0_gt0",
	#"2015_08_28_Cell.2.1_gt0",
	#"2015_08_28_Cell.2.2_gt0",
	#"2015_08_28_Cell.9.0_gt0",
	#"2015_09_03_Cell.6.0_gt0",
	#"2015_09_03_Cell.6.1_gt0",
	#"2015_09_03_Cell.9.0_gt0",
	#"2015_09_04_Cell.5.0_gt0",
	#"2015_09_04_Cell.6.0_gt0",
	#"2015_09_04_Cell.6.1_gt0",
	#"2015_09_09_Cell.4.0_gt0",
	#"2015_09_09_Cell.6.0_gt0",
	"2015_09_09_Cell.7.0_gt0",
	"2015_09_09_Cell.7.1_gt0"
]
output_dir = "data\\generated_data\\kampff\\"

for file in file_list:

	unique_config_dir = os.path.join(output_dir, file, "data_config.xml")
	save_dir = os.path.join("trained_models\\kampff\\", file)
	Path(save_dir).mkdir(parents=True, exist_ok=True)

	config={}
	config["epochs"] = 75
	config["batch_size"] = 512
	config["load_pretrained"] = True
	config["clustering_mode"] = True
	config["fine_tune_detector"] = False
	config["losses"] = {"ce_eq":3, "beta":15}
	config["beta_scheduler_enabled"] = False

	print(unique_config_dir)
	data = DatasetLoader(data_config_file=unique_config_dir, batch_size=config["batch_size"])


	config["max_clusters"] = data.max_clusters
	config["use_multi_gpu"] = False

	vae_model = VAE_sorter(config_file="model_config.xml", config=config)
	vae_model.build()
	hist = vae_model.fit(data.train_dataset, data.val_dataset)

	time.sleep(1)
	vae_model.save_weights_separate(custom_dir=save_dir)


	vae_model.calcValTestRP(data.val_dataset, data.test_dataset, target_dir = save_dir)

