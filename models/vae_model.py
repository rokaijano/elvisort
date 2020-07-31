import tensorflow as tf
import numpy as np
import sys
import copy
import pdb
import xmltodict
import csv
import os
from contextlib import nullcontext

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import * 
from tensorflow.keras.models import Model, clone_model, load_model, save_model
from tensorflow.keras.optimizers import Adam

from models.layers.LatentLayer import *
from models.layers.ClusteringLayer import *
from models.losses import *
from datetime import datetime
from os import path
from sklearn.metrics import f1_score, homogeneity_completeness_v_measure, confusion_matrix, multilabel_confusion_matrix


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class BetaScheduler(tf.keras.callbacks.Callback):

    def __init__(self, low = 2, high = 15, freq = 20, enabled=True):
        self.low = low 
        self.high = high
        self.freq = freq
        self.use_scheduler = enabled
        
    def on_epoch_end(self, epoch, logs=None):
        
        if not self.use_scheduler:
            return
        
        inside_step = epoch % self.freq
        
        if inside_step < self.freq //2:
            beta = (self.high/((self.freq//2)**2))*inside_step*inside_step
        else:
            beta = self.high
        
        if epoch % self.freq != 0:
            return

class VAE_sorter(object):

    def __init__(self, config_file="", config={}):
        
   
        self.optimizer = Adam(learning_rate = 0.001)

        self.vae = None
        self.encoder = None
        self.decoder = None
        self.detector = None
        self.clustering = None
        self.full_clustering_model = None
        self.logname = ""
        self.train_verbosity = 1

        if config_file != "" and path.exists(config_file):
            with open(config_file) as fd:
                try:
                    self.config = xmltodict.parse(fd.read())
                    self.config = self.config["root"]
                    self.config.update(config)

                except Exception:
                    self.config = config
        else:
            self.config = config

        ##losses
        self.config.setdefault("losses", {})
        self.config["losses"].setdefault("alpha", 256)
        self.config["losses"].setdefault("beta", 10)
        self.config["losses"].setdefault("delta", 10)
        self.config["losses"].setdefault("theta", 0)
        self.config["losses"].setdefault("kappa", 0)
        self.config["losses"].setdefault("ce_eq", 1)

        self.config.setdefault("use_convolutions", True)

        self.config.setdefault("clustering_mode", False)
        self.config.setdefault("use_latentloss", False)
        self.config.setdefault("load_pretrained", False)
        self.config.setdefault("fine_tune_detector", False)
        self.config.setdefault("use_multi_gpu", False)
        self.config.setdefault("beta_scheduler_enabled", True)
        self.config.setdefault("multi_label", False)

        self.config.setdefault("latent_dim", 32)
        self.config.setdefault("upper_latent_dim", 8)
        self.config.setdefault("batch_size", 32)
        self.config.setdefault("epochs", 200)
        self.config.setdefault("channels", 128)
        self.config.setdefault("timespan", 64)
        self.config.setdefault("validation_freq", 2)
        self.config.setdefault("cluster_update_interval", 15)
        self.config.setdefault("test_steps", 7607)
        self.config.setdefault("epoch_per_visualization", 5)
        self.config.setdefault("step_per_visualization", 500)
        self.config.setdefault("pretrained_model_path", "trained_models")
        self.config.setdefault("full_model_path", "trained_models\\vae_model.h5")
        self.config.setdefault("global_ae_dropout", .5)
        self.config.setdefault("global_detector_dropout", .5)
        self.config.setdefault("conv_dropout", 0.5)

        if config_file == "":
            config_file = "model_config.xml"
        with open(os.path.join(config_file), "w") as fd:
            root = {"root":config}
            fd.write(xmltodict.unparse(root, pretty=True))


        # we cleanup the fake booleans, and make them real

        boolean_keys = [
            "clustering_mode", 
            "use_latentloss", 
            "fine_tune_detector", 
            "test_on_batch", 
            "binary_detection", 
            "meta_learning_data", 
            "only_last_inner_loop", 
            "load_pretrained", 
            "load_detector", 
            "freeze_ae",
            "only_positives",
            "use_multi_gpu",
            "beta_scheduler_enabled",
            "multi_label",
            "use_convolutions",
            "exclude_decoder"
            ]

        string_keys = [
            "pretrained_model_path", 
            "full_model_path", 
            "maml_model_save_path", 
            "losses",
            "dir",
            "dataset",
            "train",
            "val",
            "test",
            "cluster_snr"
            ]

        int_keys = self.config.copy()
        for key in boolean_keys + string_keys:
            int_keys.pop(key, None)

        for key in int_keys:
            try:
                self.config[key] = int(self.config[key])
            except ValueError:
                try:
                    self.config[key] = float(self.config[key])
                except ValueError:
                    print("ValueError at key: ",key)
                    exit()
            except TypeError:
                if key == "image_shape":
                    continue
                print("TypeError at key: ",key)
                exit()

        self.config["losses"]["alpha"] = float(self.config["losses"]["alpha"])
        self.config["losses"]["beta"]  = float(self.config["losses"]["beta"])
        self.config["losses"]["delta"] = float(self.config["losses"]["delta"])
        self.config["losses"]["theta"] = float(self.config["losses"]["theta"])
        self.config["losses"]["kappa"] = float(self.config["losses"]["kappa"])
        self.config["losses"]["ce_eq"] = float(self.config["losses"]["ce_eq"])

        for key in boolean_keys:
            if key in self.config and not isinstance(self.config[key], bool):
                self.config[key] = self.config[key] == "true"


        # we dont save this to config file
        self.config["image_shape"] = [self.config["timespan"],self.config["channels"]]

        self.config.setdefault("gpu", 0)
        self.config.setdefault("exclude_decoder", False)

    def lenet(self, input, filter):
        
        x3 = Conv2D(filter, kernel_size=3, padding="same", activation="relu")(input)
        x5 = Conv2D(filter, kernel_size=5, padding="same", activation="relu")(input)
        x1 = Conv2D(filter, kernel_size=1, padding="same", activation="relu")(input)

        concat = Concatenate(axis=-1)([x1,x3,x5])

        return Conv2D(filter, kernel_size=1, padding="same", activation="relu")(concat)

    def encoder_model(self, inputs, encoder_id="1"):
        
        image_shape = self.config["image_shape"]
        latent_dim = self.config["latent_dim"]
        upper_latent_dim = self.config["upper_latent_dim"]

        x = Reshape((image_shape[0]*image_shape[1],))(inputs)
        x = BatchNormalization()(x)
        x = Reshape((image_shape[0], image_shape[1]))(x)
        
        #"""
        if self.config["use_convolutions"]:
            conv_x = Reshape((image_shape[0], 4, 32))(x)
            conv_x = tf.transpose(conv_x, perm=[0, 3, 2, 1])
            conv_x = self.lenet(conv_x, 32)
            conv_x = Dropout(self.config["conv_dropout"])(conv_x)
            conv_x = self.lenet(conv_x, 64)
            conv_x = Dropout(self.config["conv_dropout"])(conv_x)
            conv_x = Conv2D(64, kernel_size=2, padding="valid", activation="relu")(conv_x)
            conv_x = self.lenet(conv_x, 128)
            conv_x = Dropout(self.config["conv_dropout"])(conv_x)
            conv_x = Conv2D(128, kernel_size=2, padding="valid", activation="relu")(conv_x)
            conv_x = self.lenet(conv_x, 256)
            conv_x = Dropout(self.config["conv_dropout"])(conv_x)
            conv_x = Flatten()(conv_x)
        #"""


        # BiLSTM-Att block
        blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=self.config["global_ae_dropout"]))(x)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=self.config["global_ae_dropout"]))(blstm)
        x = Attention()([blstm, x])

        # BiLSTM-Att block
        blstm = Bidirectional(LSTM(64, return_sequences=True, dropout=self.config["global_ae_dropout"]))(x)
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=self.config["global_ae_dropout"]))(blstm)
        x = Attention()([blstm, x])
        
        x = Bidirectional(LSTM(32, return_sequences=True, dropout=self.config["global_ae_dropout"]))(x)

        x = Flatten()(x)

        if self.config["use_convolutions"]:
            x = Concatenate(axis=1)([x, conv_x])
            
        x = Dense(latent_dim*2, activation='relu')(x)
        
        x = BatchNormalization()(x)
        
        z_mean = Dense(latent_dim, name='z_mean'+encoder_id)(x)
        z_log_var = Dense(latent_dim, name='z_log_var'+encoder_id)(x)
        
        z = LatentLayer(beta=self.config["losses"]["beta"])([z_mean, z_log_var])
           
        # 
        # Upper latent encoding 
        #
        
        upper_z = Dense(16, activation="relu")(z)
        
        
        upper_z = BatchNormalization()(upper_z)
        upper_mean = Dense(upper_latent_dim, name="upper_mean"+encoder_id)(upper_z)
        upper_log_var = Dense(upper_latent_dim, name="upper_log_var"+encoder_id)(upper_z)
        
        upper_z = LatentLayer(beta=self.config["losses"]["beta"])([upper_mean, upper_log_var])
        
        
        final_z = Concatenate(axis=1, name="final_z_layer"+encoder_id)([z,upper_z])

        final_z_mean = Concatenate(axis=1)([z_mean, upper_mean])    
        final_z_log_var = Concatenate(axis=1)([z_log_var, upper_log_var])

        encoder = Model(inputs, final_z, name='encoder'+encoder_id)

        return encoder, final_z

    def decoder_model(self):
        input_shape = self.config["latent_dim"]+self.config["upper_latent_dim"]
        image_shape = self.config["image_shape"]

        latent_inputs = Input(shape=(input_shape,), name='z_sampling')

        first_dense_size = 32

        x = Dense(image_shape[0]*image_shape[1], activation='relu')(latent_inputs)
        x = Reshape((image_shape[0], image_shape[1]))(x)

        x = Bidirectional(LSTM(32, return_sequences=True, dropout=self.config["global_ae_dropout"]))(x)
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=self.config["global_ae_dropout"]))(x)


        x = Bidirectional(LSTM(128, return_sequences=True, dropout=self.config["global_ae_dropout"]))(x)
        blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=self.config["global_ae_dropout"]))(x)
        x = Attention()([x, blstm])

        outputs = LSTM(128, return_sequences=True, dropout=self.config["global_ae_dropout"])(x)
        outputs = Reshape((image_shape[0], image_shape[1]))(outputs)
        decoder = Model(latent_inputs, outputs, name='reconstruction')

        return decoder

    def aux_classificator(self):
        input_shape = self.config["latent_dim"]+self.config["upper_latent_dim"]
        inputs = Input(shape=(input_shape,), name='latent_dim_detector_input')
        x = Dropout(self.config["global_detector_dropout"])(inputs)
        x = Dense(512, activation="relu")(x)
        x = Dropout(self.config["global_detector_dropout"])(x) 
        x = Dense(256, activation="relu")(x)
        x = Dropout(self.config["global_detector_dropout"])(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(self.config["global_detector_dropout"])(x)
        x = Dense(64, activation="relu")(x)
        
        if self.config["multi_label"]:
            x = Dense(self.config["max_clusters"], activation="sigmoid")(x)
        else:
            x = Dense(self.config["max_clusters"], activation="softmax")(x)

        classificator = Model(inputs, x, name='detection')
        
        return classificator

    def build(self, freeze_ae=False, use_clustering = False):

        if self.config["use_multi_gpu"]:
        

            cross_device_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=1)
            
            mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops).scope()
        else:
            mirrored_strategy = tf.device("/gpu:"+str(self.config["gpu"]))
            
        with mirrored_strategy:

            if freeze_ae:
                self.beta = 0.

            inputs = Input(self.config["image_shape"], name="input")
            inputs_aux = Input((self.config["max_clusters"],), name="input_labels")

            self.encoder, latent_layer = self.encoder_model(inputs)

            self.decoder = self.decoder_model()

            self.detector = self.aux_classificator()

            if self.config["load_pretrained"]:
                self.encoder.load_weights(self.config["pretrained_model_path"]+"\\pre_trained_vae_encoder.h5")
                self.decoder.load_weights(self.config["pretrained_model_path"]+"\\pre_trained_vae_decoder.h5")

                ## DETECTOR WEIGHT LOAD

            if self.config["fine_tune_detector"]:
                self.detector.load_weights(self.config["pretrained_model_path"]+"\\pre_trained_vae_detector.h5")


            outputs = self.decoder(latent_layer)


            if freeze_ae:
                self.encoder.trainable = False
                self.decoder.trainable = False


            detector_output = self.detector(latent_layer)

            latent_layer  = BatchNormalization()(latent_layer)


            cluster_output = ClusteringLayer(self.config["max_clusters"], name="clustering")(latent_layer)

            optimizer = Adam(learning_rate = 0.001)

            
            if self.config["exclude_decoder"]:            
                self.vae = Model([inputs, inputs_aux], [detector_output, cluster_output, latent_layer, inputs_aux], name='vae')
                self.vae.compile(optimizer=optimizer, metrics={"detection":["accuracy"]}, loss=[custom_crossentropy(use_delta = True, delta=self.config["losses"]["delta"], ce_eq=self.config["losses"]["ce_eq"]), cluster_loss(valid_loss=self.config["clustering_mode"], theta=self.config["losses"]["theta"]), latent_loss(valid_loss=self.config["use_latentloss"], kappa=self.config["losses"]["kappa"]), zero_loss])
            else:
                self.vae = Model([inputs, inputs_aux], [outputs, detector_output, cluster_output, latent_layer, inputs_aux], name='vae')
                self.vae.compile(optimizer=optimizer, metrics={"detection":["accuracy"]}, loss=[vae_loss(valid_loss=not freeze_ae, alpha=self.config["losses"]["alpha"]), custom_crossentropy(use_delta = True, delta=self.config["losses"]["delta"], ce_eq=self.config["losses"]["ce_eq"]), cluster_loss(valid_loss=self.config["clustering_mode"], theta=self.config["losses"]["theta"]), latent_loss(valid_loss=self.config["use_latentloss"], kappa=self.config["losses"]["kappa"]), zero_loss])


    def fit(self, train_dataset, val_dataset):
        
        self.logname = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = "logs\\" + self.logname
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, cooldown=3, verbose=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(self.config["pretrained_model_path"], "checkpoint.h5"),save_weights_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0,patience=6,verbose=0,mode="auto",baseline=None,restore_best_weights=True)

        return self.vae.fit(train_dataset,
                epochs=self.config["epochs"],
                    #steps_per_epoch=self.config["training_steps"],
                    validation_data=(val_dataset, None, None),
                    validation_freq = self.config["validation_freq"],
                    verbose=self.train_verbosity,
                    callbacks=[
                        reduce_lr, 
                        UpdateClusters(train_dataset, self), 
                        tensorboard_callback, 
                        BetaScheduler(enabled=self.config["beta_scheduler_enabled"]),
                        early_stopping
                        ],
                    use_multiprocessing=True,
                    workers=5
                    )
  

    def save_weights_separate(self, save_vae= True, save_detector = True, save_full_model= True, custom_dir=""):
        if custom_dir == "":
            custom_dir = self.config["pretrained_model_path"]

        if save_vae:
            # save the weight of AE
            self.encoder.save_weights(custom_dir+"\\pre_trained_vae_encoder.h5")
            self.decoder.save_weights(custom_dir+"\\pre_trained_vae_decoder.h5")
        
        if save_detector:
            self.detector.save_weights(custom_dir+"\\pre_trained_vae_detector.h5")

        if save_full_model:
            save_model(self.vae, custom_dir+"\\vae_model.h5")

        with open(os.path.join(custom_dir, "\\model_config.xml"), "w") as fd:
            fd.write(xmltodict.unparse({"root":self.config}, pretty=True))
        

    def init_result_csv(self):
        res_list = ["Log name", "Reconstruction loss", "Detection loss", "mic", "mac", "wei", "completness", "homogeneity", "vmes", "Val F1/Class", "Test F1/Class"]
        with open(target_file, 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(res_list)

    def calcValTestRP(self, val_dataset, test_dataset, target_file="results.csv", target_dir=""):

        val_all_class, val_mic, val_mac, val_wei, val_completness, val_homogeneity, val_vmes, val_conf = self.calcEverything(val_dataset, self.config["test_steps"])
        test_all_class, test_mic, test_mac, test_wei, test_completness, test_homogeneity, test_vmes, test_conf = self.calcEverything(test_dataset, self.config["test_steps"])

        res_list = [self.logname, self.config["losses"]["alpha"], self.config["losses"]["delta"]]
        res_list.extend([val_mic, val_mac, val_wei, val_completness, val_homogeneity, val_vmes])
        res_list.extend([test_mic, test_mac, test_wei, test_completness, test_homogeneity, test_vmes])

        with open(os.path.join(target_dir, "f1_clusters_val.csv"), 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(val_all_class)

        with open(os.path.join(target_dir, "f1_clusters_test.csv"), 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(test_all_class)

        with open(os.path.join(target_dir, target_file), 'w') as fd:
            label = ["Log name", "Reconstruction loss", "Detection loss", "mic", "val_mac", "val_wei", "val_completness", "val_homogeneity", "val_vmes", "test_mic", "test_mac", "test_wei", "test_completness", "test_homogeneity", "test_vmes"]

            writer = csv.writer(fd)
            writer.writerow(label)
            writer.writerow(res_list)

            writer.writerow([])
            writer.writerow(["Val confusion_matrix"])

            if self.config["max_clusters"] == 2:
                writer.writerow(["tn", "fp", "fn", "tp"])
                writer.writerow(val_conf.ravel())
            else:
                writer.writerows(val_conf)

            writer.writerow([])
            writer.writerow(["Test confusion_matrix"])

            if self.config["max_clusters"] == 2:
                writer.writerow(["tn", "fp", "fn", "tp"])
                writer.writerow(test_conf.ravel())
            else:
                writer.writerows(test_conf)

        with open(os.path.join(target_dir, "model_config.xml"), "w") as fd:
            root = {"root":self.config}
            fd.write(xmltodict.unparse(root, pretty=True))

    def calcEverything(self, data, steps):
        iterator = tf.compat.v1.data.make_one_shot_iterator(data)
        y_t = []
        y_p = []

        _, y_p, cluster_y_p,_,y_t = self.vae.predict(data)

        y_t = np.asarray(y_t)
        y_p = np.asarray(y_p)
        y_t = np.argmax(y_t, axis=1)
        y_p = np.argmax(y_p, axis=1)
        #y_p = np.argmax(cluster_y_p, axis=1)
        
        all_class = f1_score(np.asarray(y_t), np.asarray(y_p), average=None)
        mic = f1_score(np.asarray(y_t), np.asarray(y_p), average="micro")
        mac = f1_score(np.asarray(y_t), np.asarray(y_p), average="macro")
        wei = f1_score(np.asarray(y_t), np.asarray(y_p), average="weighted")
            
        print("F1 micro: "+str(mic))
        print("F1 macro: "+str(mac))
        print("F1 weighted: "+str(wei))
        
        homogeneity, completness, vmes = homogeneity_completeness_v_measure(y_t, y_p)
        print("Completness: : "+str(completness))
        print("homogeneity: : "+str(homogeneity))
        print("V measure: : "+str(vmes))
        
        if self.config["multi_label"]:
            conf = np.asarray(multilabel_confusion_matrix(y_t, y_p))
            conf = np.reshape(conf, (conf.shape[0], 4))
        else:
            conf = confusion_matrix(y_t, y_p)
        
        return all_class, mic, mac, wei, completness, homogeneity, vmes, conf


    def __lshift__(self, orig):
        
        self.vae.set_weights(orig.vae.get_weights())
        return
        
        self.config = copy.deepcopy(orig.config)
        # or we could simply equal the whole system like
        self.vae = clone_model(orig.vae)
        self.vae.build((None,) +tuple(self.config["image_shape"])) 
        
        freeze_ae = False
        use_clustering = False
        optimizer = Adam(learning_rate = 0.001)

        self.vae.compile(optimizer=optimizer, metrics={"detection":["accuracy"]}, loss=[vae_loss(valid_loss=not freeze_ae, alpha=self.config["losses"]["alpha"]), custom_crossentropy(use_delta = True, delta=self.config["losses"]["delta"], ce_eq=self.config["losses"]["ce_eq"]), cluster_loss(valid_loss=use_clustering, theta=self.config["losses"]["theta"]), latent_loss(valid_loss=use_latentloss, kappa=self.config["losses"]["kappa"])])
        
        self.vae.set_weights(orig.vae.get_weights())


    def call(self, *args, **kwargs):
        return self.vae.call(*args, **kwargs)