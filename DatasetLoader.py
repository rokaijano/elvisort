import tensorflow as tf
import xmltodict
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder        
from tensorflow.python.framework import ops


class DatasetLoader(object):
    def __init__(self, image_shape=[64,128], batch_size = 32, data_config_file="config\\data_config.xml", autoload=True, only_binary_labels = False, only_positives=False, maml=False, for_evaluation = False):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.only_positives = only_positives
        self.for_evaluation = for_evaluation

        with open(data_config_file) as fd:
            self.data_config = xmltodict.parse(fd.read())
            self.data_config = self.data_config["data"]


        self.image_shape = [int(self.data_config["timespan"]), int(self.data_config["channels"])]
        self.max_clusters = int(self.data_config["max_clusters"])+1
        self.only_binary_labels = only_binary_labels

        self.data_config.setdefault("meta_learning_data", False)
        self.data_config.setdefault("multi_label", False)
        self.data_config.setdefault("filter_train_to_ratio", True)
        self.data_config.setdefault("train_filenames", "train.tfrecord")
        self.data_config.setdefault("val_filenames", "val.tfrecord")
        self.data_config.setdefault("test_filenames", "test.tfrecord")
            
        self.meta_learning_data = self.data_config["meta_learning_data"] == "true" or self.data_config["meta_learning_data"] == "True" or maml
        
        if self.meta_learning_data:
            self.batch_size = 1  ## HERE WE OVERWRITE
            self.data_config.setdefault("target_to_split", 100)
            self.target_to_split = int(self.data_config["target_to_split"])# *3 //4

        if autoload:
            self.load_data()

        self.training_steps = int(int(self.data_config["train"]["samples"]) / batch_size) #32 ## number of samples in training dataset = 189898
        self.validation_steps = int(int(self.data_config["val"]["samples"]) / batch_size) # 
        self.test_steps = int(int(self.data_config["test"]["samples"]) / batch_size)
        

    def load_data(self):
        self.load_train_dataset()
        self.load_val_dataset()
        self.load_test_dataset()
    
    def load_train_dataset(self):
        self.train_dataset = self.get_dataset(self.data_config["train_filenames"], samples=self.data_config["train"]["samples"], pos_rate=float(self.data_config["train"]["ratio"]))

    def load_val_dataset(self):
        if not self.meta_learning_data:
            self.val_dataset = self.get_dataset(self.data_config["val_filenames"], samples=self.data_config["val"]["samples"])

    def load_test_dataset(self):
        self.test_dataset = self.get_dataset(self.data_config["test_filenames"], samples=self.data_config["test"]["samples"])


    def parse_fn(self, proto):
        ml = self.data_config["multi_label"] == "true"

        if ml:
            features = {'image': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.string)}
        else:
            features = {'image': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.int64)}

        parsed_features = tf.io.parse_single_example(proto, features)
        images = tf.io.decode_raw(parsed_features['image'], tf.int16)

        if ml:
            labels = tf.io.decode_raw(parsed_features['label'], tf.int32)
            #labels = tf.cast(labels, tf.int32)
            labels = tf.reshape(labels, (self.max_clusters,))
        else:
            labels = tf.cast(parsed_features['label'], tf.int64)


        if self.only_binary_labels:
        
            labels = tf.cast(labels, tf.int32)
            labels = tf.math.minimum(labels, 1)
            labels = tf.one_hot(labels, depth=2)

        else:
            if self.max_clusters == 2:
                labels = tf.math.minimum(labels, 1)

            if not ml:
                labels = tf.one_hot(labels, depth=self.max_clusters)

        images = tf.reshape(images, self.image_shape)
        
        images = tf.cast(images, tf.float32)
        
        images = (images / tf.math.reduce_max(images))

        if self.for_evaluation:
            return (images,labels), (labels, labels, [0], [0])
        else:
            return (images,labels), (images, labels, labels, [0], [0])

    def resampler_class_func(self, data, label):
        return tf.argmax(data[1])


    def resampler_class_func(self, initial_rate, target_rate=50.):
        
        initial_rate = ops.convert_to_tensor(initial_rate, name="initial_rate")
        target_rate = ops.convert_to_tensor(target_rate, name="target_rate")

        rate = target_rate / initial_rate # 25 
        prob_rate = 1.0 / rate
        
        
        if initial_rate > 50.:
            prob_rate = 1.0

        def _map_fn(data, label):
            
            label = tf.argmax(data[1])
            label= tf.cast(label, tf.float32)
            random_num = tf.random.uniform(shape=[1])
            
            bin_layer = tf.math.reduce_min([label,1])
            
            prob_rate_for_sample = [(1-bin_layer)*prob_rate] + bin_layer
        
            if random_num <= prob_rate_for_sample:
                return True
            
            return False
            
        
        return _map_fn

    def only_positive_filter(self):

        def _map_fn(data, label):

            return tf.argmax(label[1]) > 0

        return _map_fn

    def random_flip_left_right(self, data, label):

        img, label = data
        img_flipped = tf.image.random_flip_left_right(img)

        return (img_flipped, label), (img_flipped, label, label, [0], [0])

    def get_dataset(self, fname, samples=4096, pos_rate=None, augment=False):

        samples = int(samples) #

        if type(fname) in [list,tuple]:
            training_file = []
            for name in fname:
                training_file.append(os.path.join(self.data_config["dir"],name))
        else:
            training_file = os.path.join(self.data_config["dir"],fname)

        # Build an iterator over training batches.
        
        training_dataset = tf.data.TFRecordDataset(training_file)

        training_dataset = training_dataset.map(self.parse_fn, num_parallel_calls=6)

        if pos_rate is not None and self.data_config["filter_train_to_ratio"] and not self.data_config["meta_learning_data"]:
            training_dataset = training_dataset.filter(self.resampler_class_func(pos_rate)) 
            
        if self.only_positives and self.max_clusters > 2 and not self.only_binary_labels:
            training_dataset = training_dataset.filter(self.only_positive_filter())

        if augment:
            if self.max_clusters > 2:
                raise ValueError("Warning! If you have multiple clusters, you will get poor model performance if you augment data with max_clusters > 2. To fix this put only_binary_labels = True at the constructor of DatasetLoader")
            training_dataset = training_dataset.map(self.random_flip_left_right, num_parallel_calls=6)

        if self.meta_learning_data:
            training_batches = training_dataset.take(samples).batch(self.batch_size)

        else:
            training_batches = training_dataset.shuffle(8192, reshuffle_each_iteration=True).batch(self.batch_size)#.prefetch(min(1024, self.batch_size*4))#.cache()

        if not self.for_evaluation:
            training_batches = training_batches.prefetch(min(1024, self.batch_size*4)).cache()
        else:
            training_batches = training_batches.repeat()

        #sample = tf.reshape(sample, shape=image_shape)

        return training_batches

    def get_numpy(self, dataset, return_in_batches=True):

        data = list(dataset.as_numpy_iterator())
        data = np.asarray(data)
        #print("data shape: ", data.shape)

        data = data[:,0] # at this point the data is a tuple of ( images, labels )

        l_label = []
        l_images = []

        for lab in data:
            l_images.append(lab[0])
            l_label.append(lab[1])

        im = np.asarray(l_images)
        labels = np.asarray(l_label)
        #enc = OneHotEncoder(list(range(self.max_clusters))

        #labels = enc.fit_transform(labels)

        if return_in_batches:
            

            if self.meta_learning_data:

                im = np.reshape(im, (-1, self.target_to_split)+tuple(self.image_shape))
                labels = np.reshape(labels, (-1, self.target_to_split)+(labels.shape[-1],))

            else:
                im = np.asarray(im[:-2])
                labels = np.asarray(labels[:-2])

            
            return im, labels

        im = np.reshape(im, ((-1,)+tuple(im.shape[2:])))
        labels = np.reshape(labels, (-1, labels.shape[-1]))



        im = np.concatenate(im, axis=0)
        labels = np.concatenate(labels[0], axis=0)

        ## Theoretically not needed
        """
        if not multi_label:
            y_data = np.minimum(y_data, 1)
        """
        return im, labels 