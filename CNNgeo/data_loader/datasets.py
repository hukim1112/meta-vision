import os, random
import itertools
from functools import partial
import numpy as np
import cv2
import tensorflow as tf
from .data_utils import pad_image, synthesize_with_tps

class PF_Pascal():
    #PF-Pascal dataset
    ##download : https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip
    def __init__(self, dir_path):
        '''
           dir_path : the location of PF-dataset-PASCAL directory.
        '''
        self.PATH_TO_JPEGS = os.path.join(dir_path, "PF-dataset-PASCAL", "JPEGImages")
        self.PATH_TO_ANNOTATIONS = os.path.join(dir_path, "PF-dataset-PASCAL", "Annotations")
        self.category_list = os.listdir(self.PATH_TO_ANNOTATIONS)
        #filter out .DS_Store file, Its not category.
        for i, cat in enumerate(self.category_list):
            if cat == ".DS_Store":
                self.category_list.pop(i)
        self.num_category = len(self.category_list)

    def SynthesizedImagePairMode(self, input_shape, num_examples, data_normalize):
        if num_examples == -1:
            image_list = os.listdir(self.PATH_TO_JPEGS)
        elif num_examples > 0:
            image_list = os.listdir(self.PATH_TO_JPEGS)[:num_examples]
        else:
            raise ValueError("wrong num_examples : {}".format(num_examples))
        random.shuffle(image_list)
        for imagename in image_list:
            imageA = cv2.imread(os.path.join(self.PATH_TO_JPEGS, imagename))[:,:,::-1]
            imageA = cv2.resize(imageA, input_shape, interpolation = cv2.INTER_AREA)
            #imageA => padding => tps transform => imageB
            pad_ratio = 0.3
            padded_image = pad_image(imageA.copy(), pad_ratio)
            '''
                original control points are like this.
                control_points = tf.constant([[-1.0,-1.0], [0.0,-1.0], [1.0,-1.0],
                                           [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                                           [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]], dtype=tf.float32)
                padding causes new positions of control points.
            '''
            np = 1 / (1 + pad_ratio)
            control_points = tf.constant([[-np, -np], [0.0, -np], [np, -np],
                                          [-np, 0.0], [0.0, 0.0], [np, 0.0],
                                          [-np, np], [0.0, np], [np, np]], dtype=tf.float32)
            tps_random_rate = 0.3/(1+pad_ratio)
            #original random rate is 0.3. but images to be synthesized are padded, therefore divide it with (1+pad_ratio)
            motion_vectors = (tf.random.uniform([9, 2]) - 0.5) * 2 * tps_random_rate
            # 9x2 motion vector. each element is in [-tps_random_rate,tps_random_rate]
            synth_image = synthesize_with_tps(padded_image, control_points, motion_vectors, padded_image.shape[0:2])
            dH = (padded_image.shape[0] - imageA.shape[0]) / 2
            dW = (padded_image.shape[1] - imageA.shape[1]) / 2
            dH, dW = int(dH), int(dW)
            imageB = synth_image[dH:-dH, dW:-dW]
            imageA = data_normalize(imageA)
            imageB = data_normalize(imageB)
            yield imageA, imageB, motion_vectors

    def CategoricalImagePairMode(self, input_shape, num_examples, data_normalize):
        category_images = []
        for cat in self.category_list:
            _tmp = os.listdir(os.path.join(self.PATH_TO_ANNOTATIONS, cat))
            image_lists_of_the_category = [ os.path.splitext(m_file)[0] + ".jpg" for m_file in _tmp] #convert extensions(.m -> .jpg) each file names
            category_images.append(image_lists_of_the_category)
        cats = list(range(self.num_category))
        for i in range(num_examples):
            cat = np.random.choice(cats) # select one category randomly
            pair = np.random.permutation(len(category_images[cat]))[:2] #select pair randomly in the category.
            imageA = cv2.imread(os.path.join(self.PATH_TO_JPEGS, category_images[cat][pair[0]]))[:,:,::-1]
            imageA = cv2.resize(imageA, input_shape, interpolation = cv2.INTER_AREA)
            imageA = data_normalize(imageA)
            imageB = cv2.imread(os.path.join(self.PATH_TO_JPEGS, category_images[cat][pair[1]]))[:,:,::-1]
            imageB = cv2.resize(imageB, input_shape, interpolation = cv2.INTER_AREA)
            imageB = data_normalize(imageB)
            yield  imageA, imageB

    def load_pipeline(self, mode, input_shape=(200,200), num_examples=1000, data_normalize=lambda x : x/255.):
        if mode == "CategoricalImagePair":
            gen = partial(self.CategoricalImagePairMode, input_shape, num_examples, data_normalize)
            ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
            return ds
        elif mode == "SynthesizedImagePair":
            gen = partial(self.SynthesizedImagePairMode, input_shape, num_examples, data_normalize)
            ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32))
            return ds
        else:
            print("not yet implemented")
