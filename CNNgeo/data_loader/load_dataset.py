import os
import itertools
import numpy as np
import cv2

class PF_Pascal():
    #PF-Pascal dataset
    ##download : https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip
    def __init__(dir_path):
    '''
       dir_path : the location of PF-dataset-PASCAL directory.
    '''
    self.PATH_TO_JPEGS = os.path.join(dir_path, "PF-dataset-PASCAL", "JPEGImages")
    self.PATH_TO_ANNOTATIONS = os.path.join(dir_path, "PF-dataset-PASCAL", "Annotations")
    self.category_list = os.listdir(PATH_TO_ANNOTATIONS)

    self.num_category = len(self.category_list)
    def CategoricalImagePairMode(examples):
        category_images = []
        for cat in self.category_list:
            image_lists_of_the_category = os.listdir(os.path.join(self.PATH_TO_ANNOTATIONS, cat))
            category_images.append(image_lists_of_the_category)
        cats = list(range(self.num_category))
        for i in range(examples):
            cat = np.random.choice(cats, 1) # select one category randomly
            pair = np.random.permutation(len(category_images[cat]))[:2] #select pair randomly in the category.
            imageA = cv2.imread(os.path.join(self.PATH_TO_JPEGS, category_images[cat][pair[0]]))[:,:,::-1]
            imageB = cv2.imread(os.path.join(self.PATH_TO_JPEGS, category_images[cat][pair[1]]))[:,:,::-1]
            yield  imageA, imageB






pf_pascal category_list , image folder
coco => image_id => image_id => annotation searching

mini imagenet => tensor _class x _num x image
