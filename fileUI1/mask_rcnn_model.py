import os
import gc
import sys
import time
import json
import glob
import random
from pathlib import Path
import pandas as pd

from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

import itertools
from tqdm import tqdm

from mrcnn.config import Config

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from threading import Thread

IMAGE_SIZE = 512
NUM_CATS = 4


class mask_rcnn_model():

    def __init__(self):
        self.config = SteelConfig()
        self.inference_config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode='inference', 
                    config=self.inference_config,
                    model_dir='/home/danny/Documents/git/Mask_RCNN/samples/')
        self.model.load_weights('mask_rcnn_steel_0009.h5', by_name=True)
        self.sample_df =  pd.read_csv("train.csv")
        self.sample_df = self.sample_df.dropna()
        self.test_df = pd.DataFrame(
            columns=["image_id","EncodedPixels","CategoryId"])
        self.test_dict = {}
        for idx,row in self.sample_df.iterrows():
            image = row.ImageId_ClassId.split("_")
            image_filename = image[0]
            if image_filename not in self.test_dict:
                self.test_dict[image_filename] = [int(image[1])]
            else:
                self.test_dict[image_filename].append(int(image[1]))

            self.test_df = \
            self.test_df.append({"image_id": image_filename},ignore_index=True)
        plt.ion()
        self.test_df = self.test_df.drop_duplicates()

    def resize_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
        return img

    def run_model(self):
        ccr = 0
        vis_count = 0
        category_list = ["1", "2", "3", "4"]
        for i in range(5):
            image_id = self.test_df.sample()["image_id"].values[0]
            image_path = str('./train_images/'+image_id)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = self.model.detect([self.resize_image(image_path)])
            r = result[0]
            if(result[0]['class_ids'].size != 0):
                if result[0]['class_ids'][0] in self.test_dict[image_id]:
                    ccr += 1
                    
            if r['masks'].size > 0:
                masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
                for m in range(r['masks'].shape[-1]):
                    masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                            
                y_scale = img.shape[0]/IMAGE_SIZE
                x_scale = img.shape[1]/IMAGE_SIZE
                rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
            else:
                masks, rois = r['masks'], r['rois']

            visualize.display_instances(img, rois, masks, r['class_ids'], 
                                ['bg']+category_list, r['scores'],title=image_id)
        ccr = ccr/5.0
        print(ccr)

         
class SteelConfig(Config):
    NAME = "steel"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    
    BACKBONE = 'resnet101'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    
    STEPS_PER_EPOCH = 4500
    VALIDATION_STEPS = 500

    
class InferenceConfig(SteelConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
