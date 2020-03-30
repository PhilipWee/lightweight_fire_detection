# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:22:49 2020

@author: Philip
"""

import os
import numpy as np
import cv2
import re
import time
import PIL.Image

from pydarknet import Detector, Image

YOLOV3_PATH = "yolov3-tiny-obj.cfg"
#YOLOV3_PATH = "YOLO3-4-Py/cfg/yolov3-tiny.cfg"
WEIGHTS_PATH = "yolov3-tiny-obj_3000.weights"
#WEIGHTS_PATH = "YOLO3-4-Py/weights/yolov3-tiny.weights"
COCO_DATA_PATH = "obj.data"
ENCODING = "utf-8"

class design_fire_cv():

    #Load the darknet model
    def load_darknet(self):
        self.net = Detector(bytes(YOLOV3_PATH, encoding=ENCODING),\
                            bytes(WEIGHTS_PATH, encoding=ENCODING),0,\
                            bytes(COCO_DATA_PATH, encoding=ENCODING),\
                                )
    
    #Test an image
    def test_img_from_path(self,img_path):
        img = PIL.Image.open(img_path)
        img = np.array(img)
        img = img[:,:,::-1]
        img2 = Image(img)
        return self.test_img(img2)

    def test_img(self,img):
        results = self.net.detect(img)
        return results


fire_detector = design_fire_cv()
fire_detector.load_darknet()
print(fire_detector.test_img_from_path('test.jpg'))