# Import Needed Packages
import numpy as np
import os
import shutil, sys
import cv2
import imageio
from PIL import Image

#from keras.models import load_model
#from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from collections import Counter
import operator

import time
import argparse

navya = False
sander = False
emir = False

try:
    path = r'/Users/navya/Desktop/Capstone/thermal-pedestrian-detection-cnn'
    navya = True
    print("Navya's path available")
except:
    try:
        path = r'C:\\Users\\abrah\\Documents\\GitHub\\thermal-pedestrian-detection-cnn'
        sander = True
        print("Sander's path available")
    except:
        try:
            path = r'/Users/navya/Desktop/Capstone/thermal-pedestrian-detection-cnn'
            emir = True
            print("Emir's path available")
        except:
            print("No Available Paths")



