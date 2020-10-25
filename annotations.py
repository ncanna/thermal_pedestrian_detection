# Import Needed Packages
import glob
import os
import json
from scipy.io import loadmat
from collections import defaultdict
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

# Annotations path
annotations = glob.glob('annotations/set*')
annotations = sorted(annotations,key = lambda x:x[::-2])
set_num = 0

# Provide a default value for the key that does not exists.
df = defaultdict(lambda: "Not Present")

for directory in annotations:
    # Get relative path
    directory_path = os.path.basename(directory)
    #print(directory_path)
    vbbs = sorted(glob.glob(str(directory)+'/*.vbb'))
    #print(vbbs)
    vbb_num = 0
    for vbb in vbbs:
        annotation = loadmat(vbb)
        if vbb_num == 0 and set_num == 0:
        #    print(list(annotation.keys()))
        #  ['__header__', '__version__', '__globals__', 'A', 'vers']
            print(annotation['A'][0][0][1][0])
        vbb_num += 1
    set_num += 1
