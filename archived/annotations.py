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

db_path = os.path.join(os.path.dirname(__file__), '..', 'thermal-pedestrian-detection-cnn')
vbb_inputdir = os.path.join(db_path, 'annotations')
sub_dirs = os.listdir(vbb_inputdir)

# Annotations path
annotations = glob.glob('annotations/set*')
annotations = sorted(annotations,key = lambda x:x[::-2])
set_num = 0
sum_label_count = 0

# Provide a default value for the key that does not exists.
df = defaultdict(lambda: "Not Present")

for directory in annotations:
    # Get relative path
    directory_path = os.path.basename(directory)
    print(directory_path)
    vbbs = sorted(glob.glob(str(directory)+'/*.vbb'))
    #print(vbbs)
    vbb_num = 0
    df[directory_path] = defaultdict(dict)
    for vbb in vbbs:
        annotation = loadmat(vbb)
        #if vbb_num == 0 and set_num == 0:
        #    print(list(annotation.keys()))
        #  ['__header__', '__version__', '__globals__', 'A', 'vers']
            #print(annotation['A'][0][0][1][0])

        # https://dbcollection.readthedocs.io/en/latest/_modules/dbcollection/utils/db/caltech_pedestrian_extractor/converter.html
        nFrame = int(annotation['A'][0][0][0][0][0])
        # Lists of labels
        objLists = vbb['A'][0][0][1][0]
        objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]

        nFrame = int(vbb['A'][0][0][0][0][0])
        maxObj = int(vbb['A'][0][0][2][0][0])
        objInit = vbb['A'][0][0][3][0]
        objStr = vbb['A'][0][0][5][0]
        objEnd = vbb['A'][0][0][6][0]
        objHide = vbb['A'][0][0][7][0]
        altered = int(vbb['A'][0][0][8][0][0])
        log = vbb['A'][0][0][9][0]
        logLen = int(vbb['A'][0][0][10][0][0])

        V_str = os.path.splitext(os.path.basename(vbb))[0]
        #print(V_str)
        df[directory_path][V_str]['nFrame'] = nFrame
        df[directory_path][V_str]['maxObj'] = maxObj
        df[directory_path][V_str]['log'] = log.tolist()
        df[directory_path][V_str]['logLen'] = logLen
        df[directory_path][V_str]['altered'] = altered
        df[directory_path][V_str]['frames'] = defaultdict(list)

        label_count = 0
        for frame_id, obj in enumerate(objLists):
            frame_name = '/'.join([directory_path, V_str, 'I{:05d}'.format(frame_id)])
            df[frame_name] = defaultdict(list)
            df[frame_name]["id"] = frame_name

            if len(obj[0]) > 0:
                for id, pos, occl, lock, posv in zip(
                        obj['id'][0], obj['pos'][0], obj['occl'][0],
                        obj['lock'][0], obj['posv'][0]):
                    # For matlab start from 1 not 0
                    id = int(id[0][0]) - 1
                    pos = pos[0].tolist()
                    occl = int(occl[0][0])
                    lock = int(lock[0][0])
                    posv = posv[0].tolist()

                    df[frame_name]["label"].append(objLbl[id])
                    df[frame_name]["occlusion"].append(occl)
                    df[frame_name]["bbox"].append(pos)
                    label_count += 1

        #print(directory, vbb, label_count)
        sum_label_count += label_count
        vbb_num += 1
    set_num += 1

print('Total labels:', sum_label_count)
json.dump(df, open('data/annotations.json', 'w'))