import os, glob
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
import xml.etree.ElementTree as ET
from lxml import etree, objectify
import pdb
import cv2 as cv
import numpy as np
import argparse
import random as rng

# Annotations path
annotations = glob.glob('annotations-xml/set*')
annotations = sorted(annotations,key = lambda x:x[::-2])
annotations_target = os.path.basename(annotations[2])
#print(annotations_target)
set_num = 0

# Sets path
sets = glob.glob('Sets/set*')
sets = sorted(sets,key = lambda x:x[::-2])
target_set = os.path.basename(sets[0])
#print(target_set)

main_folder = os.path.dirname(os.path.abspath(__file__))
#print(main_folder)

i = 0

# Loop through every set in annotations-xml
for directory in annotations:
    # Get relative path
    directory_path = os.path.basename(directory)
    annotation_videos = sorted(glob.glob(str(directory)+'/V*'))
    #print("Set annotation path: " + str(directory_path))

    # Loop through every video in every set in annotations-xml
    for video_dir in annotation_videos[0:2]:
        video_annotation_path = os.path.basename(video_dir)

        # Loop through every set in Sets
        for subset in sets:
            subset_path = os.path.basename(subset)
            # !!!! IF SETS MATCH
            if subset_path == directory_path:
                print("Video annotation path: " + str(directory_path) + "/" + str(video_annotation_path))
                print("Matching set path found for: " + str(subset_path))
                xml_files = sorted(glob.glob(str(video_dir) + '/*.xml'))
                print(xml_files)
                sets_videos = sorted(glob.glob(str(subset) + '/V*/lwir'))
                for set_video in sets_videos:
                    print(set_video)
                    set_lwir_path = os.path.join(main_folder, set_video)
                    for image in os.listdir(set_lwir_path)[0:2]:
                        try:
                            img_name = os.path.splitext(image)[0]
                            print("image: " + img_name)
                            # Find matching image to annotation based on base name
                            for anno in xml_files[0:2]:
                                # Get basename and look for a match
                                anno_name = os.path.splitext(os.path.basename(anno))[0]
                                if img_name == anno_name:
                                    print("match: " + anno)
                                    break # Break at a match and anno = matching xml
                            # Get objects in xml annotation
                            anno_tree = etree.parse(anno)
                            for element in anno_tree.iter():



                            # Pull base file name of XML (ie: I00000)
                            # Check = of base file name with image
                            # Pull XML bounding box data
                            # Draw bounding box
                            # Show bounding box on image
                            # blue = cyclist, red = people, green = person, purple = person?
                        except:
                            print("Error when processing: " + str(image))
            else:
                pass
    set_num += 1



