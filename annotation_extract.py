import os, glob
import shutil
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

main_folder = os.path.dirname(os.path.abspath(__file__))

# Loop through every set

i = 0
for directory in annotations:
    # Get relative path
    directory_path = os.path.basename(directory)
    annotation_videos = sorted(glob.glob(str(directory)+'/V*'))

    # Loop through every video
    for video_dir in annotation_videos:
        video_annotation_path = os.path.basename(video_dir)
        # Loop through every set in Sets
        for subset in sets:
            subset_path = os.path.basename(subset)
            if subset_path == directory_path:
                xml_files = sorted(glob.glob(str(video_dir) + '/*.xml'))
                sets_videos = sorted(glob.glob(str(subset) + '/V*/lwir'))
                set_level_base = "Sets/"+str(directory_path) + "/" + str(video_annotation_path)+"/lwir"
                for set_video in sets_videos:
                    if set_video == set_level_base:
                        print(set_video)
                        set_lwir_path = os.path.join(main_folder, set_video)
                        abs_video_path = os.path.join(main_folder, set_video)[:-5]
                        abs_anno_images_path = abs_video_path+"/annotated"
                        if os.path.exists(abs_anno_images_path):
                            shutil.rmtree(abs_anno_images_path)
                            os.makedirs(abs_anno_images_path)
                        else:
                            os.makedirs(abs_anno_images_path)

                        for image in os.listdir(set_lwir_path):
                            cv_img = cv.imread(set_lwir_path + "/" + image)
                            try:
                                img_name = os.path.splitext(image)[0]
                                for anno in xml_files:
                                    anno_name = os.path.splitext(os.path.basename(anno))[0]
                                    if img_name == anno_name:
                                        anno_tree = etree.parse(anno)

                                        for element in anno_tree.iter():
                                            if element.tag == "object":
                                                obj_type = element[0].text
                                                bottom_left = (int(float(element[1][0].text)),
                                                               int(float(element[1][1].text)))  # xmin and ymin
                                                top_right = (int(float(element[1][2].text)),
                                                             int(float(element[1][3].text)))  # xmax and ymax

                                                # Get colors based on object, format is rgb
                                                # Blue = cyclist, red = people, green = person, purple = person?
                                                color = (0, 0, 0)
                                                if obj_type == "cyclist":
                                                    color = (255, 0, 0)
                                                elif obj_type == "people":
                                                    color = (0, 0, 255)
                                                elif obj_type == "person":
                                                    color = (0, 255, 0)
                                                elif obj_type == "person?":
                                                    color = (128, 0, 128)

                                                cv.rectangle(cv_img, bottom_left, top_right, color, 1)
                                                cv.putText(cv_img, obj_type,
                                                           (int(float(element[1][0].text)), int(float(element[1][3].text))),
                                                           1, 2, color, 1)
                                            else:
                                                pass

                                        # Save cv_img
                                        annotated_image_file_name = img_name + "_bounded.jpg"
                                        annotated_image_file_path = abs_anno_images_path + "/" + annotated_image_file_name
                                        cv.imwrite(annotated_image_file_path, cv_img)
                                    else:
                                        pass
                            except Exception as e:
                                print(e)
                                print("Error when processing: " + str(image))
                    else:
                        pass
            else:
                pass
    set_num += 1



