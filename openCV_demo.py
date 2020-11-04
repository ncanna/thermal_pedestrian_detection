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
    for video_dir in annotation_videos:
        video_annotation_path = os.path.basename(video_dir)
        # Loop through every set in Sets
        for subset in sets:
            subset_path = os.path.basename(subset)
            #print("Subset" + str(subset_path))
            # If set names match
            if subset_path == directory_path:
                #print("Video annotation path: " + str(directory_path) + "/" + str(video_annotation_path))
                #print("Matching set path found for: " + str(subset_path))
                xml_files = sorted(glob.glob(str(video_dir) + '/*.xml'))
                #print(xml_files)
                sets_videos = sorted(glob.glob(str(subset) + '/V*/lwir'))
                #print(sets_videos)

                # Run script only on video with appropriate index
                set_level_base = "Sets/"+str(directory_path) + "/" + str(video_annotation_path)+"/lwir"
                #print(set_level_base)
                for set_video in sets_videos:
                    if set_video == set_level_base:
                        print(set_video)
                        # Get absolute path of LWIR directory
                        #print("Image Files Path: " + str(set_video))
                        set_lwir_path = os.path.join(main_folder, set_video)

                        # Make Annotated Images Folder if Not Exists
                        abs_video_path = os.path.join(main_folder, set_video)[:-5]
                        abs_anno_images_path = abs_video_path+"/annotated"
                        #print("Abs. Annotated Imgs Path: " + str(abs_anno_images_path))
                        if os.path.exists(abs_anno_images_path):
                            shutil.rmtree(abs_anno_images_path)
                            os.makedirs(abs_anno_images_path)
                        else:
                            os.makedirs(abs_anno_images_path)

                        for image in os.listdir(set_lwir_path):
                            cv_img = cv.imread(set_lwir_path + "/" + image)
                            try:
                                img_name = os.path.splitext(image)[0]
                                #print("Image Base: " + img_name)
                                # Find matching image to annotation based on base name
                                for anno in xml_files:
                                    # Get basename and look for a match
                                    anno_name = os.path.splitext(os.path.basename(anno))[0]
                                    #print("Annotation Base: " + str(anno_name))
                                    if img_name == anno_name:
                                        #print("Match for Base: " + anno_name)
                                        #break  # Break at a match and anno = matching xml
                                # Get objects in xml annotation
                                        anno_tree = etree.parse(anno)

                                        for element in anno_tree.iter():
                                            if element.tag == "object":
                                                obj_type = element[0].text
                                                #print(obj_type)
                                                bottom_left = (int(float(element[1][0].text)),
                                                               int(float(element[1][1].text)))  # xmin and ymin
                                                top_right = (int(float(element[1][2].text)),
                                                             int(float(element[1][3].text)))  # xmax and ymax
                                                # print(bottom_left)

                                                # Get colors based on object, format is bgr
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
                                        # print("Abs. Annotated Images Path: " + str(abs_anno_images_path))
                                        # print("Annotated Image File Name: " + str(annotated_image_file_name))
                                        annotated_image_file_name = img_name + "_bounded.jpg"
                                        annotated_image_file_path = abs_anno_images_path + "/" + annotated_image_file_name
                                        # print("Annotated Image Name: " + str(annotated_image_file_path))
                                        # Image by image check. waitkey functions waits for number of milliseconds to wait for a button press (0 mean infinite)
                                        # cv.imshow(img_name, cv_img)
                                        # cv.waitKey(0)
                                        # cv.destroyAllWindows()
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



