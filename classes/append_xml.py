import glob
import os
from lxml import etree, objectify
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring
import pandas as pd

# Annotations path
annotations = glob.glob('../annotations-xml/set*')
annotations = sorted(annotations,key = lambda x:x[::-2])
set_num = 0

df = pd.DataFrame()
i = 0
for directory in annotations:
    # Get relative path
    directory_path = os.path.basename(directory)
    video_dirs = sorted(glob.glob(str(directory)+'/V*'))
    for video_dir in video_dirs:
        video_dir_path = os.path.basename(video_dir)
        xml_files = sorted(glob.glob(str(video_dir) + '/*.xml'))
        print("Video Folder: " + str(directory_path) + "/" + str(video_dir_path))
        for xml in xml_files:
            try:
                tree = ET.parse(xml)
                root = tree.getroot()
                filename = root.find('filename').text
                #print(filename)
                for size_object in root.findall('size'):
                    width = size_object.find('width').text
                    height = size_object.find('height').text
                    depth = size_object.find('depth').text
                for object in root.findall('object'):
                    name = object.find('name').text
                    bbox = object.find('bndbox')
                    xmin = bbox[0].text
                    ymin = bbox[1].text
                    xmax = bbox[2].text
                    ymax = bbox[3].text
                    pose = object.find('pose').text
                    truncated = object.find('truncated').text
                    difficult = object.find('difficult').text
                    occlusion = object.find('occlusion').text
                    df.loc[i,"filename"] = filename
                    df.loc[i,"width"] = filename
                    df.loc[i,"height"] = height
                    df.loc[i,"depth"] = depth
                    df.loc[i,"name"] = name
                    df.loc[i,"xmin"] = xmin
                    df.loc[i,"ymin"] = ymin
                    df.loc[i,"xmax"] = xmax
                    df.loc[i,"ymax"] = ymax
                    df.loc[i,"pose"] = pose
                    df.loc[i,"truncated"] = truncated
                    df.loc[i,"difficult"] = difficult
                    df.loc[i,"occlusion"] = occlusion
                    df.loc[i, "set"] = directory_path
                    df.loc[i, "video"] = video_dir_path
            except:
                df.loc[i, "filename"] = "INVALID"
                df.loc[i, "width"] = "INVALID"
                df.loc[i, "height"] = "INVALID"
                df.loc[i, "depth"] = "INVALID"
                df.loc[i, "name"] = "INVALID"
                df.loc[i, "xmin"] = "INVALID"
                df.loc[i, "ymin"] = "INVALID"
                df.loc[i, "xmax"] = "INVALID"
                df.loc[i, "ymax"] = "INVALID"
                df.loc[i, "pose"] = "INVALID"
                df.loc[i, "truncated"] = "INVALID"
                df.loc[i, "difficult"] = "INVALID"
                df.loc[i, "occlusion"] = "INVALID"
                df.loc[i, "set"] = "INVALID"
                df.loc[i, "video"] = "INVALID"
            i += 1
    set_num += 1

df.to_csv("annotations.csv", index=False)


