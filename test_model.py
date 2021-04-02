# Imports
import pandas as pd
import os
import numpy as np
from numpy.distutils.misc_util import is_sequence
from bs4 import BeautifulSoup  # this is to extract info from the xml, if we use it in the end
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import pickle

import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
# from sklearn.metrics import f1_score, precision_score, recall_score
import statistics

import os
from datetime import datetime
from pathlib import Path
from sys import platform

############################ User Parameters ############################
torch.manual_seed(0)

user = "n"
if user == "n":
    computing_id = "na3au"
    xml_ver_string = "html.parser"
elif user == "e":
    computing_id = "es3hd"
    xml_ver_string = "xml"
elif user == "s":
    computing_id = "sa3ag"
    xml_ver_string = "xml"

local_mode = False
parallel = True

if local_mode:
    model_string = "full_model.pt"
    batch_size = 1
    selfcsv_df = pd.read_csv("frame_MasterList.csv").head(10)
    dir_path = os.getcwd()
else:
    model_string = "2021_01_04-08_23_03_PM_NOTEBOOK/full_model_25.pt"
    batch_size = 64
    selfcsv_df = pd.read_csv("frame_MasterList.csv")
    dir_path = "/scratch/" + computing_id + "/modelRuns"

##########################################################################

#print("Your platform is: ", platform)
if platform == "win32":
    unix = False
else:
    unix = True

if unix:
    # Unix
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    directory = dir_path + "/" + current_time + "_TESTING"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_output_path = directory + "/"
    modelPath = dir_path + "/" + model_string
    print(f'Creation of directory at {directory} successful')
else:
    try:
        # Windows
        current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        directory = dir_path + "\\" + current_time + "_TESTING"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_output_path = directory + "\\"
        modelPath = dir_path + "\\" + model_string
        print(f'Creation of directory at {directory} successful')
    except:
        print(f'Creation of directory at {directory} failed')

'''if unix:
    print("Unix system detected.")
else:
    print("Windows system detected")
'''

def get_box(obj):
    xmin = float(obj.find('xmin').text)
    xmax = float(obj.find('xmax').text)
    ymin = float(obj.find('ymin').text)
    ymax = float(obj.find('ymax').text)
    return [xmin, ymin, xmax, ymax]


def get_label(obj):
    if obj.find('name').text == 'person' or obj.find('name').text == 'people':
        return 1
    if obj.find('name').text == 'cyclist':
        return 2
    else:
        return 0


# Generate the target location in the image
def generate_target(image_id, file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, xml_ver_string)  # probably will have to change this
        objects = soup.find_all('object')

        num_objs = len(objects)

        boxes = []
        labels = []

        for i in objects:
            boxes.append(get_box(i))
            labels.append(get_label(i))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])

        # Creating the target for the box
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = img_id

        return target


def OHE(label):
    if label == "People" or label == "Person":
        return 1
    elif label == "Cyclist":
        return 2
    else:
        return 0


def Recode(label):
    if label == 1:
        return "Person(s)"
    elif label == 2:
        return "Cyclist"
    else:
        return "N/A"


data_transform = transforms.Compose([  # transforms.Resize((320,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.repeat(3,1,1))])


def collate_fn(batch):
    return tuple(zip(*batch))  # will need adjusting when pathing is adjusted


class FullImages(object):
    def __init__(self, transforms=None):
        self.csv = selfcsv_df
        #print(len(self.csv))
        self.imgs = self.csv.image_path.tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.csv)
        # return self.csv_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.csv.loc[idx, 'image_path']
        annotation = self.csv.loc[idx, 'annotation_path']

        img = Image.open(img).convert("L")
        target = generate_target(idx, annotation)

        # label = self.labels[idx]
        # label = OHE(label)
        # label = torch.as_tensor(label, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target



dts = FullImages(data_transform)

data_loader = torch.utils.data.DataLoader(
    dts, batch_size=batch_size, collate_fn=collate_fn)

len_dataloader = len(data_loader)
#print(f'Batches in test dataset: {len_dataloader}')


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)
    for name, param in model.named_parameters():
        param.requires_grad_(True)
        if not param.requires_grad:
            print(name)
    return model

device = torch.device('cpu')  # testing only on CPU
model = get_model_instance_segmentation(3)

state_dict = torch.load(modelPath, map_location=device)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

if parallel == True:
    model = nn.DataParallel(model)

print(f'Model {model_string} loaded.')

qt = 0
for test_imgs, test_annotations in data_loader:
    imgs = list(img_test.to(device).cpu() for img_test in test_imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in test_annotations]
    qt += 1

preds = model(imgs)
#print(f"{len(preds)} predictions loaded")


def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data
    print(img.shape)

    ax.imshow(img.permute(1, 2, 0))  # move channel to the end so that the image can be shown accordingly

    print(img.shape)
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.cpu()
        print(xmin)

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def plot_images(num):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    img_tensor = imgs[num]
    annotation = annotations[num]
    # for key, value in annotation.items():
    #         print(key, value)
    prediction = preds[num]

    img = img_tensor.cpu().data
    img = img[0, :, :]

    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(img, cmap='gray')

    ix = 0
    for box in annotation["boxes"]:
        # print(annotations[ix])
        xmin, ymin, xmax, ymax = box.tolist()
        value = annotation["labels"][ix]
        img_id = annotation["image_id"].item()
        file_name = selfcsv_df.loc[img_id, :].image_path
        if unix:
            set = file_name.split("/")[7]
            video = file_name.split("/")[8]
            file_name = file_name.split("/")[10]
        else:
            set = file_name.split("\\")[7]
            video = file_name.split("\\")[8]
            file_name = file_name.split("\\")[10]
        file_name = file_name[:-4]
        output_name = set + "_" + video + "_" + file_name
        text = Recode(value)
        colors = ["r", "#00FF00", "#0000FF"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor=colors[value], facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        ax[0].text(target_x, target_y, text, color=colors[value])
        ax[0].add_patch(rect)
        ix += 1

    ix = 0
    #print(str(len(prediction["boxes"])) + " prediction boxes made for " + str(
    #    len(annotation["boxes"])) + " actual boxes in " + str(output_name))
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()
        value = prediction["labels"][ix]
        text = Recode(value)
        colors = ["r", "#00FF00", "#0000FF"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor=colors[value], facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        ax[1].text(target_x, target_y, text, color=colors[value])
        ax[1].add_patch(rect)
        ix += 1

    # figname = file_name+"_"+input+".png"
    # fig.savefig(figname)
    if local_mode:
        plt.show()

def cluster_preds(num):
    annotation = annotations[num]
    prediction = preds[num]
    annotation_boxes = annotation["boxes"].tolist()

    th_better = 0.3
    th_X = 2
    th_Y = 2

    # Collapse predictions
    prediction_mod = prediction["boxes"].tolist()
    subset_indices = []
    c_ix = 0
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        collapsed = False
        for compare_box in prediction["boxes"]:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box.tolist()

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (
                    ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)

    prediction_superset = [[0, 0, 0, 0]]
    # prediction_mod = prediction["boxes"]
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > th_better:
                    if (xmin + mod_xmin) / th_X < xmin:
                        xmin = (xmin + mod_xmin) / th_X
                    if (ymin + mod_ymin) / th_Y < ymin:
                        ymin = (ymin + mod_ymin) / th_Y
                    if (xmax + mod_xmax) / th_X > xmax:
                        xmax = (xmax + mod_xmax) / th_X
                    if (ymax + mod_ymax) / th_Y > ymax:
                        ymax = (ymax + mod_ymax) / th_Y

                    prediction_superset[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset.append([xmin, ymin, xmax, ymax])

    prediction_mod = prediction_superset
    subset_indices = []
    c_ix = 0
    for box in prediction_superset:
        xmin, ymin, xmax, ymax = box

        collapsed = False
        for compare_box in prediction_superset:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (
                    ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)

    prediction_superset_clustered = [[0, 0, 0, 0]]
    # prediction_mod = prediction["boxes"]
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset_clustered:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > 0.1:
                    if (xmin + mod_xmin) / th_X < xmin:
                        xmin = (xmin + mod_xmin) / th_X
                    if (ymin + mod_ymin) / th_Y < ymin:
                        ymin = (ymin + mod_ymin) / th_Y
                    if (xmax + mod_xmax) / th_X > xmax:
                        xmax = (xmax + mod_xmax) / th_X
                    if (ymax + mod_ymax) / th_Y > ymax:
                        ymax = (ymax + mod_ymax) / th_Y

                    prediction_superset_clustered[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset_clustered.append([xmin, ymin, xmax, ymax])

    prediction_mod = prediction_superset_clustered
    subset_indices = []
    c_ix = 0
    for box in prediction_superset_clustered:
        xmin, ymin, xmax, ymax = box
        p_area = (xmax - xmin + 1) * (ymax - ymin + 1)

        collapsed = False
        for compare_box in prediction_superset_clustered:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area)

            if iou > 0.5 and iou != 1 and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)
    prediction_mod = prediction_mod[1:]

    ix = 0
    voc_iou_mod = []
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box

        iou_list = []
        for bound in annotation_boxes:
            a_xmin, a_ymin, a_xmax, a_ymax = bound
            xA = max(xmin, a_xmin)
            yA = max(ymin, a_ymin)
            xB = min(xmax, a_xmax)
            yB = min(ymax, a_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (a_xmax - a_xmin + 1) * (a_ymax - a_ymin + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        max_val = max(iou_list)
        voc_iou_mod.append(max_val)
        ix += 1

    ##### Calculate accuracy and IoU metrics
    ats_voc_iou_mod = []
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for mod_box in prediction_mod:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(a_area)
            iou_list.append(iou)
        max_val = max(iou_list)
        ats_voc_iou_mod.append(max_val)
    return prediction_mod

def get_iou(num):
    annotation = annotations[num]
    prediction = preds[num]
    annotation_boxes = annotation["boxes"].tolist()

    th_better = 0.3
    th_X = 2
    th_Y = 2

    ##### Original prediction boxes
    ix = 0
    voc_iou = []
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for bound in annotation_boxes:
            a_xmin, a_ymin, a_xmax, a_ymax = bound
            xA = max(xmin, a_xmin)
            yA = max(ymin, a_ymin)
            xB = min(xmax, a_xmax)
            yB = min(ymax, a_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (a_xmax - a_xmin + 1) * (a_ymax - a_ymin + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        max_val = max(iou_list)
        max_val_rounded = round(max(iou_list), 2)
        voc_iou.append(max_val)

        ix += 1

    ##### Clustered prediction boxes
    # Collapse predictions
    prediction_mod = prediction["boxes"].tolist()
    subset_indices = []
    c_ix = 0
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        collapsed = False
        for compare_box in prediction["boxes"]:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box.tolist()

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (
                    ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)

    prediction_superset = [[0, 0, 0, 0]]
    # prediction_mod = prediction["boxes"]
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > th_better:
                    if (xmin + mod_xmin) / th_X < xmin:
                        xmin = (xmin + mod_xmin) / th_X
                    if (ymin + mod_ymin) / th_Y < ymin:
                        ymin = (ymin + mod_ymin) / th_Y
                    if (xmax + mod_xmax) / th_X > xmax:
                        xmax = (xmax + mod_xmax) / th_X
                    if (ymax + mod_ymax) / th_Y > ymax:
                        ymax = (ymax + mod_ymax) / th_Y

                    prediction_superset[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset.append([xmin, ymin, xmax, ymax])

    prediction_mod = prediction_superset
    subset_indices = []
    c_ix = 0
    for box in prediction_superset:
        xmin, ymin, xmax, ymax = box

        collapsed = False
        for compare_box in prediction_superset:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (
                    ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)

    prediction_superset_clustered = [[0, 0, 0, 0]]
    # prediction_mod = prediction["boxes"]
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset_clustered:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > 0.1:
                    if (xmin + mod_xmin) / th_X < xmin:
                        xmin = (xmin + mod_xmin) / th_X
                    if (ymin + mod_ymin) / th_Y < ymin:
                        ymin = (ymin + mod_ymin) / th_Y
                    if (xmax + mod_xmax) / th_X > xmax:
                        xmax = (xmax + mod_xmax) / th_X
                    if (ymax + mod_ymax) / th_Y > ymax:
                        ymax = (ymax + mod_ymax) / th_Y

                    prediction_superset_clustered[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset_clustered.append([xmin, ymin, xmax, ymax])

    prediction_mod = prediction_superset_clustered
    subset_indices = []
    c_ix = 0
    for box in prediction_superset_clustered:
        xmin, ymin, xmax, ymax = box
        p_area = (xmax - xmin + 1) * (ymax - ymin + 1)

        collapsed = False
        for compare_box in prediction_superset_clustered:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area)

            if iou > 0.5 and iou != 1 and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)

    prediction_mod = prediction_mod[1:]

    ix = 0
    voc_iou_mod = []
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box

        iou_list = []
        for bound in annotation_boxes:
            a_xmin, a_ymin, a_xmax, a_ymax = bound
            xA = max(xmin, a_xmin)
            yA = max(ymin, a_ymin)
            xB = min(xmax, a_xmax)
            yB = min(ymax, a_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (a_xmax - a_xmin + 1) * (a_ymax - a_ymin + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        max_val = max(iou_list)
        max_val_rounded = round(max(iou_list), 2)
        voc_iou_mod.append(max_val)
        ix += 1

    ##### Calculate accuracy and IoU metrics
    ats_voc_iou_mod = []
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for mod_box in prediction_mod:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)
        max_val = max(iou_list)
        ats_voc_iou_mod.append(max_val)

    #print("\n Original Predictions")
    #print(
    #    f'{len(prediction["boxes"])} boxes made for {len(annotation["boxes"])} actual boxes in {str(output_name)} for {identifier} with note {input} (INDEX {num})')
    if len(voc_iou) == 0:
        mean_iou = 0
        #print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        og_mean_iou = sum(voc_iou) / len(voc_iou)
        fp = voc_iou.count(0)
        bp = sum((i > 0 and i < 0.4) for i in voc_iou)
        gp = sum((i >= 0.4) for i in voc_iou)
        og_accuracy = sum([1 if entry >= 0.4 else 0 for entry in ats_voc_iou_mod]) / len(annotation["boxes"])
        '''print(f'{fp} false positives (IOU = 0)')
        print(f'{bp} bad positives (0 < IOU < 0.4)')
        print(f'{gp} good positives (IOU >= 0.4)')
        print(f'Mean IOU: {mean_iou}')
        print(f'Accuracy: {accuracy*100}%')
        # print(f'Predictions for Image {num} have mean IOU: {mean_iou} and accuracy: {accuracy}')'''

    print("\n Clustered Predictions")
    if len(voc_iou_mod) == 0:
        mean_iou = 0
        #print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        mean_iou = sum(voc_iou_mod) / len(voc_iou_mod)
        fp = voc_iou_mod.count(0)
        bp = sum((i > 0 and i < 0.4) for i in voc_iou_mod)
        gp = sum((i >= 0.4) for i in voc_iou_mod)
        accuracy = sum([1 if entry >= 0.4 else 0 for entry in ats_voc_iou_mod]) / len(
            annotation["boxes"])
        '''print(
            f'{len(prediction_mod)} boxes made for {len(annotation["boxes"])} actual boxes in {str(output_name)} for {identifier} with note {input} (INDEX {num})')
        print(f'{fp} false positives (IOU = 0)')
        print(f'{bp} bad positives (0 < IOU < 0.4)')
        print(f'{gp} good positives (IOU >= 0.4)')
        print(f'Mean IOU: {mean_iou}')
        print(f'Accuracy: {accuracy*100}%')
        # print(f'Predictions for Image {num} have mean IOU: {mean_iou} and accuracy: {accuracy}')'''

    return [accuracy, mean_iou, og_accuracy, og_mean_iou]


def plot_iou(num, input="iou_plotted"):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    identifier = "test"
    img_tensor = imgs[num]
    annotation = annotations[num]
    prediction = preds[num]
    th_better = 0.3
    th_X = 2
    th_Y = 2
    #print(prediction["boxes"])

    img = img_tensor.cpu().data
    # print(f'img is {img.shape}')
    # img = img.permute(1, 2, 0)
    # print(f'img is {img.shape}')

    img = img[0, :, :]
    annotation_boxes = annotation["boxes"].tolist()

    ##### Subplot 1: Original prediction boxes
    ax1.imshow(img, cmap='gray')

    ix = 0
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()
        value = annotation["labels"][ix]
        img_id = annotation["image_id"].item()
        file_name = selfcsv_df.loc[img_id, :].image_path
        if unix:
            set = file_name.split("/")[7]
            video = file_name.split("/")[8]
            file_name = file_name.split("/")[10]
        else:
            set = file_name.split("\\")[7]
            video = file_name.split("\\")[8]
            file_name = file_name.split("\\")[10]
        file_name = file_name[:-4]
        output_name = set + "_" + video + "_" + file_name + "_" + identifier
        text = Recode(value)
        colors = ["r", "r", "r"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor=colors[value], facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        #ax1.text(target_x, target_y, text, color=colors[value])
        ax1.add_patch(rect)
        ix += 1

    ix = 0
    voc_iou = []
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for bound in annotation_boxes:
            a_xmin, a_ymin, a_xmax, a_ymax = bound
            xA = max(xmin, a_xmin)
            yA = max(ymin, a_ymin)
            xB = min(xmax, a_xmax)
            yB = min(ymax, a_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (a_xmax - a_xmin + 1) * (a_ymax - a_ymin + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        max_val = max(iou_list)
        max_val_rounded = round(max(iou_list), 2)
        voc_iou.append(max_val)

        max_ix = iou_list.index(max_val)
        map_dict = {max_ix: max_val_rounded}

        # iou_string = ', '.join((str(float) for float in iou_list))
        value = prediction["labels"][ix]
        #text = json.dumps(map_dict)
        text = max_val_rounded
        colors = ["r", "#00FF00", "#0000FF"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor=colors[value], facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        #ax1.text(target_x, target_y, text, color=colors[value])
        ax1.add_patch(rect)
        ix += 1

    ##### Subplot 2: Clustered prediction boxes
    ax2.imshow(img, cmap='gray')

    ix = 0
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()
        value = annotation["labels"][ix]
        img_id = annotation["image_id"].item()
        file_name = selfcsv_df.loc[img_id, :].image_path
        if unix:
            set = file_name.split("/")[7]
            video = file_name.split("/")[8]
            file_name = file_name.split("/")[10]
        else:
            set = file_name.split("\\")[7]
            video = file_name.split("\\")[8]
            file_name = file_name.split("\\")[10]
        file_name = file_name[:-4]
        output_name = set + "_" + video + "_" + file_name + "_" + identifier
        text = Recode(value)
        colors = ["r", "r", "r"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor=colors[value], facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        #ax2.text(target_x, target_y, text, color=colors[value])
        ax2.add_patch(rect)
        ix += 1

    # Collapse predictions
    prediction_mod = prediction["boxes"].tolist()
    subset_indices = []
    c_ix = 0
    for box in prediction["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        collapsed = False
        for compare_box in prediction["boxes"]:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box.tolist()

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)

    prediction_superset = [[0,0,0,0]]
    #prediction_mod = prediction["boxes"]
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > th_better:
                    if (xmin + mod_xmin)/th_X < xmin:
                        xmin = (xmin + mod_xmin)/th_X
                    if (ymin + mod_ymin)/th_Y < ymin:
                        ymin = (ymin + mod_ymin)/th_Y
                    if (xmax + mod_xmax)/th_X > xmax:
                        xmax = (xmax + mod_xmax)/th_X
                    if (ymax + mod_ymax)/th_Y > ymax:
                        ymax = (ymax + mod_ymax)/th_Y

                    prediction_superset[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset.append([xmin, ymin, xmax, ymax])

    prediction_mod = prediction_superset
    subset_indices = []
    c_ix = 0
    for box in prediction_superset:
        xmin, ymin, xmax, ymax = box

        collapsed = False
        for compare_box in prediction_superset:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box

            if (xmin > mod_xmin) and (xmax < mod_xmax) and (ymin > mod_ymin) and (
                    ymax < mod_ymax) and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)

    prediction_superset_clustered = [[0, 0, 0, 0]]
    # prediction_mod = prediction["boxes"]
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box
        better_match = False

        prediction_mod_ix = 0
        for mod_pred in prediction_superset_clustered:
            if not better_match:
                mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_pred
                xA = max(xmin, mod_xmin)
                yA = max(ymin, mod_ymin)
                xB = min(xmax, mod_xmax)
                yB = min(ymax, mod_ymax)
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
                iou = interArea / float(p_area + a_area - interArea)

                if iou > 0.1:
                    if (xmin + mod_xmin)/th_X < xmin:
                        xmin = (xmin + mod_xmin)/th_X
                    if (ymin + mod_ymin)/th_Y < ymin:
                        ymin = (ymin + mod_ymin)/th_Y
                    if (xmax + mod_xmax)/th_X > xmax:
                        xmax = (xmax + mod_xmax)/th_X
                    if (ymax + mod_ymax) / th_Y > ymax:
                        ymax = (ymax + mod_ymax) / th_Y

                    prediction_superset_clustered[prediction_mod_ix] = [xmin, ymin, xmax, ymax]
                    better_match = True
                    break

                prediction_mod_ix += 1

        if not better_match:
            prediction_superset_clustered.append([xmin, ymin, xmax, ymax])

    prediction_mod = prediction_superset_clustered
    subset_indices = []
    c_ix = 0
    for box in prediction_superset_clustered:
        xmin, ymin, xmax, ymax = box
        p_area = (xmax - xmin + 1) * (ymax - ymin + 1)

        collapsed = False
        for compare_box in prediction_superset_clustered:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = compare_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area)

            if iou > 0.5 and iou != 1 and not collapsed:
                subset_indices.append(c_ix)
                collapsed = True
                break
        c_ix += 1

    subset_indices.sort(reverse=True)
    for index_num in subset_indices:
        prediction_mod.pop(index_num)

    prediction_mod = prediction_mod[1:]

    ix = 0
    voc_iou_mod = []
    for box in prediction_mod:
        xmin, ymin, xmax, ymax = box

        iou_list = []
        for bound in annotation_boxes:
            a_xmin, a_ymin, a_xmax, a_ymax = bound
            xA = max(xmin, a_xmin)
            yA = max(ymin, a_ymin)
            xB = min(xmax, a_xmax)
            yB = min(ymax, a_ymax)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (a_xmax - a_xmin + 1) * (a_ymax - a_ymin + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)

        max_val = max(iou_list)
        max_val_rounded = round(max(iou_list), 2)
        voc_iou_mod.append(max_val)

        max_ix = iou_list.index(max_val)
        map_dict = {max_ix: max_val_rounded}

        # iou_string = ', '.join((str(float) for float in iou_list))
        value = prediction["labels"][ix]
        #text = json.dumps(map_dict)
        text = max_val_rounded
        colors = ["r", "#00FF00", "#0000FF"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor=colors[value], facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        if text == 0.0:
            pass
        else:
            ax2.text(target_x, target_y, text, color=colors[value])
        ax2.add_patch(rect)
        ix += 1

    ##### Calculate accuracy and IoU metrics
    ats_voc_iou_mod =[]
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()

        iou_list = []
        for mod_box in prediction_mod:
            mod_xmin, mod_ymin, mod_xmax, mod_ymax = mod_box
            xA = max(xmin, mod_xmin)
            yA = max(ymin, mod_ymin)
            xB = min(xmax, mod_xmax)
            yB = min(ymax, mod_ymax)
            p_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            a_area = (mod_xmax - mod_xmin + 1) * (mod_ymax - mod_ymin + 1)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            iou = interArea / float(p_area + a_area - interArea)
            iou_list.append(iou)
        max_val = max(iou_list)
        ats_voc_iou_mod.append(max_val)

    print("\n Original Predictions")
    print(f'{len(prediction["boxes"])} boxes made for {len(annotation["boxes"])} actual boxes in {str(output_name)} for {identifier} with note {input} (INDEX {num})')
    if len(voc_iou) == 0:
        mean_iou = 0
        print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        mean_iou = sum(voc_iou) / len(voc_iou)
        fp = voc_iou.count(0)
        bp = sum((i > 0 and i < 0.4) for i in voc_iou)
        gp = sum((i >= 0.4) for i in voc_iou)
        accuracy = sum([1 if entry >= 0.4 else 0 for entry in ats_voc_iou_mod]) / len(annotation["boxes"])
        print(f'{fp} false positives (IOU = 0)')
        print(f'{bp} bad positives (0 < IOU < 0.4)')
        print(f'{gp} good positives (IOU >= 0.4)')
        print(f'Mean IOU: {mean_iou}')
        print(f'Accuracy: {accuracy*100}%')
        #print(f'Predictions for Image {num} have mean IOU: {mean_iou} and accuracy: {accuracy}')

    print("\n Clustered Predictions")
    if len(voc_iou_mod) == 0:
        mean_iou = 0
        print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        mean_iou = sum(voc_iou_mod) / len(voc_iou_mod)
        fp = voc_iou_mod.count(0)
        bp = sum((i > 0 and i < 0.4) for i in voc_iou_mod)
        gp = sum((i >= 0.4) for i in voc_iou_mod)
        accuracy = sum([1 if entry >= 0.4 else 0 for entry in ats_voc_iou_mod]) / len(annotation["boxes"])
        print(f'{len(prediction_mod)} boxes made for {len(annotation["boxes"])} actual boxes in {str(output_name)} for {identifier} with note {input} (INDEX {num})')
        print(f'{fp} false positives (IOU = 0)')
        print(f'{bp} bad positives (0 < IOU < 0.4)')
        print(f'{gp} good positives (IOU >= 0.4)')
        print(f'Mean IOU: {mean_iou}')
        print(f'Accuracy: {accuracy*100}%')
        #print(f'Predictions for Image {num} have mean IOU: {mean_iou} and accuracy: {accuracy}')

    plt.show()
    figname = output_name + "_" + input + ".png"
    fig.savefig(file_output_path + figname)
    # print(f'Figure {figname} saved to {directory}.')

# print("Predicted:")
# for i in range(len(preds) - 1):
# print(preds[i])
# plot_image(imgs[i], preds[i])
# plot_images(i, f"Input {i}")

iou = list()
acc = list()

with torch.no_grad():
    i = 0
    for imgs, annotations in data_loader:
        preds = model(imgs)
        accuracy, io = get_iou(0)
        iou.append(io)
        acc.append(accuracy)
        i+=1
    mean_acc = np.mean(acc)
    mean_iou = np.mean(iou)
print("accuracy: ", mean_acc)
print("iou: ", mean_iou)

# print("Calculating IOU:")
# iou_df_test = pd.DataFrame(columns=["Accuracy", "Test_Mean_IOU"])
# iou_df_test_name = "full_iou_TEST.csv"
# for test_pred in range(0, len(preds)):
#     iou_function = get_iou(test_pred)
#     len_df = len(iou_df_test)
#     iou_df_test.loc[len_df, :] = iou_function
#     try:
#         if test_pred % 50 == 0:
#             partial_name = "partial_iou_TEST_" + str(test_pred) + "_images.csv"
#             iou_df_test.to_csv(file_output_path + iou_df_test_name, index=False)
#             print(f'Partial test IOUs for {len(iou_df_test)} images saved to {directory}.')
#     except:
#         pass
#
# iou_df_test.to_csv(file_output_path + iou_df_test_name, index=False)
# print(f'Full test IOUs for {len(iou_df_test)} images saved to {directory}.')
# #print(iou_df_test.sort_values(by='Test_Mean_IOU', ascending=False).head(5))
#
# max_test_ix = iou_df_test[iou_df_test['Test_Mean_IOU'] == iou_df_test['Test_Mean_IOU'].max()].index.tolist()[0]
# plot_iou(max_test_ix, "best_test")
# print(f'Test Mean IOU: {iou_df_test["Test_Mean_IOU"].mean()}')
# print(f'Test Mean Accuracy: {iou_df_test["Accuracy"].mean() * 100}')

#iterate
#get iou for that iteration
#get accuracy for that iteration
#avg all of them in the end