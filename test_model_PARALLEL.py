# Imports
import pandas as pd
import os
import numpy as np
from numpy.distutils.misc_util import is_sequence
from bs4 import BeautifulSoup #this is to extract info from the xml, if we use it in the end
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
#from sklearn.metrics import f1_score, precision_score, recall_score
import statistics

import os
from datetime import datetime
from pathlib import Path


user = "n"
if user == "n":
    computing_id = "na3au"
elif user == "e":
    computing_id = "es3hd"
elif user == "s":
    computing_id = "sa3ag"

local_mode = False
selfcsv_df = pd.read_csv("frame_MasterList.csv")
if local_mode:
    modelPath = os.getcwd()
    xml_ver_string = "xml"
else:
    modelPath = "/scratch/" + computing_id + "/modelRuns" + "/2021_01_04-08_23_03_PM_NOTEBOOK/full_model_25.pt"
    xml_ver_string = "xml"

#req
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

data_transform = transforms.Compose([#transforms.Resize((320,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]
                         )])

def collate_fn(batch):
    return tuple(zip(*batch)) #will need adjusting when pathing is adjusted

class FullImages(object):
    def __init__(self, transforms=None):
        self.csv = selfcsv_df
        print(len(self.csv))
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

dataset = FullImages(data_transform)

dataset = FullImages(data_transform)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, collate_fn=collate_fn)

len_dataloader = len(data_loader)
print(f'Length of train: {len_dataloader}')

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_model_instance_segmentation(3)
model = nn.DataParallel(model)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(modelPath))
else:
    state_dict = torch.load(modelPath, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict)
model.eval()
model.to(device)

print("Model pushed")

def plot_image(img_tensor, annotation):

    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data
    print(img.shape)

    ax.imshow(img.permute(1, 2, 0)) #move channel to the end so that the image can be shown accordingly

    print(img.shape)
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.cpu()
        print(xmin)

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

qt = 0
for test_imgs, test_annotations in data_loader:
    imgs = list(img_test.to(device).cpu() for img_test in test_imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in test_annotations]
    qt+=1
    if qt > 20:
        break

cpu_device = torch.device("cpu")

outputs = model(imgs)
print(len(imgs))

preds = model.predict(imgs)
print("preds done")

#we can adjust these but it only goes up until the total batch size.
print("Guess")
plot_image(imgs[0], preds[0])
print("Reality")
plot_image(imgs[0], annotations[0])

