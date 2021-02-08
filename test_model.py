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
    xml_ver_string = "html.parser"
elif user == "e":
    computing_id = "es3hd"
    xml_ver_string = "xml"
elif user == "s":
    computing_id = "sa3ag"
    xml_ver_string = "xml"

local_mode = True
selfcsv_df = pd.read_csv("frame_MasterList.csv").head(50)
if local_mode:
    model_string = "full_model_gpu.pt"
    modelPath = os.getcwd() + "/" +model_string
    batch_size = 10
else:
    model_string = "2021_01_04-08_23_03_PM_NOTEBOOK/full_model_25.pt"
    modelPath = "/scratch/" + computing_id + "/modelRuns" + "/" + model_string
    batch_size = 32

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
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, collate_fn=collate_fn)

len_dataloader = len(data_loader)
print(f'Length of test dataset: {len_dataloader}')

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

print(f'Model {model_string} loaded.')

qt = 0
for test_imgs, test_annotations in data_loader:
    imgs = list(img_test.to(device).cpu() for img_test in test_imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in test_annotations]
    qt+=1


preds = model(imgs)
print(f"{len(preds)} Predictions loaded")


def plot_images(num, input):
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
        set = file_name.split("/")[7]
        video = file_name.split("/")[8]
        file_name = file_name.split("/")[10]
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
    print(str(len(prediction["boxes"])) + " prediction boxes made for " + str(
        len(annotation["boxes"])) + " actual boxes in " + str(output_name))
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
    plt.show()


print("Predicted:")
for i in range(len(preds) - 1):
    #print(preds[i])
    #plot_image(imgs[i], preds[i])
    plot_images(i, f"Input {i}")

# print("Reality")
# for i in range(len(preds) - 1):
#     print(annotations[i])
#     plot_image(imgs[i], annotations[i])
