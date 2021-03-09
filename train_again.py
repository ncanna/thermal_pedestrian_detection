import matplotlib
matplotlib.use('Agg')

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
from sklearn.metrics import f1_score, precision_score, recall_score
import statistics

import os
from datetime import datetime
from pathlib import Path

local_mode = False
iou_mode = False
user = "s"

if user == "n":
    computing_id = "na3au"
elif user == "e":
    computing_id = "es3hd"
elif user == "s":
    computing_id = "sa3ag"

if local_mode:
    batch_size = 10
    num_epochs = 2
    selfcsv_df = pd.read_csv("frame_MasterList.csv").head(10)
    dir_path = os.getcwd()
    xml_ver_string = "xml"
else:
    batch_size = 128
    num_epochs = 200
    selfcsv_df = pd.read_csv("frame_MasterList.csv") #.head(50)
    dir_path = "/scratch/"+computing_id+"/modelRuns"
    xml_ver_string = "html.parser"

try:
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    directory = dir_path + "/" + current_time + "_TRAINING"
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f'Creation of directory at {directory} successful')
except:
    print(f'Creation of directory at {directory} failed')
file_output_path = directory + "/"

# Get label and encode
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
        soup = BeautifulSoup(data, xml_ver_string)
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

class FullImages(object):
    def __init__(self, transforms=None):
        self.csv = selfcsv_df
        self.csv_len = self.csv.shape[1]
        self.imgs = self.csv.image_path.tolist()
        self.imgs_len = len(self.imgs)
        self.transforms = transforms

    def __len__(self):
        return self.imgs_len
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

# Normalize
data_transform = transforms.Compose([  # transforms.Resize((80,50)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]
                         )])

# Collate images
def collate_fn(batch):
    return tuple(zip(*batch))  # will need adjusting when pathing is adjusted

dataset = FullImages(data_transform)
data_size = len(dataset)
print(f'Length of Dataset: {data_size}')

indices = list(range(data_size))
test_split = 0.2
split = int(np.floor(test_split * data_size))
# print(f'Length of Split Dataset: {split}')

train_indices, test_indices = indices[split:], indices[:split]
len_train_ind, len_test_ind = len(train_indices), len(test_indices)
print(f'Length of Train: {len_train_ind}; Length of Test: {len_test_ind}')

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collate_fn
)

len_dataloader = len(data_loader)
data_loader_test = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler,
                                               collate_fn=collate_fn)
len_testdataloader = len(data_loader_test)
print(f'Length of test: {len_testdataloader}; Length of train: {len_dataloader}')

# Instance segmentation is crucial in using the full images
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)
    return model

# print(f'{len_testdataloader} batches in test data loader.')
# print(f'{len_dataloader} batches in train data loader.')

# cnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)
model = get_model_instance_segmentation(3)
modelPath = "2021_02_19-03_40_06_PM_TRAINING/full_model.pt"

# Check if GPU
cuda = torch.cuda.is_available()
if cuda:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device = torch.device("cuda:0")
        model = nn.DataParallel(model)
        state_dict = torch.load(modelPath, map_location=device)
    else:
        device = torch.device("cuda")
        state_dict = torch.load(modelPath, map_location=device)
        print(f'Single CUDA.....baby shark doo doo doo')
else:
    device = torch.device("cpu")
    state_dict = torch.load(modelPath, map_location=device)
    print(f'But I\'m just a poor CPU and nobody loves me :(')


def train_iou(num, ges, ann):

    annotation = ann[num]
    prediction = ges[num]

    annotation_boxes = annotation["boxes"].tolist()

    ix = 0
    for box in annotation["boxes"]:
        img_id = annotation["image_id"].item()
        file_name = selfcsv_df.loc[img_id, :].image_path
        set = file_name.split("/")[7]
        video = file_name.split("/")[8]
        file_name = file_name.split("/")[10]
        file_name = file_name[:-4]
        output_name = set + "_" + video + "_" + file_name
        ix += 1

    iy = 0
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
        voc_iou.append(max_val)
        iy += 1

    print(f'{ix} true boxes, {iy} predicted boxes.')
    if len(voc_iou) == 0:
        mean_iou = 0
        print(f'No predictions made so Mean IOU: {mean_iou}')
    else:
        mean_iou = sum(voc_iou) / len(voc_iou)

    return [mean_iou, voc_iou]

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params)  # , lr = 0.005, weight_decay = 0.0005) #CONSIDER ADDING WEIGHT DECAY

tot_ats = 0
epochs = 0
epoch_iou_list = []
epoch_losses = []
for epoch in range(num_epochs):
    epochs += 1

    print(f'Epoch: {epochs}')

    model.train()

    epoch_loss = 0
    epoch_iou = 0

    i = 0
    for train_imgs, train_annotations in data_loader:
        imgs = list(img.to(device) for img in train_imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in train_annotations]
        loss_dict = model([imgs[0]], [annotations[0]])
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        i += 1
        tot_ats += 1

        #Training IoU
        if iou_mode:
            model.eval()
            guess = model(imgs[0:3])
            iteration_iou = train_iou(0,guess,annotations)
            model.train()

        epoch_loss += losses.item()

        if iou_mode:
            epoch_iou += iteration_iou[0]
            print(f'Iteration Number: {i}/{len_dataloader}, Loss: {losses}, IoU: {iteration_iou[0]}')
        else:
            print(f'Iteration Number: {i}/{len_dataloader}, Loss: {losses}')

    mean_epoch_loss = epoch_loss / i

    if iou_mode:
        mean_epoch_iou = epoch_iou / i
        epoch_iou_list.append(mean_epoch_iou)

    epoch_losses.append(mean_epoch_loss)

try:
    # Save training metrics
    full_name = "full_model_losses_" + str(epochs) + ".csv"

    if iou_mode:
        df = pd.DataFrame({'Mean_Epoch_Loss': epoch_losses, 'Mean_Training_IOU': epoch_iou_list})
    else:
        df = pd.DataFrame({'Mean_Epoch_Loss': epoch_losses})

    df.to_csv(file_output_path + full_name, index=False)
    print(f'Full model losses for {epochs} epochs saved to {directory}.')
except:
    pass

try:
    # Save model
    torch.save(model.state_dict(), file_output_path + 'full_model_extra.pt')
    print(f'Full model trained on {epochs} epochs saved to {directory}.')
except:
    pass