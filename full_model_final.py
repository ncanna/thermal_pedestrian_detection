#imports
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
from sklearn.metrics import f1_score, precision_score, recall_score
import statistics

batch_size = 128
num_epochs = 25

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
        soup = BeautifulSoup(data, 'xml')  # probably will have to change this
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
        self.csv = pd.read_csv("frame_MasterList.csv")  # will always grab this
        self.csv_len = self.csv.shape[1]
        self.imgs = self.csv.image_path.tolist()
        self.imgs_len = len(self.imgs)
        self.transforms = transforms

    def __len__(self):
        # return int(self.imgs_len/150)
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
data_transform = transforms.Compose([#transforms.Resize((80,50)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5]
                                                          )])

# Collate images
def collate_fn(batch):
    return tuple(zip(*batch)) #will need adjusting when pathing is adjusted

dataset = FullImages(data_transform)
data_size = len(dataset)
print(data_size)

indices = list(range(data_size))
test_split = 0.2
split = int(np.floor(test_split * data_size))
print(split)
train_indices, test_indices = indices[split:], indices[:split]
#print(train_indices)
#print(test_indices)
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size = batch_size,
    sampler = train_sampler,
    collate_fn = collate_fn
)
data_loader_test = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler = test_sampler, collate_fn = collate_fn)

len_dataloader = len(data_loader)
print("Length of train: " +str(len_dataloader))

len_testdataloader = len(data_loader_test)
print("Length of test: " +str(len_testdataloader))

# Check if GPU
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Instance segmentation is crucial in using the full images
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)
    return model

#cnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)
model = get_model_instance_segmentation(3)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params) #, lr = 0.005, weight_decay = 0.0005)

# Learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size = 5,
#                                                gamma = 0.2)

epoch_ats = []
epoch_losses = []
tot_ats = 0
epochs = 0
for epoch in range(num_epochs):
    epochs += 1
    print(f'Epoch: {epochs}')
    model.train()
    epoch_loss = 0
    i = 0
    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model([imgs[0]], [annotations[0]])
        losses = sum(loss for loss in loss_dict.values())

        # losses, outputs = model(imgs, annotations)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses

        i += 1
        tot_ats += 1

        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')

    mean_epoch_loss = epoch_loss / i
    epoch_losses.append(mean_epoch_loss)
    epoch_ats.append(i)

    # df = pd.DataFrame(epoch_losses, columns = list("Mean_Epoch_Loss"))

    # if epochs % 1 == 0:
    #     partial_name = "full_model_losses_partial_" + str(epochs) #+ ".csv"
    #     mean_epoch_loss
    # df.to_csv(partial_name, index=False)

    # Update learning rate
    # lr_scheduler.step()

# Save model
torch.save(model.state_dict(), 'full_model.pt')
model2 = get_model_instance_segmentation(3)

# Save training metrics
# full_name = "full_model_losses_full_" + str(epochs) + ".csv"
# df.to_csv(full_name, index=False)
with open('epoch_losses_list', 'wb') as lossFile:
    pickle.dump(epoch_losses, lossFile)

print(f'Annotations Trained: {tot_ats}')

for imgs_t, annotations_t in data_loader_test:
  imgs_test = list(img_t.to(device) for img_t in imgs_t)
  annotations_test = [{k: v.to(device) for k, v in test_t.items()} for test_t in annotations_t]
  break

