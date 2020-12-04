#imports
import pandas as pd
import os
import numpy as np
from numpy.distutils.misc_util import is_sequence
from bs4 import BeautifulSoup #this is to extract info from the xml, if we use it in the end
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

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
def generate_target(image_id,file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml') #probably will have to change this
        objects = soup.find_all('object')

        num_objs = len(objects)

        boxes = []
        labels = []

        for i in objects:
            boxes.append(get_box(i))
            labels.append(get_label(i))

        # Converting to a tensor
        # print(boxes)
        # print(labels)
        # print(image_id)
        # print(file)
        # print(is_sequence(boxes))
        # print(is_sequence(labels))
        # print(torch.as_tensor(labels, dtype=torch.int64))
        # print(torch.as_tensor(boxes, dtype=torch.int64))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])

        #creating the target for the box
        target={}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = img_id

        return target

def OHE(label):
  if label == "People" or label== "Person":
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
        self.transforms = transforms

    def __len__(self):
        return self.csv_len

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

param_batch_size = 4
dataset = FullImages(data_transform)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size = param_batch_size, #may want to adjust this
    collate_fn = collate_fn
)

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

class CNNLSTM(nn.Module):
    def __init__(self, cnn, EMBED_SIZE=1280, LSTM_UNITS=64, DO = .3):
        super(CNNLSTM, self).__init__()
        #self.cnn = cnn.module
        self.cnn = cnn
        if cuda:
            self.cnn.eval().cuda()
        else:
            self.cnn.eval()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        )

        self.lstm1 = nn.LSTM(EMBED_SIZE,LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional = True, batch_first = True)

        self.linear1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        self.linear2 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)

        self.linear_pe = nn.Linear(LSTM_UNITS * 2, 1)

    def forward(self, x, lengths = None): #forward method is the input passed into the method - data flow path
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        #embedding = self.cnn.forward(x)
        embedding = x
        b,f,_,_ = embedding.shape
        embedding = embedding.reshape(1,b,f) #trying to transform cnn output here for lstm
        self.lstm1(embedding)
        h_lstm1, _ = self.lstm1(embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(h_lstm1)

        h_conc_linear = F.relu(self.linear1(h_lstm1))
        h_conc_linear2 = F.relu(self.linear2(h_lstm2))

        hidden = h_lstm1 + h_lstm2 + h_conc_linear + h_conc_linear2

        output = self.linear_pe(hidden)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    #print(imgs)
    #print("Image input size: " + str(len(imgs)))

    #labels = list(label.to(device) for label in labels)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    #print(labels)
    #print("Labels input size: " + str(len(labels)))
    break

num_epochs = 1
len_dataloader = len(data_loader)

cnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)
#print(cnn)
#model = CNNLSTM(cnn)
#print(model.parameters())

# for name, param in cnn.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

#model = Net()
model = get_model_instance_segmentation(3)
model.to(device)
params = [p for p in cnn.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params)
#print(optimizer)

i = 0
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model([imgs[0]], [annotations[0]])
        losses = sum(loss for loss in loss_dict.values())
        #losses, outputs = model(imgs, annotations)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses

        i += 1
        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
    print(epoch_loss)

def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data
    height, width = img_tensor.size()[1], img_tensor.size()[2]

    img = img[0,:,:]
    ax.imshow(img, cmap='gray')

    # for key, value in annotation.items():
    #     print(key, value)

    ix = 0
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.tolist()
        value = annotation["labels"][ix]
        text = Recode(value)
        colors = ["r", "#0000FF", "#00FF00"]
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor=colors[value], facecolor='none')
        target_x = xmin
        target_y = ymin - 5
        ax.text(target_x, target_y, text, color=colors[value])
        ax.add_patch(rect)
        ix += 1

    plt.show()

model.eval()
preds = model(imgs)
plot_image(imgs[i], annotations[i])