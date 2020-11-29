#imports
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup #this is to extract info from the xml, if we use it in the end
import torchvision
from torchvision import transforms, datasets, models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


### These here are functions to grab information regarding the boxes...
#box coordinate generation - assuming seperate XML for each annotation. We currently do not have this,
#May be a better idea to use what we have in openCV_demo
#Keep in mind seperate XML means that we have an xml for each frame that indicates every single person/biker in the frame

#get label
def get_label(obj):
    if obj.find('name').text == 'person' or obj.find('name').text == 'people':
        xmin = int(obj.find('xmin').text) #need to adjust the labels here
        xmax = int(obj.find('xmax').text)
        ymin = int(obj.find('ymin').text)
        ymax = int(obj.find('ymax').text) #issue here is that I'm assuming we'll have XML in the end.
        return  [xmin, ymin, xmax, ymax]

def get_box(obj):
    if obj.find('name').text == 'person' or obj.find('name').text == 'people':
        return 1
    if obj.find('name').text == 'cyclist':
        return 2
    else: #assuming we ignore person?
        return 0

# Generate the target location in the image
# Based on seperate XMLs, so we may just have to adjust this part of all to fit in.
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

        #turning everything into a tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])

        #creating the target for the box
        target={}
        #target{'boxes'} = boxes
        target['labels'] = labels
        target['image_id'] = img_id

        return target


# List the files
# imgs = list(sorted(os.listdir("")))
# labels = list(sorted(os.listdir("")))

# class PedDataset(object):
#     def __init__(self,transforms):
#         self.transforms = transforms #allows transformations like the normalization and tensor
#         self.imgs = list(sorted(os.listdir("")))
#         self.labels = list(sorted(os.listdir("")))
#
#     def __getitem__(self,idx):
#         file_image = ""
#         file_label = ""
#         img_path = os.path.join()
#         label_path = ""
#         img = Image.open(img_path).convert('L') #convert to grayscale
#
#         target = generate_target(idx, label_path) #create the full image with the annotations
#
#         if self.transforms is not None:
#             img = self.transforms(img)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.imgs)

def OHE(label):
  if label == "People" or label== "Person":
      return 1
  elif label == "Cyclist":
      return 2
  else:
      return 0

class CutOutData(object):
    def __init__(self, transforms=None):
        file_out = pd.read_csv("cutout_MasterList.csv")  # will always grab this
        self.labels = file_out["Label"]
        self.cutouts = file_out["Cutout_Path"]
        self.transforms = transforms

    def __len__(self):
        return len(self.cutouts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        label = OHE(label)
        label = torch.as_tensor(label, dtype=torch.int64)
        img = self.cutouts[idx]
        img = Image.open(img).convert("L")

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

# Normalize
data_transform = transforms.Compose([transforms.Resize((80,50)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5]
                                                          )])

# Collate images
def collate_fn(batch):
    return tuple(zip(*batch)) #will need adjusting when pathing is adjusted

param_batch_size = 4
dataset = CutOutData(data_transform)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size = param_batch_size #, may want to adjust this
 #   collate_fn = collate_fn
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
    in_features = model.roi_heads.box_predictor.cls_score.infe

# c=Classes allow initialization and submodels into the main model/class
class CNNLSTM(nn.Module): #inherit nn.module so it wraps the class into a pytorch model
    def __init__(self, cnn, EMBED_SIZE=1280, LSTM_UNITS=64, DO = .3):
        super(CNNLSTM, self).__init__()
        #self.cnn = cnn.module
        self.cnn = cnn
        if cuda:
            self.cnn.eval().cuda()
        else:
            self.cnn.eval()
        self.lstm1 = nn.LSTM(EMBED_SIZE,LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional = True, batch_first = True)

        self.linear1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        self.linear2 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)

        self.linear_pe = nn.Linear(LSTM_UNITS * 2, 1)

    def forward(self, x, lengths = None): #forward method is the input passed into the method - data flow path
        embedding = self.cnn.extract_features(x)
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


for imgs, labels in data_loader:
    imgs = list(img.to(device) for img in imgs)
    #print(imgs)
    #print("Image input size: " + str(len(imgs)))

    labels = list(label.to(device) for label in labels)
    #print(labels)
    #print("Labels input size: " + str(len(labels)))
    break

num_epochs = 1
cnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)
#print("CNN: ")
#print(cnn)
model = CNNLSTM(cnn)
#print("CNN LSTM: ")
#print(model.parameters())

# for name, param in cnn.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

model.to(device)
params = [p for p in cnn.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params)
print(optimizer)

i = 0
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for imgs, labels in data_loader:
        imgs = list(img.to(device) for img in imgs)
        labels = list(label.to(device) for label in labels)
        loss_dict = model([imgs[0]], [labels[0]])
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses
        i += 1
        print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
    print(epoch_loss)
