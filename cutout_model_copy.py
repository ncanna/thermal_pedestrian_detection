#imports
import pandas as pd
import os
import numpy as np
from bs4 import BeautifulSoup #this is to extract info from the xml, if we use it in the end
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import statistics
from torch.utils.data.sampler import SubsetRandomSampler

############ USER PARAMETERS
num_epochs = 500
param_batch_size = 256

# Get label
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
        file_out = pd.read_csv("cutout_MasterList.csv")
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
    return tuple(zip(*batch))

#train and test
dataset = CutOutData(data_transform)
data_size = len(dataset)
indices = list(range(data_size))
val_split = .2
split = int(np.floor(val_split * data_size))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size = param_batch_size,
	sampler = train_sampler
 #   collate_fn = collate_fn
)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=param_batch_size, sampler = valid_sampler)

# Check if GPU
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(cuda)
# Instance segmentation is crucial in using the full images
def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)
    return model

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
        embedding = self.cnn.forward(x)
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
            nn.Linear(960, 10)
        )

        # LSTM
        # Linear layer of output

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        print(f'conv output: {x.shape}')
        x = self.linear_layers(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3400, 4000)
        self.fc2 = nn.Linear(4000, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# for imgs, labels in data_loader:
#     imgs = list(img.to(device) for img in imgs)
#     print(imgs)
#     print(f"Image input size: {imgs.shape}")
#
#     labels = list(label.to(device) for label in labels)
#     print(labels)
#     print(f"Labels input size: {labels.shape}")
#     break

len_dataloader = len(data_loader)
cnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False)
model = CNN()
#print(model)
#print(model.parameters())

# for name, param in cnn.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

#model = get_model_instance_segmentation(3)
model.to(device)

params_RCNN = [p for p in cnn.parameters() if p.requires_grad]
params = model.parameters()
optimizer = torch.optim.Adam(params, lr=0.07)
weights = [8,92]
if cuda:
    weights = torch.FloatTensor(weights).cuda()
else:
    weights = torch.FloatTensor(weights)

loss_fn = nn.CrossEntropyLoss(weight=weights)
#loss_fn = nn.BCELoss()
# print(params_RCNN)
# print(params)
# print(optimizer)

data_loaders = {"train": data_loader, "val": test_loader}
data_lengths = {'train': len(data_loader), 'val':len(test_loader)}

df_train = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1', 'Loss'])
df_val = pd.DataFrame(columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1', 'Loss'])
rows = []
epochs = 0
for epoch in range(num_epochs):
    #validation and training phase for each epoch
    epochs += 1
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)
        else:
            model.train(False)

        i= 0
        epoch_loss = 0.0
        loss_list, accuracy_list, f1_list, precision_list, recall_list = [], [], [], [], []
        #iterate
        for imgs, labels in data_loaders[phase]:
            #input and label
            batch_size = imgs.shape[0]
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = labels - 1

            out = model(imgs)
            losses = loss_fn(out, labels)
            optimizer.zero_grad()
            if phase == 'train':
                losses.backward()
                optimizer.step()
            #print loss
            epoch_loss += losses
            _, preds = torch.max(F.softmax(out, dim = 1), 1)
            accuracy = torch.sum(preds == labels.data) / float(batch_size)
            #accuracy_list.append(accuracy)

            tm_preds = preds.cpu().data.numpy()
            tm_labels = labels.cpu().data.numpy()

            f1_scores = torch.tensor(f1_score(tm_preds, tm_labels, average='micro'), device=device)
            #f1_list.append(f1_scores)

            precision = torch.tensor(precision_score(tm_preds, tm_labels, average='micro'),
                                 device=device)
            #precision_list.append(precision)

            recall_scores = torch.tensor(recall_score(tm_preds, tm_labels, average='micro'),
                                         device=device)
            recall_list.append(recall_scores)
            #loss_list.append(losses.detach().numpy())

            i += 1
            print(f'Epoch: {epochs}, Iteration: {i}/{data_lengths[phase]}, Loss: {losses}, '
                  f'F1: {f1_scores}, Accuracy: {accuracy}, Type: {phase}')
            data_list = [epochs, accuracy.item(), precision.item(), recall_scores.item(), f1_scores.item(), losses.item()]

            partial_name = "CNN_output_partial_" + str(epochs) + "_" + str(phase) + ".csv"
            modulus_num = 50
            if phase == 'train':
                df_train.loc[len(df_train)] = data_list
                if epochs % modulus_num == 0:
                    df_train.to_csv(partial_name, index=False)
            else:
                df_val.loc[len(df_val)] = data_list
                if epochs % modulus_num == 0:
                    df_val.to_csv(partial_name, index=False)
        print(f'Epoch: {epochs},  Final Loss: {losses}, 'f'Final F1: {f1_scores}, Final Accuracy: {accuracy}, Type: {phase}')


# Save training metrics
full_name_train = "cnn_output_full_" + str(epochs) + "_train.csv"
df_train.to_csv(full_name_train, index=False)

full_name_val = "cnn_output_full_" + str(epochs) + "_val.csv"
df_val.to_csv(full_name_val, index=False)

# Save model and weights
torch.save(model, "cnn_model.pt")
torch.save(model.state_dict(), "cnn_model_state_dict.pt")
torch.save(optimizer.state_dict(), "cnn_model_optimizer_dict.pt")

# print(imgs[0])
# print(labels[0])
