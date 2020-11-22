import os, glob
import argparse
import imageio
import numpy as np
import pandas as pd
import torch
import shutil
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torchvision import models
import torch.nn as nn

#from src.utils import config
#from src.dataloaders.uva_dar_dataset import *
import time

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x

parser = argparse.ArgumentParser()
parser.add_argument("-sfn", "--start_file_num", help="start_file_num",
                        type=int, default=-1)
parser.add_argument("-cdn", "--cuda_device_no", help="cuda device no",
                        type=int, default=0)
parser.add_argument("-ddbp", "--data_dir_base_path", help="data_dir_base_path",
                        default='/train')
parser.add_argument("-edbp", "--embed_dir_base_path", help="data_dir_base_path",
                        default='/fe_embed')
args = parser.parse_args()
data_dir_base_path = args.data_dir_base_path
embed_dir_base_path = args.embed_dir_base_path
main_folder = os.path.dirname(os.path.abspath(__file__))
print("Main: " + str(main_folder))
print("Data dir:" + str(data_dir_base_path))
print("Embed dir:" + str(embed_dir_base_path))

rgb_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ##### Are these numbers flexible?
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

# def collate_fn(batch):
#     return tuple(zip(*batch))
# dataset = MaskDataset(data_transform)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

original_model = models.resnet50(pretrained=True)
feature_extractor = ResNet50Bottom(original_model)
#device = torch.device(f'cuda:{args.cuda_device_no}')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

feature_extractor.to(device)
feature_extractor.eval()
# print(feature_extractor)

train = glob.glob('train/*')

total_parsing = 0
for class_folder in train:
    seq = []
    class_path = os.path.basename(class_folder)
    print("Class: " + str(class_path))
    abs_class_path = os.path.join(main_folder, class_folder)
    total_parsing += 1
    #seq, seq_list = video.get_all_frames()
    for img_path in os.listdir(abs_class_path):
        abs_img_path = os.path.join(abs_class_path, img_path)
        img = Image.open(abs_img_path) #.convert("RGB")
        trans = transforms.ToTensor()
        seq.append(trans(img))
    print("Images in " + str(class_path) + ": " + str(len(seq)))
    embed = feature_extractor(seq)
    embed = embed.detach()
    embed_dir_path = f'{embed_dir_base_path}/{class_path}'
    try:
        shutil.rmtree(embed_dir_path)
        os.makedirs(embed_dir_path)
    except:
        pass
    # Remove file extension from name
    filename = filename[:-4]
    torch.save(embed, f'{embed_dir_path}/{filename}.pt')
    if(total_parsing%100==0):
        print(f'parsing completed:{total_parsing}')


# #### REFERENCE CODE
# #activity_types = ['change_lane', 'checking_mirror_middle', 'checking_speed_stack',
# #                  'checking_mirror_driver', 'checking_mirror_passenger']
# for activity_type in activity_types:
#     data_dir_path = f'{data_dir_base_path}/{activity_type}'
#     total_parsing = 0
#     print(f'start parsing {activity_type}')
#     for count, filename in enumerate(sorted(os.listdir(data_dir_path), reverse=False)):
#         # print('filename',filename)
#         # tm_filename = filename.split('.')[0]
#         # if(total_parsing<=args.start_file_num and args.start_file_num!=-1):
#         #     total_parsing += 1
#         #     continue
#         # if(os.path.exists(f'{embed_dir_base_path}/{tm_filename}.pt')):
#         #     total_parsing += 1
#         #     continue
#         try:
#             video = Video(f'{data_dir_path}/{filename}', transforms=rgb_transforms)
#         except:
#             continue
#         seq, seq_len = video.get_all_frames()
#         seq = seq.to(device)
#         embed = feature_extractor(seq)
#         embed = embed.detach()
#
#         embed_dir_path = f'{embed_dir_base_path}/{activity_type}'
#         try:
#             os.makedirs(embed_dir_path)
#         except:
#             pass
#         filename = filename[:-4]
#     ######## Per file vs. video vs. set, granularity?
#         torch.save(embed, f'{embed_dir_path}/{filename}.pt')
#         total_parsing += 1
#         # if(total_parsing%100==0):
#         #     print(f'parsing completed:{total_parsing}')
#     print(f'Total parsed files: {total_parsing}')
