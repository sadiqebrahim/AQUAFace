from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import os
import sys
# sys.path.append('../gmm-torch/')
# from gmm import GaussianMixture
sys.path.append("/workspace")
sys.path.append("/workspace/arcface_pytorch")
sys.path.append("/workspace/arcface_pytorch/models")
sys.path.append("/workspace/arcface_pytorch/utils")
sys.path.append("/workspace/arcface_pytorch/config")
from PIL import Image
import os
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
from models import metrics
from utils.visualizer import *
import torchvision
# from utils import Visualizer, view_model
import torch
import numpy as np
import random
random.seed(16)
import time
from config.config import *
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from scipy.stats import hmean
from sklearn.metrics import accuracy_score
# from test import *
# from data import Dataset
from sklearn.mixture import GaussianMixture
from scipy import spatial
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
#import test_get_features as test
import arcface_pytorch.test as test

import torch.distributed as dist
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
torch.manual_seed(16)
torch.autograd.set_detect_anomaly(True)

# local_rank = 0
# os.environ['RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['MASTER_ADDR'] = '127.0.0.3'
# os.environ['MASTER_PORT'] = '8003'
# dist.init_process_group(backend='nccl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming you have two lists of image features: features1 and features2, and corresponding labels
# features1 and features2 are arrays of shape (n_samples, feature_dim)
# labels are binary (1 for same class, 0 for different class)

class FDataset(Dataset):
        def __init__(self, data, transforms=None):
            super(FDataset, self).__init__()
            self.data = data
            self.transforms = transforms
            self.subject_id_map = self.create_subject_id_map()
        def create_subject_id_map(self):
            """Create a dictionary that maps each unique subject id to a unique integer from 0 to 999"""
            subject_ids = set()
            for sample in self.data:
                try:
                    subject_ids.add("_".join(sample[0].split('/')[-1].split("_")[:2]))
                    subject_ids.add("_".join(sample[1].split('/')[-1].split("_")[:2]))
                except:
                    print(sample)
                    raise 
            subject_id_map = {id: idx for idx, id in enumerate(sorted(subject_ids))}
            # print(subject_id_map)
            print(len(subject_id_map))
            # quit()
            return subject_id_map
        def __getitem__(self, idx):
            st = ""
            # image_path = '/workspace/face_data_v1.2_rearranged/images/training/'+ self.data[idx][0]
            image_path1 = self.data[idx][0]
            # print(image_path1)
            label1 = self.subject_id_map["_".join(image_path1.split('/')[-1].split("_")[:2])]
            # print(label1)
            image_path2 = self.data[idx][1]
            # print(image_path2)
            label2 = self.subject_id_map["_".join(image_path2.split('/')[-1].split("_")[:2])]
            # print(label2)
            label = int(self.data[idx][2])
            #quality = int(self.data[idx][3])
            #age = int(self.data[idx][4])
            # print(label, quality, age)
            sample1 = Image.open(st + image_path1).convert('RGB')
            sample2 = Image.open(st + image_path2).convert('RGB')
            # quit()
            if self.transforms is not None:
                sample1 = self.transforms(sample1)
                sample2 = self.transforms(sample2)
            # return sample1, sample2, label1, label2, label, quality, age
            return sample1, sample2, label1, label2, label

        def __len__(self):
            return len(self.data)


def compute_cosine_similarity(features1, features2):
    feature1=features1.data.cpu().numpy()
    feature2=features2.data.cpu().numpy()
    similarities = []
    for i in range(feature1.shape[0]):
        sim = 1-spatial.distance.cosine(feature1[i].flatten(),feature2[i].flatten())
        if sim < 0:
            sim=0.0
        similarities.append(sim)
    # similarities = cosine_similarity(features1.detach().cpu().numpy(), features2.detach().cpu().numpy())
    # return similarities.diagonal().reshape(-1, 1)  # Return as (16, 1) shape
    
    return np.array(similarities)


def generate_labels(n_samples):
    labels = np.zeros((n_samples, n_samples))
    np.fill_diagonal(labels, 1)
    return labels.flatten()


opt = Config('/workspace/data/codes/resnet18_110.pth', '/workspace/arcface_pytorch/lfw_test_pair.txt')
if opt.display:
    visualizer = Visualizer()
accelerator = Accelerator()
device = accelerator.device
data = np.loadtxt("/workspace/codes/swiggy_pairs/selfie_all_hard.txt", dtype=object)
np.random.shuffle(data)
level = '2'
# data = data[data[:, 4] == level]
# data = data[data[:, 3] == level]
print(data.shape)
# train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = np.split(data, [int(0.99 * len(data))])
# print(train_data.shape, val_data.shape)
# quit()
file_path = data
transform = transforms.Compose(
        [
        # transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Resize((128,128),antialias=True)
            ])
val_dataset = FDataset(data=val_data, transforms=transform)
valloader = DataLoader(val_dataset,
                           batch_size=opt.train_batch_size,
                           shuffle=True,
                           num_workers=opt.num_workers, pin_memory=True, drop_last=True)

if opt.backbone == 'resnet18':
    model = resnet_face18(use_se=opt.use_se)
elif opt.backbone == 'resnet34':
    model = resnet34()
elif opt.backbone == 'resnet50':
    model = resnet50()

if opt.metric == 'add_margin':
    metric_fc = metrics.AddMarginProduct(1000, opt.num_classes, s=30, m=0.35)
elif opt.metric == 'arc_margin':
    metric_fc = metrics.ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
elif opt.metric == 'sphere':
    metric_fc = metrics.SphereProduct(1000, opt.num_classes, m=4)
elif opt.metric == 'adaface':
    metric_fc = metrics.AdaFace(1000, opt.num_classes, m=0.4, s=30.)
elif opt.metric == 'aquaface':
    metric_fc = metrics.Aquaface(1024, opt.num_classes, m=0.4, s=30.)
else:
    metric_fc = torch.nn.Linear(512, opt.num_classes)

checkpoint = torch.load('/workspace/codes/aquaface.pth')
mystatedict = {}
for key in checkpoint.keys():
    m = key.replace("module.","")
    mystatedict[m] = checkpoint[key]
model.load_state_dict(mystatedict, strict=False)

model = DataParallel(model).to(device)
metric_fc = metric_fc.to(device)
model.module.metric_fc = metric_fc

model.eval()

y_pred = []
y_true = []
distance = []

for ii, data in enumerate(tqdm(valloader)):
            image1, image2, label1, label2, label = data
            feature1 = model(image1)
            feature2 = model(image2)

            #pos_scores, neg_scores = cal_score(feature1, feature2, label, quality, age)

            similarities = compute_cosine_similarity(feature1, feature2)

            y_true.extend(label.numpy().tolist())
            distance.extend(similarities.tolist())

def cal_score(feature1, feature2, label, quality, age):
    feature1=feature1.data.cpu().numpy()
    feature2=feature2.data.cpu().numpy()
            
    data=defaultdict(list)
    for i in range(feature1.shape[0]):
        sim = 1-spatial.distance.cosine(feature1[i].flatten(),feature2[i].flatten())
        if sim < 0:
            sim=0.0
        data[label[i].item()].append([sim,label[i].item(),quality[i].item(),age[i].item()])
    # print(data, len(data))
    # print(data[0],data[1])
    pos_scores=np.array(data[1])
    p_scores=np.around(pos_scores,4)
    # print(pos_scores.shape,pos_scores.size,pos_scores)
    # quit()
    neg_scores=np.array(data[0])
    # print(neg_scores.shape, neg_scores.size, neg_scores)
    # scores = np.concatenate((pos_scores,neg_scores))
    # # print(scores)
    # quit()
    # indices = np.random.choice(neg_scores.shape[0], pos_scores.shape[0], replace=False)
    # select random samples
    # n_scores = neg_scores[indices]
    n_scores = np.around(neg_scores,4)
    # print(n_scores.shape,n_scores)
    # quit()
    return p_scores, n_scores

def evaluate_model(model, val_data, device):
    model.eval()  # Set the model to evaluation mode
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Resize((128,128), antialias=True)
    ])
    
    val_dataset = FDataset(data=val_data, transforms=transform)
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    all_pos_scores = []
    all_neg_scores = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for data in tqdm(valloader):
            image1, image2, label1, label2, label, quality, age = data
            image1, image2 = image1.to(device), image2.to(device)
            label = label.to(device)

            feature1 = model(image1)
            feature2 = model(image2)

            pos_scores, neg_scores = cal_score(feature1, feature2, label, quality, age)

            # if pos_scores.shape[0] !=0:
            #     all_pos_scores.extend(pos_scores[:,0])  # Similarity scores for positive pairs
            # if neg_scores.shape[0] !=0:
            #     all_neg_scores.extend(neg_scores[:,0])  # Similarity scores for negative pairs

            if pos_scores.shape[0] !=0:
                all_pos_scores.append(pos_scores)
            if neg_scores.shape[0] !=0:
                all_neg_scores.append(neg_scores)
            
            all_labels.extend([1] * len(pos_scores) + [0] * len(neg_scores))
            # all_labels.extend(torch.hstack((label1,label2)))

    return np.array(all_pos_scores), np.array(all_neg_scores), np.array(all_labels)


# opt = Config('/workspace/data/codes/resnet18_110.pth', '/workspace/arcface_pytorch/lfw_test_pair.txt')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Assuming model is already loaded
# checkpoint = torch.load('/workspace/codes/aquaface.pth')
# model = resnet_face18(use_se=opt.use_se)
# model = DataParallel(model).to(device)
# model.load_state_dict(checkpoint, strict= False)
# model = model.to(device)

# # Load validation data
# data = np.loadtxt("/workspace/sadiq/degraded/train_pairs_verifyme.txt", dtype=object)
# level = '2'
# # data = data[data[:, 4] == level]
# # data = data[data[:, 3] == level]
# _, val_data = np.split(data, [int(0.999 * len(data))])

# pos_scores, neg_scores, labels = evaluate_model(model, val_data, device)

# if pos_scores.size == 0 and neg_scores.size == 0:
#     all_scores = np.empty((0, 4))  # If both are empty, create an empty array with shape (0, 4)
# elif pos_scores.size == 0:
#     all_scores = neg_scores  # If only pos_scores is empty, use neg_scores
# elif neg_scores.size == 0:
#     all_scores = pos_scores  # If only neg_scores is empty, use pos_scores
# else:
#     all_scores = np.concatenate((pos_scores, neg_scores), axis=0)

# # all_scores = np.concatenate([pos_scores, neg_scores])
# fpr, tpr, thresholds = roc_curve(labels, all_scores)
# roc_auc = auc(fpr, tpr)


print(len(y_true))
print(len(distance))
# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_true, distance)
roc_auc = auc(fpr, tpr)

# Print TPR values at FPR of 0.1% and 1%
fpr_values = [0.001, 0.01]
tpr_values = []

for fpr_value in fpr_values:
    # Find the TPR value at the given FPR value
    index = np.where(fpr >= fpr_value)[0][0]
    tpr_values.append(tpr[index])
    print(f"TPR at FPR={fpr_value*100:.1f}%: {tpr[index]:.4f}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlim([1e-5, 1.0])  # Set limits for the x-axis
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (log scale)')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(f'roc_curvea{level}.png')
