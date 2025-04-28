import matplotlib.pyplot as plt
import glob
#import config
from numpy import argmax, sqrt
from torch.autograd import Variable
from scipy import spatial
from sklearn.metrics import roc_auc_score, roc_curve, auc
import sys

from models import resnet
from collections import defaultdict
import torch
import numpy as np
import cv2
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
# import torch.nn.functional as F
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.nn import DataParallel
#from deepface.commons import functions
import time
import pickle
# from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
# from light_cnn_v4 import LightCNN_V4
import argparse
import sklearn
import sys
# sys.path.append('ElasticFace')
# from backbones.iresnet import iresnet100, iresnet50
torch.manual_seed(15)
# import net
from PIL import Image
import seaborn as sns
# sys.path.append('gmm-torch')
# from gmm import GaussianMixture
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getembs(i):
        #print("name",i)
        if i in embeddings.keys():
              
              return embeddings[i]
        im1=cv2.imread(i, 0)
        im1 = cv2.cvtColor(im1,cv2.COLOR_GRAY2RGB)
        im=cv2.resize(im1,(128,128))
        Img1 = np.dstack((im, np.fliplr(im)))
        Img1 = Img1.transpose((2, 0, 1))       
        Img1 = Img1[:, np.newaxis, :, :]
        Img1 = Img1.astype(np.float32, copy=False)
        Img1-=127.5
        Img1 /= 127.5
        im=torch.from_numpy(Img1)
        img1 = im.to(device)
        emb1 = model_mugs1(img1)
        output1 = emb1
        output1 = emb1.data.cpu().numpy()
        # print(output1.shape)
        # quit()
        fe_11 = output1[::2]
        fe_21 = output1[1::2]
        feature1 = np.hstack((fe_11, fe_21))
        feature1 = feature1.flatten()
        embeddings[i]=feature1   
        return feature1

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
                    # subject_ids.add(sample[0])
                    # subject_ids.add(sample[1])
                except:
                    print(sample)
                    raise 
            subject_id_map = {id: idx for idx, id in enumerate(sorted(subject_ids))}
            # print(subject_id_map)
            print(len(subject_id_map))
            # quit()
            return subject_id_map
        def __getitem__(self, idx):
            # image_path = '/workspace/face_data_v1.2_rearranged/images/training/'+ self.data[idx][0]
            image_path1 = self.data[idx][0]
            # print(image_path1)
            label1 = self.subject_id_map["_".join(image_path1.split('/')[-1].split("_")[:2])]
            #label1 = self.subject_id_map[image_path1]
            # print(label1)
            image_path2 = self.data[idx][1]
            # print(image_path2)
            label2 = self.subject_id_map["_".join(image_path2.split('/')[-1].split("_")[:2])]
            #label2 = self.subject_id_map[image_path2]
            # print(label2)
            label = int(self.data[idx][2])
            quality = int(self.data[idx][3])
            age = int(self.data[idx][4])
            # print(label, quality, age)
            sample1 = Image.open(image_path1).convert('RGB')
            sample2 = Image.open(image_path2).convert('RGB')
            # quit()
            if self.transforms is not None:
                sample1 = self.transforms(sample1)
                sample2 = self.transforms(sample2)
            return sample1, sample2, label1, label2, label, quality, age
        def __len__(self):
            return len(self.data)


######## Arcface
model_mugs1 = resnet.resnet_face18()
# model_mugs = DataParallel(model_mugs)

checkpoint = torch.load('/workspace/codes/aquaface.pth')
# checkpoint = torch.load('best_checkpoint_scface/Arc_best_checkpoint.pth')

mystatedict = {}
for key in checkpoint.keys():
  m = key.replace("module.","")
  mystatedict[m] = checkpoint[key]
model_mugs1.load_state_dict(mystatedict, strict=False)
# model_mugs.load_state_dict(checkpoint)
model_mugs1 = model_mugs1.to(device)
model_mugs1.eval()
embeddings = {}


model_mugs2 = resnet.resnet_face18()
# model_mugs = DataParallel(model_mugs)

checkpoint = torch.load('/workspace/codes/checkpoints__/resnet18_103560.pth')
# checkpoint = torch.load('best_checkpoint_scface/Arc_best_checkpoint.pth')

mystatedict = {}
for key in checkpoint.keys():
  m = key.replace("module.","")
  mystatedict[m] = checkpoint[key]
model_mugs2.load_state_dict(mystatedict, strict=False)
# model_mugs.load_state_dict(checkpoint)
model_mugs2 = model_mugs2.to(device)
model_mugs2.eval()
######## Arcface



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

    n_scores = np.around(neg_scores,4)
    # print(n_scores.shape,n_scores)
    # quit()
    return p_scores, n_scores


# pos_npy = np.load("postest_sc_new.npy",allow_pickle=True)
# neg_npy = np.load("negtest_sc_new.npy",allow_pickle=True)
# pos_npy = np.loadtxt("/workspace/sadiq/degraded/train_pairs_verifyme.txt",dtype=object)
pos_npy = np.loadtxt("/workspace/codes/swiggy_pairs/train_pairs_swiggy.txt",dtype=object)

np.random.shuffle(pos_npy)
train_data, pos_npy = np.split(pos_npy, [int(0.1 * len(pos_npy))])

transform = transforms.Compose(
            [
            # transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
            #  transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5]),
             transforms.Resize((128,128),antialias=True)
             ])
train_dataset = FDataset(data=pos_npy, transforms=transform)

trainloader = DataLoader(train_dataset,
                                batch_size=64,
                                shuffle=True,
                                num_workers=4, pin_memory=True, drop_last=True)
    


pos1=[]
pos2=[]
neg1=[]
neg2=[]
grey = transforms.Grayscale()
for ii, data in enumerate(tqdm(trainloader)):
    image1, image2, label1, label2, label, quality, age = data
 
    feature1 = model_mugs2(grey(image1).to("cuda"))
    feature2 = model_mugs2(grey(image2).to("cuda"))
    pos_scores, neg_scores = cal_score(feature1.to("cuda"), feature2.to("cuda"), label.to("cuda"), quality.to("cuda"), age.to("cuda"))
    if pos_scores.shape[0] !=0:
        pos2.append(pos_scores)
    if neg_scores.shape[0] !=0:
        neg2.append(neg_scores)
    
# pos_X1 = np.concatenate(pos1)
pos_X2 = np.concatenate(pos2)

X = pos_X2[:,[0,2,3]]
# X = neg_X[:,[0,2,3]]
gmm = GaussianMixture(n_components=3, covariance_type='full')  # Adjust components based on your data
gmm.fit(X)

# Predict clusters
labels = gmm.predict(X)

# Plot the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')
ax.set_xlim(0.3, 1)
# Add labels and titles
ax.set_xlabel('Sim')
ax.set_ylabel('Quality')
ax.set_zlabel('Age')
plt.title('GMM Clustering')

# Show the color legend
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

# plt.savefig("gmmm_.jpg")
plt.savefig("gmm_arc_pos_real.jpg")
quit()

# Fit GMM to quality_labels
n_components = 3  # Choose an appropriate number of components
gmm1 = GaussianMixture(n_components=n_components)
# gmm.fit(np.column_stack((similarity_scores, quality_labels)))
gmm1.fit(pos_X1)

# Predict the GMM components
gmm_labels1 = gmm1.predict(pos_X1)
probs1 = gmm1.predict_proba(pos_X1)
score1 = gmm1.score_samples(pos_X1)
print(gmm_labels1[:10])
print(probs1[:10])
print(score1.shape)


n_components = 3  # Choose an appropriate number of components
gmm2 = GaussianMixture(n_components=n_components)
# gmm.fit(np.column_stack((similarity_scores, quality_labels)))
gmm2.fit(pos_X2)

# Predict the GMM components
gmm_labels2 = gmm2.predict(pos_X2)
probs2 = gmm2.predict_proba(pos_X2)
score2 = gmm2.score_samples(pos_X2)
print(gmm_labels2[:10])
print(probs2[:10])
print(score2.shape)

# Create scatter plot with fitted GMM components
plt.figure(figsize=(10, 6))
plt.scatter(score1, score2, c=pos_X1[:, 3], s=10, cmap='viridis', alpha=0.5)
plt.title('Scatter Plot with Fitted GMM Components')
plt.xlabel('score1')
plt.ylabel('score2')
plt.colorbar(label='GMM Component')
plt.savefig("hma.jpg")

