from __future__ import print_function
import os
import sys
# sys.path.append('../gmm-torch/')
# from gmm import GaussianMixture
from models import get_model
from loss.focal import *
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
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
#import test_get_features as test
# import arcface_pytorch.test as test
# import onnx
# from onnx2pytorch import ConvertModel

import torch.distributed as dist
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
torch.manual_seed(16)
torch.autograd.set_detect_anomaly(True)

local_rank = 0
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.3'
os.environ['MASTER_PORT'] = '8005'
dist.init_process_group(backend='nccl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FixedDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(FixedDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return torch.nn.functional.dropout(x, p=self.p, training=self.training)

def save_model(model, save_path, name, iter_cnt, last_tpr, totalTrainLoss):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    # torch.save(model.state_dict(), save_name)
    # current_tpr = os.system('python test_get_features.py --model_path model --
    # pair_list = '/workspace/codes/pairCreation/pairs_test_v2_single/selfie_all.txt'
    os.makedirs(save_path,exist_ok=True)
    # pair_list = '/workspace/data/codes/selfie_all.txt'
    pair_list_path = '/workspace/sadiq/degraded/train_pairs_verifyme.txt'

    with open(pair_list_path,"r") as f:
        pair_list = f.readlines()

    model.eval()
    current_tpr = check_pairs(model, random.sample(pair_list,100))
    # print("last_tpr:", last_tpr)
    # print("current_tpr:", current_tpr)
    print('Last_tpr {l_tpr} \t'
    'Current_tpr {c_tpr} \t'.format(l_tpr=last_tpr, c_tpr=current_tpr))
    # if current_tpr is not None and float(current_tpr) > last_tpr and current_tpr != 1.0:
    #     last_tpr = float(current_tpr)
    #     torch.save(model.state_dict(), save_name)
    if totalTrainLoss is not None and float(totalTrainLoss) < last_tpr:
        last_tpr = float(totalTrainLoss)
        torch.save(model.state_dict(), save_name)
    return last_tpr


def compute_cosine_similarity(features1, features2):
    similarities = cosine_similarity(features1.detach().cpu().numpy(), features2.detach().cpu().numpy())
    return similarities.diagonal().reshape(-1, 1)  # Return as (16, 1) shape

def find_optimal_threshold(y_true, distances):
    fpr, tpr, thresholds = roc_curve(y_true, distances)
    fpr_values = [0.001, 0.01]
    tpr_values = []

    for fpr_value in fpr_values:
        # Find the TPR value at the given FPR value
        index = np.where(fpr >= fpr_value)[0][0]
        tpr_values.append(tpr[index])
        print(f"TPR at FPR={fpr_value*100:.1f}%: {tpr[index]:.4f}")

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"TPR:: {tpr[optimal_idx]}, FPR:: {fpr[optimal_idx]}")
    #return optimal_threshold
    print(tpr_values[1])
    return tpr_values[1]

def evaluate_pairs(model, pairs, device='cuda'):
    model.eval()
    embeddings = {}
    
    for pair in tqdm(pairs):
        img1_path, img2_path, label, _, _ = pair.split(" ")
        if img1_path not in embeddings:
            img1 = load_image(img1_path)  # Load and preprocess the image
            embeddings[img1_path] = model(img1.to(device)).detach().cpu().numpy()
        if img2_path not in embeddings:
            img2 = load_image(img2_path)
            embeddings[img2_path] = model(img2.to(device)).detach().cpu().numpy()
    
    y_true = []
    distances = []
    
    for pair in pairs:
        img1_path, img2_path, label, _, _ = pair.split(" ")
        emb1 = embeddings[img1_path]
        emb2 = embeddings[img2_path]
        distance = spatial.distance.cosine(emb1.flatten(),emb2.flatten())
        y_true.append(int(label))
        distances.append(distance)
    
    threshold =find_optimal_threshold(y_true,distances)
    
    y_pred = [1 if d < threshold else 0 for d in distances]
    
    #return accuracy_score(y_true, y_pred), threshold
    return threshold


def load_image(img_path):
    from PIL import Image
    import torchvision.transforms as transforms
    
    transform = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                # transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Resize((112,112),antialias=True)
            ])    
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def check_pairs(model, pair_list, device='cuda'):
    tpr = evaluate_pairs(model, pair_list, device)
    return tpr

def cal_score(feature1, feature2, label, quality, age):
    feature1=feature1.data.cpu().numpy()
    feature2=feature2.data.cpu().numpy()
            
    data=defaultdict(list)
    for i in range(feature1.shape[0]):
        sim = 1-spatial.distance.cosine(feature1[i].flatten(),feature2[i].flatten())
        if sim < 0:
            sim=0.0
        data[label[i].item()].append([sim,label[i].item(),quality[i].item(),age[i].item()])
    
    pos_scores=np.array(data[1])
    p_scores=np.around(pos_scores,4)
    neg_scores=np.array(data[0])
    n_scores = np.around(neg_scores,4)
    return p_scores, n_scores

cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-06)
class ContrastiveLoss(torch.nn.Module):
        def __init__(self, margin=0.5):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, f1, f2, label, lr_norms):
            
            dists = 1-cosine_sim(f1, f2)
            # print(dists.shape,label.shape,lr_norms.shape,lr_norms)
            m_norms = self.margin*lr_norms
            # print(m_norms.shape,m_norms)
            loss_contrastive = torch.mean((label) * torch.pow(dists, 2) +
                            (1-label) * torch.pow(torch.clamp(m_norms - dists, min=0.0), 2))
            # print(loss_contrastive)
            # quit()
            return loss_contrastive

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

def main():
    opt = Config('/workspace/data/codes/resnet18_110.pth', '/workspace/arcface_pytorch/lfw_test_pair.txt')
    if opt.display:
        visualizer = Visualizer()
    accelerator = Accelerator()
    device = accelerator.device
    data = np.loadtxt("/workspace/CASIA-WebFace_split/val_pairs.txt",dtype=object)
    np.random.shuffle(data)
    print(data.shape)
    train_data, val_data = np.split(data, [int(0.01 * len(data))])

# train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    # train_data, val_data = np.split(data, [int(0.8 * len(data))])
    # print(train_data.shape, val_data.shape)
    # quit()
    file_path = data
    transform = transforms.Compose(
            [
            # transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
            #  transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5]),
             transforms.Resize((112,112),antialias=True)
             ])
    train_dataset = FDataset(data=train_data, transforms=transform)
    
    trainloader = DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    
    
    # val_dataset = FDataset(data=val_data, transforms=transform)
    # valloader = DataLoader(val_dataset,
                                #   batch_size=opt.train_batch_size,
                                #   shuffle=True,
                                #   num_workers=opt.num_workers, pin_memory=True, drop_last=True)
    print('{} train iters per epoch:'.format(len(trainloader)))
    # print('{} val iters per epoch:'.format(len(valloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    elif opt.loss == 'L1_loss':
        criterion = torch.nn.L1Loss()
    elif opt.loss == 'con_loss':
        criterion = ContrastiveLoss().to(device)
    # else:
    criterion1 = torch.nn.CrossEntropyLoss().to(device)
    criterion2 = torch.nn.BCEWithLogitsLoss().to(device)

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    model = resnet101()

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
    
    checkpoint = torch.load('/workspace/codes/quickmatch/backbone.pth')
    mystatedict = {}
    # for key in checkpoint.keys():
    #     m = key.replace("module.","")
    #     mystatedict[m] = checkpoint[key]
    # model.load_state_dict(mystatedict, strict=False)

    model = get_model("r100", fp16=False)
    model.load_state_dict(checkpoint)

    # onnx_model = onnx.load('/workspace/codes/model.onnx')
    # model = ConvertModel(onnx_model)

    # model = torch.load("/workspace/codes/r100_ms1mv2.pth")
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Dropout):
    #         setattr(model, name, FixedDropout(p=module.p))

    model = DataParallel(model).to(device)
    metric_fc = metric_fc.to(device)
    model.module.metric_fc = metric_fc



    for name, param in model.named_parameters():
        # print(name)
        if 'bn4' not in name and 'fc5' not in name and 'bn5' not in name and 'metric_fc' not in name:
        # if 'fc5' not in name and 'metric_fc' not in name:
            # print(name)
            param.requires_grad = False


    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

        # optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    # lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                     lr=2e-3, weight_decay=opt.weight_decay)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # trainloader, valloader, my_model, optimizer, scheduler = accelerator.prepare(trainloader, valloader, my_model, optimizer, scheduler)
    trainloader, model, optimizer = accelerator.prepare(trainloader, model, optimizer)
    # trainloader, model, optimizer, scheduler = accelerator.prepare(trainloader, model, optimizer, scheduler)
    steps=120
    # steps=80
    best_tpr = 0.0
    best_loss = 1000
    # pos_feat_dict={'sims': [], 'labels': [], 'quality': [], 'age': []}
    # neg_feat_dict={'sims': [], 'labels': [], 'quality': [], 'age': []}
    pos=[]
    neg=[]
    for epoch in range(opt.max_epoch):
        model.train()
        totalTrainLoss = 0.0
        totalAcc = 0.0
        for ii, data in enumerate(tqdm(trainloader)):
            image1, image2, label1, label2, label, quality, age = data
            # img1, img2 = image1.float().to(device), image2.float().to(device)
            # label = label.to(device).long()
            # label1 = label1.float().to(device).long()
            # label2 = label2.float().to(device).long()
            optimizer.zero_grad()
            
            feature1 = model(image1)
            feature2 = model(image2)
            pos_scores, neg_scores = cal_score(feature1, feature2, label, quality, age)

            if pos_scores.shape[0] !=0:
                pos.append(pos_scores)
            if neg_scores.shape[0] !=0:
                neg.append(neg_scores)
            # out = model.module.metric_fc(feature2,label2)
            features = torch.vstack((feature1, feature2))

            labels = torch.hstack((label1,label2))
            # print(labels.shape)
            outputs = model.module.metric_fc(features,labels)
            cl_loss = criterion1(outputs,labels)
            
            iters = epoch * len(trainloader) + ii
            
            if (iters+1) % steps == 0:
               

                pos_X = np.concatenate(pos)
                # print(pos_X,pos_X.shape)
                neg_X = np.concatenate(neg)
                # print(neg_X,neg_X.shape)
                # quit()
                pos_gmm = GaussianMixture(n_components=3)
                pos_gmm.fit(pos_X)
                # print(pos_gmm.get_params)
                # print(pos_gmm.aic(pos_X), pos_gmm.bic(pos_X))
                # print(pos_gmm.converged_)
                
                neg_gmm = GaussianMixture(n_components=3)
                neg_gmm.fit(neg_X)
    
                pos_scores, neg_scores = cal_score(feature1, feature2, label, quality, age)
                # Check if pos_scores or neg_scores are empty and handle concatenation accordingly
                if pos_scores.size == 0 and neg_scores.size == 0:
                    scores = np.empty((0, 4))  # If both are empty, create an empty array with shape (0, 4)
                elif pos_scores.size == 0:
                    scores = neg_scores  # If only pos_scores is empty, use neg_scores
                elif neg_scores.size == 0:
                    scores = pos_scores  # If only neg_scores is empty, use pos_scores
                else:
                    scores = np.concatenate((pos_scores, neg_scores), axis=0)
                # print(scores,scores.shape)

                pdf1 = pos_gmm.score_samples(scores)
                # print(pdf1.shape)
                pdf2 = neg_gmm.score_samples(scores)
                # print(pdf2.shape)
                llr = pdf1 / pdf2
                # print(llr,llr.shape)
                llr = torch.tensor(llr)
                lr_norms = torch.clip(llr, min=-1, max=1) # for stability
                lr_norms = lr_norms.to(device)
                # print(lr_norms.shape,lr_norms)
                # quit()
                # dists=torch.tensor(dists).to(device)
                # print(dists.shape,dists)
                # out = model.module.metric_fc(feats,label,lr_norms)
                # print(out.shape)
                # labels = torch.nn.functional.one_hot(label, num_classes=2)
                # labels=labels.float()
                # cl_loss = criterion2(out,labels)
                # print(cl_loss)
                # con_loss = criterion(dists,label,lr_norms)
                con_loss = criterion(feature1,feature2,label,lr_norms)
                # print('con_loss:',con_loss)
                loss = cl_loss + con_loss
                # loss = con_loss
                # print('comb_loss:',loss)
                totalTrainLoss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                # optimizer.zero_grad()
                
                # print(f'Epoch {epoch + 1}')
                # print(f'Iter {iters + 1}')
                print('\nIter {it} \t'.format(it=iters+1))
                print(f'Train_Loss: {totalTrainLoss / len(trainloader)}')
                print(f'Train_acc: {totalAcc, totalAcc / len(trainloader)}')
                print('\nTrain set: Loss {loss:.4f} \t'
               .format(loss=(totalTrainLoss / steps)))
                # best_tpr = save_model(model, opt.checkpoints_path, opt.backbone, iters+1, best_tpr)
                # best_tpr = best_tpr
                print(totalTrainLoss / steps)
                best_loss = save_model(model, opt.checkpoints_path, opt.backbone, iters+1, best_loss, totalTrainLoss / steps)
                best_loss = best_loss
                totalTrainLoss = 0.0
                totalAcc = 0.0
                if ((iters+1) % (steps)) == 0:
                    with torch.no_grad():
                        for level in ['0','1','2']:
                            val_data_ = train_data[train_data[:, 3] == level]
                            print(val_data.shape)

                            val_dataset = FDataset(data=val_data_, transforms=transform)
                            valloader = DataLoader(val_dataset,
                                                batch_size=opt.train_batch_size,
                                                shuffle=False,  # No need to shuffle for validation
                                                num_workers=opt.num_workers,
                                                pin_memory=True,
                                                drop_last=True)
                                    

                            y_pred = []
                            y_true = []
                            distance = []

                            for ii, data in enumerate(tqdm(valloader)):
                                        image1, image2, label1, label2, label, quality, age = data
                                        feature1 = model(image1)
                                        feature2 = model(image2)

                                        #pos_scores, neg_scores = cal_score(feature1, feature2, label, quality, age)

                                        similarities = compute_cosine_similarity(feature1, feature2)

                                        y_true.extend(label.numpy().tolist())
                                        distance.extend(similarities.tolist())

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
            else:
                totalTrainLoss += cl_loss.item()
                outputs = outputs.data.cpu().numpy()
                outputs = np.argmax(outputs, axis=1)
                labels = labels.data.cpu().numpy()
                acc = np.mean((outputs == labels).astype(int))
                totalAcc += acc
                accelerator.backward(cl_loss)
                optimizer.step()
                # optimizer.zero_grad()
        print('\nEpoch {ep} \t'.format(ep=epoch+1))
        # print(f'Train_Loss: {totalTrainLoss / len(trainloader)}')
        # print(f'Train_acc: {(100*(totalAcc / len(trainloader)))}')
        print('\nTrain_Loss {loss:.4f} \t'
        'Acc {acc:.3f} \t'.format(loss=(totalTrainLoss / len(trainloader)), acc=(100.0*(totalAcc/len(trainloader)))))
        best_tpr = save_model(model, opt.checkpoints_path, opt.backbone, epoch+1, best_tpr,(totalTrainLoss / len(trainloader))*100)
        best_tpr = best_tpr

    
    model.eval()

    for level in ['0','1','2']:
        val_data_ = val_data[val_data[:, 3] == level]
        print(val_data.shape)

        val_dataset = FDataset(data=val_data_, transforms=transform)
        valloader = DataLoader(val_dataset,
                            batch_size=opt.train_batch_size,
                            shuffle=False,  # No need to shuffle for validation
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=True)
                

        y_pred = []
        y_true = []
        distance = []

        for ii, data in enumerate(tqdm(valloader)):
                    image1, image2, label1, label2, label, quality, age = data
                    feature1 = model(image1)
                    feature2 = model(image2)

                    #pos_scores, neg_scores = cal_score(feature1, feature2, label, quality, age)

                    similarities = compute_cosine_similarity(feature1, feature2)

                    y_true.extend(label.numpy().tolist())
                    distance.extend(similarities.tolist())

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
    




if __name__ == '__main__':
    main()
        
