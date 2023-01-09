# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:53:06 2022

@author: andre
"""
import torch 
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.model_selection import StratifiedKFold


folder = "db/"

metadades_files = sorted([folder+f for f in os.listdir(folder) if f.endswith(".parquet")])
windows_files = sorted([folder+f for f in os.listdir(folder) if f.endswith(".npz")])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

@torch.no_grad()  
def validate(criterion, model, test_data_loader):
    
    total_loss = 0.0
    correct = 0
    true_positives = 0
    possible_positives = 0
    
    model.eval()

    for batchidx, (data, target) in enumerate(test_data_loader):
        data, target = data.to(device), target.to(device)

        output = model(data.float())
        loss = criterion(output, target)
                                                                     
        pred = output.data.max(1, keepdim=True)[1]                                          
        correct += pred.eq(target.view_as(pred)).sum().item()
        true_positives += (target * pred).sum().item()
        possible_positives += target.sum().item()
        
        total_loss += loss.item()

    accuracy = 100. * correct / len(test_data_loader.dataset)
    recall = true_positives / possible_positives
    
    return total_loss / len(test_data_loader), accuracy, recall


def train(epoch, criterion, model, optimizer, train_data_loader):
    
    total_loss = 0.0

    model.train()

    for batchidx, (data, target) in enumerate(train_data_loader):
        
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)

        output = model(data.float())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() 
        
        print('Epochs', epoch)

    return total_loss / len(train_data_loader)

      
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.backbone = nn.Sequential(
          nn.AvgPool2d((21,1)),
          nn.Conv1d(1, 16, 5, padding=2, stride=1), nn.ReLU(), 
          nn.MaxPool1d(4),
          nn.Conv1d(16, 32, 5, padding=2, stride=1), nn.ReLU(), 
          nn.MaxPool1d(4), 
          nn.Conv1d(32, 64, 5, padding=2, stride=1), nn.ReLU(), 
          nn.MaxPool1d(4), 
          nn.Flatten(),
          )
          
        self.fc1 = nn.Linear(128,128) 
        self.fc2 = nn.Linear(128,2) 
        
    def forward(self, x):
        features = self.backbone(x)
        fc1 = self.fc1(features)
        fc2 = self.fc2(fc1)
        
        return fc2
        


def load_patient(metadades, windows):

    EEG_GT = pd.read_parquet(metadades)['class'].values
    EEG_Feat = np.load(windows)['data']
    
    skf = StratifiedKFold(n_splits=10)
    
    metrics = {"acc": [], "recall": []}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cudnn.benchmark = True  
    learning_rate = 1e-2
    lambda_l2 = 1e-5
    momentum = 0.5
    torch.manual_seed(0) 
    
    for idxtr, idxts in skf.split(np.arange(len(EEG_GT)), EEG_GT):
        
        train_data = EEG_Feat[idxtr[0::5], :]
        
                
        train_labels = EEG_GT[idxtr[0::5]]
        
        
        test_data = EEG_Feat[idxts[0::5], :]
        
        test_labels = EEG_GT[idxts[0::5]]      
        
        train_data2 = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
        test_data2 = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))
        
        train_data_loader = DataLoader(train_data2, shuffle=False, batch_size=64)
        test_data_loader = DataLoader(test_data2, shuffle=False, batch_size=1000)
        
        model = CNN()
        
        model.to(device)
        
        criterion = torch.nn.CrossEntropyLoss() 

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=lambda_l2) # built-in L2
        
        
        llista_acc = []
        
        for epoch in range(5):
            train(epoch, criterion, model, optimizer, train_data_loader)
            val_loss, acc, recall = validate(criterion, model, test_data_loader)
            llista_acc.append(acc)
            metrics["recall"].append(recall)
        mitjana_acc = np.mean(llista_acc)           
        metrics["acc"].append(mitjana_acc)         
        #metrics["acc"].append(acc)      
    return metrics
        

global_dict = dict()

for metadades, windows in zip(metadades_files, windows_files):

    metrics = load_patient(metadades, windows)
    global_dict[windows[:-4]] = metrics
    
print(global_dict)
