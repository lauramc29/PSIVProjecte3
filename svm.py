import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


folder = "db/"

subfolders_metadata = sorted([folder+f for f in os.listdir(folder) if f.endswith(".parquet")])
subfolders_stacks = sorted([folder+f for f in os.listdir(folder) if f.endswith(".npz")])

def load_patient(metadata, stack): 
    
    EEG_GT = pd.read_parquet(metadata)['class'].values
    EEG_Feat = np.load(stack)['data']

    best_config = {'kernel': 'rbf', 'C': 1000}
    
    skf = StratifiedKFold(n_splits=10)
    
    accs = []
    recalls = []
    total_pred = 0
    total = 0
    
    for idxtr, idxts in skf.split(np.arange(len(EEG_GT)), EEG_GT):
    
    
        svm = SVC(class_weight='balanced', C = best_config["C"], kernel = best_config["kernel"])
        
        X_Train = EEG_Feat[idxtr[0::5], :] 
        X_Train = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1] * X_Train.shape[2]))
        
        y_Train = EEG_GT[idxtr[0::5]]
                    
        model = svm.fit(X_Train,y_Train)
        
        X_Test = EEG_Feat[idxts[0::5], :]        
        X_Test = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1] * X_Test.shape[2]))
        
        y_Test = EEG_GT[idxts[0::5]]
                    
        y_pred = model.predict(X_Test)
        
        total += sum(y_Test)
        total_pred += sum(y_pred)
        
        acc = accuracy_score(y_Test, y_pred)
        recall = sum(y_Test*y_pred)/sum(y_Test)
        print(sum(y_Test*y_pred), sum(y_Test))
        accs.append(acc)
        recalls.append(recall)
        
    return accs, recalls, total, total_pred

global_dict = dict()

for metadata, stack in zip(subfolders_metadata, subfolders_stacks):

    accs, recalls, total, total_pred = load_patient(metadata, stack)
    global_dict[stack[:-4]] = [total, total_pred, np.mean(accs), np.mean(recalls)]
    
    
print(global_dict)





    


    