
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV


folder = r"/export/home/mapiv07/resultats/"

subfolders_metadata = sorted([folder+f for f in os.listdir(folder) if f.endswith(".parquet")])
subfolders_stacks = sorted([folder+f for f in os.listdir(folder) if f.endswith(".npz")])

subfolders_metadata_train = subfolders_metadata[:-1]
subfolders_stacks_train = subfolders_stacks[:-1]

subfolders_metadata_test = subfolders_metadata[-1]
subfolders_stacks_test = subfolders_stacks[-1]


def random_search(metadades, stack):
    
    EEG_GT = pd.read_parquet(metadades)['class'].values
    EEG_Feat = np.load(stack)['data']

    skf = StratifiedKFold(n_splits=10)
    
    configs = []
    
    
    for idxtr, idxts in skf.split(np.arange(len(EEG_GT)), EEG_GT):
       
        
        svm = SVC(class_weight='balanced')
        distributions = dict(C=[0.01, 0.1, 1, 10, 100, 1000],
                         kernel=['poly', 'rbf'])
    
        clf = RandomizedSearchCV(svm, distributions, refit=True)
        
        print(idxtr[0::30])
        
        X = EEG_Feat[idxtr[0::30], :]      
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
        
        y = EEG_GT[idxtr[0::30]]
                    
        search = clf.fit(X,y)
        
        if search.best_params_ not in configs:
            configs.append(search.best_params_)
        
    return configs
    
configs_final = []   

for metadata, stack in zip(subfolders_metadata_train, subfolders_stacks_train):
    configs = random_search(metadata, stack)
    configs_final += [x for x in configs if x not in configs_final] 
    
print("---------------\n")
print(configs_final)
print("\n---------------\n")



    


    