import pandas as pd
import numpy as np
from skimage import measure
import os
from fastparquet import write


subfolders_data = sorted(["/export/home/mapiv/database/"+f for f in os.listdir("/export/home/mapiv/database/") if f.endswith(".parquet")])

def load_patient(patient_path): 
    data = pd.read_parquet(patient_path)
    excel = pd.read_excel("/export/home/mapiv/database/df_annotation_full.xlsx") 
    
    dic_pacient = {}
    pacient = np.unique(data['filename'])
    n_pacient = np.unique(data['PatID'])[0]
    dataPat = excel[excel.PatID==n_pacient]    
    metadades = pd.DataFrame()
    s0 = []
    s1 = []
    s = []
    
    for i in range(len(pacient)):
        recording = np.zeros_like(data[data['filename']==pacient[i]]['type'])
        nom_pacient = pacient[i]
        print(nom_pacient)
        dic_pacient[nom_pacient] = recording 
        dataPatfile = dataPat[dataPat['filename']==pacient[i]]
        recording_parket = data[data['filename']==pacient[i]]
        
        
        for d in dataPatfile.iloc:
            if d['seizure_id'] == 1:
                inici = int(d['seizure_start']) 
                final = int(d['seizure_end'])
                dic_pacient[nom_pacient][inici*128:final*128+1] = 1   
                dic_pacient[nom_pacient][(inici-30)*128:inici*128] = 2  
                dic_pacient[nom_pacient][final*128:(final+30)*128] = 3
    
            LABEL_0 = measure.label(dic_pacient[nom_pacient] == 0)
            LABEL_1 = measure.label(dic_pacient[nom_pacient] == 1)
            
            n_lab0 = np.unique(LABEL_0)
            n_lab0 = n_lab0[n_lab0!=0]
            n_lab1 = np.unique(LABEL_1)
            n_lab1 = n_lab1[n_lab1!=0]
            
            
            for lab0 in n_lab0:
                component_conexa0 = np.nonzero(LABEL_0==lab0)
                longitud = len(component_conexa0[0])
                long_component = int(longitud/128)
                longitud_final = long_component * 128
                origin=np.arange(0, longitud_final, 129)
                for n0 in range(0, len(origin), 2):
                    inici = origin[n0]
                    final = inici+129
                    pos_w0 = component_conexa0[0][inici:final]
             
                    df1 = recording_parket.iloc[pos_w0[0]: pos_w0[-1]]
                    if df1.shape[0] == 128:
                        
                        df_window = df1[df1.columns[0:21]]
                        
                        s0.append(df_window.transpose()) 
                        label = LABEL_0[inici]
                        window = pd.DataFrame({'class':0, 'label':label, 'filename':n_pacient}, index=[0])
                        metadades = pd.concat([metadades, window], ignore_index=True) 
        
            
            for lab1 in n_lab1:
                component_conexa1 = np.nonzero(LABEL_1==lab1)
                longitud = len(component_conexa1[0])
                long_component = int(longitud/128)
                longitud_final = long_component * 128
                origin1=np.arange(0, longitud_final, 129) 
                for n1 in range(0, len(origin1)): 
                    inici1 = origin1[n1]
                    final1 = inici1+129
                    pos_w1 = component_conexa1[0][inici1:final1]
                    df1 = recording_parket.iloc[pos_w1[0]: pos_w1[-1]]
                    if df1.shape[0] == 128:
                        
                        df_window1 = df1[df1.columns[0:21]]
                        s1.append(df_window1.transpose()) 
                        label1 = LABEL_1[inici1]
                        window1 = pd.DataFrame({'class':1, 'label':label1, 'filename':n_pacient}, index=[0])
                        metadades = pd.concat([metadades, window1], ignore_index=True) 
                    
                
                long_component1  = int((longitud-64)/128) 
                longitud_final1 = long_component1 * 128
                origin2=np.arange(0, longitud_final1, 129) 
                for n1 in range(0, len(origin2)):
                    inici1 = origin1[n1]+64
                    final1 = inici1+129
                    pos_w1 = component_conexa1[0][inici1:final1]
                    df1 = recording_parket.iloc[pos_w1[0]: pos_w1[-1]]
                    if df1.shape[0] == 128:
                        
                        df_window1 = df1[df1.columns[0:21]]
                        s1.append(df_window1.transpose()) 
                        label1 = LABEL_1[inici1]
                        window2 = pd.DataFrame({'class':1, 'label':label1, 'filename':n_pacient}, index=[0])
                        metadades = pd.concat([metadades, window2], ignore_index=True)  
        
    s = s0+s1
       
    stack = np.stack(s)
    
    outDir = '/export/home/mapiv07/resultats/'

    
    np.savez(os.path.join(outDir, n_pacient), data = stack)
    
    nom_parquet = n_pacient+".parquet"


    write(os.path.join(outDir, nom_parquet), metadades)
       

            
            
for patient_path in subfolders_data:
    print(patient_path)
    load_patient(patient_path)
    
