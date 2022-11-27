
# Prima classificatore binario : Grasp / no Grasp , poi classificatore multiclasse solo con Grasp
import numpy as np
import math
import os
import math
import pandas as pd
import pickle
import argparse
import logging
import pathlib

from math import sin,cos,pi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler("window_split_data.log", mode='w'),
        logging.StreamHandler()
    ]
)


def One_dof(train_data):
    one_dof_=[]
    for i in range(len(train_data)):
        valori=train_data[i]
        x0=valori[0]
        x1=valori[1]
        v=math.sqrt(pow(x0,2)+pow(x1,2))
        one_dof_.append(v)
        
    one_dof_=np.array(one_dof_)
    return one_dof_
        
    


def prepare_detection_dataset(dirpath,df,lookback, lookahead):
    X = []
    label= []

    
    go=df['Go']
    hold = df['Hold']
    rew = df['Rew']
    end=df['End']    
    obj = df['obj_id']
    
    hold_go=hold-go
    Rew_hold=rew-hold+1 
    End_rew=end-rew


    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)

    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        indice=int(df[df['img_id']==int(filenames[idx])].index.values)

        
        matrice=np.zeros([2,np_matrix.shape[1]])
        
        init=go[indice]
        arr_init=np.zeros(int(init))
        
        x=hold_go[indice]
        arr=np.arange(x)
        arr_sigm_x=np.asarray(norma_sig(arr))
        
        y=Rew_hold[indice]
        arr_sigm_y=np.ones(int(y))
        
        z=End_rew[indice]
        arr=-(np.arange(z))
        arr_sigm_z=np.asarray(norma_sig(arr))
        
        arr_conc=pd.Series(np.concatenate([arr_init,arr_sigm_x,arr_sigm_y,arr_sigm_z]))
        
        x0=(arr_conc*cos(angle[indice])).to_numpy().T
        x1=(arr_conc*sin(angle[indice])).to_numpy().T
        
        matrice[0,0:len(x0)]=x0
        matrice[1,0:len(x1)]=x1
        
        for i in range(np_matrix.shape[1]-lookback-lookahead):
            t = []
            for j in range(0, lookback):
                matrix=np_matrix[:, [(i+j)]]                
                t.append(matrix)
            X.append(t)
            label.append(matrice[:,i+lookback+lookahead])
        
    data_set = np.array(X)
    y_label = np.array(label) 
    
        
    
    
    return data_set, y_label
        
        
        

        

            

               
         
    
    
    


def norma_sig(a):
    d = 2.*(a - np.min(a))/np.ptp(a)-1
    d=d*5
    sig=[]
    for i in range(len(d)):
        sig.append(funzione(d[i]))
        
    return sig

def obj_mapping(id): #mappatura oggetti uguali
    if id == 11:
        id = 41
    if  id == 13:
        id = 34
    if id == 14:
        id = 44
    if id == 16:
        id = 25
    if id == 12:
        id = 54
        
    if id == 21 or id == 22 or id == 31 or id == 41 or id == 42 or id == 43:
       return 3 #classe gialla
    if id == 23 or id == 24 or id == 25 or id == 26:
       return 2 #classe azzurra
    if id == 32 or id == 33 or id == 34 or id == 35 or id == 36:
       return 7 #classe rosa
    if id == 44 or id == 45 or id == 46:
       return 6 #classe nera
    if id == 51 or id == 52 or id == 53 or id == 54 or id == 55 or id == 56:
       return 4 #classe blu
    if id == 15 or id == 61 or id == 62 or id == 63 or id == 64 or id == 65 or id == 66: 
       return 5 #classe rossa
    if id == 71 or id == 72 or id == 73 or id == 74 or id == 75 or id == 76:
       return 1 #classe verde
    else: 
        return id
    
def angle_mapping(obj,total_class):
    arr_angle=[]
    for i in range(len(obj)):
        angle=(2*pi/total_class)*(int(obj[i])-1)
        arr_angle.append(angle)
    return np.array(arr_angle)

    

def funzione(x):
    return 1 / (1 + math.exp(-x))
    

#sostituisce i valori di obj id nella sua corretta clusterizzazione
def prepare_binned(binned_samples_df):   
    file_ids = binned_samples_df['obj_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    
    for idx in range(len(filenames)):
        filenames[idx]=obj_mapping(int(filenames[idx]))  
    
    binned_samples_df['obj_id']=filenames
    
    return binned_samples_df
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/ZRec50_Mini_40_binned_spiketrains',
                        type=str)
    parser.add_argument("-o", "--outdir", help="Output data directory", default='./data/Continuous_control',
                        type=str)
    
    parser.add_argument("-k", "--name", help="Name of dataset from previous step, without extension", default='MRec40_40_binned_spiketrains',
                        type=str)
    
    


    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.outdir,args.name)):
        os.makedirs(os.path.join(args.outdir,args.name))
        
        
    data_prefix = os.path.basename(os.path.normpath(args.dataset))
    csv_file = os.path.join(args.dataset,data_prefix+'.csv')
    
    
    binned_samples_df = pd.read_csv(csv_file)
    go=binned_samples_df['Go']
    hold = binned_samples_df['Hold']
    rew = binned_samples_df['Rew']
    end=binned_samples_df['End']    
    obj = binned_samples_df['obj_id']
    
    
    
    
    binned_samples_df=prepare_binned(binned_samples_df)
    obj_max = max(binned_samples_df['obj_id'].to_numpy())
    angle=angle_mapping(obj,obj_max)
    
        
    train_df, test_df = train_test_split(binned_samples_df, test_size=0.2, random_state=42, stratify=binned_samples_df.obj_id) 
    lookahead=0
    lookback=12
    
    
    X_Train,y_train=prepare_detection_dataset(args.dataset,train_df,lookback, lookahead)
    one_dof_train=One_dof(y_train)
    X_Test,y_test=prepare_detection_dataset(args.dataset,test_df,lookback, lookahead)
    one_dof_test=One_dof(y_test)
    
    logging.info('\n')
    logging.info('Save dataset')
    with open(os.path.join(args.outdir,data_prefix,'trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_Train, y=y_train, z=one_dof_train)
    with open(os.path.join(args.outdir,data_prefix,'testset.npz'), 'bw') as testfile:
        np.savez(testfile, X=X_Test, y=y_test,z=one_dof_test)
    

        
        
        
        
        
        
        
        
    
    
    


   