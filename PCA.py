from sklearn.decomposition import PCA
from neo.io.neomatlabio import NeoMatlabIO
import argparse
import logging
import os
import pandas as pd
import numpy as np


def P_C_A(X_train):
    pca=PCA(n_components=0.95)
    X_reduced=pca.fit_transform(X_train)
    return X_reduced
    
    

def load_data(dirpath,df):
    global X_bin,np_matrix
    X_bin = []

    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    hold = df['Hold'].to_numpy(dtype=int)
    rew = df['Rew'].to_numpy(dtype=int)

    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        X_bin.append(np_matrix)

    
    return X_bin

if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/ZRec50_Mini_40_binned_spiketrains',
                        type=str)
    args = parser.parse_args()
    
    logging.info('\n')
    logging.info('----------------')
    logging.info('Loading Dataset')
    
    dataset_path = os.path.join(args.dataset)
    x=dataset_path.split('/')
    csv_name=x[1]+'.csv'
    csv_file = os.path.join(args.dataset,csv_name)
    binned_samples_df=pd.read_csv(csv_file)
    
    data_set=load_data(dataset_path,binned_samples_df)
    PCA_data_complete=[]
    size_complete=[]
    for i in range(len(X_bin)):
        PCA_data_set=P_C_A(data_set[i].T)
        size=PCA_data_set.shape[0]
        PCA_data_complete.append(PCA_data_set)        
        size_complete.append(size)
    minimo=min(size_complete)
    print(minimo)

    