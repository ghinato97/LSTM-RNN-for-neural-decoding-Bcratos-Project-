# Prima classificatore binario : Grasp / no Grasp , poi classificatore multiclasse solo con Grasp
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import matplotlib.pyplot as plt
import pickle
import argparse
import logging
import pathlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler("window_split_data.log", mode='w'),
        logging.StreamHandler()
    ]
)


def obj_mapping(id): #mappatura oggetti uguali
     if id == 11:
         return 41
     if id == 13:
         return 34
     if id == 14:
         return 44
     if id == 16:
         return 25
     if id == 12:
        return 54    
     return id
 


def prepare_Binary_Dataset(dirpath,df): #dataset per classificazione binaria
    lookback = 8 
    x = []
    y = []
    
    for idx, row in df.iterrows():
        s = str(int(row['img_id']))
        obj = int(row['obj_id'])
        stringa = 'img_'+s
        print(stringa)
        filepath = os.path.join(dirpath, stringa)
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']

        for i in range(np_matrix.shape[1]-lookback-1):
            t = []
            for j in range(0, lookback):
                t.append(np_matrix[:, [(i+j)]])
            if (i+lookback) >= row['Hold'] and (i+lookback) < row['Rew']:  # attivazione
                y.append(1)
            else: 
                y.append(0)
                
            x.append(t)
            
            

        data_set = np.array(x)
        y_label = np.array(y)    
        
        label_encoder = LabelEncoder()
        label_encoder.fit(y_label)
        integer_encoded = label_encoder.transform(y_label).reshape(len(y_label), 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit(integer_encoded)
        label = onehot_encoder.transform(integer_encoded)
     
    return data_set, label
            
def prepare_Multiclass_Dataset(dirpath, df): #dataset per attivazione multiclasse si considerano solo le attivazioni
    lookback = 8 
    x = []
    y = []
    
    for idx, row in df.iterrows():
        s = str(int(row['img_id']))
        obj = int(row['obj_id'])
        stringa = 'img_'+s
        filepath = os.path.join(dirpath, stringa)
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        
        for i in range(np_matrix.shape[1]-lookback-1):
            t = []
            if (i+lookback) >= (row['Hold']) and (i+lookback) < (row['Rew']):
                for j in range(0, lookback):
                    t.append(np_matrix[:, [(i+j)]])
                y.append(obj_mapping(obj))
            
                x.append(t)
                
                
        data_set = np.array(x)
        y_label = np.array(y)    
        
        label_encoder = LabelEncoder()
        label_encoder.fit(y_label)
        integer_encoded = label_encoder.transform(y_label).reshape(len(y_label), 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit(integer_encoded)
        label = onehot_encoder.transform(integer_encoded)
     
    return data_set, label





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/ZRec50_Mini_40_binned_spiketrains',
                        type=str)

    parser.add_argument("--ref-start-event", help="Reference start event for data window definition and grasp labelling", default='Hold',
                        type=str)
    parser.add_argument("--ref-end-event", help="Reference end event for data window definition and grasp labelling", default='Rew',
                        type=str)

    args = parser.parse_args()
    data_prefix = os.path.basename(os.path.normpath(args.dataset))
    csv_file = args.dataset+'.csv'

    logging.info('\n')
    logging.info('----------------')
    logging.info('Building windows')
    logging.info(args.dataset)
    logging.info('----------------')
    logging.info('\n')
    logging.info('Loading data...')

    binned_samples_df = pd.read_csv(csv_file)
    hold = binned_samples_df['Hold']
    rew = binned_samples_df['Rew']
    img = binned_samples_df['img_id']
    obj = binned_samples_df['obj_id']

    train_df, test_df = train_test_split(binned_samples_df, test_size=0.2, random_state=42, stratify=binned_samples_df.obj_id)    
    
    train_bin_set, label_bin_train = prepare_Binary_Dataset(args.dataset, train_df)
    test_bin_set, label_bin_test = prepare_Binary_Dataset(args.dataset, test_df)
    
    train_multi_set, label_multi_train = prepare_Multiclass_Dataset(args.dataset, train_df)
    test_multi_set, label_multi_test = prepare_Multiclass_Dataset(args.dataset, test_df)
    
    
    logging.info('\n')
    logging.info('Save binary dataset')
    with open(os.path.join('data',data_prefix+'_trainset_bin.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=train_bin_set, y=label_bin_train)
    with open(os.path.join('data',data_prefix+'_testset_bin.npz'), 'bw') as testfile:
        np.savez(testfile, X=test_bin_set, y=label_bin_test)
            
        
    logging.info('\n')
    logging.info('Save multiclass dataset')
    with open(os.path.join('data',data_prefix+'_trainset_multi.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=train_multi_set, y=label_multi_train)
    with open(os.path.join('data',data_prefix+'_testset_multi.npz'), 'bw') as testfile:
        np.savez(testfile, X=test_multi_set, y=label_multi_test)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        