from sklearn.decomposition import PCA
from neo.io.neomatlabio import NeoMatlabIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import argparse
import logging
import os
import pandas as pd
import numpy as np

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


def P_C_A(X_train):
    pca=PCA(n_components=150)
    X_reduced=pca.fit_transform(X_train)
    return X_reduced,pca
    
def dataset_PCA(dirpath,df):
    X = []
    size_matrix_bin=[]
    size_matrix_multi=[]

    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        size_matrix_bin.append(np_matrix.shape[1])
        size_matrix_multi.append(np_matrix.shape[1])
        
        X.append(np_matrix.T)

    X_=np.vstack(X)   
    return X_,size_matrix_bin,size_matrix_multi

def Prepare_dataset_bin(PCA_Dataset,size_matrix,df):
    X=PCA_Dataset
    X_arr=[]
    tt=[]
    y=[]
    y_bin=[]
    lookback=12
    lookahead=0
    
    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    hold = df['Hold'].to_numpy(dtype=int)
    rew = df['Rew'].to_numpy(dtype=int)
    
    for idx in range(len(size_matrix)):
        n=size_matrix.pop(0)
        n_col=np.arange(n)
        np_matrix=X[n_col,:].T
        X=np.delete(X,n_col,axis=0)
        tt.append(np_matrix)
        for i in range(np_matrix.shape[1]-lookback-lookahead):
            t = []
            for j in range(0, lookback):
                t.append(np_matrix[:, [(i+j)]])
            X_arr.append(t)
            if (i+lookback+lookahead) >= hold[idx] and (i+lookback+lookahead) < rew[idx]:  # attivazione
                y_bin.append(1)
                y.append(obj_mapping(obj_ids[idx]))
            else: 
                y_bin.append(0)
                y.append(-1)
    data_set= np.array(X_arr)
    y_label_bin = np.array(y_bin)
    y_label = np.array(y) 
    
    label_encoder_bin = LabelEncoder()
    label_encoder_bin.fit(y_label_bin)
    integer_encoded_bin = label_encoder_bin.transform(y_label_bin).reshape(len(y_label_bin), 1)
    onehot_encoder_bin = OneHotEncoder(sparse=False)
    onehot_encoder_bin.fit(integer_encoded_bin)
    label_bin = onehot_encoder_bin.transform(integer_encoded_bin)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(y_label)
    integer_encoded = label_encoder.transform(y_label).reshape(len(y_label), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    label = onehot_encoder.transform(integer_encoded)
        
    return data_set,label,label_bin
    
def Prepare_dataset_multi(PCA_Dataset,size_matrix,df):
    X=PCA_Dataset
    X_arr=[]
    tt=[]
    y=[]
    y_bin=[]
    lookback=12
    lookahead=0
    
    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    hold = df['Hold'].to_numpy(dtype=int)
    rew = df['Rew'].to_numpy(dtype=int)
    
    for idx in range(len(size_matrix)):
        n=size_matrix.pop(0)
        n_col=np.arange(n)
        np_matrix=X[n_col,:].T
        X=np.delete(X,n_col,axis=0)
        tt.append(np_matrix)
        for i in range(np_matrix.shape[1]-lookback-lookahead):
            t = []
            for j in range(0, lookback):
                t.append(np_matrix[:, [(i+j)]])
            if (i+lookback+lookahead) >= hold[idx] and (i+lookback+lookahead) < rew[idx]:  # attivazione
                X_arr.append(t)
                y_bin.append(1)
                y.append(obj_mapping(obj_ids[idx]))

    data_set= np.array(X_arr)
    y_label = np.array(y) 
    
    
    # label_encoder = LabelEncoder()
    # label_encoder.fit(y_label)
    # integer_encoded = label_encoder.transform(y_label).reshape(len(y_label), 1)
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoder.fit(integer_encoded)
    # label = onehot_encoder.transform(integer_encoded)
        
    return data_set,y_label        


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help=".mat file containing brain recordings", default='data/ZRec50_Mini_40_binned_spiketrains',
                        type=str)
    
    parser.add_argument("-o", "--outdir", help="Output data directory", default='./data',
                        type=str)
    
    args = parser.parse_args()
    
    logging.info('\n')
    logging.info('Loading data...')
    
    data_prefix = os.path.basename(os.path.normpath(args.dataset))
    csv_path=os.path.join(args.dataset,data_prefix+'.csv')
    df = pd.read_csv(csv_path)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.obj_id)
    
    X_Train,size_matrix_bin_Train,size_matrix_multi_Train=dataset_PCA(args.dataset,train_df)
    X_Test,size_matrix_bin_Test,size_matrix_multi_Test=dataset_PCA(args.dataset,test_df)
    
    
    logging.info('\n')
    logging.info('Creating PCA dataset...')    
    Data_PCA_Train,pca =P_C_A(X_Train)
    Data_PCA_Test=pca.transform(X_Test)

    
    logging.info('\n')
    logging.info('Save PCA dataset')
    if not os.path.exists(os.path.join(args.outdir,data_prefix,'PCA_Dataset')):
        os.makedirs(os.path.join(args.outdir,data_prefix,'PCA_Dataset'))
        
    
    logging.info('\n')
    logging.info('Processing Bin PCA dataset ..')        
    data_set_train_bin,label_train,label_train_bin=Prepare_dataset_bin(Data_PCA_Train, size_matrix_bin_Train,train_df)
    data_set_test_bin,label_test,label_test_bin=Prepare_dataset_bin(Data_PCA_Test, size_matrix_bin_Test,test_df)
    
    with open(os.path.join(args.outdir,data_prefix,'PCA_Dataset','binary_trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=data_set_train_bin, y=label_train_bin)
    with open(os.path.join(args.outdir,data_prefix,'PCA_Dataset','binary_testset.npz'), 'bw') as testfile:
            np.savez(testfile, X=data_set_test_bin ,y=label_test_bin)
            
            
    
    logging.info('\n')
    logging.info('Processing Multi PCA dataset ..')        
    data_set_train_multi,label_train_multi=Prepare_dataset_multi(Data_PCA_Train, size_matrix_multi_Train,train_df)
    data_set_test_multi,label_test_multi=Prepare_dataset_multi(Data_PCA_Test, size_matrix_multi_Test,test_df)
    
    with open(os.path.join(args.outdir,data_prefix,'PCA_Dataset','multi_trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=data_set_train_multi, y=label_train_multi)
    with open(os.path.join(args.outdir,data_prefix,'PCA_Dataset','multi_testset.npz'), 'bw') as testfile:
            np.savez(testfile, X=data_set_test_multi ,y=label_test_multi)
    