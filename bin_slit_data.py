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


# def obj_mapping(id):
#     if id == 11:
#         return 41
#     if id == 13:
#         return 34
#     if id == 14:
#         return 44
#     if id == 16:
#         return 25
#     return id

def Prepare_Dataset(dirpath,img_id,hold,rew):
    global np_matrix,t
    lookback=8
    x=[]
    y=[]
    for im in img_id:
        s=str(int(im))
        stringa='img_'+s
        # print(stringa)
        filepath=os.path.join(dirpath,stringa)
        binned_spk=np.load(filepath)
        np_matrix = binned_spk['arr_0']
        
        for i in range(np_matrix.shape[1]-lookback-1):
            t=[]
            for j in range(0,lookback):
                n=np_matrix[:,[(i+j)]]
                t.append(n)
            if (i+lookback)>=hold[im] and (i+lookback)<rew[im]:
                 y.append(1)  # 1 è quando c'è Hold 
            else:
                 y.append(0)  #0 è quando non c'è Hold
                  
            x.append(t)
    y=np.array(y)
    data_set=np.array(x) # dimensionalmente [bin_in_considerazione,bin_precedenti,somma_sui_canali]
    data_set=np.reshape(data_set,(data_set.shape[0],data_set.shape[1],data_set.shape[2]))
    
    return data_set,y


def somma_canali(x):
    somma=[]
    for i in range(len(x)):
        k=x[i]
        p=[]
        for j in range(len(k)):
            kk=sum(k[j])
            p.append(kk)
        somma.append(p)
    return somma
            




def prepare_mask_dataset(binned_samples_df):
    df_mask_train, df_mask_test = train_test_split(binned_samples_df, test_size=0.2, random_state=42, stratify=binned_samples_df.obj_id)
    df_mask_train=df_mask_train['img_id']
    df_mask_test=df_mask_test['img_id']
    return df_mask_train, df_mask_test

    
  
    
if __name__=="__main__":
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
    
    binned_samples_df=pd.read_csv(csv_file)
    hold=binned_samples_df['Hold']
    rew=binned_samples_df['Rew']
    img=binned_samples_df['img_id']
    
    mask_train,mask_test=prepare_mask_dataset(binned_samples_df)
    train_set,label_train=Prepare_Dataset(args.dataset,mask_train,hold,rew)
    test_set,label_test=Prepare_Dataset(args.dataset,mask_test,hold,rew)

