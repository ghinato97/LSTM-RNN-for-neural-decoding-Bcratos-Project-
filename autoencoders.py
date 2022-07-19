
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import os
import pandas as pd
import tensorflow.keras as keras
from keras import Sequential,optimizers
from keras.layers import Dense,Flatten,Reshape,RepeatVector,TimeDistributed,LSTM


# def load_data(filepath):
#     # load existing data
#     windows_dataset = np.load(filepath)
#     return windows_dataset


def prepare_dataset(dirpath,df):
    X = []
    y=[]
    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))

    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        X.append(np_matrix)


    for i in range(len(X)):
        x=X[i].shape[1]
        y.append(x)
    minimo=min(y)
    
    for i in range(len(X)):
        xx=X[i]
        xx=xx[:,0:minimo]
        X[i]=xx
    X_=np.array(X)  
      
    return X_



class Simple:
    def __init__(self,train_set):       
        opt=keras.optimizers.SGD(learning_rate=1)
        
        shape_tot=train_set.shape[1]*train_set.shape[2]
        

        model_enco=Sequential()
        model_enco.add(Flatten(input_shape=[train_set.shape[1],train_set.shape[2]]))
        model_enco.add(Dense(200,activation="relu"))
        model_enco.add(Dense(85,activation="relu"))
        model_enco.summary()
        
        model_deco=Sequential()
        model_deco.add(Dense(200,activation="selu", input_shape=[85]))
        model_deco.add(Dense(shape_tot,activation="sigmoid"))
        model_deco.add(Reshape([train_set.shape[1], train_set.shape[2]]))   
        model_deco.summary()
        
        autoencoders=Sequential([model_enco,model_deco])
        autoencoders.summary()
        autoencoders.compile(loss="binary_crossentropy",optimizer=opt)
        history = autoencoders.fit(train_set, train_set, epochs=12)
        


class Recu_Auto:
    def __init__(self,train_set):
        # trainset_path = os.path.join(args.dataset, 'full_trainset.npz')
        # testset_path = os.path.join(args.dataset, 'full_testset.npz')
        # train_set=load_data(trainset_path)
        # test_set=load_data(testset_path)
        
        
        shape_tot=train_set.shape[1]*train_set.shape[2]
        
        model_enco=Sequential()
        model_enco.add(LSTM(units=150,return_sequences=True,input_shape=(train_set.shape[1],train_set.shape[2])))
        model_enco.add(LSTM(units=85))
        model_enco.summary()
        
        
        model_deco=Sequential()
        model_deco.add(RepeatVector(train_set.shape[1],input_shape=[85]))
        model_deco.add(LSTM(150,return_sequences=True))
        model_deco.add(TimeDistributed(Dense(train_set.shape[2],activation="sigmoid")))
        model_deco.summary()
        
        
        autoencoders=Sequential([model_enco,model_deco])
        autoencoders.summary()
        
        autoencoders.compile(optimizer='Adam', loss="binary_crossentropy",metrics='accuracy') 
        autoencoders.fit(train_set,train_set, epochs=100, batch_size=16)
        
        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help=".mat file containing brain recordings", default='data/ZRec50_Mini_40_binned_spiketrains',
                        type=str)

    args = parser.parse_args()

    logging.info('\n')
    logging.info('Loading data...')
    
    data_prefix = os.path.basename(os.path.normpath(args.dataset))
    csv_path=os.path.join(args.dataset,data_prefix+'.csv')
    df = pd.read_csv(csv_path)
    X=prepare_dataset(args.dataset,df)
    Recu_Auto(X)

    

    