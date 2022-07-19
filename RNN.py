import math 
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential,optimizers
from keras.layers import Dense,LSTM,Dropout
from sklearn.metrics import confusion_matrix, classification_report
import os
import argparse
import logging
import numpy as np


def load_data(filepath):
    # load existing data
    windows_dataset = np.load(filepath)
    X = windows_dataset['X']
    y = windows_dataset['y']
    return X, y




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/ZRec50_Mini_40_binned_spiketrains/lookback_12_lookahead_0',
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
    
    # binned_samples_df=pd.read_csv(csv_file)
    # hold=binned_samples_df['Hold']
    # rew=binned_samples_df['Rew']
    # img=binned_samples_df['img_id']
    
    # train_df,test_df=train_test_split(binned_samples_df, test_size=0.2, random_state=42, stratify=binned_samples_df.obj_id)
    # train_set,label_train=Prepare_Dataset(args.dataset,train_df)
    # test_set,label_test=Prepare_Dataset(args.dataset,test_df)

    data_prefix = os.path.basename(os.path.normpath(args.dataset))
    trainset_bin_path = args.dataset+'/binary_trainset.npz'
    testset_bin_path = args.dataset+'/binary_testset.npz'
    testlist_bin_path = args.dataset+'/testlist.pkl'
    
    train_bin_set, label_train_bin = load_data(trainset_bin_path)
    test_bin_set, label_test_bin = load_data(testset_bin_path)
    


    print('\n')
    model = Sequential()
    model.add(LSTM(units=50, return_sequences= True, input_shape=(train_bin_set.shape[1],train_bin_set.shape[2])))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    # model.add(Dropout(.5))
    model.add(Dense(2, activation="softmax"))
    model.summary()
    
    # optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer='Adam', loss="binary_crossentropy",metrics='accuracy')
    
    
    # model.fit(train_bin_set,label_train_bin, epochs=100, batch_size=16)

    # predicted_value= model.predict(test_bin_set)
    # predicted_value=predicted_value.round()
    
    # predicted_value=predicted_value.argmax(axis=1)
    # label_test_bin=label_test_bin.argmax(axis=1)
    
    # #confusion matrix
    
    # cm = confusion_matrix(label_test_bin,predicted_value)
    # print(cm)
    
    
    
    
