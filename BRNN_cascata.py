import logging
import argparse
import os
import seaborn as sns
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
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


def load_data(filepath):
    # load existing data
    windows_dataset = np.load(filepath)
    X = windows_dataset['X']
    # X = X.reshape(-1, X[0].shape[0], X[0].shape[1], 1)
    # resize with 3 channels, but it is still a problem the input dimension
    # X = np.repeat(X, 3, axis=3)
    y = windows_dataset['y']
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Name of BRNN model", default='ZRec50_Mini_40_binned_spiketrains_lookback_12_lookahead_0_',
                        type=str)
    parser.add_argument("--lookback", help="length of the sequence to be classified", default=12,
                        type=int)
    parser.add_argument("--lookahead", help="length of the sequence to be classified", default=0,
                        type=int)
    parser.add_argument("-d", "--dataset",help="Name of dataset from previous step, without extension", 
                        default='data/ZRec50_Mini_40_binned_spiketrains/lookback_12_lookahead_0',
                        type=str)
    
    args = parser.parse_args()
    
    data_prefix = os.path.basename(os.path.normpath(args.model))
    path_bin=data_prefix+'binary_model'
    path_multi=data_prefix+'multi_model'
    
    
    logging.info('Caricamento dataset')
    testset_path = os.path.join(args.dataset, 'full_testset.npz')
    test_set,label_test=load_data(testset_path)
    test_set=np.reshape(test_set,(test_set.shape[0],test_set.shape[1],test_set.shape[2]))
    
    objID_path = os.path.join(args.dataset, 'full_testset_objid.npz')
    objID_set,label_objID=load_data(objID_path)
    
    logging.info('\n')
    logging.info('----------------')
    logging.info('Loading Binary Models')
    binary_model=keras.models.load_model(path_bin)
    logging.info('Binary Model Info')
    binary_model.summary()    
       
    logging.info('\n')
    logging.info('----------------')
    logging.info('Loading Multi Models')
    multi_model=keras.models.load_model(path_multi)
    logging.info('Multi Model Info')
    multi_model.summary()    
    
    list_pred_bin=[]
    list_pred_multi=[]
    for i in range(test_set.shape[0]):
        xx=test_set[i,:,:]
        xx=np.reshape(xx,(1,xx.shape[0],xx.shape[1]))
        bin_pred=binary_model.predict(xx)
        bin_pred=int(bin_pred.argmax(axis=1))
        list_pred_bin.append(bin_pred)
        if bin_pred==1 and label_objID[i]!=-1:
            multi_pred=multi_model.predict(xx)
            multi_pred=int(multi_pred.argmax(axis=1))
            list_pred_multi.append(multi_pred)
        else:
            list_pred_multi.append(-1)
    
    #confusion matrix
    
    cm2 = confusion_matrix(label_objID,list_pred_multi)
    
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm2, annot=cm2, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.savefig('all_classes_matrix_BRNN.png')
    # plt.close()
    print('All classes report')
    print(classification_report(label_objID,list_pred_multi))
            


