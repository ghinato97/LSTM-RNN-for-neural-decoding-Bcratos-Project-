import tensorflow.keras as keras
import math 
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential,optimizers
from keras.layers import Dense,LSTM,Dropout
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import logging
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    # load existing data
    windows_dataset = np.load(filepath)
    X = windows_dataset['X']
    # X = X.reshape(-1, X[0].shape[0], X[0].shape[1], 1)
    # resize with 3 channels, but it is still a problem the input dimension
    # X = np.repeat(X, 3, axis=3)
    y = windows_dataset['y']
    return X, y




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
    
    
    data_prefix = os.path.basename(os.path.normpath(args.dataset))
    trainset_bin_path = args.dataset+'_trainset_bin.npz'
    testset_bin_path = args.dataset+'_testset_bin.npz'
    
    trainset_multi_path = args.dataset+'_trainset_multi.npz'
    testset_multi_path = args.dataset+'_testset_multi.npz'
    
    train_bin_set, label_bin_train = load_data(trainset_bin_path)
    test_bin_set, label_bin_test = load_data(testset_bin_path)
    
    train_multi_set, label_multi_train = load_data(trainset_multi_path)
    test_multi_set, label_multi_test = load_data(testset_multi_path)



    # Primo classificatore Grasp / no Grasp
    print('\n')
    keras.mixed_precision.set_global_policy('mixed_float16')
    model = Sequential()
    model.add(LSTM(units = 20, return_sequences = True, input_shape = (train_bin_set.shape[1], train_bin_set.shape[2])))
    model.add(LSTM(units = 20))
    model.add(Dense(label_bin_train.shape[1], activation = "softmax"))
    model.summary()
    
    model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = "accuracy") # binary perchè sono due classi
    model.fit(train_bin_set, label_bin_train, epochs = 20, batch_size = 128)
  

    predicted_bin_value = model.predict(test_bin_set)
    predicted_bin_value = predicted_bin_value.round()
    
    predicted_bin_value = predicted_bin_value.argmax(axis=1)
    label_bin_test = label_bin_test.argmax(axis=1)
    
    #confusion matrix
    
    cm1 = confusion_matrix(label_bin_test,predicted_bin_value)
        
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm1, annot=cm1, cmap='Blues', vmin=1, vmax=100)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('binary_classes_matrix.png')
    plt.close()
    print('Binary classes report')
    print(classification_report(label_bin_test, predicted_bin_value))
    
    
    # Secondo classificatore multiclasse
    print('\n')
    keras.mixed_precision.set_global_policy('mixed_float16')
    model = Sequential()
    model.add(LSTM(units = 10, return_sequences = True, input_shape = (train_multi_set.shape[1], train_multi_set.shape[2])))
    model.add(LSTM(units = 10))
    model.add(Dense(label_multi_train.shape[1], activation = "softmax"))
    model.summary()
    
    model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = "accuracy")  #categorical perchè sono n classi
    model.fit(train_multi_set, label_multi_train, epochs = 20, batch_size = 128)
  

    predicted_multi_value = model.predict(test_multi_set)
    # predicted_bin_value = predicted_bin_value.round()
    
    predicted_multi_value = predicted_multi_value.argmax(axis=1)
    label_multi_test = label_multi_test.argmax(axis=1)
    
    #confusion matrix
    
    cm2 = confusion_matrix(label_multi_test,predicted_multi_value)
    
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm2, annot=cm2, cmap='Blues', vmin=1, vmax=100)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('all_classes_matrix.png')
    plt.close()
    print('All classes report')
    print(classification_report(label_multi_test, predicted_multi_value))
