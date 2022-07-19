import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import math 
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential,optimizers
from keras.layers import Dense,LSTM,Dropout,TimeDistributed,Bidirectional
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import logging
import argparse

def load_data(filepath):
    # load existing data
    windows_dataset = np.load(filepath)
    X = windows_dataset['X']
    # X = X.reshape(-1, X[0].shape[0], X[0].shape[1], 1)
    # resize with 3 channels, but it is still a problem the input dimension
    # X = np.repeat(X, 3, axis=3)
    y = windows_dataset['y']
    return X, y

def lr_scheduler(epoch, lr):
  if epoch < 50:
    return lr
  else:
    return lr/2.

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/MRec40_40_binned_spiketrains/lookback_12_lookahead_0',
                        type=str)

    args = parser.parse_args()
    data_prefix = '_'.join(os.path.normpath(args.dataset).split(os.sep)[-2:])

    logging.info('\n')
    logging.info('----------------')
    logging.info('Building windows')
    logging.info(data_prefix)
    logging.info('----------------')
    logging.info('\n')
    logging.info('Loading data...')
    
    
    trainset_bin_path = os.path.join(args.dataset, 'binary_trainset.npz')
    testset_bin_path = os.path.join(args.dataset, 'binary_testset.npz')
    
    train_bin_set, label_bin_train = load_data(trainset_bin_path)
    test_bin_set, label_bin_test = load_data(testset_bin_path)


    # Primo classificatore Grasp / no Grasp
    print('\n')
    model = Sequential()
    model.add(Bidirectional(LSTM(40, return_sequences = True, dropout=0.6), input_shape = (train_bin_set.shape[1], train_bin_set.shape[2])))
    model.add(Bidirectional(LSTM(units = 40, return_sequences = True, dropout=0.6)))
    model.add(Bidirectional(LSTM(units = 40, return_sequences = False, dropout=0.6)))
    model.add(Dense(label_bin_train.shape[1], activation = "softmax"))
    model.summary()
    
    opt = keras.optimizers.Adam(learning_rate=0.0002)    #     callback = []
    #     callback = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    
    
    callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    
    model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = "accuracy") # binary perchè sono due classi
    # model.fit(train_bin_set, label_bin_train, epochs = 50, batch_size = 128, validation_data=(test_bin_set, label_bin_test))
    model.fit(train_bin_set, label_bin_train, epochs = 50, batch_size = 128, validation_data=(test_bin_set, label_bin_test), callbacks=[callback])
  
    model.save(data_prefix+'_binary_model')

    predicted_bin_value = model.predict(test_bin_set)
    predicted_bin_value = predicted_bin_value.argmax(axis=1)
    label_bin_test = label_bin_test.argmax(axis=1)
    
    #confusion matrix
    
    cm1 = confusion_matrix(label_bin_test,predicted_bin_value)
        
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm1, annot=cm1, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('binary_classes_matrix_BRNN.png')
    plt.close()
    print('Binary classes report')
    print(classification_report(label_bin_test, predicted_bin_value))