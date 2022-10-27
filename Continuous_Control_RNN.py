import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import math 
import os
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential,optimizers
from keras.layers import Dense,LSTM,Dropout,TimeDistributed,Bidirectional
from sklearn.metrics import confusion_matrix, classification_report,mean_squared_error

def cartesian_plot(x1,x2,y1,y2):
    # Enter x and y coordinates of points and colors
    xs = x1
    ys = x2
    xxs=y1
    yys=y2
    
    colors = ['m', 'g', 'r', 'b']

    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -1, 1, -1, 1
    ticks_frequency = 1

    # Plot points
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(xs, ys)
    ax.scatter(xxs, yys)


    # Set identical scales for both axes
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])



    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)



    plt.savefig(args.dataset+'cartesian_X1.png')
    
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
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/Continuous_control/ZRec50_Mini_40_binned_spiketrains',
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
    
    
    trainset_bin_path = os.path.join(args.dataset, 'trainset.npz')
    testset_bin_path = os.path.join(args.dataset, 'testset.npz')
    
    train_set, label_train = load_data(trainset_bin_path)
    test_set, label_test = load_data(testset_bin_path)


    print('\n')
    model = Sequential()
    model.add(Bidirectional(LSTM(40, return_sequences = True, dropout=0.6), input_shape = (train_set.shape[1], train_set.shape[2])))
    model.add(Bidirectional(LSTM(units = 40, return_sequences = True, dropout=0.6)))
    model.add(Bidirectional(LSTM(units = 40, return_sequences = False, dropout=0.6)))
    model.add(Dense(2))
    model.summary()
    
    opt = keras.optimizers.Adam(learning_rate=0.0002)    #     callback = []
    #     callback = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    
    
    callback = keras.callbacks.EarlyStopping(monitor="val_mean_squared_error", patience=10, restore_best_weights=True)
    
    model.compile(optimizer = opt, loss = "mean_squared_error",metrics=['MeanSquaredError', 'MeanAbsoluteError','RootMeanSquaredError']) # binary perchè sono due classi
    # model.fit(train_bin_set, label_bin_train, epochs = 50, batch_size = 128, validation_data=(test_bin_set, label_bin_test))
    model.fit(train_set, label_train, epochs = 50, batch_size = 128, validation_data=(test_set, label_test), callbacks=[callback])
  
    model.save(data_prefix+'model')

    predicted_value = model.predict(test_set)
    
    print('---------- Evaluation on Test Data ----------')
    print("MSE: ", mean_squared_error(label_test, predicted_value))
    print("")
    

    fig, ax = plt.subplots(figsize=(16,14))
    plt.plot(label_test[:,0],label='True Values')  # green dots
    plt.plot(predicted_value[:,0],label='Predicted_Values')  # blue stars
    plt.title(' X0 Prediction Plots')  
    plt.xlabel('Time_Stamp')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.savefig(args.dataset+'Xo.png')
    plt.close()
    
    
    fig, ax = plt.subplots(figsize=(16,14))
    plt.plot(label_test[:,1], label='True Values')  # green dots
    plt.plot(predicted_value[:,1], label='Predicted_Values')  # blue stars
    plt.title(' X1 Prediction Plots')  
    plt.xlabel('Time_Stamp')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.savefig(args.dataset+'X1.png')
    plt.close()
    
    cartesian_plot(label_test[1:150,0],label_test[1:150,1],predicted_value[1:150,0],predicted_value[1:150,1])