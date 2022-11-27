import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import math 
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential,optimizers
from keras.layers import Dense,LSTM,Dropout,TimeDistributed,Bidirectional
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers
import os
import numpy as np
import logging
import argparse
import pickle
import keras_tuner

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

def build_model(hp):
    model = keras.Sequential()
    # Tune the number of layers.
    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(
            layers.Bidirectional(layers.LSTM(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=5, max_value=100, step=5),
                return_sequences=True,
                dropout=hp.Float('dropout', min_value=0.0, max_value=0.6, step=0.2),
                input_shape=(12,369)
            )
        ))
    model.add(layers.Bidirectional(layers.LSTM(
        units=hp.Int("units", min_value=5, max_value=100, step=5),
        return_sequences=False, 
        dropout=hp.Float('Dropout', min_value=0.0, max_value=0.6, step=0.2)
        )))
    model.add(layers.Dense(2, activation="softmax"))
    opt = keras.optimizers.Adam(learning_rate=hp.Choice("Learning_Rate", [1e-2,1e-3,1e-4]),)
    model.compile(
        optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"],
    )
    return model


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/ZRec50_Mini_40_binned_spiketrains/lookback_12_lookahead_0',
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
    
    
    trainset_multi_path = os.path.join(args.dataset, 'multi_trainset.npz')
    testset_multi_path = os.path.join(args.dataset, 'multi_testset.npz')
    
    train_multi_set, label_multi_train = load_data(trainset_multi_path)
    test_multi_set, label_multi_test = load_data(testset_multi_path)
    
    
    # Secondo classificatore multiclasse
    print('\n')
    build_model(keras_tuner.HyperParameters())
    
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=2,
        overwrite=True,
        directory="optimization",
        project_name="BRNN_optimization",
    )
    
    best_hps = pickle.load(open("best_hps.pkl","rb"))
    print(best_hps)

    model = tuner.hypermodel.build(best_hps)
    model.build(input_shape=(None,12,369))
    model.summary()
    
    diz=best_hps.values
    lr=diz['Learning_Rate']
    opt = keras.optimizers.Adam(learning_rate=lr)
    #     callback = []
    #     callback = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True)
    
    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])  #categorical perchè sono n classi
    # model.fit(train_multi_set, label_multi_train, epochs = 100, batch_size = 128, validation_data=(test_multi_set, label_multi_test))
    model.fit(train_multi_set, label_multi_train, epochs = 100, batch_size = 128, validation_data=(test_multi_set, label_multi_test), callbacks=[callback])
  
    model.save(data_prefix+'_multi_model')
    
    predicted_multi_value = model.predict(test_multi_set)
    predicted_multi_value = predicted_multi_value.argmax(axis=1)
    label_multi_test = label_multi_test.argmax(axis=1)
    
    #confusion matrix
    
    cm2 = confusion_matrix(label_multi_test,predicted_multi_value)
    
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm2, annot=cm2, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('all_classes_matrix_BRNN_PCA.png')
    plt.close()
    print('All classes report')
    print(classification_report(label_multi_test, predicted_multi_value))
