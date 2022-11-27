
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
import logging
import argparse
import os
import numpy as np
import pickle
import time

def load_data(filepath):
    # load existing data
    windows_dataset = np.load(filepath)
    X = windows_dataset['X']
    # X = X.reshape(-1, X[0].shape[0], X[0].shape[1], 1)
    # resize with 3 channels, but it is still a problem the input dimension
    # X = np.repeat(X, 3, axis=3)
    y = windows_dataset['y']
    return X, y


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
    
    
    trainset_path = os.path.join(args.dataset, 'multi_trainset.npz')
    testset_path = os.path.join(args.dataset, 'multi_testset.npz')
    
    train_set, label_train = load_data(trainset_path)
    test_set, label_test = load_data(testset_path)

    train_set=train_set.reshape(train_set.shape[0],train_set.shape[1],train_set.shape[2])
    train_set=train_set.astype(float)
    test_set=test_set.reshape(test_set.shape[0],test_set.shape[1],test_set.shape[2])
    test_set=test_set.astype(float)
    
    canali=train_set.shape[2]
    

    
    
    
    build_model(keras_tuner.HyperParameters())
    
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=2,
        overwrite=True,
        directory="optimization",
        project_name="BRNN_optimization_bayesian",
    )
    
    tuner.search_space_summary()
    tuner.search(train_set, label_train, epochs=10, validation_data=(test_set, label_test))
    
    best_hps = tuner.get_best_hyperparameters()[0]   
    
    with open("bayesian_best_hps.pkl", "wb") as f:
       pickle.dump(best_hps, f)
    
    model = tuner.hypermodel.build(best_hps)
    model.build(input_shape=(None,12,369))
    model.summary()

    
    
    
    
    
    


