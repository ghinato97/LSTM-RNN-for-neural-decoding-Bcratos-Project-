import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler("cnn_training.log", mode='w'),
        logging.StreamHandler()
    ]
    )


def load_data(filepath):
    # load existing data
    windows_dataset = np.load(filepath)
    X = windows_dataset['X']
    X = X.reshape(-1, X[0].shape[0], X[0].shape[1], 1)
    # resize with 3 channels, but it is still a problem the input dimension
    # X = np.repeat(X, 3, axis=3)
    y = windows_dataset['y']
    return X, y

def build_model(input_shape, num_classes):
    model = keras.Sequential(
        [   
            keras.Input(shape=input_shape),           
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax")
        ])
    # model = keras.models.Sequential([
    #         keras.layers.Input(shape=input_shape),
    #         keras.layers.Conv2D(64, 
    #                             kernel_size=(3, 3),
    #                             padding="same",
    #                             activation="relu",
    #                             name="conv1"),
    #         keras.layers.MaxPooling2D(padding='same', 
    #                                      name="pool1"),
    #         keras.layers.Conv2D(32, 
    #                             kernel_size=(3, 3),
    #                             padding="same",
    #                             activation="relu",
    #                             name="conv2"),
    #         keras.layers.MaxPooling2D(padding='same', 
    #                                      name="pool2"),
    #         keras.layers.Flatten(),
    #         keras.layers.Dense(512, activation='relu', name="fc1"),
    #         keras.layers.Dropout(0.4),
    #         keras.layers.Dense(num_classes, activation='softmax', name="predictions")
    #     ])

    return model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="File from previous pre-processing, without extension", default='data\ZRec50_Mini_40_binned_spiketrains_12_Hold_Rew_-5_5_sorted',
                        type=str)
                        
    args = parser.parse_args()

    logging.info('\n')
    logging.info('----------------')
    logging.info('Training convnet on DPZ data with following params')
    logging.info(args)
    logging.info('On Intel, run Tensorflow>=2.5 with export TF_ENABLE_ONEDNN_OPTS=1')
    logging.info('On Nvidia, check that keras.mixed_precision.set_global_policy\(\'mixed_float16\'\) is added to the script')
    logging.info('----------------')
    logging.info('\n')
    logging.info('Loading data...')

    keras.mixed_precision.set_global_policy('mixed_float16')

    data_prefix = os.path.basename(os.path.normpath(args.dataset))
    trainset_path = args.dataset+'_trainset.npz'
    testset_path = args.dataset+'_testset.npz'
    testlist_path = args.dataset+'_testlist.pkl'

    X_train, y_train = load_data(trainset_path)
    X_test, y_test = load_data(testset_path)

    with open (testlist_path, 'rb') as fp:
        windows_position_test = pickle.load(fp)
    num_classes = y_train.shape[1]
    input_shape = X_train[0].shape    
    CNN_model = build_model(input_shape, num_classes)
    # CNN_model = ResNet50(input_shape=input_shape, classes=num_classes, include_top=False)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    CNN_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy', 'top_k_categorical_accuracy'])

    # Train model on dataset
    history = CNN_model.fit(X_train, y_train, epochs=20, batch_size=256, validation_data=(X_test, y_test), shuffle=True)
    score = CNN_model.evaluate(X_test, y_test, verbose=1)

    CNN_model.save(data_prefix+'_model')

    # Plot training curves
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('curve.png')

    y_pred = CNN_model.predict(X_test)
    classes_pred = y_pred.argmax(axis=1)
    classes_test = y_test.argmax(axis=1)

    # confusionn matrix with all the classes
    cm = confusion_matrix(classes_test, classes_pred)
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm, annot=cm, cmap='Blues', vmin=1, vmax=100)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('all_classes_matrix.png')
    plt.close()
    print('All classes report')
    print(classification_report(classes_test, classes_pred))

    # confusion matrix for only the grasps
    cm_grasp = cm[1:,1:]
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm_grasp, annot=cm_grasp, cmap='Blues');
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('Only_grasps_matrix.png')
    plt.close()

    sns.set(font_scale=2.5)
    # binary matrix grasp vs no grasp
    cm_binary = np.array([[cm[0,0], cm[0,1:].sum()],[cm[1:,0].sum(), cm[1:,1:].sum()]])
    fig, ax = plt.subplots(figsize=(16,14))
    sns.heatmap(cm_binary, annot=cm_binary, cmap='Blues');
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('binary_matrix.png')
    plt.close()
    classes_pred[classes_pred!=0]=1
    classes_test[classes_test!=0]=1
    print('Binary classes report')
    print(classification_report(classes_test, classes_pred))

    error_dict = {}
    for error_pos in np.where(classes_test!=classes_pred)[0]:
        error_done = windows_position_test[error_pos]
        if error_done in error_dict.keys():
            if classes_test[error_pos]==0 and classes_pred[error_pos]==1:
                error_dict[error_done][1]+=1
            else:
                error_dict[error_done][0]+=1
        else:
            if classes_test[error_pos]==0 and classes_pred[error_pos]==1:
                error_dict[error_done]=[0,1]
            else:
                error_dict[error_done]=[1,0]
            
    for error_key, error_val in error_dict.items():
        print('Number of error in phase {}: Grasp as NoGrasp {} - NoGrasp as Grasp {}'.format(error_key, error_val[0], error_val[1]))

