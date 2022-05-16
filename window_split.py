import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import matplotlib.pyplot as plt
import pickle
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler("window_split_data.log", mode='w'),
        logging.StreamHandler()
    ]
    )

#Rimappo oggetti
def obj_mapping(id):
    if id == 11:
        return 41
    if id == 13:
        return 34
    if id == 14:
        return 44
    if id == 16:
        return 25
    return id

X_list = None
y_list = None
X_4_sorting = None
windows_position_test = []

def apply_sliding(line, dirpath, window_size, ref_start_event, ref_end_event, shift_left, shift_right, apply_sorting=False, channel_sorting=None, grasp_only=False, test=False):
    global  X_list, y_list
    filepath = os.path.join(dirpath, 'img_'+str(int(line.img_id)))
    binned_spk = np.load(filepath)
    np_matrix = binned_spk['arr_0']

    vectorization_sliding_windows = np.expand_dims(np.arange(window_size), 0) + np.expand_dims(np.arange(np_matrix.shape[1]-window_size + 1), 0).T
    if apply_sorting:
        np_matrix=np_matrix[channel_sorting,:]
    # 3D matrices: the shape is (#windows, windows_size, #channels)
    windows_matrices=np_matrix.T[vectorization_sliding_windows]
    num_samples = windows_matrices.shape[0]    

    if test:
        # save the positions of the windows for the test set
        global windows_position_test
        for w in range(num_samples):
            if len(line[line==w])>0:
                possible_bin = line[line==w].keys()[0]                
                if possible_bin!='img_id' and possible_bin!='obj_id':
                    reference_bin = possible_bin
            windows_position_test.append(reference_bin)
    # label -1 stands for no grasp (the majority of them)
    evt_labels = -1*np.ones(num_samples)
    # assign the object label to the windows that lie in the hold area
    evt_labels[int(line[ref_start_event]+shift_left): int(line[ref_end_event]-window_size+1+shift_right)] = obj_mapping(line.obj_id)

    evt_labels_list = evt_labels.tolist()

    if grasp_only:
        for i in range(len(evt_labels)):
            if evt_labels[i] == -1:
                continue
            else:
                X_list.append(windows_matrices[i])
                y_list.append(evt_labels_list[i])
    else:
        for w in windows_matrices:
            X_list.append(w)
        y_list.extend(evt_labels_list)
    
    return X_list, y_list

def create_sliding_windows(dirpath, df, window_size, ref_start_event, ref_end_event, shift_left, shift_right, apply_sorting, channel_sorting, grasp_only=False, test=False):
    global X_list, y_list
    X_list = []
    y_list = []
    logging.info('Apply sliding windows')
    df.apply(lambda line: apply_sliding(line, dirpath, window_size, ref_start_event, ref_end_event, shift_left, shift_right, apply_sorting, channel_sorting, grasp_only, test), axis=1)
    X = np.asarray(X_list)
    y = np.asarray(y_list)
    return X,y
        

def prepare_dataset(binned_samples_df, window_size):
    # split into training and test set the samples, stratified with respect to the object identifier
    
    #dati, dimensione test
    df_train, df_test = train_test_split(binned_samples_df, test_size=0.2, random_state=42, stratify=binned_samples_df.obj_id)
    # df_val, df_test = train_test_split(df_test, test_size=2/3, random_state=42, stratify=df_test.obj_id)
    # compute the average size (in bins) of the hold period and set it as window size    
    # df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42, stratify=df_train.obj_id)
    return df_train, df_test

def read_matrix(line, dirpath):
    global  X_4_sorting
    filepath = os.path.join(dirpath, 'img_'+str(int(line.img_id)))
    binned_spk = np.load(filepath)
    np_matrix = binned_spk['arr_0']
    if len(X_4_sorting)==0:
        X_4_sorting=np_matrix.sum(axis=1)
    else:
        X_4_sorting += np_matrix.sum(axis=1)

def compute_sorting(df_train_4_sorting, dirpath):
    global X_4_sorting
    X_4_sorting = np.array([])
    df_train_4_sorting.apply(lambda line: read_matrix(line, dirpath), axis=1)
    # sorting of the channel by firing rate
    channel_order=np.argsort(X_4_sorting)
    return channel_order

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/ZRec50_Mini_40_binned_spiketrains',
                        type=str)
    parser.add_argument("--window-size", help="Number of bins used for the running window (integer)", default=12,
                        type=int)
    parser.add_argument("--ref-start-event", help="Reference start event for data window definition and grasp labelling", default='Hold',
                        type=str)
    parser.add_argument("--ref-end-event", help="Reference end event for data window definition and grasp labelling", default='Rew',
                        type=str)
    parser.add_argument("--shift-left", help="Data window left extension wrt to reference event (bins)", default=-5,
                        type=int)
    parser.add_argument("--shift-right", help="Data window right extension wrt to reference event (bins)", default=5,
                        type=int)
    parser.add_argument("-s", "--no-channel-sorting", help="Disable sorting stream channels by firing rate",
                        action="store_true")
    parser.add_argument("-g", "--only-grasp", help="Remove no-grasp windows",
                        action="store_true")

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

    # flag to decide whether apply sorting to the channels
    apply_sorting = not args.no_channel_sorting
    channel_sorting = None
    # read the csv file
    binned_samples_df=pd.read_csv(csv_file)

    data_prefix = data_prefix+ '_' + str(args.window_size) + '_' + args.ref_start_event + '_' + args.ref_end_event + '_' + str(args.shift_left) + '_' + str(args.shift_right)
    if apply_sorting:
        data_prefix = data_prefix + '_sorted'
    if args.only_grasp:
        data_prefix = data_prefix + '_grasponly'
    
    # some samples can be under-represented (in this case object 55), I have to remove it
    # Rimozione oggetti poco rappresentai
    logging.info('Remove under-represented classes')
    obj_to_remove = binned_samples_df.groupby('obj_id').Start.count().loc[lambda p: p < 2]
    for obj_r in obj_to_remove.keys():
        logging.info('Removed object: ', obj_r)
        binned_samples_df = binned_samples_df[binned_samples_df.obj_id!=obj_r]
        
    #Preparo il dataset
    logging.info('Split dataset in test and train')
    df_train, df_test = prepare_dataset(binned_samples_df, args.window_size)
    
    # choose a training set on which compute the sorted order of the channels
    df_train_4_sorting = df_train.head(int(0.25*df_train.shape[0]))
    logging.info('Sorting channels')
    channel_sorting = compute_sorting(df_train_4_sorting, args.dataset)
    
    # create the sliding windows for each dataset: train, test and val
    logging.info('Create training windows')
    X_train, y_train = create_sliding_windows(args.dataset, df_train, args.window_size, args.ref_start_event, args.ref_end_event, args.shift_left, args.shift_right, apply_sorting, channel_sorting, args.only_grasp)
    logging.info('Create test windows')
    X_test, y_test = create_sliding_windows(args.dataset, df_test, args.window_size, args.ref_start_event, args.ref_end_event, args.shift_left, args.shift_right, apply_sorting, channel_sorting, args.only_grasp, test=True)
    # logging.info('Create validation windows')
    # X_val, y_val = create_sliding_windows(args.dataset, df_val, args.window_size, args.ref_start_event, args.ref_end_event, args.shift_left, args.shift_right, apply_sorting, channel_sorting, args.only_grasp)
    
    # encode the labels
    logging.info('Label encoding')
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    integer_encoded_train = label_encoder.transform(y_train).reshape(len(y_train), 1)
    integer_encoded_test = label_encoder.transform(y_test).reshape(len(y_test), 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit(integer_encoded_train)

    y_train = onehot_encoder.transform(integer_encoded_train)
    y_test = onehot_encoder.transform(integer_encoded_test)
    # y_val = label_encoder.transform(y_val)
    with open(os.path.join('data',data_prefix+'_label_encoding.txt'), 'w') as fp:
        for cl in label_encoder.classes_:
            fp.write(str(label_encoder.transform([cl])[0])+' -> obj '+str(cl)+'\n')
    
    # save binary files
    logging.info('Save binary datasets')
    with open(os.path.join('data',data_prefix+'_trainset.npz'), 'bw') as trainfile:
        np.savez(trainfile, X=X_train, y=y_train)
    with open(os.path.join('data',data_prefix+'_testset.npz'), 'bw') as testfile:
        np.savez(testfile, X=X_test, y=y_test)
    # with open(os.path.join('data',data_prefix+'_valset.npz'), 'bw') as valfile:
    #     np.savez(valfile, X=X_val, y=y_val)
    with open(os.path.join('data',data_prefix+'_testlist.pkl'), 'wb') as list_file:
        pickle.dump(windows_position_test, list_file)
