# from bin_slit_data_sum import *
from bin_slit_data_sum import *
import math 
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential,optimizers
from keras.layers import Dense,LSTM,Dropout
from sklearn.metrics import confusion_matrix, classification_report





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
    
    binned_samples_df=pd.read_csv(csv_file)
    hold=binned_samples_df['Hold']
    rew=binned_samples_df['Rew']
    img=binned_samples_df['img_id']
    
    train_df,test_df=train_test_split(binned_samples_df, test_size=0.2, random_state=42, stratify=binned_samples_df.obj_id)
    train_set,label_train=Prepare_Dataset(args.dataset,train_df)
    test_set,label_test=Prepare_Dataset(args.dataset,test_df)



    print('\n')
    model = Sequential()
    model.add(LSTM(units=50, return_sequences= True, input_shape=(train_set.shape[1],train_set.shape[2])))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    # model.add(Dropout(.5))
    model.add(Dense(2, activation="softmax"))
    model.summary()
    
    # optimizer = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer='Adam', loss="binary_crossentropy",metrics='accuracy')
    
    
    model.fit(train_set,label_train, epochs=100, batch_size=16)

    predicted_value= model.predict(test_set)
    predicted_value=predicted_value.round()
    
    predicted_value=predicted_value.argmax(axis=1)
    label_test=label_test.argmax(axis=1)
    
    #confusion matrix
    
    cm = confusion_matrix(label_test,predicted_value)
    
    
    
    
