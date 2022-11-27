import logging
import argparse
import os
import math
from math import sin,cos,pi
import seaborn as sns
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler("window_split_data.log", mode='w'),
        logging.StreamHandler()
    ]
)


def prepare_detection_dataset(dirpath,df,lookback, lookahead):
    X = []
    label= []

    
    go=df['Go']
    hold = df['Hold']
    rew = df['Rew']
    end=df['End']    
    obj = df['obj_id']
    
    hold_go=hold-go
    Rew_hold=rew-hold+1 
    End_rew=end-rew


    file_ids = df['img_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    obj_ids = df['obj_id'].to_numpy(dtype=int)
    dizionario_val={}
    dizionario_label={}
    obj_val={}
    obj_label={}
    list_indici=[]
    for idx in range(len(filenames)):
        filepath = os.path.join(dirpath, 'img_'+filenames[idx])
        binned_spk = np.load(filepath)
        np_matrix = binned_spk['arr_0']
        #cerco l'indice dell'immagine relativa
        indice=int(df[df['img_id']==int(filenames[idx])].index.values)
        list_indici.append(indice)
        
        matrice=np.zeros([2,np_matrix.shape[1]])
        
        init=go[indice]
        arr_init=np.zeros(int(init))
        
        x=hold_go[indice]
        arr=np.arange(x)
        arr_sigm_x=np.asarray(norma_sig(arr))
        
        y=Rew_hold[indice]
        arr_sigm_y=np.ones(int(y))
        
        z=End_rew[indice]
        arr=-(np.arange(z))
        arr_sigm_z=np.asarray(norma_sig(arr))
        
        arr_conc=pd.Series(np.concatenate([arr_init,arr_sigm_x,arr_sigm_y,arr_sigm_z]))
        
        x0=(arr_conc*cos(angle[indice])).to_numpy().T
        x1=(arr_conc*sin(angle[indice])).to_numpy().T
        
        matrice[0,0:len(x0)]=x0
        matrice[1,0:len(x1)]=x1
        
        for i in range(np_matrix.shape[1]-lookback-lookahead):
            t = []
            for j in range(0, lookback):
                matrix=np_matrix[:, [(i+j)]]                
                t.append(matrix)
            X.append(t)
            label.append(matrice[:,i+lookback+lookahead])
            
            
        
        data_set = np.array(X)
        y_label = np.array(label)    
        nome=filenames[idx]
        
        dizionario_val[indice]=data_set
        dizionario_label[indice]=y_label
        
        
        obj_id=obj_ids[idx]
        obj_val[str(obj_id)]=data_set
        obj_label[str(obj_id)]=y_label
        X=[]
        label=[]

        
        
            
        
        
    return dizionario_val,dizionario_label,obj_val,obj_label
        

            

               
         
    
    
    


def norma_sig(a):
    d = 2.*(a - np.min(a))/np.ptp(a)-1
    d=d*5
    sig=[]
    for i in range(len(d)):
        sig.append(funzione(d[i]))
        
    return sig

def obj_mapping(id): #mappatura oggetti uguali
    if id == 11:
        id = 41
    if  id == 13:
        id = 34
    if id == 14:
        id = 44
    if id == 16:
        id = 25
    if id == 12:
        id = 54
        
    if id == 21 or id == 22 or id == 31 or id == 41 or id == 42 or id == 43:
       return 3 #classe gialla
    if id == 23 or id == 24 or id == 25 or id == 26:
       return 2 #classe azzurra
    if id == 32 or id == 33 or id == 34 or id == 35 or id == 36:
       return 7 #classe rosa
    if id == 44 or id == 45 or id == 46:
       return 6 #classe nera
    if id == 51 or id == 52 or id == 53 or id == 54 or id == 55 or id == 56:
       return 4 #classe blu
    if id == 15 or id == 61 or id == 62 or id == 63 or id == 64 or id == 65 or id == 66: 
       return 5 #classe rossa
    if id == 71 or id == 72 or id == 73 or id == 74 or id == 75 or id == 76:
       return 1 #classe verde
    else: 
        return id
    
def angle_mapping(obj,total_class):
    arr_angle=[]
    for i in range(len(obj)):
        angle=(2*pi/total_class)*(int(obj[i])-1)
        arr_angle.append(angle)
    return np.array(arr_angle)

    

def funzione(x):
    return 1 / (1 + math.exp(-x))
    

#sostituisce i valori di obj id nella sua corretta clusterizzazione
def prepare_binned(binned_samples_df):   
    file_ids = binned_samples_df['obj_id'].to_numpy(dtype=int).tolist()
    filenames = list(map(str, file_ids))
    
    for idx in range(len(filenames)):
        filenames[idx]=obj_mapping(int(filenames[idx]))  
    
    binned_samples_df['obj_id']=filenames
    
    return binned_samples_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Name of CC BRNN model", default='Continuous_control_MRec40_40_binned_spiketrainsmodel',
                        type=str)

    parser.add_argument("-b", "--binned_data", help="Name of binned dataset ", default='data/MRec40_40_binned_spiketrains',
                        type=str)
    
    parser.add_argument("-d", "--dataset", help="Name of dataset from previous step, without extension", default='data/MRec40_40_binned_spiketrains',
                        type=str)
    
    args = parser.parse_args()
    
    data_prefix = os.path.basename(os.path.normpath(args.binned_data))
    csv_file = os.path.join(args.binned_data,data_prefix+'.csv')    
    binned_samples_df = pd.read_csv(csv_file)
    obj = binned_samples_df['obj_id']
    
    train_df, test_df = train_test_split(binned_samples_df, test_size=0.2, random_state=42, stratify=binned_samples_df.obj_id) 
    obj_max = max(binned_samples_df['obj_id'].to_numpy())
    angle=angle_mapping(obj,obj_max)
    
    lookahead=0
    lookback=12
    
    X_Test,y_test,obj_X_test,obj_y_test=prepare_detection_dataset(args.dataset,test_df,lookback, lookahead)
    
    with open("y_test.pkl", "wb") as f:
       pickle.dump(y_test, f)

    
    logging.info('\n')
    logging.info('----------------')
    logging.info('Loading Binary Models')
    data_prefix = os.path.basename(os.path.normpath(args.model))
    model=keras.models.load_model(data_prefix)
    logging.info('Binary Model Info')
    model.summary()
    
    predicted_dict={}
    lista_indici=list(test_df.index)
    lista_oggetti=list(test_df.obj_id)
    data = {'obj_id': [],
	'index_id': [],
	'MSE': []}

    MSE_df=pd.DataFrame(data)
    
    for i in range(len(lista_indici)):
        n=int(lista_indici.pop())
        p=int(lista_oggetti.pop())
        
        Go=int(test_df.loc[[n]]['Go'].to_numpy()[0]-12)
        End=int(test_df.loc[[n]]['End'].to_numpy()[0]-12)
        
        test=X_Test[n]
        label=y_test[n]
        
        predicted_value = model.predict(test)
        
        predicted_dict[str(n)]=predicted_value
        
        mse=mean_squared_error(y_test[n][(Go):(End)],predicted_value[(Go):(End)])
        new_row = {'obj_id':p, 'index_id':n, 'MSE':mse}
        MSE_df=MSE_df.append(new_row,ignore_index=True)
        

    MSE_df=MSE_df.sort_values(by=['MSE'])
    
    
    #utilizzo il valore MSE come asse x
    asse_x=MSE_df.MSE.to_numpy()
    asse_x=np.round(asse_x,2)
    #utilizzo il numero del trial come valore y
    asse_y=MSE_df.index_id.to_numpy()
    
    #Plotting MSE 
    plt.figure(figsize=(9,9))
    plt.hist(asse_x,bins=20)
    plt.xlabel('MSE',fontsize=20)
    plt.ylabel('#Trial',fontsize=20)
    plt.plot([0.3, 0.3], [0, 16], 'k-', lw=2)
    plt.savefig('MSE evaluetion.png')
    plt.show()
    
    valori_minori=asse_x[asse_x<=0.35]
    
    print('\n')
    print('-------------------------------')
    percentuale= len(valori_minori)/len(asse_x)*100
    print(f'Percentuale MSE al di sotto della soglia:{percentuale}')
    
    with open("predicted_value.pkl", "wb") as f:
       pickle.dump(predicted_dict, f)
    
    
    
    # cerco i numeri di oggetti che hanno il valore MSE minore di 0.35 e plotto i valori
    #predetti e reali
    minore_soglia=MSE_df.where(MSE_df.MSE<0.35).dropna()
    
    with open("sotto_soglia_value.pkl", "wb") as f:
       pickle.dump(minore_soglia, f)
       
    # for i in range(len(minore_soglia)):
    #     indice_peggio_035=minore_soglia.iloc[i]['index_id']
    #     peggio_predizione=predicted_dict[str(int(indice_peggio_035))]
    #     peggio_true_label=y_test[int(indice_peggio_035)]
        
    #     fig, ax = plt.subplots(figsize=(16,14))
    #     plt.plot(peggio_true_label[:,0],label='True Values')  # green dots
    #     plt.plot(peggio_predizione[:,0],label='Predicted_Values')  # blue stars
    #     plt.title('Comparison')  
    #     plt.xlabel('Time_Stamp')
    #     plt.ylabel('Value')
    #     plt.legend(loc='best')
    
    


    
    


    

    # # Plotting MSE-Trial
    # plt.figure()
    # plt.bar(asse_x,asse_y,width = 0.01)
    # plt.xlabel('MSE')
    # plt.ylabel('#Trial')
    # plt.savefig('MSE evaluetion on trial.png')
    # plt.show()
                
    # #raggruppo il db in base all'oggetto e faccio valore medio MSE     
    # df=MSE_df.drop(['index_id'],axis=1)
    # df=df.groupby('obj_id').mean()
          
    
    
    
    
        
        
    


    
    
