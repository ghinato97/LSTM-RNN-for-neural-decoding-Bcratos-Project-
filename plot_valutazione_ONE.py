import pickle
import matplotlib.pyplot as plt
import math
import numpy as np

file=open('sotto_soglia_value_ONE.pkl','rb')
minore_soglia=pickle.load(file)
file.close()

file=open('predicted_value_ONE.pkl','rb')
predicted_dict=pickle.load(file)
file.close()

file=open('y_test_ONE.pkl','rb')
y_test=pickle.load(file)
file.close()

def One_dof(train_data):
    one_dof_=[]
    for i in range(len(train_data)):
        valori=train_data[i]
        x0=valori[0]
        x1=valori[1]
        v=math.sqrt(pow(x0,2)+pow(x1,2))
        one_dof_.append(v)
        
    one_dof_=np.array(one_dof_)
    return one_dof_


for i in range(len(minore_soglia)):
    
    indice_peggio_035=minore_soglia.iloc[i]['index_id']
    peggio_predizione=predicted_dict[str(int(indice_peggio_035))]
    peggio_true_label=y_test[int(indice_peggio_035)]
    one_dof=One_dof(peggio_true_label)
    
    fig, ax = plt.subplots(figsize=(16,14))
    plt.plot(one_dof,label='True Values')  # green dots
    plt.plot(peggio_predizione,label='Predicted_Values')  # blue stars
    plt.title('Comparison',fontsize=15)  
    plt.xlabel('Time_Stamp',fontsize=15)
    plt.ylabel('Value',fontsize=15)
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig(f'Plot{i}.png')  
  
