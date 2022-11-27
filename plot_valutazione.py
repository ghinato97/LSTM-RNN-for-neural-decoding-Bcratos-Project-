import pickle
import matplotlib.pyplot as plt

file=open('sotto_soglia_value.pkl','rb')
minore_soglia=pickle.load(file)
file.close()

file=open('predicted_value.pkl','rb')
predicted_dict=pickle.load(file)
file.close()

file=open('y_test.pkl','rb')
y_test=pickle.load(file)
file.close()


for i in range(len(minore_soglia)):
    
    indice_peggio_035=minore_soglia.iloc[i]['index_id']
    peggio_predizione=predicted_dict[str(int(indice_peggio_035))]
    peggio_true_label=y_test[int(indice_peggio_035)]
   
    fig, ax = plt.subplots(figsize=(16,14))
    plt.plot(peggio_true_label[:,0],label='True Values')  # green dots
    plt.plot(peggio_predizione[:,0],label='Predicted_Values')  # blue stars
    plt.title('Comparison',fontsize=15)  
    plt.xlabel('Time_Stamp',fontsize=15)
    plt.ylabel('Value',fontsize=15)
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig(f'Plot{i}.png')  
  
