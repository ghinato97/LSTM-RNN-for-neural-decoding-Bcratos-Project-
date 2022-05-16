from neo.io.neomatlabio import NeoMatlabIO
from elephant.conversion import BinnedSpikeTrain
import quantities as pq
import numpy as np
import os
import pandas as pd
import argparse
import logging

#Configurazione per salvare e mostrare i risultati
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:  %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler("all_stream_data.log", mode='w'),
        logging.StreamHandler()
    ]
    )


def from_matlab_to_numpy_data(filename, bin_size, out_prefix, not_only_correct=False):
    
    #Se l'output path non esiste te lo crea nella cartella Data_Binned spiketrains
    if not os.path.exists(os.path.join('data',out_prefix+'binned_spiketrains')):
        os.makedirs(os.path.join('data',out_prefix+'binned_spiketrains'))
        
    #apro e leggo i dati
    neodata = NeoMatlabIO(filename)
    blk = neodata.read_block()
    
    #creo un dataframe vuoto
    binned_samples_df = pd.DataFrame()
    
    
    # number of samples
    num_trials = blk.size['segments']
    # iterate over the samples
    for i in range(num_trials):
        # segment of the considered sample
        seg = blk.segments[i]
        
        # save only correct grasps and the non problematic ones (identifier of the object less than 85)
        object_id = seg.annotations['obj']
        if (seg.annotations['correct'] == 1 or not_only_correct) and object_id < 85:
            # list of the events and the corresponding timing
            evt = seg.events[0]
            evt_labels = []
            for lab in evt.labels:
                evt_labels.append(lab.strip())
                
                
            #lista degli spikeTrains per ogni canale    
            spk = list(seg.spiketrains) 
            #Conta quanti spike sono avvenuti in finestre temporali di tot secondi
            bst = BinnedSpikeTrain(spk, bin_size=bin_size*pq.ms)
            
            
            # convert the times of the event into the corresponding start bins
            events_bin = np.array((evt.times-bst.t_start)*1000/bin_size, dtype=int)
            #crea un dizionario vuoto
            events_dict = {}
            for j in range(len(events_bin)):
                events_dict[evt_labels[j]]=events_bin[j]
            
            #al dizionario si aggiunge l'id dell'oggetto e l'id dell'immagine
            events_dict['obj_id']=object_id
            events_dict['img_id']=i
            
            #aggiunge al dataFrame il dizionario
            binned_samples_df=binned_samples_df.append(events_dict, ignore_index=True)
            
            # binned spiketrain as numpy array
            bst_array = bst.to_array()
            output_file = open(os.path.join(os.getcwd(),'data',out_prefix+'binned_spiketrains','img_'+str(i)), "wb")
            # save array to the file
            np.savez(output_file, bst_array)
            # close the file
            output_file.close
    # save the binning of the events and the identifier of the held object for each sample        
    binned_samples_df.to_csv(os.path.join(os.getcwd(),'data',out_prefix+'binned_spiketrains.csv'), index=False)    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help=".mat file containing brain recordings", default='data/ZRec50_Mini.neo.mat',
                        type=str)
    parser.add_argument("--bin-size", help="Time lenght for the spike train binning (milliseconds)", default=40,
                        type=int)      
    args = parser.parse_args()

    logging.info('\n')
    logging.info('----------------')
    logging.info('Building dataset with following params')
    logging.info(args)
    logging.info('----------------')
    logging.info('\n')
    logging.info('Loading data...')
    
    #Prefisso dell'output finale del path
    output_prefix = os.path.basename(os.path.normpath(args.dataset)).split('.')[0]+'_'+str(args.bin_size)+'_'
    from_matlab_to_numpy_data(args.dataset, args.bin_size, output_prefix)
