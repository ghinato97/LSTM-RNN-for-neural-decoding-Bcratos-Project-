import time

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
    
from neo.io.neomatlabio import NeoMatlabIO
from elephant.conversion import BinnedSpikeTrain

from viziphant.rasterplot import eventplot
from viziphant.events import add_event

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

r = NeoMatlabIO(filename='data\ZRec50_Mini.neo.mat')
# r = NeoMatlabIO(filename='data/ZRec50_Mini.neo.mat')
blk = r.read_block()

for i in range(blk.size['segments']): #per tutti i segmenti, da 1 fino a fine
    seg = blk.segments[i]
    evt = seg.events[0]
    spk = list(seg.spiketrains)
    t0 = spk[0].t_start
    t1 = spk[0].t_stop

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16,12))
    eventplot(spk, axes=axes, histogram_bins=50)
    add_event(axes[0], evt)
    add_event(axes[1], evt)

    bst = BinnedSpikeTrain(spk, bin_size=40*pq.ms)
    bsta = bst.to_array()
    
    plt.imshow(bsta, aspect='auto', interpolation='none', 
                extent=[t0,t1,bsta.shape[0],0],clim=(0,2))
    plt.ylabel('Spikes/bin')
    plt.show()
    
    print(seg.annotations)

    print(blk.size)
    time.sleep(5.5)    # Pause 5.5 seconds

   

def get_event_time(evt, label):
    idx = np.where(evt.labels == label)
    return evt.times[idx]

# Fix labels of events
for i in range(blk.size['segments']):
    seg = blk.segments[i]
    evt = seg.events[0]
    temp_labels = []
    for lab in evt.labels:
        temp_labels.append(lab.strip())
    evt.labels = temp_labels
    

#Grasp
grasp_data = []

for i in range(blk.size['segments']):
    seg = blk.segments[i]
    if seg.annotations['correct'] == 1:
        evt = seg.events[0]
        spk = list(seg.spiketrains)
        bst = BinnedSpikeTrain(spk, bin_size=40*pq.ms)
        mid = get_event_time(evt, 'Hold')
        sample = {}
        start = (mid-0.3*pq.s)
        stop = (mid+0.3*pq.s)
        sample['spike_train'] = bst.time_slice(t_start=start, t_stop=stop, copy=True)
        sample['object'] = seg.annotations['obj']
        sample['graps'] = True
        # print(sample['spike_train'])
        grasp_data.append(sample)
        # Other no-grasp  datax
        trial_start = get_event_time(evt, 'Start')
        # for step in np.arange(0,2,0.25):
        #     sample = {}
        #     start = trial_start+step*pq.s
        #     stop = start+0.6*pq.s
        #     sample['spike_train'] = bst.time_slice(t_start=start, t_stop=stop, copy=True)
        #     sample['object'] = 99
        #     sample['graps'] = False
        #     grasp_data.append(sample)

print(grasp_data[0]['spike_train'].to_array().shape)
classes = []
for item in grasp_data:
    classes.append(item['object'])
num_classes = np.unique(classes)
print(num_classes.size)


X_list = []
y_list = []
for item in grasp_data:
    array = item['spike_train'].to_array()
    X_list.append(array.reshape(array.shape[0], array.shape[1], 1))
    # X_list.append(array.flatten())
    y_list.append(item['object'])

X = np.array(X_list)
Y = np.array(y_list).reshape(-1, 1)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


input_shape = X[0].shape
print(input_shape)
print(X.shape)

# Design model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(onehot_encoded[0]), activation="softmax"),
    ]
)

model.summary()


optimizer = keras.optimizers.Adam(lr=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.summary()
# Train model on dataset
model.fit(X, onehot_encoded, epochs=200, batch_size=32, verbose=1, validation_split=0.1)