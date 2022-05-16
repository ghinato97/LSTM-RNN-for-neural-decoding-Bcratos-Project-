import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
    
from neo.io.neomatlabio import NeoMatlabIO
from elephant.conversion import BinnedSpikeTrain

from viziphant.rasterplot import eventplot
from viziphant.events import add_event

# from ephyviewer import mkQApp
# from ephyviewer import get_sources_from_neo_segment, compose_mainviewer_from_sources

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# %%
#"""
# r = NeoMatlabIO(filename='data\MRec40.neo.mat')
# r = NeoMatlabIO(filename='data/ZRec50_Mini.neo.mat')
# blk = r.read_block()

# i = 0
# seg = blk.segments[i]
# evt = seg.events[0]
# spk = seg.spiketrains
# t0 = spk[0].t_start
# t1 = spk[0].t_stop
# print(seg.annotations)
# print(evt.labels) 

# fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16,12))
# eventplot(spk, axes=axes, histogram_bins=50)
# add_event(axes[0], evt)
# add_event(axes[1], evt)

# bst = BinnedSpikeTrain(spk, bin_size=40*pq.ms)
# bsta = bst.to_array()

# plt.imshow(bsta, aspect='auto', interpolation='none', 
#            extent=[t0,t1,bsta.shape[0],0],clim=(0,2))
# plt.ylabel('Spikes/bin')
# plt.show()

# def get_event_time(evt, label):
#     idx = np.where(evt.labels == label)
#     return evt.times[idx]

# # Fix labels of events
# for i in range(blk.size['segments']):
#     seg = blk.segments[i]
#     evt = seg.events[0]
#     temp_labels = []
#     for lab in evt.labels:
#         temp_labels.append(lab.strip())
#     evt.labels = temp_labels

r = NeoMatlabIO(filename='data\c.mat')
seg = r.read_block()