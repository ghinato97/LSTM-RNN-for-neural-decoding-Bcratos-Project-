This Master Thesis aims at developing a strategy for Continuous Control (CC) of a prosthetic hand to evaluate how a Machine Learning (ML) algorithm can
enable a Non-Human-Primate to control a number of degree of freedom(DOF). Before focusing on continuous DOF control, I focused on a discrete classification
task based on a dataset col-lected by researchers of DPZ (Deutsches Primatenzentrum), who recorded brain activity via implanted microe-lectrode arrays (FMAs)
from NHPs trained to grasp different objects that vary in size and shape for a total of 48 different objects. Classification task were two: the first was a binary
classification with purpose of finding movement activation while the second consist of detection of graspable object. For these tasks I investigated different NN-based
approaches, focusing on Bidirectional Recurrent Neural Newtork (BRNN). Eventually,in order to test the capabilities of BRNNs for Continuous Control, due
to the lack of proper data readily available, with support of DPZ researchers I created a new virtual kinematics. To do so I grouped ob-jects into 7 macro-classes,
based on their shape and the typical grasp behaviour of NHPs. For each object group, a synthetic dataset is recreated containing the values of two coordinates
describing the virtual kinematics. As final step I implemented a regression task on this kinematics by means of a BRNN.
