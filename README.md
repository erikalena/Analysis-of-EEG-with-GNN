# Analysis of EEG using GNN

This project explores EEG classification using Graph Neural Networks (GNNs), focusing on distinguishing rest vs. computation states and analyzing solver performance. Subject-specific models and a global model were trained, with edge masks learned to highlight class-relevant brain connectivity. 

### Dataset

The dataset used in this project is the [EEG During Mental Arithmetic Tasks](https://physionet.org/content/eegmat/1.0.0/). The dataset contains EEG signals captured from 36 participants performing mental arithmetic tasks. The dataset contains two .edf files for each participant (e.g. Subject00_1.edf, Subject00_2.edf), one for the CQ task and one for the MS task. The dataset also contains a .csv file with the labels (Good or Bad counting quality) for each participant. 

