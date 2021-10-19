"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from modules.NN_data import load_data
from modules.NN_evaluate import evaluate_test_data
from modules.control_plots import plot_Fa_vs_Fs_v2

import pandas as pd

from modules.NN_config import *
from modules.CD_parameters_RELAB import path_relab

filename_train_data = 'OC-norm-denoised.dat'
x_test, y_test = load_data(filename_train_data)

model_names = ['20211018154939_FC.h5',
               '20211018154949_FC.h5',
               '20211018155002_FC.h5',
               '20211018155013_FC.h5',
               '20211018155023_FC.h5']

predictions, accuracy = evaluate_test_data([model_names[4]], x_test, y_test)
plot_Fa_vs_Fs_v2(y_test, predictions)

ind_Fa, ind_Fs = num_minerals, num_minerals + subtypes[0]
Fa_pred, Fs_pred = predictions[:, ind_Fa] * 100, predictions[:, ind_Fs] * 100

filename = path_relab + 'OC-norm-denoised_meta.dat'
data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
types = data[:, 7]
inds_H = np.array(['H' in type for type in types])
inds_L = np.array([('L' in type) and ('LL' not in type) for type in types])
inds_LL = np.array(['LL' in type for type in types])

HFa, HFs = (16.2, 16.2+3.8), (14.5, 14.5+3.5)
LFa, LFs = (22.0, 22.0+4.0), (19.0, 19.0+3.0)
LLFa, LLFs = (26.0, 26.0+6.0), (22.0, 22.0+4.2)

FaH, FsH = Fa_pred[inds_H], Fs_pred[inds_H]
FaL, FsL = Fa_pred[inds_L], Fs_pred[inds_L]
FaLL, FsLL = Fa_pred[inds_LL], Fs_pred[inds_LL]

ctot = 0

NN = 100

#####

tmp = np.linspace(FaH - accuracy[ind_Fa], FaH + accuracy[ind_Fa], NN)
c1 = np.array([np.array([HFa[1]>=tmp[i,j]>=HFa[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c1 = np.array([c1[:, j].any() for j in range(np.shape(tmp)[1])])

tmp = np.linspace(FsH - accuracy[ind_Fs], FsH + accuracy[ind_Fs], NN)
c2 = np.array([np.array([HFs[1]>=tmp[i,j]>=HFs[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c2 = np.array([c2[:, j].any() for j in range(np.shape(tmp)[1])])

ctot += sum(c1 + c2)

#####

tmp = np.linspace(FaL - accuracy[ind_Fa], FaL + accuracy[ind_Fa], NN)
c1 = np.array([np.array([LFa[1]>=tmp[i,j]>=LFa[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c1 = np.array([c1[:, j].any() for j in range(np.shape(tmp)[1])])

tmp = np.linspace(FsL - accuracy[ind_Fs], FsL + accuracy[ind_Fs], NN)
c2 = np.array([np.array([LFs[1]>=tmp[i,j]>=LFs[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c2 = np.array([c2[:, j].any() for j in range(np.shape(tmp)[1])])

ctot += sum(c1 + c2)

#####

tmp = np.linspace(FaLL - accuracy[ind_Fa], FaLL + accuracy[ind_Fa], NN)
c1 = np.array([np.array([LLFa[1]>=tmp[i,j]>=LLFa[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c1 = np.array([c1[:, j].any() for j in range(np.shape(tmp)[1])])

tmp = np.linspace(FsLL - accuracy[ind_Fs], FsLL + accuracy[ind_Fs], NN)
c2 = np.array([np.array([LLFs[1]>=tmp[i,j]>=LLFs[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c2 = np.array([c2[:, j].any() for j in range(np.shape(tmp)[1])])

ctot += sum(c1 + c2)

ctot /= sum(inds_H + inds_L + inds_LL)
ctot *= 100

"""
