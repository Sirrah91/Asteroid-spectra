from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from modules.NN_data import load_data
from modules.NN_evaluate import evaluate_test_data, evaluate
from modules.control_plots import plot_Fa_vs_Fs_v2, plot_Fa_vs_Fs_Tuomas

from modules.NN_config import *
from modules.CD_parameters_RELAB import path_relab

# Name of the new data in ./Datasets/RELAB/ or ./Datasets/Tuomas/
# filename_data = 'Tuomas_HB_spectra-norm-denoised_nolabel.dat'
# filename_data = 'AP_spectra-norm-denoised_nolabel.dat'
# filename_train_data = 'OC-norm-denoised.dat'
filename_train_data = 'achondrites-norm-denoised.dat'

x_test, y_test = load_data(filename_train_data)

model_names1 = ['20211018154939_FC.h5',
                '20211018154949_FC.h5',
                '20211018155002_FC.h5',
                '20211018155013_FC.h5',
                '20211018155023_FC.h5']  # this one is good

model_names2 = ['20211020134525_FC.h5',
                '20211020140040_FC.h5',
                '20211020141700_FC.h5',
                '20211020143237_FC.h5',  # this one is good
                '20211020144856_FC.h5',
                '20211020150621_FC.h5',
                '20211020152738_FC.h5',
                '20211020154836_FC.h5',
                '20211020160605_FC.h5',
                '20211020161912_FC.h5']

model_names3 = ['20211020205324_FC.h5']


predictions, accuracy = evaluate_test_data([model_names1[4]], x_test, y_test)
# predictions, accuracy = evaluate_test_data([model_names2[3]], x_test, y_test)
# predictions, accuracy = evaluate_test_data(model_names3, x_test, y_test)

plot_Fa_vs_Fs_v2(y_test, predictions)
plot_Fa_vs_Fs_Tuomas(predictions)


ind_Fa, ind_Fs = num_minerals, num_minerals + subtypes[0]
Fa, Fs = predictions[:, ind_Fa] * 100, predictions[:, ind_Fs] * 100

filename = path_relab + 'OC-norm-denoised_meta.dat'
data = np.loadtxt(filename, delimiter='\t')
types = data[:, 7]
inds_H = np.array(['H' in OC_type for OC_type in types])
inds_L = np.array([('L' in OC_type) and ('LL' not in OC_type) for OC_type in types])
inds_LL = np.array(['LL' in OC_type for OC_type in types])

HFa, HFs = (16.2, 16.2 + 3.8), (14.5, 14.5 + 3.5)
LFa, LFs = (22.0, 22.0 + 4.0), (19.0, 19.0 + 3.0)
LLFa, LLFs = (26.0, 26.0 + 6.0), (22.0, 22.0 + 4.2)

FaH, FsH = Fa[inds_H], Fs[inds_H]
FaL, FsL = Fa[inds_L], Fs[inds_L]
FaLL, FsLL = Fa[inds_LL], Fs[inds_LL]

NN = 100
sigma = 1

#####

tmp = np.linspace(FaH - sigma * accuracy[ind_Fa], FaH + sigma * accuracy[ind_Fa], NN)
c1 = np.array(
    [np.array([HFa[1] >= tmp[i, j] >= HFa[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c1 = np.array([c1[:, j].any() for j in range(np.shape(tmp)[1])])

tmp = np.linspace(FsH - sigma * accuracy[ind_Fs], FsH + sigma * accuracy[ind_Fs], NN)
c2 = np.array(
    [np.array([HFs[1] >= tmp[i, j] >= HFs[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c2 = np.array([c2[:, j].any() for j in range(np.shape(tmp)[1])])

cH = sum(c1 * c2) / sum(inds_H) * 100

#####

tmp = np.linspace(FaL - sigma * accuracy[ind_Fa], FaL + sigma * accuracy[ind_Fa], NN)
c1 = np.array(
    [np.array([LFa[1] >= tmp[i, j] >= LFa[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c1 = np.array([c1[:, j].any() for j in range(np.shape(tmp)[1])])

tmp = np.linspace(FsL - sigma * accuracy[ind_Fs], FsL + sigma * accuracy[ind_Fs], NN)
c2 = np.array(
    [np.array([LFs[1] >= tmp[i, j] >= LFs[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c2 = np.array([c2[:, j].any() for j in range(np.shape(tmp)[1])])

cL = sum(c1 * c2) / sum(inds_L) * 100

#####

tmp = np.linspace(FaLL - sigma * accuracy[ind_Fa], FaLL + sigma * accuracy[ind_Fa], NN)
c1 = np.array(
    [np.array([LLFa[1] >= tmp[i, j] >= LLFa[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c1 = np.array([c1[:, j].any() for j in range(np.shape(tmp)[1])])

tmp = np.linspace(FsLL - sigma * accuracy[ind_Fs], FsLL + sigma * accuracy[ind_Fs], NN)
c2 = np.array(
    [np.array([LLFs[1] >= tmp[i, j] >= LLFs[0] for j in range(np.shape(tmp)[1])]) for i in range(np.shape(tmp)[0])])
c2 = np.array([c2[:, j].any() for j in range(np.shape(tmp)[1])])

cLL = sum(c1 * c2) / sum(inds_LL) * 100

np.round(np.mean((FaH, FsH), axis=1), 1)
np.round(np.mean((FaL, FsL), axis=1), 1)
np.round(np.mean((FaLL, FsLL), axis=1), 1)

# error
np.round(accuracy[ind_Fa] / np.sqrt(np.sum(inds_H)), 1)
np.round(accuracy[ind_Fa] / np.sqrt(np.sum(inds_L)), 1)
np.round(accuracy[ind_Fa] / np.sqrt(np.sum(inds_LL)), 1)

np.round(accuracy[ind_Fs] / np.sqrt(np.sum(inds_H)), 1)
np.round(accuracy[ind_Fs] / np.sqrt(np.sum(inds_L)), 1)
np.round(accuracy[ind_Fs] / np.sqrt(np.sum(inds_LL)), 1)
