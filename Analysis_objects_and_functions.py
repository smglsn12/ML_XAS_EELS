import matplotlib.pyplot as plt
import ncempy.io as nio
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import savgol_filter
import math
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
# from Materials_Project_Functions import *
from scipy.interpolate import interp1d
import glob
import matplotlib
import matplotlib.font_manager as font_manager
from sklearn.metrics import r2_score
from matplotlib.cm import ScalarMappable
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from scipy import interpolate

def flatten(list1):
    return [item for sublist in list1 for item in sublist]
def scale_spectra_flex(df_w_spectra, zero_energy='default', energy_col='Energies',
                       intensity_col='Spectrum', broadened_col=None, output_col_energy='Scaled Energy (eV)',
                       output_col_intensity='Scaled Intensity', show_plot = False):
    spectra_energies_scaled = []
    spectra_intensities_scaled = []
    spectra_broadened_scaled = []
    mins = []
    maxes = []
    for i in range(0, len(df_w_spectra)):
        energy = df_w_spectra.iloc[i][energy_col]
        mins.append(min(energy))
        maxes.append(max(energy))
    # print(mins)
    min_lock = max(mins)
    interp_max = round(min(maxes) - 0.05, 1)
    for i in range(0, len(df_w_spectra)):
        full_energies = []
        full_intens = []

        energy = df_w_spectra.iloc[i][energy_col]
        intensity = df_w_spectra.iloc[i][intensity_col]
        if broadened_col != None:
            broadened_intensity = df_w_spectra.iloc[i]['Corrected Broadened Intensities exp alignment']
        interp_min = round(min(energy) + 0.05, 1)
        # print(i)
        # print(energy)
        # print(len(energy))
        # print(len(intensity))
        # print(len(broadened_intensity))

        f = interpolate.interp1d(energy, intensity)
        if broadened_col != None:
            f_broadened = interpolate.interp1d(energy, broadened_intensity)
        interp_energies = np.arange(interp_min, interp_max + 0.1, 0.1)
        interp_energies_final = []
        for en in interp_energies:
            en_rounded = round(en, 1)
            if en_rounded <= interp_max:
                interp_energies_final.append(en_rounded)
        # print([min(interp_energies_final), max(interp_energies_final)])

        # if round(max(interp_energies), 1) != 774.5:
        try:
            interped_intens = f(interp_energies_final)
            if broadened_col != None:
                interped_broadened_intens = f_broadened(interp_energies_final)
        except ValueError:
            # print(energy)
            print(interp_energies_final)

        if show_plot:
            plt.plot(energy, intensity, label='Simulated Spectrum', linewidth=3)
            plt.title('Illustration of Scaled Baseline', fontsize=22, fontweight = 'bold')
            plt.xticks(fontsize=18, fontweight = 'bold')
            plt.yticks(fontsize=18, fontweight = 'bold')
            plt.xlabel('Energy (eV)', fontsize=20, fontweight = 'bold')
            plt.ylabel('Intensity', fontsize=20, fontweight = 'bold')
            # plt.plot(interp_energies_final, interped_intens, label = 'interpolated')
            # plt.legend(fontsize = 16)
            # plt.show()
            print('interp min = ' + str(interp_min))

        if zero_energy == 'default':
            zero_energy = interp_min - 1

        x = [zero_energy, interp_energies_final[
            0]]  # two given datapoints to which the exponential function with power pw should fit
        y = [10 ** -10, interped_intens[0]]

        if broadened_col != None:
            y_broadened = [10 ** -10, interped_broadened_intens[0]]

        def func(x, adj1, adj2):
            return ((x + adj1) ** pw) * adj2

        pw = 6
        A = np.exp(np.log(y[0] / y[1]) / pw)
        a = (x[0] - x[1] * A) / (A - 1)
        b = y[0] / (x[0] + a) ** pw

        end_energy = interp_energies_final[0] - 0.1
        gap = int(round(end_energy - zero_energy, 1) * 10)
        # print(gap)

        xf = np.linspace(zero_energy, end_energy, gap + 1)
        # plt.plot(x, y, 'ko', label="Original Data")
        ys = func(xf, a, b)

        if broadened_col != None:
            A = np.exp(np.log(y_broadened[0] / y_broadened[1]) / pw)
            a = (x[0] - x[1] * A) / (A - 1)
            b = y_broadened[0] / (x[0] + a) ** pw
            ys_broad = func(xf, a, b)
        if show_plot:
            # ys-min(ys)
            plt.plot(xf, ys, 'r', label="Fitted Baseline", linewidth=3)
            # plt.xlim([925, 940])
            plt.legend(fontsize=16)
            plt.show()
            # print(ys - min(ys))
            # print(xf)
            # print(interp_energies)
            plt.show()

        interp_energies_rounded = [round(num, 1) for num in interp_energies_final]
        extrapolated_energies_rounded = [round(num, 1) for num in xf]

        # if min(interped_intens) < 0.01:
        # interped_intens = interped_intens + 0.05
        if np.isnan(ys[0]):
            # print(interped_intens)
            # print(ys)
            # print(full_intens)
            ys = np.zeros((len(ys)))
        full_energies = list(extrapolated_energies_rounded) + list(interp_energies_rounded)
        full_intens = list(ys) + list(interped_intens)
        full_intens = full_intens - min(full_intens)
        if min(full_intens) > 10 ** -13:
            print('fail')

        if broadened_col != None:
            full_intens_broad = list(ys_broad) + list(interped_broadened_intens)
            full_intens_broad = full_intens_broad - min(full_intens_broad)
            if min(full_intens_broad) > 10 ** -13:
                print('fail')

        # print(full_energies)
        # plt.plot(x, y, 'ko', label="Original Data")
        # plt.plot(full_energies, full_intens)
        # plt.show()
        # print(full_intens-min(full_intens))
        spectra_energies_scaled.append(full_energies)
        spectra_intensities_scaled.append(full_intens)
        if broadened_col != None:
            spectra_broadened_scaled.append(full_intens_broad)

    df_w_spectra[output_col_energy] = spectra_energies_scaled
    df_w_spectra[output_col_intensity] = spectra_intensities_scaled
    if broadened_col != None:
        df_w_spectra['Aligned Scaled Broadened'] = spectra_broadened_scaled

    return df_w_spectra

def build_L2_3(l3, l2, show_plot=True):
    interp_min_L3 = round(min(l3.T[0]) + 0.15, 1)
    # print(interp_min_L3)
    interp_max_L3 = round(max(l3.T[0]) - 0.15, 1)
    # print(interp_max_L3)
    L3_energies = np.arange(interp_min_L3, interp_max_L3, 0.1)
    # print(L3_energies)

    interp_min_L2 = round(min(l2.T[0]) + 0.15, 1)
    interp_max_L2 = round(max(l2.T[0]) - 0.15, 1)
    L2_energies = np.arange(interp_min_L2, interp_max_L2, 0.1)

    # plt.vlines(l3_fermi, 0,1.5, color = 'green')
    f_l2 = interpolate.interp1d(l2.T[0], l2.T[1])
    f_l3 = interpolate.interp1d(l3.T[0], l3.T[1])

    interped_l3 = f_l3(L3_energies)
    interped_l2 = f_l2(L2_energies)

    zero_energy = interp_min_L3
    x = [zero_energy, L2_energies[0]]  # two given datapoints to which the exponential function with power pw should fit
    y = [10 ** -10, interped_l2[0]]

    def func(x, adj1, adj2):
        return ((x + adj1) ** pw) * adj2

    pw = 10
    A = np.exp(np.log(y[0] / y[1]) / pw)
    a = (x[0] - x[1] * A) / (A - 1)
    b = y[0] / (x[0] + a) ** pw

    end_energy = L2_energies[0] - 0.1
    gap = int(round(end_energy - zero_energy, 1) * 10)

    xf = np.linspace(zero_energy, end_energy, gap + 1)
    # plt.plot(x, y, 'ko', label="Original Data")
    ys = func(xf, a, b)

    full_energies = list(xf) + list(L2_energies)
    full_intens = list(ys) + list(interped_l2)
    full_energies_final = []
    for i in full_energies:
        full_energies_final.append(round(i, 1))

    big_bad = False
    for j in range(1, len(full_energies_final)):
        if round(full_energies_final[j] - full_energies_final[j - 1], 8) != 0.1:
            big_bad = True
    if big_bad == True:
        raise ValueError

    L2_3 = interped_l3 + full_intens[0:len(L3_energies)]
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(L3_energies, interped_l3, label='L3', linewidth = 3)
        plt.plot(full_energies_final[0:len(L3_energies)], full_intens[0:len(L3_energies)], label='L2', linewidth = 3)
        plt.plot(L3_energies, L2_3, label='L2,3', linewidth = 3)
        plt.xticks(fontsize=18, fontweight = 'bold')
        plt.yticks(fontsize=18, fontweight = 'bold')
        plt.legend(fontsize=14)
        plt.xlabel('Energy (eV)', fontsize = 20, fontweight = 'bold')
        plt.ylabel('Intensity', fontsize = 20, fontweight = 'bold')
        plt.title('L2, L3, L2,3 For mp-30 (Cu(0))', fontsize=22, fontweight = 'bold')
        plt.show()

    L3_energies_rounded = []
    for i in L3_energies:
        L3_energies_rounded.append(round(i, 1))

    return [L3_energies_rounded, L2_3]


def visualize_full_noise_test_set(noise_dfs, interp_ranges, show_err = True, savefigure=False):
    if type(noise_dfs) != list:
        noise_dfs = [noise_dfs]
    if type(interp_ranges) != list:
        interp_ranges = [interp_ranges]
    for vis in ['R2', 'RMSE']:
        count = -1
        plt.figure(figsize=(8,7))
        for noise_df in noise_dfs:
            count += 1
            mean_01 = np.mean(np.asarray(noise_df.loc[noise_df['noise_std'] == 1000][vis]))
            mean_05 = np.mean(np.asarray(noise_df.loc[noise_df['noise_std'] == 500][vis]))
            mean_1 = np.mean(np.asarray(noise_df.loc[noise_df['noise_std'] == 100][vis]))
            mean_2 = np.mean(np.asarray(noise_df.loc[noise_df['noise_std'] == 50][vis]))

            std_01 = np.std(np.asarray(noise_df.loc[noise_df['noise_std'] == 1000][vis]))
            std_05 = np.std(np.asarray(noise_df.loc[noise_df['noise_std'] == 500][vis]))
            std_1 = np.std(np.asarray(noise_df.loc[noise_df['noise_std'] == 100][vis]))
            std_2 = np.std(np.asarray(noise_df.loc[noise_df['noise_std'] == 50][vis]))


            if vis == 'R2':
                plt.title('R2 vs Noise', fontsize = 36, fontweight='bold')
                plt.xlabel('Noise STD', fontsize = 36, fontweight='bold')
                plt.ylabel('R2', fontsize = 36, fontweight='bold')
                plt.xticks([0,0.1, 0.2], fontsize = 36, fontweight='bold')
                plt.yticks([0.3,0.6,0.9], fontsize = 36, fontweight='bold')
                plt.ylim([0.28, 0.95])
                plt.xlim([-0.01, 0.225])
                plt.scatter([0, 0.01, 0.05, 0.1, 0.2], [0.88, mean_01, mean_05, mean_1, mean_2], color = 'k', s=200, zorder=5)

                if show_err:
                    plt.plot([0, 0.01, 0.05, 0.1, 0.2], [0.88, mean_01, mean_05, mean_1, mean_2], color = 'k')
                    eb1 = plt.errorbar([0, 0.01, 0.05, 0.1, 0.2], [0.88, mean_01, mean_05, mean_1, mean_2], yerr=[0, std_01, std_05, std_1, std_2],
                                 ecolor='k', errorevery=1, capsize=15, linewidth = 4, label = str(interp_ranges[count]))
                    eb1[-1][0].set_linestyle(':')
                    if savefigure:
                        plt.savefig('R2 Noise Profile '+str(interp_ranges[count])+'.pdf',  bbox_inches='tight', transparent=True)

                else:
                    plt.plot([0, 0.01, 0.05, 0.1, 0.2], [0.88, mean_01, mean_05, mean_1, mean_2],
                             linewidth = 4, label = str(interp_ranges[count]))

            if vis == 'RMSE':
                plt.title('RMSE vs Noise', fontsize = 36, fontweight='bold')
                plt.xlabel('Noise STD', fontsize = 36, fontweight='bold')
                plt.ylabel('RMSE', fontsize = 36, fontweight='bold')
                plt.xticks([0,0.1, 0.2], fontsize = 36, fontweight='bold')
                plt.yticks([0.2,0.35,0.5], fontsize = 36, fontweight='bold')
                plt.ylim([0.19, 0.525])
                plt.xlim([-0.01, 0.225])
                plt.scatter([0, 0.01, 0.05, 0.1, 0.2], [0.214, mean_01, mean_05, mean_1, mean_2], color = 'k', s=200, zorder=5)

                if show_err:
                    plt.plot([0, 0.01, 0.05, 0.1, 0.2], [0.214, mean_01, mean_05, mean_1, mean_2], color = 'k')
                    eb1 = plt.errorbar([0, 0.01, 0.05, 0.1, 0.2], [0.214, mean_01, mean_05, mean_1, mean_2], yerr=[0, std_01, std_05, std_1, std_2],
                                 ecolor='k', errorevery=1, capsize=15, linewidth = 4, label = str(interp_ranges[count]))
                    eb1[-1][0].set_linestyle(':')
                    if savefigure:
                        plt.savefig('RMSE Noise Profile '+str(interp_ranges[count])+'.pdf',  bbox_inches='tight', transparent=True)

                else:
                    plt.plot([0, 0.01, 0.05, 0.1, 0.2], [0.214, mean_01, mean_05, mean_1, mean_2],
                             linewidth = 4, label = str(interp_ranges[count]))
        if show_err == False:
            plt.legend(fontsize = 22, title="Sampling Interval (eV)", title_fontsize = 22)
            if vis == 'RMSE':
                if savefigure:
                    plt.savefig('RMSE Noise Profile'+str(' all')+'.pdf',  bbox_inches='tight', transparent=True)
            if vis == 'R2':
                if savefigure:
                    plt.savefig('R2 Noise Profile'+str(' all')+'.pdf',  bbox_inches='tight', transparent=True)


def poisson(x, std, random_state=32):
    np.random.seed(random_state)

    noise = np.random.poisson(100, len(x))-100
    noise = noise/std

    x_noisy = np.asarray(x) + np.asarray(noise)

    return x_noisy

def gaussian_noise(x, mu, std, random_state=32):
    """
    :param x: int/float - value to be augmented with gaussian noise
    :param mu:
    :param std:
    :param random_state:
    :return:
    """
    np.random.seed(random_state)

    noise = []
    for entry in x:
        noise.append(np.random.normal(mu, std, size=1)[0])

    x_noisy = np.asarray(x) + np.asarray(noise)

    return x_noisy

def spectrum(E,osc,sigma,x):
    gE=[]
    for Ei in x:
        tot=0
        for Ej,os in zip(E,osc):
            tot+=os*np.exp(-((((Ej-Ei)/sigma)**2)))
        gE.append(tot)
    return gE



class eels_rf_setup():
    def __init__(self, spectra_df_filepath):
        self.spectra_df_filepath = spectra_df_filepath
        self.rf_model = None
        self.rf_error_df = None
        self.spectra_df = None
        self.rf_input_parameters = []
        self.rf_training_set = None
        self.smoothing_params = []
        self.interped_intens = None
        self.interped_energies = None
        self.interp_spacing = None
        self.energy_col = None
        self.points_to_average = None
        self.mixture_ids = []

    def load_spectra_df(self):
        self.spectra_df = joblib.load(self.spectra_df_filepath)

    def full_noise_setup(self, column, energy_col, interp_range, smoothing_window, filename=None, show_plots=False,
                         baseline_subtract=False):
        full_noise_analysis_output = []
        accuracies = []
        rmses = []
        count = 0
        count_1 = 0
        for random_seed in np.linspace(0, 99, 100):
            random_seed = int(random_seed)
            # print(random_seed)
            # for std in [0.01, 0.05, 0.1, 0.2]:
            for std in [1000, 500, 100, 50]:
                # print(std)
                noisy_test = []
                interp_spec = np.asarray(self.rf_error_df[column])

                energy = np.arange(925, 970.1, 0.1)[0:451]
                energy[450] = 970
                interp_energies = np.arange(925, 970 + interp_range, interp_range)

                smooth_interp_spec = []
                interp_energies_final = []
                for en in interp_energies:
                    en_rounded = round(en, 2)
                    if en_rounded <= 970:
                        interp_energies_final.append(en_rounded)
                if show_plots:
                    if count == 0:
                        plt.plot(interp_spec[0])
                        plt.title('Interpolate Spectra ' + str(interp_range) + ' eV', fontsize=18)
                        plt.show()

                for spec in interp_spec:
                    noisy_test.append(poisson(spec, std, random_seed))

                # interp_energies_further = np.arange(925, 970 + 0.001, 0.001)
                for noisy_spec in noisy_test:
                    # print(count)
                    if show_plots:
                        if count_1 == 0:
                            plt.plot(self.rf_error_df.iloc[0][energy_col], noisy_spec)
                            plt.title('add noise original')
                            plt.show()
                            count_1 += 1
                    # print(len(test_rf_obj.rf_error_df.iloc[0][energy_col]))
                    # f = interpolate.interp1d(test_rf_obj.rf_error_df.iloc[0][energy_col], noisy_spec)

                    # print(interp_energies_final)
                    # interped_noisy_spec = f(interp_energies_final)

                    smoothed_spec = savgol_filter(noisy_spec, smoothing_window, 3)

                    # print(len(interp_energies_final))
                    # print(len(smoothed_spec))
                    f = interpolate.interp1d(interp_energies_final, smoothed_spec)
                    smoothed_spec = f(energy)
                    if baseline_subtract:
                        smoothed_spec = smoothed_spec - min(smoothed_spec)
                    smooth_interp_spec.append(smoothed_spec)
                if show_plots:
                    if count == 0:
                        plt.plot(interp_energies_final[0:100], noisy_spec[0:100])
                        plt.plot(energy[0:30], smoothed_spec[0:30])
                        plt.title('Zoom in on baseline')
                        plt.show()

                        plt.figure(figsize=(8, 7))
                        plt.xlabel('Energy (eV)', fontsize=36, fontweight='bold')
                        plt.ylabel(' Intensty', fontsize=36, fontweight='bold')
                        plt.xticks([930, 950, 970], fontsize=36, fontweight='bold')
                        plt.yticks([0, 0.5, 1, 1.5], fontsize=36, fontweight='bold')
                        plt.plot(self.rf_error_df.iloc[0][energy_col], noisy_test[0], linewidth=3,
                                 label='Noisy Spec')
                        plt.title('Add Noise std = ' + str(std), fontsize=36, fontweight='bold')
                        # plt.show()
                        plt.plot(energy, smooth_interp_spec[0], linewidth=3, label='Smoothed')
                        # plt.title('Smooth Using Same Window As Exp', fontsize = 18)
                        plt.legend(fontsize=21)
                        # plt.savefig('Example Noisy Spec' + str(std) + '.pdf',  bbox_inches='tight', transparent=True)
                        plt.show()

                # plt.figure(figsize=(8,7))

                # plt.plot(noisy_test[0], linewidth = 3, label = 'Noisy Spectra')
                # plt.plot(energy, smooth_interp_spec[0], linewidth = 2, label = 'Smoothed')

                # plt.title('Noisy Spectra', fontsize = 36, fontweight='bold')

                # font = font_manager.FontProperties(
                #                                    weight='bold',
                #                                    style='normal', size=22)
                # plt.xlabel('Energy (eV)', fontsize = 36, fontweight='bold')
                # plt.ylabel(' Intensty', fontsize = 36, fontweight='bold')
                # plt.xticks([930, 950, 970], fontsize = 36, fontweight='bold')
                # plt.yticks([0,1], fontsize = 36, fontweight='bold')
                # plt.legend(fontsize = 21)
                # plt.show()

                noisy_test_cum = []
                for spec in smooth_interp_spec:
                    temp_intens = []
                    for k in range(0, len(spec)):
                        temp_intens.append(sum(spec[0:k]))
                    noisy_test_cum.append(temp_intens / max(temp_intens))
                if show_plots:
                    if count == 0:
                        plt.plot(noisy_test_cum[0])
                        plt.show()

                accuracy = self.rf_model.score(noisy_test_cum, self.rf_error_df['Labels Test'])

                # noisy_spectra_test_pre = np.stack(noisy_test_cum).astype(np.float32)
                # noisy_spectra_test = preprocessing.scale(noisy_spectra_test_pre, axis = 1)

                # plt.plot(noisy_spectra_test[0])
                # plt.plot(preprocessing.scale(noisy_spectra_test_pre[0]))
                # plt.show()

                # predictions = dnn_test[1].predict(noisy_spectra_test)
                # print(predictions)
                # labels_test = np.asarray(test_rf_obj.rf_error_df['Labels Test'])
                # accuracy = r2_score(np.asarray(labels_test), np.asarray(predictions))

                # print(accuracy)

                predictions = self.rf_model.predict(noisy_test_cum)
                # predictions_full = []
                # trees = test_rf_obj.rf_model.estimators_
                # for tree in trees:
                #     predictions_full.append(tree.predict(np.asarray(noisy_test_cum)))
                # predictions_ordered = np.asarray(predictions_full).T
                # predictions_std = []
                # count = 0
                # for prediction in predictions_ordered:
                #     predictions_std.append(np.std(prediction))
                #     count += 1
                # print(predictions_std)

                predictions_rounded = []
                predictions_clean = []
                for pred in predictions:
                    predictions_rounded.append(round(pred, 1))
                for pred in predictions:
                    predictions_clean.append(pred)
                predictions = predictions_clean

                errors_noisy = np.abs(self.rf_error_df['Labels Test'] - predictions)
                # test_rf_obj.rf_error_df['Predictions_noisy_std'] = predictions_std
                self.rf_error_df['Predictions_noisy'] = predictions_rounded

                scatter_spot_multiplier = 15

                print('model accuracy (R^2) on simulated test data ' + str(accuracy))

                # plt.figure(figsize=(8, 6))
                true = []
                pred = []
                count_list = []
                # condensed_stds = []
                for i in np.asarray(
                        self.rf_error_df[['Predictions_noisy', 'Labels Test Rounded']].value_counts().index):
                    pred.append(round(i[0], 1))
                    true.append(round(i[1], 1))
                    # condensed_stds.append(np.mean(test_rf_obj.rf_error_df.loc[
                    #                                   (test_rf_obj.rf_error_df['Predictions_noisy'] == round(i[0], 1)) & (
                    #                                           test_rf_obj.rf_error_df['Labels Test Rounded'] == round(i[1], 1))][
                    #                                   'Predictions_noisy_std']))

                for k in np.asarray(
                        self.rf_error_df[['Predictions_noisy', 'Labels Test Rounded']].value_counts()):
                    count_list.append(k)
                count_list = np.asarray(count_list)

                # plt.figure(figsize=(8, 6))
                # plt.scatter(true, pred, s=count_list * scatter_spot_multiplier, c=count_list)

                # cb = plt.colorbar(label='Num Predictions')
                # ax = cb.ax
                # text = ax.yaxis.label
                # font = matplotlib.font_manager.FontProperties(size=22)
                # text.set_font_properties(font)
                # for t in cb.ax.get_yticklabels():
                #     t.set_fontsize(22)
                # min_plot = round(min(test_rf_obj.rf_error_df['Labels Test']) - 0.5, 0)
                # max_plot = round(max(test_rf_obj.rf_error_df['Labels Test']) + 1.5, 0)
                # plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3,
                #          linestyle='--')
                # plt.title('Predicted vs True', fontsize=24)
                # plt.xticks(fontsize=18)
                # plt.yticks(fontsize=18)

                # plt.ylabel('Bond Valance Prediction', fontsize=22)
                # plt.xlabel('True Bond Valance', fontsize=22)
                # plt.show()

                # plt.figure(figsize=(8, 6))
                # plt.title('Predicted vs True', fontsize=28)
                # plt.xticks(fontsize=24)
                # plt.yticks(fontsize=24)
                # plt.ylabel('Bond Valance Prediction', fontsize=28)
                # plt.xlabel('True Bond Valance', fontsize=28)
                # min_plot = round(min(test_rf_obj.rf_error_df['Labels Test']) - 0.5, 0)
                # max_plot = round(max(test_rf_obj.rf_error_df['Labels Test']) + 1.5, 0)
                # plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3,
                #          linestyle='--')
                # plt.scatter(true, pred, s=count * scatter_spot_multiplier, c=condensed_stds)
                # cb = plt.colorbar(label='Prediction Std')
                # ax = cb.ax
                # text = ax.yaxis.label
                # font = matplotlib.font_manager.FontProperties(size=28)
                # text.set_font_properties(font)
                # for t in cb.ax.get_yticklabels():
                #     t.set_fontsize(24)

                # plt.savefig('r^2 noisy plot.pdf',  bbox_inches='tight', transparent=True)

                # show_type = 'Abs Error'
                # nbins=20
                # show_rmse = True

                MSE = np.square(errors_noisy).mean()
                RMSE = math.sqrt(MSE)
                print('RMSE ' + str(RMSE))
                # plt.figure(figsize=(8, 6))
                # plt.title('Error Histogram', fontsize=24)
                # if show_type == 'Abs Error':
                #     hist = plt.hist(errors_noisy, bins=nbins)
                # elif show_type == 'MSE':
                #     hist = plt.hist(np.square(errors_noisy), bins=nbins)

                # if show_rmse:
                #     plt.vlines(RMSE, max(hist[0]), min(hist[0]), color='limegreen', linewidth=5, label='RMSE')
                #     plt.text(RMSE + 0.25, max(hist[0]) - 0.1 * max(hist[0]), 'RMSE = ' + str(round(RMSE, 3)),
                #              horizontalalignment='center', fontsize=16)
                # plt.xticks(fontsize=24)
                # plt.yticks(fontsize=24)
                # plt.xlabel(show_type, fontsize=28)
                # plt.ylabel('Frequency', fontsize=28)
                # plt.savefig('Error Histogram.pdf',  bbox_inches='tight', transparent=True)
                # plt.show()
                # mp_ids = np.asarray(test_rf_obj.rf_error_df['Materials Ids'])

                # full_noise_analysis_output.append([labels_test, mp_ids, predictions, predictions_ordered, predictions_std,
                #                                    errors_noisy, interp_spec, noisy_test, smooth_interp_spec,
                #                                    noisy_test_cum, std, random_seed, accuracy, RMSE])
                # accuracies.append(accuracy)
                # rmses.append(RMSE)

                # print('mean accuracy  = ' + str(np.mean(accuracies)))
                # print('std accuracy = ' + str(np.std(accuracies)))

                # print('mean RMSE = ' + str(np.mean(rmses)))
                # print('std RMSE = ' + str(np.std(rmses)))

                full_noise_analysis_output.append([std, random_seed, accuracy, RMSE])
            count += 1
        full_noise_df = pd.DataFrame(full_noise_analysis_output, columns=['noise_std', 'random_state', 'R2', 'RMSE'])
        if filename != None:
            joblib.dump(full_noise_df, filename)

    def compare_simulation_to_experiment(self, material='CuO', savgol_params=[51, 3],
                                         xlims=[920, 980], show_feff=False, feff_shift=-10,
                                         compare_to_lit=False, exp_spectrum=None, lit_shift=0.0, title=None,
                                         show_experiment = True):
        output = nio.dm.dmReader("C:/Users/smgls/Materials_database/" + material + " Deconvolved Spectrum.dm4")
        intens = output['data']
        energies = output['coords'][0]

        plt.figure(figsize=(8, 7))
        plt.xlabel('Energy (eV)', fontsize=22, fontweight='bold')
        plt.ylabel('Intensity', fontsize=22, fontweight='bold')
        plt.title(title, fontsize=24, fontweight='bold')
        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')

        interped_intens_smoothed = savgol_filter(intens / max(intens), savgol_params[0], savgol_params[1])

        if show_experiment:
            plt.plot(energies, interped_intens_smoothed / max(interped_intens_smoothed), color='k',
                     label='This Work', linewidth=3)

        if compare_to_lit:
            output = pd.read_csv(exp_spectrum)
            intens_temp = output['Intensity'] - min(output['Intensity'])
            intens = intens_temp / max(intens_temp)
            energies = output['Energy (eV)']
            energies_interp = np.arange(round(min(energies) + 1, 1), round(max(energies) - 1, 1), 0.1)
            energies_interp_use = []
            for i in energies_interp:
                energies_interp_use.append(round(i, 1))
                # print(round(i,1))
            f = interp1d(energies, intens)
            interp_intens = f(energies_interp_use)

            intens = interp_intens - min(interp_intens)
            intens = intens / max(intens)
            energies = energies_interp_use

            broadened_intens = spectrum(energies, intens, 0.2, energies)
            broadened_intens = broadened_intens - min(broadened_intens)
            broadened_intens = broadened_intens / max(broadened_intens)

        if show_feff:
            feff_spec_energies = self.spectra_df.iloc[0]['new Scaled Energies use']
            if material == 'Cu metal':
                feff_spec_intens = np.asarray(
                    self.spectra_df.loc[self.spectra_df.mpid_string == 'mp-30']['TEAM_1_aligned_925_970'])
            if material == 'Cu2O':
                feff_spec_intens = np.asarray(
                    self.spectra_df.loc[self.spectra_df.mpid_string == 'mp-361']['TEAM_1_aligned_925_970'])
            if material == 'CuO':
                feff_spec_intens = np.asarray(
                    self.spectra_df.loc[self.spectra_df.mpid_string == 'mp-1692']['TEAM_1_aligned_925_970'])

        if compare_to_lit:
            plt.plot(np.asarray(energies) - lit_shift, broadened_intens, label='Liturature XAS', linewidth=3,
                     color='#1f77b4')

        if show_feff:
            plt.plot(np.asarray(feff_spec_energies) + feff_shift, feff_spec_intens[0] / max(feff_spec_intens[0]),
                     color='#ff7f0e',
                     linewidth=3, linestyle='-', label='FEFF9 Simulation')
        plt.xlim(xlims)
        plt.legend(fontsize=18)

    def visualize_mixtures(self, cu_metal_id, cu2o_id, cuo_id, cu_fraction=0.5, cu2o_fraction=0.5, cuo_fraction=0.0,
                           include_ticks=False, include_title=True, savefigure = False):
        plt.figure(figsize=(8, 7))
        energies = np.asarray(
            self.spectra_df.loc[self.spectra_df['mpid_string'] == cu_metal_id][
                'new Scaled Energies use'])[0]
        # label_cu_metal = np.asarray(test_rf_obj.spectra_df.loc[test_rf_obj.spectra_df['mpid_string']==cu_metal_id]['pretty_formula'])[0]+ ' BV = ' + str(0)
        # label_cu2o = np.asarray(test_rf_obj.spectra_df.loc[test_rf_obj.spectra_df['mpid_string']==cu2o_id]['pretty_formula'])[0]+ ' BV = ' + str(1)
        # label_cuo = np.asarray(test_rf_obj.spectra_df.loc[test_rf_obj.spectra_df['mpid_string']==cuo_id]['pretty_formula'])[0]+ ' BV = ' + str(2)
        label_list = []
        if cu_fraction > 0:
            label_list.append('Cu(0)')
        if cu2o_fraction > 0:
            label_list.append('Cu(I)')
        if cuo_fraction > 0:
            label_list.append('Cu(II)')

        mix_str = 'Mixture'
        for comp in label_list:
            mix_str = mix_str + ' ' + comp

        cu_metal = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cu_metal_id]['TEAM_1_aligned_925_970'])[0]
        cu2o = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cu2o_id]['TEAM_1_aligned_925_970'])[0]
        cuo = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cuo_id]['TEAM_1_aligned_925_970'])[0]

        mix_spec = cu_metal * cu_fraction + cu2o * cu2o_fraction + cuo * cuo_fraction

        plt.plot(energies, cu_metal * cu_fraction, linewidth=4, linestyle='--')
        plt.plot(energies, cu2o * cu2o_fraction, linewidth=4, linestyle='--')
        plt.plot(energies, cuo * cuo_fraction, linewidth=4, linestyle='--')
        plt.plot(energies, mix_spec, label=mix_str, linewidth=4)

        font = font_manager.FontProperties(
            weight='bold',
            style='normal', size=24)

        # print(label_list)

        print(include_ticks)
        if include_ticks:
            plt.xlabel('Energy (eV)', fontsize=36, fontweight='bold')
            plt.xticks([930, 950, 970], fontsize=36, fontweight='bold')
        else:
            plt.xticks([], fontsize=36, fontweight='bold')

        if include_title:
            plt.title('Mixture Spectrum Example', fontsize=24, fontweight='bold')

        plt.xlim([922.5, 975])
        # plt.ylabel(' Intensty', fontsize = 36, fontweight='bold')
        plt.yticks([], fontsize=36, fontweight='bold')
        plt.ylim([-0.05, max(mix_spec) + 0.25])
        plt.legend(prop=font)
        if savefigure:
            plt.savefig(str([cu_metal_id, cu2o_id, cuo_id, cu_fraction, cu2o_fraction, cuo_fraction]) + '.pdf',
                    bbox_inches='tight', transparent=True)

    def visualize_mixture_addition(self, column = 'BV Used For Alignment', include_mixtures=False, savefigure = False):
        ys1 = self.spectra_df[column].value_counts().values[0:3]
        xs1 = self.spectra_df[column].value_counts().index[0:3]

        ys2 = self.spectra_df[column].value_counts().values[3:212]
        xs2 = self.spectra_df[column].value_counts().index[3:212]

        colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

        fig, ax1 = plt.subplots(figsize=(8, 7))
        # if include_mixtures:
            # ax1.yaxis.tick_right()
        ax1.bar(xs1, ys1, color=colors, width=0.3)

        # ax = plt.figure(figsize=(8,7))
        # ax.add_axes(ax1)
        if include_mixtures:
            mixture_values = []
            for i in range(0, len(self.spectra_df)):
                if self.spectra_df.iloc[i][column] not in [0.0, 1.0, 2.0]:
                    mixture_values.append(self.spectra_df.iloc[i][column])
            ax1.hist(mixture_values, bins=20, color='#d62728')

        plt.title('Spectra Per Oxidation State', fontsize=26, fontweight='bold')

        font = font_manager.FontProperties(
            weight='bold',
            style='normal', size=22)
        plt.xlabel('Oxidation State', fontsize=36, fontweight='bold')
        # plt.ylabel('Count', fontsize = 36, fontweight='bold')
        plt.xlim([-0.2, 2.2])
        plt.xticks([0, 1, 2], fontsize=36, fontweight='bold')
        plt.yticks([0, 400, 800, 1200], fontsize=36, fontweight='bold')
        if savefigure:
            if include_mixtures:
                plt.savefig('Full Dataset Mixtures.pdf', bbox_inches='tight', transparent=True)
            else:
                plt.savefig('Full Dataset.pdf', bbox_inches='tight', transparent=True)

    def visualize_predictions(self, indicies, show_cu_oxide_reference = False):
        plt.figure(figsize=(10, 8))

        for index in indicies:
            if show_cu_oxide_reference:

                ox_states = [0, 1, 2]
                count = 0
                for mp_id in ['mp-30', 'mp-361', 'mp-704645']:
                    spec = np.asarray(
                        self.spectra_df.loc[self.spectra_df.mpid_string == mp_id]['TEAM_1_aligned_925_970'])
                    energies = np.asarray(
                        self.spectra_df.loc[self.spectra_df.mpid_string == mp_id]['new Scaled Energies use'])
                    plt.plot(energies[0], spec[0], linewidth=3, label='Oxidation State = ' + str(ox_states[count]))
                    count += 1

            plt.title(self.rf_error_df.iloc[index]['Materials Ids'], fontsize=34, fontweight='bold')
            plt.xticks(fontsize=30, fontweight='bold')
            plt.yticks(fontsize=30, fontweight='bold')
            plt.xlabel('Energy (eV)', fontsize=34, fontweight='bold')
            plt.ylabel('Intensity', fontsize=34, fontweight='bold')
            plt.plot(self.rf_error_df.iloc[index]['Energies'], self.rf_error_df.iloc[index]['XAS Spectrum'],
                     linewidth=5, label='True = ' + str(self.rf_error_df.iloc[index]['Labels Test Rounded'])
                                        + ' Predicted = ' + str(self.rf_error_df.iloc[index]['Predictions Rounded']))

            plt.legend(fontsize=20)
            # plt.xlim([705, 725])
            plt.show()

            plt.figure(figsize=(10, 8))
            hist = plt.hist(self.rf_error_df.iloc[index]['Full Predictions'], edgecolor='k', facecolor='grey', fill=True, linewidth=3)
            ax = plt.gca()
            # ax.get_xticks() will get the current ticks
            # ax.set_yticklabels(map(str, ax.get_yticks()))
            plt.xticks([0, 0.5, 1, 1.5, 2, 2.5])

            height = max(hist[0])

            # plt.vlines(np.median(df.iloc[index]['Full Predictions']), 0, height, color = 'purple', label = 'Median Prediction', linewidth = 5)
            plt.vlines(self.rf_error_df.iloc[index]['Labels Test'], 0, height, color='blue', label='Labeled BV', linewidth=5)
            plt.vlines(np.mean(self.rf_error_df.iloc[index]['Full Predictions']), 0, height, color='red', label='Prediction',
                       linewidth=5,
                       linestyle='-')
            mean = np.mean(self.rf_error_df.iloc[index]['Full Predictions'])
            std = np.std(self.rf_error_df.iloc[index]['Full Predictions'])
            low = mean - std
            high = mean + std

            plt.hlines(height / 2, high, low, color='limegreen', linewidth=5, label='Std From Mean')
            plt.vlines(high, height / 2 - height * 0.075, height / 2 + height * 0.075, color='limegreen', linewidth=5)
            plt.vlines(low, height / 2 - height * 0.075, height / 2 + height * 0.075, color='limegreen', linewidth=5)

            plt.title('Prediction Histogram ' + self.rf_error_df.iloc[index]['Materials Ids'], fontsize=34, fontweight='bold')
            plt.xticks(fontsize=30, fontweight='bold')
            plt.xlim([-0.1, 2.6])
            plt.ylim([-10, 310])
            plt.yticks([0, 100, 200, 300], fontsize=30, fontweight='bold')
            plt.xlabel('Prediction', fontsize=34, fontweight='bold')
            plt.ylabel('Num Trees', fontsize=34, fontweight='bold')
            plt.legend(fontsize=20)
            # plt.plot(np.arange(0,max(errors),0.1), np.arange(0,max(errors),0.1), color = 'k', linewidth = 3, linestyle = '--')
            plt.show()

    def visualize_mixture_components(self, cu_metal_id, cu2o_id, cuo_id, include_title=True, include_ticks=True,
                                     savefigure=False):
        plt.figure(figsize=(8, 7))
        energies = np.asarray(
            self.spectra_df.loc[self.spectra_df['mpid_string'] == cu_metal_id][
                'new Scaled Energies use'])[0]
        label_cu_metal = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] ==
                                                        cu_metal_id]['pretty_formula'])[0] + ' BV = ' + str(0)
        label_cu2o = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] ==
                                                    cu2o_id]['pretty_formula'])[0] + ' BV = ' + str(1)
        label_cuo = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] ==
                                                   cuo_id]['pretty_formula'])[0] + ' BV = ' + str(2)

        cu_metal = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cu_metal_id]['TEAM_1_aligned_925_970'])[0]
        cu2o = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cu2o_id]['TEAM_1_aligned_925_970'])[0]
        cuo = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cuo_id]['TEAM_1_aligned_925_970'])[0]

        plt.plot(energies, cu_metal, label=label_cu_metal, linewidth=4)
        plt.plot(energies, cu2o, label=label_cu2o, linewidth=4)
        plt.plot(energies, cuo, label=label_cuo, linewidth=4)
        if include_title:
            plt.title('Random Sample Cu(0), Cu(I) and Cu(II)', fontsize=20, fontweight='bold')

        font = font_manager.FontProperties(
            weight='bold',
            style='normal', size=22)
        # plt.ylabel(' Intensty', fontsize = 36, fontweight='bold')
        if include_ticks:
            plt.xlabel('Energy (eV)', fontsize=36, fontweight='bold')
            plt.xticks([930, 950, 970], fontsize=36, fontweight='bold')
        else:
            plt.xticks([])
        plt.yticks([], fontsize=36, fontweight='bold')
        plt.legend(prop=font)

        if savefigure:
            plt.savefig('Mixture Components ' + str([cu_metal_id, cu2o_id, cuo_id]) + '.pdf', bbox_inches='tight',
                    transparent=True)

    def add_interp_spec_to_ERROR_df(self, interpolation_ranges, add_zeros=False, energy_zeros=920):

        count = 0
        for interpolation_range in interpolation_ranges:
            print(interpolation_range)
            energy = np.arange(925, 970.1, 0.1)[0:451]
            energy[450] = 970
            interp_spec = []
            interp_energy_for_df = []
            interp_energies = np.arange(925, 970 + interpolation_range, interpolation_range)
            if add_zeros:
                interp_energies_front = np.arange(energy_zeros, 925, interpolation_range)
                interp_energies_front = list(interp_energies_front)
                interp_energies_front.reverse()
            for spec in np.asarray(self.rf_error_df['XAS Spectrum']):
                # print(count)
                f = interpolate.interp1d(energy, spec)
                interp_energies_final = []
                for en in interp_energies:
                    en_rounded = round(en, 3)
                    if en_rounded <= 970:
                        interp_energies_final.append(en_rounded)
                # print(interp_energies_final)
                interped_spec = f(interp_energies_final)
                if add_zeros:
                    interped_spec = list(interped_spec)
                    for i in range(0, len(interp_energies_front)):
                        interped_spec.insert(0, 0)
                        interp_energies_final.insert(0, round(interp_energies_front[i], 3))

                # plt.plot(interp_energies_final, interped_spec)
                # plt.show()
                # print(interp_energies_final)
                # print(interped_spec)
                interp_spec.append(interped_spec)
                interp_energy_for_df.append(interp_energies_final)
                # count += 1
            self.rf_error_df['Interpolated_spec_' + str(interpolation_range)] = interp_spec
            self.rf_error_df['Interpolated_spec_energies_' + str(interpolation_range)] = interp_energy_for_df



    def scale_experimental_spectra(self, intens, energies, scale = 0.1):
        energies_interp = np.arange(round(min(energies)+0.5, 1),round(max(energies)-0.5, 1), scale)
        energies_interp_use = []
        for i in energies_interp:
            energies_interp_use.append(round(i, 1))
        f = interp1d(energies, intens)
        interp_intens = f(energies_interp)

        # intens = interp_intens - min(interp_intens)
        interp_intens = interp_intens / interp_intens[len(interp_intens) - 5]
        energies = energies_interp

        self.interped_intens = interp_intens
        self.interped_energies = energies
        self.interp_spacing = scale
        # plt.legend()
        plt.show()




    def predict_experiment_folder(self, folder_path, shifts = [0.0], smoothings = [[51,3]], edge_points = [10],
                                  spectra_type = 'csv', cumulative_spec = True,
                                  theory_column = 'XAS_aligned_925_970', energies_range = [925,970], show_hist = False,
                                  show_plots = False, show_inputted_spectrum=False, print_details = False,
                                  savefigure = False):
        paper_paths = glob.glob(folder_path+'/*')
        # print('clean')

        # print(paper_paths)
        predictions_set = []
        for path in paper_paths:
            # cu_spectra = glob.glob(path_temp )
            # for path in cu_spectra:
            for shift in shifts:
                for smooth in smoothings:
                    for edge in edge_points:
                        if print_details:
                            print('Predicting From ' + path)
                            print('Energy Axis Shift = ' + str(shift))
                            print('Smoothing Parameters = ' + str(smooth))
                        # try:
                        # print(path)
                        if 'Cu Metal' in path:
                            self.predict_experiment_random_forest(path, 1640, 'Cu Metal',
                                                                  smoothing='inputted single',
                                                                  exp_spectrum_type=spectra_type,
                                                                  smoothing_parms=smooth,
                                                                  points_to_average=edge,
                                                                  cumulative_spectrum=cumulative_spec,
                                                                  spectrum_energy_shift=shift,
                                                                  theory_column=theory_column,
                                                                  energies_range=energies_range,
                                                                  show_hist=show_hist,
                                                                  show_plots=show_plots,
                                                                  show_inputted_spectrum = show_inputted_spectrum,
                                                                  print_prediction=print_details,
                                                                  savefigure=savefigure)

                        elif 'Cu2O' in path:
                            self.predict_experiment_random_forest(path, 2240, 'Cu2O',
                                                                  smoothing='inputted single',
                                                                  exp_spectrum_type=spectra_type,
                                                                  smoothing_parms=smooth,
                                                                  points_to_average=edge,
                                                                  cumulative_spectrum=cumulative_spec,
                                                                  spectrum_energy_shift=shift,
                                                                  theory_column=theory_column,
                                                                  energies_range=energies_range,
                                                                  show_hist=show_hist,
                                                                  show_plots=show_plots,
                                                                  show_inputted_spectrum = show_inputted_spectrum,
                                                                  print_prediction=print_details,
                                                                  savefigure=savefigure)

                        elif 'CuO' in path:
                            self.predict_experiment_random_forest(path, 2191, 'CuO',
                                                                  smoothing='inputted single',
                                                                  exp_spectrum_type=spectra_type,
                                                                  smoothing_parms=smooth,
                                                                  points_to_average=edge,
                                                                  cumulative_spectrum=cumulative_spec,
                                                                  spectrum_energy_shift=shift,
                                                                  theory_column=theory_column,
                                                                  energies_range=energies_range,
                                                                  show_hist=show_hist,
                                                                  show_plots=show_plots,
                                                                  show_inputted_spectrum = show_inputted_spectrum,
                                                                  print_prediction=print_details,
                                                                  savefigure=savefigure)

                        else:
                            self.predict_experiment_random_forest(path, 1640, 'Unlabeled Cu',
                                                                  smoothing='inputted single',
                                                                  exp_spectrum_type=spectra_type,
                                                                  smoothing_parms=smooth,
                                                                  points_to_average=edge,
                                                                  cumulative_spectrum=cumulative_spec,
                                                                  spectrum_energy_shift=shift,
                                                                  theory_column=theory_column,
                                                                  energies_range=energies_range,
                                                                  show_hist=show_hist,
                                                                  show_plots=show_plots,
                                                                  show_inputted_spectrum = show_inputted_spectrum,
                                                                  print_prediction=print_details,
                                                                  savefigure=savefigure)


                        predictions_set.append(
                            [self.prediction, self.prediction_std, self.true_val,
                             self.points_to_average, self.smoothing_params[0], self.smoothing_params[1],
                             self.intensities_final, self.theory_index, self.exp_spec_path, self.material,
                             self.spectrum_energy_shift])
                        # except:
                            # pass

        self.prediction_df = pd.DataFrame(predictions_set,
                                columns=['Prediction', 'Predictions Std', 'True', 'Num Tail Points', 'Smoothing Window',
                                         'Smoothing Poly Order', 'Predicted Spectrum', 'Theory Index',
                                         'Spectrum filepath', 'Material', 'Spectrum Energy Shift'])

    def reset_mixed_valent_series(self):
        self.mixed_valent_pred = []



    def predict_Experimental_Cu_non_integers(self, cu_metal=1, cu2o=0, cuo=0,
                                             show_predicted_spectrum = True,
                                 show_plots = False, print_predictions = False,
                                folder_path = 'C:/Users/smgls/Materials_database/Cu_deconvolved_spectra',
                                smoothing_params = [51,3],
                                energies_range = [925, 970],
                                exp_scale = 0.1):

        cu_metal = float(cu_metal)
        cu2o = float(cu2o)
        cuo = float(cuo)
        self.smoothing_params = smoothing_params
        cu_fractions = [cu_metal, cu2o, cuo]
        true_val = (cu_metal*0 + cu2o*1 + cuo*2)/sum(cu_fractions)
        paper_paths = glob.glob(folder_path + '/*')

        # print(paper_paths)
        self.exp_spectra_raw = {}
        self.exp_spectra_cumulative = {}
        for exp_spectrum in paper_paths:

            output = nio.dm.dmReader(exp_spectrum)
            intens = output['data']
            energies = output['coords'][0]

            # plt.figure(figsize=(8, 7))
            if show_plots:
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.ylabel('Intensity', fontsize=20)
                plt.xlabel('Energy (eV)', fontsize=20)
                plt.title('Exp vs sim pre smoothing', fontsize=22)
                # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
                plt.plot(energies, intens / max(intens),
                         label='Experimental Spectrum', linewidth=1, zorder=10)
                plt.legend(fontsize=12)
                plt.xlim([927, 945])



            intens = savgol_filter(intens, self.smoothing_params[0], self.smoothing_params[1])
            if show_plots:
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.ylabel('Intensity', fontsize=20)
                plt.xlabel('Energy (eV)', fontsize=20)
                # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
                plt.plot(energies, intens / max(intens),
                         label='Experimental Spectrum Smoothed', linewidth=1, zorder=10)
                plt.legend(fontsize=12)
                plt.xlim([927, 945])
                plt.show()

            self.scale_experimental_spectra(intens, energies, scale = exp_scale)

            self.final_energies = np.arange(energies_range[0], energies_range[1] + 0.2, 0.1)[
                                  0:((energies_range[1] - energies_range[0]) * 10) + 1]
            energies_interp_use = []
            for i in self.final_energies:
                energies_interp_use.append(round(i, 1))
                # print(round(i,1))
            self.final_energies = energies_interp_use
            f = interp1d(self.interped_energies, self.interped_intens)

            self.intensities_final = f(self.final_energies)

            temp_intens = []

            for k in range(0, len(self.intensities_final)):
                temp_intens.append(sum(self.intensities_final[0:k]))
            self.standard_cum_spec = temp_intens / max(temp_intens)

            if 'Cu Metal' in exp_spectrum:
                self.exp_spectra_raw['Cu Metal'] = self.intensities_final
                self.exp_spectra_cumulative['Cu Metal'] = self.standard_cum_spec
            if 'Cu2O' in exp_spectrum:
                self.exp_spectra_raw['Cu2O'] = self.intensities_final
                self.exp_spectra_cumulative['Cu2O'] = self.standard_cum_spec

            if 'CuO' in exp_spectrum:
                self.exp_spectra_raw['CuO'] = self.intensities_final
                self.exp_spectra_cumulative['CuO'] = self.standard_cum_spec

        mixed_spec = self.exp_spectra_raw['Cu Metal'] * cu_fractions[0] + \
                     self.exp_spectra_raw['Cu2O'] * cu_fractions[1] + \
                     self.exp_spectra_raw['CuO'] * cu_fractions[2]

        # self.mixed_cum_spec = self.mixed_cum_spec/max(self.mixed_cum_spec)



        self.mixed_cum_spec = self.exp_spectra_cumulative['Cu Metal'] * cu_fractions[0] + \
                     self.exp_spectra_cumulative['Cu2O'] * cu_fractions[1] + \
                     self.exp_spectra_cumulative['CuO'] * cu_fractions[2]

        self.mixed_cum_spec = self.mixed_cum_spec/max(self.mixed_cum_spec)

        if show_plots:
            plt.figure(figsize=(6,5))
            plt.title('Mixed Spectrum Components', fontsize = 24)
            plt.plot(self.final_energies, mixed_spec, label = 'Mixed Spectrum', linewidth = 3)
            plt.plot(self.final_energies, self.exp_spectra_raw['Cu Metal'] * cu_fractions[0], label = 'Cu Metal', linewidth = 3)
            plt.plot(self.final_energies, self.exp_spectra_raw['Cu2O'] * cu_fractions[1], label = 'Cu2O', linewidth = 3)
            plt.plot(self.final_energies, self.exp_spectra_raw['CuO'] * cu_fractions[2], label = 'CuO', linewidth = 3)
            plt.legend(fontsize = 16, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()



        if show_plots:
            plt.figure(figsize=(6,5))
            plt.title('Mixed Cumulative Spectrum Components', fontsize = 24)
            plt.plot(self.final_energies, self.mixed_cum_spec, label = 'Mixed Cumulative Spectrum', linewidth = 3)
            plt.plot(self.final_energies, self.exp_spectra_cumulative['Cu Metal'] * cu_fractions[0], label = 'Cu Metal', linewidth = 3)
            plt.plot(self.final_energies, self.exp_spectra_cumulative['Cu2O'] * cu_fractions[1], label = 'Cu2O', linewidth = 3)
            plt.plot(self.final_energies, self.exp_spectra_cumulative['CuO'] * cu_fractions[2], label = 'CuO', linewidth = 3)
            plt.legend(fontsize = 16, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

        self.mixed_cum_spec = self.mixed_cum_spec / max(self.mixed_cum_spec)

        if show_predicted_spectrum:
            plt.figure(figsize=(6,5))
            plt.plot(self.final_energies, self.mixed_cum_spec, linewidth = 3)
            plt.title('Predicted Spectrum', fontsize = 24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

        pred = self.rf_model.predict([np.asarray(self.mixed_cum_spec)])
        predictions_full = []
        trees = self.rf_model.estimators_
        for tree in trees:
            predictions_full.append(tree.predict([np.asarray(self.mixed_cum_spec)]))


        predictions_full = np.asarray(predictions_full).T[0]
        predictions_std = np.std(predictions_full)
        if print_predictions:
            print('prediction = ' + str(round(pred[0], 2)))
            print('prediction std = ' + str(round(predictions_std, 2)))

        if show_plots:
            plt.figure(figsize=(6,5))
            hist = plt.hist(predictions_full, edgecolor='k', facecolor='grey', fill=True, linewidth=3)
            height = max(hist[0])

            # plt.vlines(np.median(predictions_full), 0, height,
            # color = 'purple', label = 'Median Prediction', linewidth = 5)
            plt.vlines(true_val, true_val,
                       height, color='blue', label='Labeled BV', linewidth=5)
            plt.vlines(pred[0], 0, height, color='red', label='Average Prediction', linewidth=5, linestyle=':')
            mean = pred
            std = predictions_std
            low = mean - std
            high = mean + std

            plt.hlines(height / 2, high, low, color='limegreen', linewidth=5, label='Std From Mean')
            plt.vlines(high, height / 2 - height * 0.075, height / 2 + height * 0.075, color='limegreen', linewidth=5)
            plt.vlines(low, height / 2 - height * 0.075, height / 2 + height * 0.075, color='limegreen', linewidth=5)

            plt.title('Prediction Histogram Mixed Valence', fontsize = 26)

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel('Prediction', fontsize=20)
            plt.ylabel('Count', fontsize=20)
            plt.legend(fontsize = 16, loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.plot(np.arange(0,max(errors),0.1), np.arange(0,max(errors),0.1), color = 'k', linewidth = 3, linestyle = '--')
            plt.show()

        self.prediction = round(pred[0], 2)
        self.prediction_std = round(predictions_std, 2)
        self.true_val = true_val

        self.mixed_valent_pred.append([self.prediction, self.prediction_std, self.true_val])

    def predict_Cu_non_integers(self, cu_metal=1, cu2o=0, cuo=0, cu_int_indicies = (1640, 1471, 3011),
                                show_standards = False, show_plots = False, energy_col ='new Scaled Energies use',
                                theory_col = 'TEAM_1_aligned_925_970',
                                predict_col = 'Cumulative_Spectra_TEAM_1_aligned_925_970',
                                show_predictions = False):

        cu_metal = float(cu_metal)
        cu2o = float(cu2o)
        cuo = float(cuo)


        if show_standards:
            titles = ['Cu metal', 'Cu2O', 'CuO']
            for i in range(0,3):
                plt.plot(self.spectra_df.iloc[cu_int_indicies[i]][energy_col],
                         self.spectra_df.iloc[cu_int_indicies[i]][theory_col])
                plt.title(titles[i])
                plt.show()

        cu_fractions = [cu_metal, cu2o, cuo]
        true_val = (cu_metal*0 + cu2o*1 + cuo*2)/sum(cu_fractions)

        energies = self.spectra_df.iloc[cu_int_indicies[0]][energy_col]



        mixed_spec = self.spectra_df.iloc[cu_int_indicies[0]][theory_col] * cu_fractions[0] + \
                     self.spectra_df.iloc[cu_int_indicies[1]][theory_col] * cu_fractions[1] + \
                     self.spectra_df.iloc[cu_int_indicies[2]][theory_col] * cu_fractions[2]
        if show_plots:
            plt.figure(figsize=(6,5))
            plt.title('Mixed Spectrum Components', fontsize = 24)
            plt.plot(energies, mixed_spec, label = 'Mixed Spectrum', linewidth = 3)
            plt.plot(energies, self.spectra_df.iloc[cu_int_indicies[0]][theory_col] * cu_fractions[0], label = 'Cu Metal', linewidth = 3)
            plt.plot(energies, self.spectra_df.iloc[cu_int_indicies[1]][theory_col] * cu_fractions[1], label = 'Cu2O', linewidth = 3)
            plt.plot(energies, self.spectra_df.iloc[cu_int_indicies[2]][theory_col] * cu_fractions[2], label = 'CuO', linewidth = 3)
            plt.legend(fontsize = 16, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

        self.mixed_cum_spec = self.spectra_df.iloc[cu_int_indicies[0]][predict_col] * cu_fractions[0] + \
                     self.spectra_df.iloc[cu_int_indicies[1]][predict_col] * cu_fractions[1] + \
                     self.spectra_df.iloc[cu_int_indicies[2]][predict_col] * cu_fractions[2]

        if show_plots:
            plt.figure(figsize=(6,5))
            plt.title('Mixed Cumulative Spectrum Components', fontsize = 24)
            plt.plot(energies, self.mixed_cum_spec, label = 'Mixed Cumulative Spectrum', linewidth = 3)
            plt.plot(energies, self.spectra_df.iloc[cu_int_indicies[0]][predict_col] * cu_fractions[0], label = 'Cu Metal', linewidth = 3)
            plt.plot(energies, self.spectra_df.iloc[cu_int_indicies[1]][predict_col] * cu_fractions[1], label = 'Cu2O', linewidth = 3)
            plt.plot(energies, self.spectra_df.iloc[cu_int_indicies[2]][predict_col] * cu_fractions[2], label = 'CuO', linewidth = 3)
            plt.legend(fontsize = 16, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

        self.mixed_cum_spec = self.mixed_cum_spec / max(self.mixed_cum_spec)

        if show_plots:
            plt.figure(figsize=(6,5))
            plt.plot(energies, self.mixed_cum_spec, linewidth = 3)
            plt.title('Predicted Spectrum', fontsize = 24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

        pred = self.rf_model.predict([np.asarray(self.mixed_cum_spec)])
        predictions_full = []
        trees = self.rf_model.estimators_
        for tree in trees:
            predictions_full.append(tree.predict([np.asarray(self.mixed_cum_spec)]))


        predictions_full = np.asarray(predictions_full).T[0]
        predictions_std = np.std(predictions_full)
        if show_predictions:
            print('prediction = ' + str(round(pred[0], 2)))
            print('prediction std = ' + str(round(predictions_std, 2)))

        if show_plots:
            plt.figure(figsize=(6,5))
            hist = plt.hist(predictions_full, edgecolor='k', facecolor='grey', fill=True, linewidth=3)
            height = max(hist[0])

            # plt.vlines(np.median(predictions_full), 0, height,
            # color = 'purple', label = 'Median Prediction', linewidth = 5)
            plt.vlines(true_val, true_val,
                       height, color='blue', label='Labeled BV', linewidth=5)
            plt.vlines(pred[0], 0, height, color='red', label='Average Prediction', linewidth=5, linestyle=':')
            mean = pred
            std = predictions_std
            low = mean - std
            high = mean + std

            plt.hlines(height / 2, high, low, color='limegreen', linewidth=5, label='Std From Mean')
            plt.vlines(high, height / 2 - height * 0.075, height / 2 + height * 0.075, color='limegreen', linewidth=5)
            plt.vlines(low, height / 2 - height * 0.075, height / 2 + height * 0.075, color='limegreen', linewidth=5)

            plt.title('Prediction Histogram Mixed Valence', fontsize = 26)

            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel('Prediction', fontsize=20)
            plt.ylabel('Count', fontsize=20)
            plt.legend(fontsize = 16, loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.plot(np.arange(0,max(errors),0.1), np.arange(0,max(errors),0.1), color = 'k', linewidth = 3, linestyle = '--')
            plt.show()

        self.prediction = round(pred[0], 2)
        self.prediction_std = round(predictions_std, 2)
        self.true_val = true_val

        self.mixed_valent_pred.append([self.prediction, self.prediction_std, self.true_val])

    def simulated_mixed_valent(self, catagory = '0-1', colorbar_range = [0.1, 0.5], savefigure=False):
        min_std = colorbar_range[0]
        max_std = colorbar_range[1]
        self.reset_mixed_valent_series()
        first_comp = list(np.linspace(1, 0, 51))
        second_comp = list(np.linspace(0, 1, 51))

        if catagory == '0-1':
            for i in range(0, len(first_comp)):
                self.predict_Cu_non_integers(first_comp[i], second_comp[i], 0, cu_int_indicies=(1640, 2240, 824))
            plt.figure(figsize=(8, 7))
            plt.title('Simulated Mixtures Cu(0) to Cu(I)', fontsize=23, fontweight = 'bold')

        elif catagory == '1-2':
            for i in range(0, len(first_comp)):
                self.predict_Cu_non_integers(0, first_comp[i], second_comp[i], cu_int_indicies=(1640, 2240, 824))
            plt.figure(figsize=(8, 7))
            plt.title('Simulated Mixtures Cu(I) to Cu(II)', fontsize=23, fontweight = 'bold')

        elif catagory == '0-2':
            for i in range(0, len(first_comp)):
                self.predict_Cu_non_integers(first_comp[i], 0, second_comp[i], cu_int_indicies=(1640, 2240, 824))
            plt.figure(figsize=(8, 7))
            plt.title('Simulated Mixtures Cu(0) to Cu(II)', fontsize=23)

        trues = np.asarray(self.mixed_valent_pred).T[2]
        predictions = np.asarray(self.mixed_valent_pred).T[0]
        prediction_std = np.asarray(self.mixed_valent_pred).T[1]
        # print(len(count))

        sc = plt.scatter(trues, predictions, s=100, c=prediction_std, vmin=min_std,
                         vmax=max_std)
        cb = plt.colorbar(sc, label='Prediction Std')


        plt.plot(trues, trues, color='k', linestyle='--', linewidth=3)
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=22, weight='bold')
        text.set_font_properties(font)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(22)
            t.set_weight('bold')
        plt.xticks(fontsize=22, fontweight = 'bold')
        plt.yticks(fontsize=22, fontweight = 'bold')
        plt.xlabel('True Value', fontsize=24, fontweight = 'bold')
        plt.ylabel('Bond Valance Prediction', fontsize=24, fontweight = 'bold')
        if savefigure:
            plt.savefig('Simulated Mixtures ' + catagory + '.pdf', bbox_inches='tight', transparent=True)

    def experimental_mixed_valent(self, catagory = '0-1', smoothing_params = [51,3], show_plots = False,
                                  show_predicted_spectrum = False, colorbar_range = [0.1, 0.5], savefigure=False):

        min_std = colorbar_range[0]
        max_std = colorbar_range[1]
        first_comp = list(np.linspace(1,0,51))
        second_comp = list(np.linspace(0,1,51))
        self.reset_mixed_valent_series()

        if catagory == '0-1':
            for i in range(0, len(first_comp)):
                # print(first_comp[i] + second_comp[i])
                self.predict_Experimental_Cu_non_integers(cu_metal=first_comp[i], cu2o=second_comp[i], cuo=0,
                                                                 show_plots=show_plots, show_predicted_spectrum=show_predicted_spectrum,
                                                             smoothing_params=smoothing_params)
            trues = []
            for i in range(0, len(first_comp)):
                trues.append(first_comp[i] * 0.31 + second_comp[i] * 1.08)
            # trues = np.asarray(test_rf_obj.mixed_valent_pred).T[2]
            plt.figure(figsize=(8, 7))
            plt.title('Experimental Mixtures Cu(0) to Cu(I)', fontsize=22, fontweight = 'bold')
            plt.xticks([0.3,0.5,0.7,0.9,1.1])
            plt.yticks([0.3,0.5,0.7,0.9,1.1])
        elif catagory == '1-2':
            for i in range(0, len(first_comp)):
                # print(first_comp[i] + second_comp[i])
                self.predict_Experimental_Cu_non_integers(cu_metal=0, cu2o=first_comp[i], cuo=second_comp[i],
                                                                 show_plots=show_plots, show_predicted_spectrum=show_predicted_spectrum,
                                                                 smoothing_params=smoothing_params)
            trues = []
            for i in range(0, len(first_comp)):
                trues.append(first_comp[i] * 1.08 + second_comp[i] * 1.99)
            # trues = np.asarray(test_rf_obj.mixed_valent_pred).T[2]
            plt.figure(figsize=(8, 7))
            plt.title('Experimental Mixtures Cu(I) to Cu(II)', fontsize=22, fontweight = 'bold')


        elif catagory == '0-2':
            for i in range(0, len(first_comp)):
                # print(first_comp[i] + second_comp[i])
                self.predict_Experimental_Cu_non_integers(cu_metal=first_comp[i], cu2o=0, cuo=second_comp[i],
                                                                 show_plots=show_plots, show_predicted_spectrum=show_predicted_spectrum,
                                                                 smoothing_params=smoothing_params)
            trues = []
            for i in range(0, len(first_comp)):
                trues.append(first_comp[i] * 0.31 + second_comp[i] * 1.99)
            plt.figure(figsize=(8, 7))
            plt.title('Experimental Mixtures Cu(0) to Cu(II)', fontsize=22, fontweight = 'bold')


        predictions = np.asarray(self.mixed_valent_pred).T[0]
        prediction_std = np.asarray(self.mixed_valent_pred).T[1]


        sc = plt.scatter(trues, predictions, s=125, c=prediction_std, vmin=min_std,
                         vmax=max_std)
        cb = plt.colorbar(sc, label='Prediction Std')
        # print(len(count))
        plt.plot(trues, trues, color='k', linestyle='--', linewidth=3)
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=22, weight='bold')
        text.set_font_properties(font)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(22)
            t.set_weight('bold')
        plt.xticks(fontsize=22, fontweight = 'bold')
        plt.yticks(fontsize=22, fontweight = 'bold')
        plt.xlabel('True Value', fontsize=22, fontweight = 'bold')
        plt.ylabel('Bond Valance Prediction', fontsize=22, fontweight = 'bold')
        if savefigure:
            plt.savefig('Experimental Mixtures ' + catagory + '.pdf', bbox_inches='tight', transparent=True)

    def visualize_shift(self, material = 'All', show_stds = False, show_table = False, savefigure=False):
        if material == 'All':
            mins = []
            maxes = []
            for mat in ['Cu Metal', 'Cu2O', 'CuO']:
                subdf = self.prediction_df.loc[self.prediction_df['Material'] == mat]
                prediction_std = np.asarray(subdf['Predictions Std'])

                mins.append(min(np.asarray(prediction_std)))
                maxes.append(max(np.asarray(prediction_std)))

            min_std = min(mins)
            max_std = max(maxes)

            bv = 0
            for mat in ['Cu Metal', 'Cu2O', 'CuO']:
                full_output = [['Shift (eV)', 'Prediction', 'Prediction STD', 'True Oxidation State']]
                subdf = self.prediction_df.loc[self.prediction_df['Material'] == mat]
                shifts = np.asarray(subdf['Spectrum Energy Shift'])
                ticks = []
                count = 0
                for shift in np.asarray(shifts):
                    if count %6 ==0:
                        ticks.append(shift)
                    count += 1
                predictions = np.asarray(subdf['Prediction'])
                prediction_std = np.asarray(subdf['Predictions Std'])
                plt.figure(figsize=(8, 7))
                # print(len(count))
                sc = plt.scatter(shifts, predictions, s=200, c=prediction_std, vmin = round(min_std-0.05,1),
                            vmax = round(max_std+0.05, 1),)
                cb = plt.colorbar(sc, label='Prediction Std')
                cb.set_ticks(np.linspace(round(min_std-0.05,1), round(max_std+0.05, 1), 5))
                ax = cb.ax
                text = ax.yaxis.label
                font = matplotlib.font_manager.FontProperties(size=26, weight='bold')
                text.set_font_properties(font)
                cb.ax.set_yticklabels([0.1, 0.3, 0.5, 0.7, 0.9], fontsize=26, weight='bold')

                plt.title(mat + ' vs Spectrum Shift', fontsize=28, fontweight='bold')
                plt.xticks(ticks, fontsize=24, fontweight='bold')
                # if mat == 'CuO':
                plt.ylim([0.15,2.3])
                # plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
                plt.yticks(fontsize=24, fontweight='bold')


                plt.xlabel('Spectrum Shift (eV)', fontsize=26, fontweight='bold')
                plt.ylabel('Bond Valance Prediction', fontsize=26, fontweight='bold')
                if savefigure:
                    plt.savefig('XAS Spectrum Shift Analysis ' + mat + '.pdf', bbox_inches='tight', transparent=True)

                for i in range(0, len(shifts)):
                    full_output.append([round(shifts[i],2), round(predictions[i],2), prediction_std[i], bv])
            # shifts, predictions, prediction_std
                if show_table:
                    print(tabulate(full_output, headers='firstrow', tablefmt='fancy_grid'))
                bv += 1


            if show_stds:
                stds = pd.DataFrame(np.asarray([prediction_std, shifts, predictions]).T, columns = ['std', 'shift', 'predictions'])
                print(stds.sort_values('std').head())



        if material == 'Unlabeled':
            subdf = self.prediction_df.loc[self.prediction_df['Material'] == 'Unlabeled Cu']
            shifts = np.asarray(subdf['Spectrum Energy Shift'])
            predictions = np.asarray(subdf['Prediction'])
            prediction_std = np.asarray(subdf['Predictions Std'])
            plt.figure(figsize=(8, 7))
            # print(len(count))
            plt.scatter(shifts, predictions, s=1 / prediction_std * 50, c=prediction_std)
            cb = plt.colorbar(label='Prediction Std')
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(size=22)
            text.set_font_properties(font)
            for t in cb.ax.get_yticklabels():
                t.set_fontsize(22)
            plt.title('Predicted ' + 'Unlabeled Cu' + ' vs Spectrum Shift', fontsize=24)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=18)
            plt.xlabel('Spectrum Shift (eV)', fontsize=22)
            plt.ylabel('Bond Valance Prediction', fontsize=22)
            if show_stds:
                stds = pd.DataFrame(np.asarray([prediction_std, shifts, predictions]).T,
                                    columns=['std', 'shift', 'predictions'])
                print(stds.sort_values('std').head())



    def visualize_smoothings(self, material = 'All', show_stds = False, show_table = False, savefigure = False):
        if material == 'All':
            mins = []
            maxes = []
            for mat in ['Cu Metal', 'Cu2O', 'CuO']:
                subdf = self.prediction_df.loc[self.prediction_df['Material'] == mat]
                prediction_std = np.asarray(subdf['Predictions Std'])

                mins.append(min(np.asarray(prediction_std)))
                maxes.append(max(np.asarray(prediction_std)))

            min_std = min(mins)
            max_std = max(maxes)
            bv = 0

            for mat in ['Cu Metal', 'Cu2O', 'CuO']:
                full_output = [['Smoothing Window (eV)', 'Prediction', 'Prediction STD', 'True Oxidation State']]

                subdf = self.prediction_df.loc[self.prediction_df['Material'] == mat]
                shifts = np.asarray(subdf['Smoothing Window'])
                predictions = np.asarray(subdf['Prediction'])
                prediction_std = np.asarray(subdf['Predictions Std'])
                plt.figure(figsize=(8, 7))
                # print(len(count))
                sc = plt.scatter(shifts*0.03, predictions, s=200, c=prediction_std, vmin=round(min_std - 0.05, 1),
                                 vmax=round(max_std + 0.05, 1))
                plt.vlines(1.5, 0.15, 2.3, linewidth = 3, color = 'darkorange', linestyles=['--'], label = 'Default Smoothing')
                plt.legend(fontsize=16)
                cb = plt.colorbar(sc, label='Prediction Std')
                cb.set_ticks([0.1, 0.25, 0.4, 0.55])
                ax = cb.ax
                text = ax.yaxis.label
                font = matplotlib.font_manager.FontProperties(size=26, weight='bold')
                text.set_font_properties(font)
                for t in cb.ax.get_yticklabels():
                    t.set_fontsize(22)
                    t.set_weight('bold')
                plt.title(mat + ' vs Smoothing', fontsize=28, fontweight='bold')
                plt.xticks(fontsize=24, fontweight='bold')
                # if mat == 'CuO':
                plt.ylim([0.15, 2.3])
                # plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
                plt.yticks(fontsize=24, fontweight='bold')
                plt.xlabel('Smoothing Window (eV)', fontsize=26, fontweight='bold')
                plt.ylabel('Bond Valance Prediction', fontsize=26, fontweight='bold')

                for i in range(0, len(shifts)):
                    full_output.append([shifts[i]*0.03, round(predictions[i], 2), prediction_std[i], bv])
                bv += 1

                if show_stds:
                    stds = pd.DataFrame(np.asarray([prediction_std, shifts, predictions]).T, columns = ['std', 'Smoothing Window', 'predictions'])
                    print(stds.sort_values('std').head())
                if show_table:
                    print(tabulate(full_output, headers='firstrow', tablefmt='fancy_grid'))
                if savefigure:
                    plt.savefig('Spectrum Smoothing Analysis ' + mat + '.png', bbox_inches='tight', transparent=True)

        if material == 'Unlabeled':
            subdf = self.prediction_df.loc[self.prediction_df['Material'] == 'Unlabeled Cu']
            shifts = np.asarray(subdf['Smoothing Window'])
            predictions = np.asarray(subdf['Prediction'])
            prediction_std = np.asarray(subdf['Predictions Std'])
            plt.figure(figsize=(8, 7))
            # print(len(count))
            plt.scatter(shifts, predictions, s=1 / prediction_std * 50, c=prediction_std)
            cb = plt.colorbar(label='Prediction Std')
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(size=22)
            text.set_font_properties(font)
            for t in cb.ax.get_yticklabels():
                t.set_fontsize(22)
            plt.title('Predicted ' + 'Unlabeled Cu' + ' vs Smoothing', fontsize=24)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=18)
            plt.xlabel('Smoothing', fontsize=22)
            plt.ylabel('Bond Valance Prediction', fontsize=22)
            if show_stds:
                stds = pd.DataFrame(np.asarray([prediction_std, shifts, predictions]).T,
                                    columns=['std', 'shift', 'predictions'])
                print(stds.sort_values('std').head())
    def predict_experiment_random_forest(self, exp_spectrum, theory_index, material,
                                         smoothing = 'default', exp_spectrum_type = 'TEAM I',
                                          smoothing_parms = (), exp_scale = 0.1,
                                         points_to_average = 20, cumulative_spectrum = True,
                                         spectrum_energy_shift = 0.0, energies_range = [925, 970],
                                         theory_column = 'XAS_aligned_925_970', print_prediction = True,
                                         use_dnn = False, dnn_model = None, show_plots = False, show_hist = False,
                                         show_inputted_spectrum = True, savefigure = False):

        self.cumulative_spectrum = cumulative_spectrum
        self.theory_index = theory_index
        self.exp_spec_path = exp_spectrum
        self.material = material
        self.points_to_average = points_to_average

        self.spectrum_energy_shift = spectrum_energy_shift

        if exp_spectrum_type == 'TEAM I':
            output = nio.dm.dmReader(exp_spectrum)
            intens = output['data']
            energies = output['coords'][0]

            # plt.figure(figsize=(8, 7))
            if show_plots:
                plt.xticks(fontsize=18, fontweight = 'bold')
                plt.yticks(fontsize=18, fontweight = 'bold')
                plt.ylabel('Intensity', fontsize=20, fontweight = 'bold')
                plt.xlabel('Energy (eV)', fontsize=20, fontweight = 'bold')
                plt.title('Raw Experimental Spectrum', fontsize=22, fontweight = 'bold')
                # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
                # noise = []
                # np.random.seed(23)
                # for entry in intens:
                    # print(entry)
                    # noise.append(np.random.normal(0, 50, size=1)[0])
                # plt.plot(energies, noise, label = 'Noise')
                plt.plot(energies, intens, zorder=10)
                # intens = np.asarray(intens) + np.asarray(noise)
                # plt.plot(energies, intens, label = 'Summed w/Noise')
                plt.show()
                new_noise = []

                # plt.plot(energies, intens/max(intens))
                # plt.plot(energies, intens / max(intens),
                #          label='Experimental Spectrum', linewidth=1, zorder=10)
                # plt.plot(self.spectra_df.iloc[theory_index][self.energy_col],
                #          self.spectra_df.iloc[theory_index][theory_column][0:451] / max(
                #              self.spectra_df.iloc[theory_index][theory_column][0:451]),
                #          label='From FEFF', linewidth=1)
                # plt.legend(fontsize=12)
                # plt.xlim([927, 945])
                # plt.show()


            if smoothing == 'default':
                self.smoothing_params = [51, 3]
                intens = savgol_filter(intens, self.smoothing_params[0], self.smoothing_params[1])
            elif smoothing == 'default series':
                if len(self.smoothing_params) == 0:
                    windows = np.arange(5,151,14)
                    for window in windows:
                        self.smoothing_params.append([window, 3])
                else:
                    print('values already exist in smoothing params, please reset')
                    raise ValueError
            elif smoothing == 'inputted single':
                self.smoothing_params = smoothing_parms
                intens = savgol_filter(intens, self.smoothing_params[0], self.smoothing_params[1])
            elif smoothing == 'inputted series':
                self.smoothing_params = smoothing_parms


        if exp_spectrum_type == 'csv':
            output = pd.read_csv(exp_spectrum)
            # intens_temp = output['Intensity']
            # YOU ARE BASELINE SUBTRACTING THE XAS SPECTRUM!! DO NOT FORGET THIS! This is because
            # the baseline is far above zero due to the axes of the literature extraction. There really isn't
            # another good option. Luckily the noise is low and this isn't an issue
            intens_temp = output['Intensity'] - min(output['Intensity'])
            intens = intens_temp / intens_temp[len(intens_temp) - 5]
            energies = output['Energy (eV)']
            if show_plots:
                plt.xticks(fontsize=18, fontweight = 'bold')
                plt.yticks(fontsize=18, fontweight = 'bold')
                plt.ylabel('Intensity', fontsize=20, fontweight = 'bold')
                plt.xlabel('Energy (eV)', fontsize=20, fontweight = 'bold')
                plt.title('Raw Experimental Spectrum', fontsize=22, fontweight = 'bold')
                # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
                # noise = []
                # np.random.seed(23)
                # for entry in intens:
                    # print(entry)
                    # noise.append(np.random.normal(0, 50, size=1)[0])
                # plt.plot(energies, noise, label = 'Noise')
                plt.plot(energies, intens, zorder=10)
                # intens = np.asarray(intens) + np.asarray(noise)
                # plt.plot(energies, intens, label = 'Summed w/Noise')
                plt.show()
            self.scale_experimental_spectra(intens, energies, scale=exp_scale)
            intens = self.interped_intens
            energies = self.interped_energies

            if smoothing == 'default':
                self.smoothing_params = [51, 3]
                intens = savgol_filter(intens, self.smoothing_params[0],
                                                     self.smoothing_params[1])
            elif smoothing == 'default series':
                if len(self.smoothing_params) == 0:
                    windows = np.arange(5, 151, 14)
                    for window in windows:
                        self.smoothing_params.append([window, 3])
                else:
                    print('values already exist in smoothing params, please reset')
                    raise ValueError
            elif smoothing == 'inputted single':
                # print(material)
                # print('Metal' in material)

                self.smoothing_params = smoothing_parms
                # print(self.smoothing_params[0])
                intens = savgol_filter(intens, self.smoothing_params[0],
                                                     self.smoothing_params[1])
            elif smoothing == 'inputted series':
                self.smoothing_params = smoothing_parms
        if show_plots:
            plt.xticks(fontsize=18, fontweight = 'bold')
            plt.yticks(fontsize=18, fontweight = 'bold')
            plt.ylabel('Intensity', fontsize=20, fontweight = 'bold')
            plt.xlabel('Energy (eV)', fontsize=20, fontweight = 'bold')
            plt.title('Post Smoothing', fontsize=22, fontweight = 'bold')
            # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
            plt.plot(energies, intens / max(intens),
                     label='Experimental Spectrum', linewidth=3, zorder=10)
            # plt.plot(self.spectra_df.iloc[theory_index][self.energy_col],
            #          self.spectra_df.iloc[theory_index][theory_column][0:451] / max(
            #              self.spectra_df.iloc[theory_index][theory_column][0:451]),
            #          label='From FEFF', linewidth=3)
            # plt.xlim([927, 945])
            plt.show()

        self.scale_experimental_spectra(intens, energies, scale = exp_scale)
        if show_plots:
            plt.plot(self.interped_energies, self.interped_intens / max(self.interped_intens),
                     label='Experimental Spectrum', linewidth=3, zorder=10)
            plt.xticks(fontsize=18, fontweight = 'bold')
            plt.yticks(fontsize=18, fontweight = 'bold')
            plt.ylabel('Intensity', fontsize=20, fontweight = 'bold')
            plt.xlabel('Energy (eV)', fontsize=20, fontweight = 'bold')
            plt.title('Post Interpolation', fontsize=22, fontweight = 'bold')
            plt.show()

        if self.spectrum_energy_shift != 0.0:
            diff = self.spectrum_energy_shift * 10
            spectrum_use = list(self.interped_intens)
            energies_use = list(energies)
            energies_min = 924.9
            energies_max = max(energies_use)
            if diff > 0:
                points_to_add = np.zeros((int(diff)))
                for point in points_to_add:
                    spectrum_use.insert(0, point)
                    energies_use.insert(0, energies_min)
                    energies_min = round(energies_min - 0.1, 1)
                spectrum_use = spectrum_use[0:len(self.interped_intens)]
            if diff < 0:
                points_to_add = np.zeros((int(np.abs(diff))))
                intens_max = np.mean(spectrum_use[len(spectrum_use) - points_to_average:len(spectrum_use)])
                for point in points_to_add:
                    energies_max = round(energies_max + 0.1, 1)
                    spectrum_use.append(intens_max)
                    energies_use.append(energies_max)
                # print('Scaled < 0')
                # print(len(spectrum_use[int(np.abs(scale)):len(spectrum_use)]))
                spectrum_use = spectrum_use[int(np.abs(diff)):len(spectrum_use)]

            spectrum_use = np.asarray(spectrum_use)
            self.interped_intens = spectrum_use / np.mean(spectrum_use[len(spectrum_use) - points_to_average:len(spectrum_use)])


        self.broadened_intens = self.interped_intens


        self.broadened_intens =  self.broadened_intens / np.mean(self.broadened_intens[len(self.broadened_intens) - points_to_average:len(self.broadened_intens)])

        self.final_energies = np.arange(energies_range[0], energies_range[1]+0.2, 0.1)[0:((energies_range[1]-energies_range[0])*10)+1]
        energies_interp_use = []
        for i in self.final_energies :
            energies_interp_use.append(round(i, 1))
            # print(round(i,1))
        self.final_energies = energies_interp_use
        # print(len(self.interped_intens))
        # print(len(self.broadened_intens))
        # print(len(self.interped_energies))
        f = interp1d(self.interped_energies, self.broadened_intens)

        self.intensities_final = f(self.final_energies)

        if show_plots:
            plt.xticks(fontsize=18, fontweight = 'bold')
            plt.yticks(fontsize=18, fontweight = 'bold')
            plt.ylabel('Intensity', fontsize=20, fontweight = 'bold')
            plt.xlabel('Energy (eV)', fontsize=20, fontweight = 'bold')
            plt.title('Post Cropping', fontsize=22, fontweight = 'bold')
            # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
            plt.plot(self.final_energies, self.intensities_final,
                     label='Experimental Spectrum', linewidth=4, zorder=10)
            # plt.plot(self.spectra_df.iloc[theory_index][self.energy_col],
            #          self.spectra_df.iloc[theory_index][theory_column][0:451],
            #          label='From FEFF', linewidth=4)

            # plt.xlim([927, 945])

            # if material =='Cu2O':
        # if show_plots:
        #     plt.figure(figsize=(8, 7))


        #     plt.xlabel('Energy (eV)', fontsize=36, fontweight='bold')
        #     plt.ylabel(' Intensty', fontsize = 36, fontweight='bold')
        #     plt.xticks([930, 950, 970], fontsize=36, fontweight='bold')
        #     plt.yticks([], fontsize=36, fontweight='bold')


        #     plt.title('EELS Spectrum ' + material, fontsize=36, fontweight='bold')
        # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
        self.intensities_final = self.intensities_final / np.mean(self.intensities_final[len(self.intensities_final) - points_to_average:len(self.intensities_final)])
        # if show_plots:
            # plt.plot(self.final_energies, self.intensities_final,
            #          label='Experimental Spectrum', linewidth=4, zorder=10)
            # plt.plot(self.spectra_df.iloc[theory_index][self.energy_col],
            #          self.spectra_df.iloc[theory_index][theory_column][0:451],
            #          label='From FEFF', linewidth=4)

            # plt.xlim([927, 945])


            # if material =='Cu2O':
            # plt.savefig(material + ' Exp Spectrum.pdf', bbox_inches='tight', transparent=True)
            # plt.show()

            # plt.xticks(fontsize=18, fontweight = 'bold')
            # plt.yticks(fontsize=18, fontweight = 'bold')
            # plt.ylabel('Intensity', fontsize=20, fontweight = 'bold')
            # plt.xlabel('Energy (eV)', fontsize=20, fontweight = 'bold')
            # plt.title('Cumulative Spectrum', fontsize=22, fontweight = 'bold')

        # self.intensities_final = self.intensities_final - min(self.intensities_final)

        if self.cumulative_spectrum:
            temp_intens = []
            for k in range(0, len(self.intensities_final)):
                temp_intens.append(sum(self.intensities_final[0:k]))
            self.intensities_final = temp_intens/max(temp_intens)
        # print(self.intensities_final)
        if show_inputted_spectrum:
            plt.figure(figsize=(8, 7))
            plt.xlabel('Energy (eV)', fontsize=36, fontweight='bold')
            plt.ylabel(' Intensty', fontsize = 36, fontweight='bold')
            plt.xticks([930, 950, 970], fontsize=36, fontweight='bold')
            # plt.yticks([0, 0.5, 1], fontsize=36, fontweight='bold')
            plt.yticks([])
            plt.title('Cumulative Spectrum ' + material, fontsize=29, fontweight='bold')

            # self.intensities_final = self.intensities_final - min(self.intensities_final)

            # print(self.intensities_final)
            plt.plot(self.final_energies, self.intensities_final,
                          linewidth=4, zorder=10)


            # if material =='Cu2O':
            if savefigure:
                plt.savefig(material + ' Outline Cumulative Spectrum.pdf', bbox_inches='tight', transparent=True)


        # plt.hlines(1, energies_range[0], energies_range[1], color = 'k')
        # plt.plot(self.spectra_df.iloc[theory_index][self.energy_col],
        #          self.spectra_df.iloc[theory_index][self.predicted_col][0:451],
        #          label='From FEFF', linewidth=4)

        # gE=spectrum(xin,yin,0.9,xz[0:401])
        # plt.plot(xin[100:501], gE/gE[len(gE)-20], linewidth = 3, label = 'Broadened')
        # plt.legend(fontsize=20)

        rf_model = self.rf_model

        if use_dnn:
            updated_spectra_test = []
            spec = self.intensities_final
            # print(spec)
            # spec = preprocessing.scale(spec)
            # spec = spec[0]
            plt.figure(figsize=(8,6))
            plt.plot(spec)
            plt.show()
            for i in [spec]:
                temp_spectrum = []
                for k in i:
                    temp_spectrum.append(np.float32(k))
                temp_spectrum = [temp_spectrum]
                updated_spectra_test.append(np.asarray(temp_spectrum))
            # plt.plot(updated_spectra_test[0][0])
            pred = dnn_model.predict(updated_spectra_test)
            if print_prediction:
                print('prediction = ' + str(round(pred[0][0], 2)))
                # print('prediction std = ' + str(round(predictions_std, 2)))

        else:
            pred = self.rf_model.predict([np.asarray(self.intensities_final)])
            predictions_full = []
            trees = rf_model.estimators_
            for tree in trees:
                predictions_full.append(tree.predict([np.asarray(self.intensities_final)]))


            predictions_full = np.asarray(predictions_full).T[0]
            predictions_std = np.std(predictions_full)
            if print_prediction:
                print('prediction = ' + str(round(pred[0], 2)))
                print('prediction std = ' + str(round(predictions_std, 2)))
                print(' ')

       
        # if show_plots:
            # if 'Metal' in material:
            #     plt.title('Prediction Histogram ' + material[0:10], fontsize=32)
            #     plt.title('Prediction Histogram', fontsize=32)

            # elif 'Cu2O' in material:
            #     plt.title('Prediction Histogram ' + material[0:4], fontsize=32)
            #     plt.title('Prediction Histogram', fontsize=32)

            # elif 'CuO' in material:
            #     plt.title('Prediction Histogram ' + material[0:3], fontsize=32)
            #     plt.title('Prediction Histogram', fontsize=32)

            # elif 'FeS' in material:
            #     plt.title('Prediction Histogram ' + material[0:3], fontsize=32)
            # elif 'Fe2O3' in material:
            #     plt.title('Prediction Histogram ' + material[0:5], fontsize=32)
            # elif 'Fe3O4' in material:
            #     plt.title('Prediction Histogram ' + material[0:5], fontsize=32)

        # plt.xticks(fontsize=28)
        # plt.yticks(fontsize=28)
        # plt.xlabel('Prediction', fontsize=30)
        # plt.ylabel('Num Trees', fontsize=30)
        # plt.legend(fontsize = 20)
        # plt.savefig('Prediction Histogram ' + material[0:5] + '.pdf', bbox_inches='tight', transparent=True)
        # plt.legend(fontsize=20,  loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.plot(np.arange(0,max(errors),0.1), np.arange(0,max(errors),0.1), color = 'k', linewidth = 3, linestyle = '--')
        # plt.show()
        if show_hist:
            plt.figure(figsize=(8, 7))
            plt.title('Prediction Histogram ' + material[0:4], fontsize=30, fontweight='bold')
            hist = plt.hist(predictions_full, edgecolor='k', facecolor='grey', fill=True, linewidth=3)
            height = max(hist[0])

            # plt.vlines(np.median(predictions_full), 0, height,
            # color = 'purple', label = 'Median Prediction', linewidth = 5)
            plt.vlines(self.spectra_df.iloc[theory_index]['BV Used For Alignment'],
                       self.spectra_df.iloc[theory_index]['BV Used For Alignment'],
                       height, color='blue', label='True Value', linewidth=5)
            plt.vlines(pred[0], 0, height, color='red', label='Prediction', linewidth=5, linestyle=':')
            mean = pred
            std = predictions_std
            low = mean - std
            high = mean + std

            plt.hlines(height / 2, high, low, color='limegreen', linewidth=5, label='Std')
            plt.vlines(high, height / 2 - height * 0.075, height / 2 + height * 0.075, color='limegreen', linewidth=5)
            plt.vlines(low, height / 2 - height * 0.075, height / 2 + height * 0.075, color='limegreen', linewidth=5)
            plt.xlabel('Prediction', fontsize=36, fontweight='bold')
            plt.ylabel('Num Trees', fontsize=36, fontweight='bold')

            plt.xticks([0, 0.5, 1, 1.5, 2, 2.5], fontsize=36, fontweight='bold')
            plt.yticks([100, 200, 300], fontsize=36, fontweight='bold')
            font = font_manager.FontProperties(
                weight='bold',
                style='normal', size=26)
            plt.legend(prop=font)
            if savefigure:
                plt.savefig('Prediction Histogram ' + material[0:5] + '.pdf', bbox_inches='tight', transparent=True)
            # plt.legend(fontsize=20,  loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.plot(np.arange(0,max(errors),0.1), np.arange(0,max(errors),0.1), color = 'k', linewidth = 3, linestyle = '--')
            plt.show()

        self.prediction = round(pred[0], 2)
        self.prediction_std = round(predictions_std, 2)
        self.true_val = self.spectra_df.iloc[theory_index]['BV Used For Alignment']

    def show_feature_importances(self, material_type, savefigure=False):
        plt.figure(figsize=(8, 6))
        plt.xticks([930, 950, 970], fontsize=32, fontweight='bold')
        plt.yticks([0.0, 0.05, 0.1, 0.15], fontsize=32, fontweight='bold')
        # plt.ylabel('Normalized Variance Reduction', fontsize=30)
        plt.ylabel('Importance', fontsize=36, fontweight='bold')
        plt.xlabel('Energy (eV)', fontsize=36, fontweight='bold')
        # plt.title('Feature Importances ' + self.predicted_col, fontsize=22)
        plt.title('Feature Importances', fontsize=36, fontweight='bold')

        plt.plot(np.asarray(self.spectra_df.iloc[0][self.energy_col]),
                 self.rf_model.feature_importances_, linewidth=5, label=material_type, color='#ff7f0e')
        # plt.legend(fontsize=24)
        if savefigure:
            plt.savefig('Feature Importances.pdf', bbox_inches='tight', transparent=True)
        plt.show()


    def show_errors_histogram(self, nbins = 50, title = 'Error Histogram', show_rmse = True, show_type = 'Abs Error',
                              savefigure=False, error_df = None, yticks = None):

        if type(error_df) == type(None):
            error_df = self.rf_error_df



        errors = error_df['Errors']

        MSE = np.square(errors).mean()
        RMSE = math.sqrt(MSE)
        print('RMSE ' + str(RMSE))
        plt.figure(figsize=(8, 6))
        plt.title(title, fontsize=24)
        if show_type == 'Abs Error':
            hist = plt.hist(errors, bins=nbins)
        elif show_type == 'MSE':
            hist = plt.hist(np.square(errors), bins=nbins)

        if show_rmse:
            plt.vlines(RMSE, max(hist[0]), min(hist[0]), color='limegreen', linewidth=5, label='RMSE')
            plt.text(RMSE + 0.25, max(hist[0]) - 0.1 * max(hist[0]), 'RMSE = ' + str(round(RMSE, 3)),
                     horizontalalignment='left', fontsize=28, fontweight='bold')
        plt.xticks([0.0, 0.4, 0.8, 1.2], fontsize=32, fontweight='bold')
        if type(yticks) != list:
            plt.yticks([0.0, 250, 500, 750], fontsize=32, fontweight='bold')

        else:
            plt.yticks(yticks, fontsize=32, fontweight='bold')
        # plt.ylabel('Normalized Variance Reduction', fontsize=30)
        plt.ylabel('Frequency', fontsize=36, fontweight='bold')
        plt.xlabel('Abs Error', fontsize=36, fontweight='bold')
        # plt.title('Feature Importances ' + self.predicted_col, fontsize=22)
        plt.title('Error Histogram', fontsize=36, fontweight='bold')
        if savefigure:
            plt.savefig('Error Histogram.pdf',  bbox_inches='tight', transparent=True)
        plt.show()


    def show_mixture_samples_accuracy(self):
        mixture_ids = []
        for i in range(0, len(self.spectra_df)):
            mp_id = self.spectra_df.iloc[i]['mpid_string']
            if type(mp_id) == list:
                for m_id in mp_id:
                    if m_id not in mixture_ids:
                        mixture_ids.append(m_id)

        mixture_df_slice = self.spectra_df.loc[self.spectra_df['mpid_string'].isin(mixture_ids)]
        error_df_mixtures = self.predict_set_of_spectra(mixture_df_slice)

        self.show_r2(error_df=error_df_mixtures)
        self.show_errors_histogram(error_df=error_df_mixtures, yticks=[0, 50, 100, 150])

    def plot_errors_vs_std(self, scatter_spot_multiplier = 15):
        pred = []
        true = []
        condensed_stds = []
        count = []
        for i in np.asarray(self.rf_error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts().index):
            pred.append(round(i[0], 1))
            true.append(round(i[1], 1))
            condensed_stds.append(np.mean(self.rf_error_df.loc[
                                              (self.rf_error_df['Predictions Rounded'] == round(i[0], 1)) & (
                                                      self.rf_error_df['Labels Test Rounded'] == round(i[1], 1))][
                                              'Predictions Std']))
        for k in np.asarray(self.rf_error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts()):
            count.append(k)
        count = np.asarray(count)

        plt.figure(figsize=(8, 7))
        plt.scatter(np.abs(np.asarray(pred) - np.asarray(true)), condensed_stds, s=count * scatter_spot_multiplier,
                    c=count)
        min_plot = round(min(np.abs(np.asarray(pred) - np.asarray(true))), 2)
        max_plot = round(max(np.abs(np.asarray(pred) - np.asarray(true))), 2)
        plt.plot([min_plot, max_plot], [min_plot, max_plot], color='k', linewidth=3, linestyle='--')

        cb = plt.colorbar(label='Num Predictions')
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=22)
        text.set_font_properties(font)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(22)
        plt.title('Errors vs Prediction Std', fontsize=18)

        plt.xlim([-0.1, 1.4])
        plt.ylim([-0.1, 1.4])

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Error in BV Prediction', fontsize=16)
        plt.ylabel('Prediction Std', fontsize=16)
        plt.figure(figsize=(8, 7))
        plt.show()

    def show_r2(self, scatter_spot_multiplier = 15, savefigure=False, error_df = None, show_value_counts_plot = False):
        if type(error_df) == type(None):
            error_df = self.rf_error_df
        print('num spectra = ' +str(len(error_df)))

        print('model accuracy (R^2) on simulated test data ' + str(self.accuracy))

        true = []
        pred = []
        count = []
        condensed_stds = []
        for i in np.asarray(error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts().index):
            pred.append(round(i[0], 1))
            true.append(round(i[1], 1))
            condensed_stds.append(np.mean(error_df.loc[
                                              (error_df['Predictions Rounded'] == round(i[0], 1)) & (
                                                      error_df['Labels Test Rounded'] == round(i[1], 1))][
                                              'Predictions Std']))
        for k in np.asarray(error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts()):
            count.append(k)

        count = np.asarray(count)

        if show_value_counts_plot:
            plt.figure(figsize=(8, 6))
            plt.scatter(true, pred, s=count * scatter_spot_multiplier, c=count)

            cb = plt.colorbar(label='Num Predictions')
            ax = cb.ax
            text = ax.yaxis.label
            print(text)
            font = matplotlib.font_manager.FontProperties(size=32)
            text.set_font_properties(font)
            cb.ax.set_yticklabels(fontsize=32, weight='bold')

            min_plot = round(min(error_df['Labels Test']) - 0.5, 0)
            max_plot = round(max(error_df['Labels Test']) + 1.5, 0)
            plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3,
                     linestyle='--')
            plt.title('Predicted vs True', fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.ylabel('Bond Valance Prediction', fontsize=22)
            plt.xlabel('True Bond Valance', fontsize=22)
            plt.show()

        plt.figure(figsize=(8, 6))

        plt.xticks([0.0, 1.0, 2.0, 3.0], fontsize=32, fontweight='bold')
        plt.yticks([0.0, 1.0, 2.0, 3.0], fontsize=32, fontweight='bold')
        # plt.ylabel('Normalized Variance Reduction', fontsize=30)
        plt.ylabel('Predicted Valence', fontsize=36, fontweight='bold')
        plt.xlabel('True Valence', fontsize=36, fontweight='bold')
        # plt.title('Feature Importances ' + self.predicted_col, fontsize=22)
        plt.title('Prediction vs True', fontsize=36, fontweight='bold')

        min_plot = round(min(error_df['Labels Test']) - 0.5, 0)
        max_plot = round(max(error_df['Labels Test']) + 1.5, 0)
        plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3,
                 linestyle='--')
        plt.scatter(true, pred, s=count * scatter_spot_multiplier, c=condensed_stds)
        cb = plt.colorbar(label='Prediction Std', ticks = [0.1, 0.2, 0.3, 0.4, 0.5])
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=28, weight='bold')
        text.set_font_properties(font)
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_weight("bold")
            t.set_fontsize(32)

        if savefigure:
            plt.savefig('r^2 plot.pdf',  bbox_inches='tight', transparent=True)

    def predictions_from_threshold(self, threshold, show_plot=False):
        low_std = self.rf_error_df.loc[self.rf_error_df['Predictions Std'] <= threshold]
        predictions = low_std['Predictions']
        labels_test = low_std['Labels Test']
        predictions_std = low_std['Predictions Std']
        if show_plot:
            plt.figure(figsize=(8, 6))
            min_plot = round(min(labels_test) - 0.5, 0)
            max_plot = round(max(labels_test) + 1.5, 0)
            plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3,
                     linestyle='--')
            plt.title('Predicted vs True Bond Valance', fontsize=24)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.xlabel('Bond Valance Prediction', fontsize=22)
            plt.ylabel('True Bond Valance', fontsize=22)

            plt.scatter(predictions, np.asarray(labels_test), c=predictions_std, s=50)
            cb = plt.colorbar(label='Prediction Std')
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(size=22)
            text.set_font_properties(font)
            for t in cb.ax.get_yticklabels():
                t.set_fontsize(22)
            plt.show()

        MSE = np.square(np.subtract(labels_test, predictions)).mean()

        RMSE = math.sqrt(MSE)
        # print("Root Mean Square Error:\n")
        # print(RMSE)

        errors = np.abs(np.subtract(labels_test, predictions))

        # self.rf_error_df = self.rf_error_df.loc[self.rf_error_df['Predictions Std'] <= threshold]
        # predictions = low_std['Predictions']
        # labels_test = low_std['Labels Test']
        # predictions_std = low_std['Predictions Std']
        if show_plot:
            plt.show()
            plt.figure(figsize=(8, 7))
            plt.title('Error Histogram', fontsize=18)
            hist = plt.hist(errors, bins=50)
            plt.vlines(RMSE, max(hist[0]), min(hist[0]), color='limegreen', linewidth=5, label='RMSE')
            plt.text(RMSE + 0.25, max(hist[0]) - 0.1 * max(hist[0]), 'RMSE = ' + str(round(RMSE, 3)),
                     horizontalalignment='center', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Error', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.show()

            # print(labels_test.index)
            # print(spectra_df.iloc[labels_test.index]['task id (MAY BE WRONG)'])

            plt.figure(figsize=(8, 7))
            true = []
            pred = []
            count = []
            condensed_stds = []
            for i in np.asarray(error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts().index):
                pred.append(round(i[0], 1))
                true.append(round(i[1], 1))
                condensed_stds.append(np.mean(error_df.loc[
                                                  (error_df['Predictions Rounded'] == round(i[0], 1)) & (
                                                          error_df['Labels Test Rounded'] == round(i[1], 1))][
                                                  'Predictions Std']))

            for k in np.asarray(error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts()):
                count.append(k)
            count = np.asarray(count)
            # print(count)
            plt.scatter(pred, true, s=count * 18, c=count)
            # plt.xlim([1.9, 3.1])
            # plt.ylim([1.9, 3.1])
            cb = plt.colorbar(label='Num Predictions')
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(size=22)
            text.set_font_properties(font)
            for t in cb.ax.get_yticklabels():
                t.set_fontsize(22)
            min_plot = round(min(labels_test) - 0.5, 0)
            max_plot = round(max(labels_test) + 1.5, 0)
            plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3,
                     linestyle='--')
            plt.title('Predicted vs True Bond Valance', fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel('Bond Valance Prediction', fontsize=22)
            plt.ylabel('True Bond Valance', fontsize=22)
            plt.show()

            plt.figure(figsize=(8, 7))
            # print(len(count))
            plt.scatter(pred, true, s=count * 18, c=condensed_stds)
            # plt.xlim([1.9, 3.1])
            # plt.ylim([1.9, 3.1])
            cb = plt.colorbar(label='Prediction Std')
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(size=22)
            text.set_font_properties(font)
            for t in cb.ax.get_yticklabels():
                t.set_fontsize(22)

            min_plot = round(min(labels_test) - 0.5, 0)
            max_plot = round(max(labels_test) + 1.5, 0)
            plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3,
                     linestyle='--')
            plt.title('Predicted vs True Bond Valance', fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel('Bond Valance Prediction', fontsize=22)
            plt.ylabel('True Bond Valance', fontsize=22)
            plt.show()

            plt.figure(figsize=(8, 7))
            plt.scatter(np.abs(np.asarray(pred) - np.asarray(true)), condensed_stds, s=count * 18, c=count)
            cb = plt.colorbar(label='Num Predictions')
            ax = cb.ax
            text = ax.yaxis.label
            font = matplotlib.font_manager.FontProperties(size=22)
            text.set_font_properties(font)
            for t in cb.ax.get_yticklabels():
                t.set_fontsize(22)
            plt.title('Errors vs Prediction Std', fontsize=18)
            min_plot = round(min(np.abs(np.asarray(pred) - np.asarray(true))), 1)
            max_plot = round(max(np.abs(np.asarray(pred) - np.asarray(true))), 1)
            plt.xlim([-0.2, 2.5])
            plt.ylim([-0.2, 2.5])
            plt.plot(np.arange(0, 2.4, 0.1), np.arange(0, 2.4, 0.1), color='k', linewidth=3, linestyle='--')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Error in BV Prediction', fontsize=16)
            plt.ylabel('Prediction Std', fontsize=16)
            plt.figure(figsize=(8, 7))

        return [r2_score(labels_test, predictions), len(low_std), RMSE]

    def visualize_all_thresholds(self, thresholds, ylims = (0.7, 1.02), width_multiplier = 0.04, text_height = 0.005,
                                 text_fontsize = 12, savefigure = False, show_type = 'percentage_predicted',
                                 color_scheme = 'viridis', yticks_to_use = ()):

        r2s = []
        percentage_predicted = []
        rmse = []
        for thresh in thresholds:
            output = self.predictions_from_threshold(thresh, show_plot=False)
            # print(output[0])
            # print(thresh)
            r2s.append(output[0])
            # print(output[1])
            percentage_predicted.append(output[1] / len(self.rf_error_df))
            rmse.append(output[2])


        data_x = thresholds
        data_hight = np.asarray(r2s)
        data_color = percentage_predicted
        data_color_temp = np.asarray(data_color) - min(data_color)
        data_color_normalized = [x / max(data_color_temp) for x in data_color_temp]

        fig, ax = plt.subplots(figsize=(8, 6))

        my_cmap = plt.cm.get_cmap(color_scheme)
        colors = my_cmap(data_color_normalized)

        plt.bar(data_x, data_hight, color=colors, width=np.asarray(data_color) * width_multiplier)

        sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(min(data_color), max(data_color)))

        sm.set_array([])

        cb = plt.colorbar(sm, label='Fraction Test Set Predicted')
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=22, weight='bold')
        text.set_font_properties(font)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(22)
            t.set_weight('bold')

        plt.xticks(fontsize=22, fontweight = 'bold')
        if len(yticks_to_use) == 0:
            plt.yticks(fontsize=22, fontweight = 'bold')
        else:
            plt.yticks(yticks_to_use, fontsize = 22, fontweight = 'bold')
        plt.xlabel('Std Cutoff', fontsize=22, fontweight = 'bold')
        plt.ylabel('Accuracy (R^2)', fontsize=22, fontweight = 'bold')
        plt.title('Std Cutoff Threshold vs Accuracy', fontsize=22, fontweight = 'bold')
        plt.ylim(ylims)
        for i in range(0, len(r2s)):
            if show_type == 'percentage_predicted':
                plt.text(thresholds[i], r2s[i] + text_height, str(round(percentage_predicted[i], 2)), horizontalalignment='center',
                 fontsize=text_fontsize, fontweight = 'bold')
            if show_type == 'RMSE':
                plt.text(thresholds[i], r2s[i] + text_height, str(round(rmse[i], 2)), horizontalalignment='center',
                 fontsize=text_fontsize, fontweight = 'bold')
        if savefigure:
            plt.savefig(show_type + 'bar_chart_with_colorbar.png', bbox_inches='tight')
        plt.show()

    def augment_df_with_mixtures(self, len_mixtures=100, len_combinations=20):
        cu_metal = list(np.linspace(1, 0, 101))
        cu_1 = list(np.linspace(0, 1, 101))
        cu_2 = list(np.linspace(0, 1, 101))

        zeros = self.spectra_df.loc[self.spectra_df['BV Used For Alignment'] == 0.0]
        ones = self.spectra_df.loc[self.spectra_df['BV Used For Alignment'] == 1.0]
        twos = self.spectra_df.loc[self.spectra_df['BV Used For Alignment'] == 2.0]

        mixture_df = self.add_mixed_valent_spectra(zeros, ones, twos, mixture_type='all', len_mixtures=len_mixtures,
                                              len_combinations=len_combinations, cu_metal=cu_metal, cu_1=cu_1,
                                                   cu_2=cu_2)
        mixture_0_1 = self.add_mixed_valent_spectra(zeros, ones, twos, mixture_type='0-1', len_mixtures=len_mixtures,
                                               len_combinations=len_combinations, cu_metal=cu_metal, cu_1=cu_1,
                                                    cu_2=cu_2)
        mixture_1_2 = self.add_mixed_valent_spectra(zeros, ones, twos, mixture_type='1-2', len_mixtures=len_mixtures,
                                               len_combinations=len_combinations, cu_metal=cu_metal, cu_1=cu_1,
                                                    cu_2=cu_2)

        mixture_df_use = pd.DataFrame(mixture_df, columns=['TEAM_1_aligned_925_970', 'mpid_string',
                                                           'pretty_formula', 'BV Used For Alignment',
                                                           'Cumulative_Spectra_TEAM_1_aligned_925_970',
                                                           'new Scaled Energies use', 'Mixture Components'])

        mixture_df_use_0_1 = pd.DataFrame(mixture_0_1, columns=['TEAM_1_aligned_925_970', 'mpid_string',
                                                                'pretty_formula', 'BV Used For Alignment',
                                                                'Cumulative_Spectra_TEAM_1_aligned_925_970',
                                                                'new Scaled Energies use', 'Mixture Components'])

        mixture_df_use_1_2 = pd.DataFrame(mixture_1_2, columns=['TEAM_1_aligned_925_970', 'mpid_string',
                                                                'pretty_formula', 'BV Used For Alignment',
                                                                'Cumulative_Spectra_TEAM_1_aligned_925_970',
                                                                'new Scaled Energies use', 'Mixture Components'])

        mixture_df_full = pd.concat([mixture_df_use, mixture_df_use_0_1, mixture_df_use_1_2])
        self.spectra_df = pd.concat([self.spectra_df, mixture_df_full])
        self.spectra_df.reset_index(inplace=True)

    def add_mixed_valent_spectra(self, zeros, ones, twos, mixture_type='all', len_mixtures=100, len_combinations=20,
                                 cu_metal=tuple(np.linspace(1, 0, 101)), cu_1=tuple(np.linspace(1, 0, 101)),
                                 cu_2=tuple(np.linspace(1, 0, 101))):

        mixture_df = []
        np.random.seed(32)

        for k in range(0, len_mixtures):
            # for each mixture (in this case there are 100) we draw a random Cu(0), a random Cu(I) and a random Cu(II)
            # therefore 900 draws are occuring, and the mixture spectra are comprised of 541 unique integer valence
            # spectra. 359 draws result in the same spectrum as a previous draw I suppose
            index_zero = np.random.randint(0, len(zeros))
            zero = zeros.iloc[index_zero]
            self.mixture_ids.append(zero.mpid_string)

            index_one = np.random.randint(0, len(ones))
            one = ones.iloc[index_one]
            self.mixture_ids.append(one.mpid_string)

            index_two = np.random.randint(0, len(twos))
            two = twos.iloc[index_two]
            self.mixture_ids.append(two.mpid_string)

            if mixture_type == 'all':
                metal_fractions = np.random.choice(cu_metal, len_combinations)
                cuo_fractions = np.random.choice(cu_2, len_combinations)

            elif mixture_type == '0-1':
                metal_fractions = np.random.choice(cu_metal, len_combinations)
                cuo_fractions = np.zeros((len_combinations))

            elif mixture_type == '1-2':
                metal_fractions = np.zeros((len_combinations))
                cuo_fractions = np.random.choice(cu_2, len_combinations)

            cu2o_fractions = np.random.choice(cu_1, len_combinations)

            for j in range(0, len_combinations):
                sum_comp = metal_fractions[j] + cu2o_fractions[j] + cuo_fractions[j]
                if sum_comp == 0:
                    pass
                else:
                    components = [metal_fractions[j], cu2o_fractions[j], cuo_fractions[j]] / sum_comp
                    # print(components)

                    mixed_spec = zero['TEAM_1_aligned_925_970'] * components[0] + \
                                 one['TEAM_1_aligned_925_970'] * components[1] + \
                                 two['TEAM_1_aligned_925_970'] * components[2]

                    mixed_cum_spec = zero['Cumulative_Spectra_TEAM_1_aligned_925_970'] * components[0] + \
                                     one['Cumulative_Spectra_TEAM_1_aligned_925_970'] * components[1] + \
                                     two['Cumulative_Spectra_TEAM_1_aligned_925_970'] * components[2]

                    mixed_cum_spec = mixed_cum_spec / max(mixed_cum_spec)

                    if np.isnan((max(mixed_cum_spec))):
                        print(metal_fractions[j], cu2o_fractions[j], cuo_fractions[j])
                        print(sum_comp)
                        print(components)
                        raise ValueError

                    mat_ids = [zero.mpid_string, one.mpid_string, two.mpid_string]
                    formulas = [zero.pretty_formula, one.pretty_formula, two.pretty_formula]
                    # print(mat_ids)
                    bv = round(components[0] * 0 + components[1] * 1 + components[2] * 2, 2)
                    # print(bv)

                    mixture_df.append(
                        [mixed_spec, mat_ids, formulas, bv, mixed_cum_spec, zero['new Scaled Energies use'], components])

        return mixture_df


    def predict_set_of_spectra(self, df_slice, using_error_df = False):
        df_slice.reset_index(inplace = True)
        try:
            labels_to_predict = df_slice['BV Used For Alignment']
        except:
            labels_to_predict = df_slice['Labels Test']

        try:
            spectra_to_predict_temp = np.asarray(df_slice['Cumulative_Spectra_TEAM_1_aligned_925_970'])
        except:
            spectra_to_predict_temp = np.asarray(df_slice['Spectrum'])
        spectra_to_predict = []
        for spec in spectra_to_predict_temp:
            spectra_to_predict.append(np.asarray(spec))

        predictions = self.rf_model.predict(spectra_to_predict)
        accuracy = self.rf_model.score(spectra_to_predict, labels_to_predict)
        print('Accuracy = ' + str(accuracy))

        predictions_full = []
        trees = self.rf_model.estimators_
        for tree in trees:
            predictions_full.append(tree.predict(np.asarray(spectra_to_predict)))
        predictions_ordered = np.asarray(predictions_full).T
        predictions_std = []
        count = 0
        for prediction in predictions_ordered:
            predictions_std.append(np.std(prediction))
            count += 1
            # print(predictions_std)
        errors = np.abs(labels_to_predict - predictions)
        errors = np.asarray(errors)

        MSE = np.square(errors).mean()
        RMSE = math.sqrt(MSE)
        print('RMSE = ' + str(RMSE))
        error_list = []


        if using_error_df == False:
            task_ids = np.asarray(df_slice['mpid_string'])
            for i in range(0, len(df_slice)):
                error_list.append(
                    [np.asarray(labels_to_predict)[i], round(np.asarray(labels_to_predict)[i], 1), predictions[i],
                     round(predictions[i], 1), errors[i], round(errors[i], 1),
                     predictions_std[i], predictions_ordered[i], task_ids[i],
                     df_slice.iloc[i]['pretty_formula'],
                     np.asarray(df_slice['new Scaled Energies use'].iloc[0]),
                     spectra_to_predict[i],
                     df_slice.iloc[i]['TEAM_1_aligned_925_970']])

        else:
            task_ids = np.asarray(df_slice['Materials Ids'])
            for i in range(0, len(df_slice)):
                error_list.append(
                    [np.asarray(labels_to_predict)[i], round(np.asarray(labels_to_predict)[i], 1), predictions[i],
                     round(predictions[i], 1), errors[i], round(errors[i], 1),
                     predictions_std[i], predictions_ordered[i], task_ids[i],
                     df_slice.iloc[i]['Pretty Formula'],
                     np.asarray(df_slice['Energies'].iloc[0]),
                     spectra_to_predict[i],
                     df_slice.iloc[i]['XAS Spectrum']])
        columns = [
            'Labels Test', 'Labels Test Rounded', 'Predictions', 'Predictions Rounded', 'Errors',
            'Errors Rounded', 'Predictions Std', 'Full Predictions', 'Materials Ids', 'Pretty Formula',
            'Energies', 'Spectrum', 'XAS Spectrum']
        error_df = pd.DataFrame(np.asarray(error_list, dtype=object), columns=columns)
        return error_df

    def random_forest_train_bond_valance(self, bv_column='BV Used For Alignment',
                                             spectra_to_predict='925-970 Onset Energy Removed Broadened 0.3 eV',
                                             pcas=None, test_fraction=0.25,  show_uncertianty=True,
                                            num_trees=500, use_new_scale=False,
                                             new_scale_range=1, new_scale_spacing=0.1, use_full_train=True, use_full_test=True,
                                             energy_col='new Scaled Energies use', max_features='auto',
                                         remove_cu_oxides=True):
            self.energy_col = energy_col
            self.predicted_col = spectra_to_predict
            error_df = None
            if type(self.spectra_df) == type(None):
                print('no spectra df, loading ' + self.spectra_df_filepath)
                self.load_spectra_df()

            self.spectra_df_no_oxides = self.spectra_df.drop([1640, 2240, 824])
            # self.spectra_df_no_oxides = self.spectra_df
            self.spectra_df_no_oxides = self.spectra_df_no_oxides.reset_index()

            bond_valences = self.spectra_df_no_oxides[bv_column]
            scaled_spectra = np.asarray(self.spectra_df_no_oxides[spectra_to_predict])

            spectra_train, spectra_test, labels_train, labels_test = train_test_split(scaled_spectra, bond_valences,
                                                                                      test_size=test_fraction,
                                                                                      random_state=32)

            updated_spectra_train = []
            for i in spectra_train:
                updated_spectra_train.append(np.asarray(i))

            updated_spectra_test = []
            for i in spectra_test:
                updated_spectra_test.append(np.asarray(i))

            self.labels_train = labels_train
            self.labels_test = labels_test

            self.spectra_train = updated_spectra_train
            self.spectra_test = updated_spectra_test


            print('len training data = ' + str(len(self.labels_train)))
            print('Using column: ' + spectra_to_predict + ' to predict: ' + bv_column)

            rf_model = RandomForestRegressor(n_estimators=num_trees, n_jobs=-1, max_features=max_features,
                                             random_state=32)
            # print(np.asarray(self.spectra_train).shape)

            # self.spectra_train = np.stack(updated_spectra_train).astype(np.float32)
            # self.spectra_test = np.stack(updated_spectra_test).astype(np.float32)

            # print(self.spectra_train.shape)

            # self.spectra_train = preprocessing.scale(self.spectra_train, axis=1)
            # self.spectra_test = preprocessing.scale(self.spectra_test, axis=1)
            # print(self.spectra_train.shape)

            # plt.plot(self.spectra_train[0])
            # plt.show()

            # print(np.mean(self.spectra_train[0]))
            # print(np.std(self.spectra_train[0]))

            rf_model.fit(self.spectra_train, self.labels_train)
            accuracy = rf_model.score(np.asarray(updated_spectra_test), np.asarray(labels_test))
            print('model accuracy (R^2) on simulated test data ' + str(accuracy))
            self.accuracy = accuracy
            predictions = rf_model.predict(updated_spectra_test)

            if show_uncertianty:
                predictions_full = []
                trees = rf_model.estimators_
                for tree in trees:
                    predictions_full.append(tree.predict(np.asarray(updated_spectra_test)))
                predictions_ordered = np.asarray(predictions_full).T
                predictions_std = []
                count = 0
                for prediction in predictions_ordered:
                    predictions_std.append(np.std(prediction))
                    count += 1
                    # print(predictions_std)
                errors = np.abs(labels_test - predictions)
                errors = np.asarray(errors)

                error_list = []

                task_ids = np.asarray(self.spectra_df_no_oxides.iloc[labels_test.index]['mpid_string'])
                stable = np.asarray(self.spectra_df_no_oxides.iloc[labels_test.index]['is_stable'])
                theoretical = np.asarray(self.spectra_df_no_oxides.iloc[labels_test.index]['is_theoretical'])

                for i in range(0, len(labels_test)):
                    error_list.append(
                        [np.asarray(labels_test)[i], round(np.asarray(labels_test)[i], 1), predictions[i],
                         round(predictions[i], 1), errors[i], round(errors[i], 1),
                         predictions_std[i], predictions_ordered[i], task_ids[i],
                         self.spectra_df_no_oxides['pretty_formula'].iloc[labels_test.index[i]],
                         np.asarray(self.spectra_df_no_oxides[energy_col].iloc[0]),
                         updated_spectra_test[i],
                         self.spectra_df_no_oxides['TEAM_1_aligned_925_970'].iloc[labels_test.index[i]],
                         stable[i], theoretical[i]])
                columns = [
                    'Labels Test', 'Labels Test Rounded', 'Predictions', 'Predictions Rounded', 'Errors',
                    'Errors Rounded',
                    'Predictions Std', 'Full Predictions', 'Materials Ids', 'Pretty Formula',
                    'Energies', 'Spectrum', 'XAS Spectrum', 'is_stable', 'is_theoretical']
                error_df = pd.DataFrame(np.asarray(error_list, dtype=object), columns=columns)

            self.rf_error_df = error_df
            self.rf_model = rf_model
            self.rf_training_set = [updated_spectra_train, labels_train]
            self.rf_test_set = [updated_spectra_test, labels_test]


class PCA_spectra():
    def __init__(self, spectra_df_filepath, column_to_pca, nmf, type_to_pca):
        self.spectra_df_filepath = spectra_df_filepath
        self.spectra_df = joblib.load(self.spectra_df_filepath)
        self.type_to_pca = type_to_pca
        self.nmf = nmf
        self.principle_components = None
        self.principle_df = None
        self.num_components = None
        self.bond_valances = None
        self.column = column_to_pca
        self.pca = None
        if type_to_pca != 'all':
            self.pca_subdf = self.spectra_df.loc[self.spectra_df['Spectra_type'] == type_to_pca]
        if type_to_pca == 'all':
            self.pca_subdf = self.spectra_df

    def run_pca(self, num_components, bv_col):
        self.num_components = num_components
        spectra = self.pca_subdf[self.column]
        spectra_list = list(spectra)
        spectra_v1 = pd.DataFrame(spectra_list).apply(pd.Series)
        self.bond_valances = self.pca_subdf[bv_col]
        cols = []
        for i in range(0, num_components):
            cols.append('principal component ' + str(i + 1))

        if self.nmf:
            nmf = NMF(n_components=num_components)
            self.pca = nmf
            self.principle_components = nmf.fit_transform(spectra_v1)

        if self.nmf == False:
            pca = PCA(n_components=num_components)
            self.pca = pca
            self.principle_components = pca.fit_transform(spectra_v1)

        self.principalDf = pd.DataFrame(data=self.principle_components, columns=cols)

    def pca_by_bv(self):
        plt.figure(figsize = (8,6))
        """
        if self.num_components >= 3:
            plt.scatter(self.principalDf['principal component 1'], self.principalDf['principal component 2'],
                        self.principalDf['principal component 3'], c=self.bond_valances, s = 50)
        if self.num_components == 2:
        """
        plt.scatter(self.principalDf['principal component 1'], self.principalDf['principal component 2'],
                        c=self.bond_valances, s = 50)
        plt.title(str(self.num_components) + ' Component PCA', fontsize=24)
        plt.ylabel('Principal Component 1', fontsize=22)
        plt.xlabel('Principal Component 2', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        cb = plt.colorbar(label='Bond Valance')
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=22)
        text.set_font_properties(font)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(22)

    def visualize_pcs(self, num_pcs_to_plot, energy_col):
        plt.figure(figsize=(8, 7))
        self.energies = self.spectra_df.iloc[0][energy_col]
        for i in range(0, num_pcs_to_plot):
            plt.plot(self.energies, self.pca.components_[i], linewidth=3., label = 'PC ' + str(i+1))
        plt.title(str(self.num_components)+' Component PCA', fontsize=24)
        plt.ylabel('PC Value', fontsize=22)
        plt.xlabel('Energy (eV)', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize = 14)

    def scree_plot(self):
        plt.figure(figsize=(8, 7))
        PC_values = np.arange(self.pca.n_components_) + 1
        plt.plot(PC_values, self.pca.explained_variance_ratio_, 'ro-', linewidth=2)
        plt.title('Scree Plot', fontsize=24)
        plt.xlabel('Principal Component', fontsize=24)
        plt.ylabel('Proportion of Variance Explained', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.show()

    """"
    scaled_spectra = spectra_df['Scaled Intensities']
    scaled_spectra_v1 = []
    for spectrum in scaled_spectra:
        scaled_spectra_v1.append(np.asarray(spectrum))
    scaled_spectra_v1 = np.asarray(scaled_spectra_v1)
    if standardize:
        scaled_spectra_v1 = StandardScaler().fit_transform(scaled_spectra_v1)

    bond_valances = spectra_df['BV Sum']
    ce = spectra_df['Condensed_Labels']
    if non_neg == False:
        if dimension == 2:
            pca = PCA(n_components=2)
        if dimension == 3:
            pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(scaled_spectra_v1)

    if non_neg == True:
        if dimension == 2:
            pca = NMF(n_components=2)
        if dimension == 3:
            pca = NMF(n_components=3)
        principalComponents = pca.fit_transform(scaled_spectra_v1)
    if show_pcs:
        fig = plt.figure(figsize=(10, 8))
        xs = np.arange(700, 774.5, 0.1)
        if max(xs) != 774.5:
            xs = list(xs)
            xs.append(774.5)
        for i in range(0, len(pca.components_)):
            plt.plot(xs, pca.components_[i], linewidth=3)
            plt.title('Principle Components', fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
        plt.show()
    if dimension == 2:
        principalDf = pd.DataFrame(data=principalComponents,
                                   columns=['principal component 1', 'principal component 2'])
    if dimension == 3:
        principalDf = pd.DataFrame(data=principalComponents,
                                   columns=['principal component 1', 'principal component 2', 'principal component 3'])
    principalDf['ce'] = ce
    principalDf['BV'] = bond_valances

    if dimension == 2:
        if color_by == 'BV':
            fig = plt.figure(figsize=(12, 8))
            plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'],
                        c=principalDf[color_by])
            plt.colorbar(label=color_by)
        if color_by == 'ce':
            fig = plt.figure(figsize=(10, 8))
            for ce in principalDf['ce'].unique():
                sub_df = principalDf[principalDf['ce'] == ce]
                plt.scatter(sub_df['principal component 1'], sub_df['principal component 2'], label=ce)
            plt.legend(fontsize=14)
        plt.title('PCA Visualization', fontsize=24)
        plt.ylabel('Principal Component 1', fontsize=22)
        plt.xlabel('Principal Component 2', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    if dimension == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(projection='3d')
        ax = fig.add_subplot(projection='3d')
        if color_by == 'BV':
            ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'],
                       principalDf['principal component 3'], c=bond_valances)
        if color_by == 'ce':
            for ce in principalDf['ce'].unique():
                sub_df = principalDf[principalDf['ce'] == ce]
                ax.scatter(sub_df['principal component 1'], sub_df['principal component 2'],
                           sub_df['principal component 3'], label=ce)
            ax.legend(fontsize=14)
    if non_neg == False:
        print(pca.explained_variance_ratio_)
    return principalDf
    """
# class rf_output():
#     def __init__(self, rf_model_output):


