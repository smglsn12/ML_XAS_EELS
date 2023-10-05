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
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)




class eels_rf_setup():
    def __init__(self, spectra_df_filepath):
        self.spectra_df_filepath = spectra_df_filepath # filepath to the dataframe containing spectra, energies,
        # labeled oxidation state, and other columns
        self.rf_model = None # attribute storing the trained random forest model
        self.rf_error_df = None # dataframe containing test set spectra and prediction statistics
        self.spectra_df = None # attribute storing the loaded dataframe from the path provided above
        self.rf_input_parameters = [] # doesn't appear to be used anywhere currently
        self.rf_training_set = None # training set spectra and oxidation state, assigned after train test split
        self.smoothing_params = [] # window size and polynomial order used by the savgol filter
        self.interped_intens = None # analyzed spectrum post interpolation to 0.1 eV energy axis
        self.interped_energies = None # energy axis for above
        self.interp_spacing = None # energy axis for interpolation of inputted spectrum (currently fixed at 0.1 eV)
        self.energy_col = None # column of the dataframe holding the spectral energy values
        self.points_to_average = None # amount of the tail of the spectrum to average into a normalization factor.
        # while this is still being used in the code base, it is no longer relevant due to the normalization of the
        # cumulative spectrum to one.
        self.mixture_ids = [] # materials IDs of the spectra that build the mixture database

    def load_spectra_df(self):
        """
        Loads the spectra df from the filepath provided when the object is initialized and stores it as self.spectra_df
        :return: stores spectra dataframe as an attribute
        """
        self.spectra_df = joblib.load(self.spectra_df_filepath)

    def full_noise_setup(self, column, energy_col, interp_range, smoothing_window, filename=None, show_plots=False,
                         baseline_subtract=False, stds = (1000, 200, 100, 50), num_random_seeds = 100):
        """
        This function takes test set spectra, adds poisson noise to them at a specified set of standard deviations, and
        computes this over a series of random states for each standard deviation. At each unique set of spectra, the
        rf model is used to re predict each test set spectrum, thereby producing a detailed picture of the effect
        of simulated noise on the test set spectra.

        :param column: column of the error dataframe containing the spectrum (string)
        :param energy_col: energy column corresponding to above (string)
        :param interp_range: energy axis spectra have been interpolated to (float)
        :param smoothing_window: window size for savgol filter (int)
        :param filename: the name of the resulting dataframe (string)
        :param show_plots: whether to show various plots as the function runs (bool)
        :param baseline_subtract: whether to baseline subtract the noisy spectra (bool) should remain false
        :return: nothing, outputted dataframe is saved as the inputted filename (unless no filename is inputted,
        then the outputted dataframe is returned)
        """
        # define accumulators used throughout the script
        full_noise_analysis_output = []

        count = 0
        count_1 = 0
        for random_seed in np.linspace(0, num_random_seeds-1, num_random_seeds): # runs the following analysis over
            # the number of specified random seeds
            random_seed = int(random_seed)
            print(random_seed)
            for std in stds:
                noisy_test = []
                interp_spec = np.asarray(self.rf_error_df[column])

                energy = np.arange(925, 970.1, 0.1)[0:451]
                energy[450] = 970
                interp_energies = np.arange(925, 970 + interp_range, interp_range) # define energies for interpolation.
                # This is done to mimic the process done on an experimental spectrum

                smooth_interp_spec = []
                interp_energies_final = []
                for en in interp_energies: # ensure no floating point errors in energies
                    en_rounded = round(en, 2)
                    if en_rounded <= 970:
                        interp_energies_final.append(en_rounded)
                if show_plots:
                    if count == 0:
                        plt.plot(interp_spec[0])
                        plt.title('Interpolate Spectra ' + str(interp_range) + ' eV', fontsize=18)
                        plt.show()

                for spec in interp_spec: # add noise to spectra
                    noisy_test.append(poisson(spec, std, random_seed))

                for noisy_spec in noisy_test:
                    # print(count)
                    if show_plots:
                        if count_1 == 0:
                            plt.plot(self.rf_error_df.iloc[0][energy_col], noisy_spec)
                            plt.title('Add Noise Original', fontsize=18)
                            plt.show()
                            count_1 += 1

                    smoothed_spec = savgol_filter(noisy_spec, smoothing_window, 3) # smooth generated noisy spectrum
                    # with same method as experimental spectra


                    f = interpolate.interp1d(interp_energies_final, smoothed_spec) # interpolate noisy spectrum back to
                    # 0.1 eV energy axis
                    smoothed_spec = f(energy)
                    if baseline_subtract:
                        smoothed_spec = smoothed_spec - min(smoothed_spec)
                    smooth_interp_spec.append(smoothed_spec)
                if show_plots:
                    if count == 0:
                        plt.plot(interp_energies_final[0:100], noisy_spec[0:100])
                        plt.plot(energy[0:30], smoothed_spec[0:30])
                        plt.title('Zoom in on baseline', fontsize = 18)
                        plt.show()

                        plt.figure(figsize=(8, 7))
                        plt.xlabel('Energy (eV)', fontsize=36)
                        plt.ylabel(' Intensty', fontsize=36)
                        plt.xticks([930, 950, 970], fontsize=36)
                        plt.yticks([0, 0.5, 1, 1.5], fontsize=36)
                        plt.plot(self.rf_error_df.iloc[0][energy_col], noisy_test[0], linewidth=3,
                                 label='Noisy Spec')
                        plt.title('Add Noise std = ' + str((1/std)*10), fontsize=36)
                        # plt.show()
                        plt.plot(energy, smooth_interp_spec[0], linewidth=3, label='Smoothed')
                        # plt.title('Smooth Using Same Window As Exp', fontsize = 18)
                        plt.legend(fontsize=21)
                        # plt.savefig('Example Noisy Spec' + str(std) + '.pdf',  bbox_inches='tight', transparent=True)
                        plt.show()

                noisy_test_cum = []
                # calculate cumulative spectrum for noisy spectra
                for spec in smooth_interp_spec:
                    temp_intens = []
                    for k in range(0, len(spec)):
                        temp_intens.append(sum(spec[0:k]))
                    noisy_test_cum.append(temp_intens / max(temp_intens))
                if show_plots:
                    if count == 0:
                        plt.plot(noisy_test_cum[0])
                        plt.show()

                # run predictions on the noisy test set
                accuracy = self.rf_model.score(noisy_test_cum, self.rf_error_df['Labels Test'])

                predictions = self.rf_model.predict(noisy_test_cum)

                # eliminate floating point errors in predictions and round to 0.1
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

                # display R2 of noisy data
                print('model accuracy (R^2) on simulated test data ' + str(accuracy))

                true = []
                pred = []
                count_list = []
                for i in np.asarray(
                        self.rf_error_df[['Predictions_noisy', 'Labels Test Rounded']].value_counts().index):
                    pred.append(round(i[0], 1))
                    true.append(round(i[1], 1))


                for k in np.asarray(
                        self.rf_error_df[['Predictions_noisy', 'Labels Test Rounded']].value_counts()):
                    count_list.append(k)

                MSE = np.square(errors_noisy).mean()
                RMSE = math.sqrt(MSE)
                # display RMSE of noisy data
                print(RMSE)
                full_noise_analysis_output.append([std, random_seed, accuracy, RMSE])
            count += 1
        full_noise_df = pd.DataFrame(full_noise_analysis_output, columns=['noise_std', 'random_state', 'R2', 'RMSE'])
        if filename != None:
            joblib.dump(full_noise_df, filename)
        else:
            return full_noise_df

    def compare_simulation_to_experiment(self, material='CuO', savgol_params=[51, 3],
                                         xlims=[920, 980], show_feff=False, feff_shift=-10,
                                         compare_to_lit=False, lit_spectrum=None, lit_shift=0.0, title=None,
                                         show_experiment = True, savefigure=False):
        """
        Creates a plot comparing any combination of simulated XAS, experimental EELS and literature XAS spectra

        :param material: The material to compare to simulations/literature. Options are Cu Metal, Cu2O and CuO (string)
        :param savgol_params: smoothing parameters, window size and polynomial order for the savgol filter (list of int)
        :param xlims: the limits for the plots (list of int)
        :param show_feff: whether to show simulated spectra in the comparison plots (bool)
        :param feff_shift: energy axis shift for the simulated spectrum (float) This value is added to the
        energy axis, so a positive number moves the spectrum in the positive direction
        :param compare_to_lit: whether to show literature xas spectra in the comparison plots (bool)
        :param lit_spectrum: filepath to the literature spectrum for comparison (string) loads from csv files
        and does not currently support other file types
        :param lit_shift: energy axis shift for the literature spectrum (float) This value is subtracted from the
        energy axis, so a positive number moves the spectrum in the negative direction
        :param title: title of the comparison plot (string)
        :param show_experiment: whether to show the experimental spectrum in the comparison plot (bool)
        :return: Shows comparison plot set to the above specifications
        """

        # load and plot experimental spectrum
        output = nio.dm.dmReader("C:/Users/smgls/Materials_database/" + material + " Deconvolved Spectrum.dm4")
        intens = output['data']
        energies = output['coords'][0]

        plt.figure(figsize=(8, 7))
        plt.xlabel('Energy (eV)', fontsize=36)
        plt.ylabel('Intensity', fontsize=36)
        plt.title(title, fontsize=36)
        plt.xticks([930,950,970], fontsize=36)
        plt.yticks(fontsize=36)

        interped_intens_smoothed = savgol_filter(intens / max(intens), savgol_params[0], savgol_params[1])

        if show_experiment:
            plt.plot(energies, interped_intens_smoothed / max(interped_intens_smoothed), color='k',
                     label='This Work', linewidth=3)
        # load and plot literature spectrum
        if compare_to_lit:
            output = pd.read_csv(lit_spectrum)
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
        # load and plot simulated spectrum
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
            plt.plot(np.asarray(energies) + lit_shift, broadened_intens, label='Liturature XAS', linewidth=3,
                     color='#1f77b4')

        if show_feff:
            plt.plot(np.asarray(feff_spec_energies) + feff_shift, feff_spec_intens[0] / max(feff_spec_intens[0]),
                     color='#ff7f0e',
                     linewidth=3, linestyle='-', label='FEFF9 Simulation')
        plt.xlim(xlims)
        plt.legend(fontsize=28)
        if savefigure:
            plt.savefig(material + ' Comparison.pdf',
                    bbox_inches='tight', transparent=True)
        plt.show()


    def visualize_mixtures(self, cu_metal_id, cu2o_id, cuo_id, cu_fraction=0.5, cu2o_fraction=0.5, cuo_fraction=0.0,
                           include_ticks=False, include_title=True, savefigure = False):

        """
        Creates a plot showing a mixture spectrum and each of the components that comprise it, with each component
        displayed scaled by its proportion to the mixture spectrum
        :param cu_metal_id: materials ID of the Cu metal material in the mixture (string)
        :param cu2o_id: materials ID of the Cu (I) material in the mixture (string)
        :param cuo_id: materials ID of the Cu (II) material in the mixture (string)
        :param cu_fraction: amount the Cu metal material contributes to the mixture (float between 0 and 1)
        :param cu2o_fraction: amount the Cu (I) material contributes to the mixture (float between 0 and 1)
        :param cuo_fraction: amount the Cu (II) material contributes to the mixture (float between 0 and 1)

        NOTE - cu_fraction, cu2o_fraction and cuo_fraction must add to 1!

        :param include_ticks: whether to include ticks in the plot (bool)
        :param include_title: whether to include a title in the plot (bool)
        :param savefigure: whether to save the plot as a pdf (bool)
        :return: none, plot is displayed
        """

        plt.figure(figsize=(8, 7))
        energies = np.asarray(
            self.spectra_df.loc[self.spectra_df['mpid_string'] == cu_metal_id][
                'new Scaled Energies use'])[0]

        # define labels for component and mixture spectra
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

        # pull component spectra from dataframe
        cu_metal = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cu_metal_id]['TEAM_1_aligned_925_970'])[0]
        cu2o = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cu2o_id]['TEAM_1_aligned_925_970'])[0]
        cuo = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cuo_id]['TEAM_1_aligned_925_970'])[0]

        # build mixture spectrum using component weights
        mix_spec = cu_metal * cu_fraction + cu2o * cu2o_fraction + cuo * cuo_fraction

        # plot spectra
        plt.plot(energies, cu_metal * cu_fraction, linewidth=4, linestyle='--')
        plt.plot(energies, cu2o * cu2o_fraction, linewidth=4, linestyle='--')
        plt.plot(energies, cuo * cuo_fraction, linewidth=4, linestyle='--')
        plt.plot(energies, mix_spec, label=mix_str, linewidth=4)

        font = font_manager.FontProperties(
            style='normal', size=24, weight='bold')


        if include_ticks:
            plt.xlabel('Energy (eV)', fontsize=36, fontweight = 'bold')
            plt.xticks([930, 950, 970], fontsize=36, fontweight = 'bold')
        else:
            plt.xticks([], fontsize=36)

        if include_title:
            plt.title('Mixture Spectrum Example', fontsize=24)

        plt.xlim([922.5, 975])
        # plt.ylabel(' Intensty', fontsize = 36, fontweight='bold')
        plt.yticks([], fontsize=36)
        plt.ylim([-0.05, max(mix_spec) + 0.25])
        plt.legend(prop=font)

        # save figure if specified
        if savefigure:
            plt.savefig(str([cu_metal_id, cu2o_id, cuo_id, cu_fraction, cu2o_fraction, cuo_fraction]) + '.pdf',
                    bbox_inches='tight', transparent=True)

    def visualize_mixture_addition(self, column = 'BV Used For Alignment', include_mixtures=False, right_y_ticks = False,
                                   savefigure = False):

        """
        Displays the spread of spectra in the dataset corresponding to each oxidation state. Can show either only
        integer spectra or full dataset including mixed valent spectra

        :param column: column from the dataframe to extract the oxidation state labels from (string)
        :param include_mixtures: whether to display mixed valent spectra in the plot (bool)
        :param savefigure: whether to save the plot as a pdf (bool)
        :return: none, the plot is displayed
        """

        # find number of spectra for each oxidation state, split into
        # integer oxidation state
        ys1 = self.spectra_df[column].value_counts().values[0:3]
        xs1 = self.spectra_df[column].value_counts().index[0:3]

        # mixed valent
        ys2 = self.spectra_df[column].value_counts().values[3:212]
        xs2 = self.spectra_df[column].value_counts().index[3:212]

        colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

        fig, ax1 = plt.subplots(figsize=(8, 7))
        if right_y_ticks:
            ax1.yaxis.tick_right()
        ax1.bar(xs1, ys1, color=colors, width=0.3)

        # ax = plt.figure(figsize=(8,7))
        # ax.add_axes(ax1)

        # add mixtures if specified
        if include_mixtures:
            mixture_values = []
            for i in range(0, len(self.spectra_df)):
                if self.spectra_df.iloc[i][column] not in [0.0, 1.0, 2.0]:
                    mixture_values.append(self.spectra_df.iloc[i][column])
            ax1.hist(mixture_values, bins=20, color='#d62728')

        # plt.title('Spectra Per Oxidation State', fontsize=32)

        font = font_manager.FontProperties(
            weight='bold',
            style='normal', size=22)
        plt.xlabel('Oxidation State', fontsize=36, fontweight = 'bold')
        plt.ylabel('Number of Spectra', fontsize = 36, fontweight = 'bold')
        plt.xlim([-0.2, 2.2])
        plt.xticks([0, 1, 2], fontsize=36, fontweight = 'bold')
        plt.yticks([0, 400, 800, 1200], fontsize=36, fontweight = 'bold')
        if savefigure:
            if include_mixtures:
                plt.savefig('Full Dataset Mixtures.pdf', bbox_inches='tight', transparent=True)
            else:
                plt.savefig('Full Dataset.pdf', bbox_inches='tight', transparent=True)

    def visualize_predictions(self, indicies, show_cu_oxide_reference = False):
        """
        Visualize a specific prediction from the test set. Includes a plot of the spectrum and the
        prediction histogram for that spectrum

        :param indicies: row in error dataframe to extract the prediction from
        :param show_cu_oxide_reference: whether to also plot reference spectra for Cu metal, Cu (I) and Cu (II)
        :return: None, the plot is displayed
        """
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

            # extract the specified spectrum from test set and plot
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

            # plot the full predictions into a prediction histogram
            plt.figure(figsize=(10, 8))
            hist = plt.hist(self.rf_error_df.iloc[index]['Full Predictions'], edgecolor='k', facecolor='grey', fill=True, linewidth=3)
            ax = plt.gca()
            # ax.get_xticks() will get the current ticks
            # ax.set_yticklabels(map(str, ax.get_yticks()))
            plt.xticks([0, 0.5, 1, 1.5, 2, 2.5])

            height = max(hist[0])

            # plt.vlines(np.median(df.iloc[index]['Full Predictions']), 0, height, color = 'purple', label = 'Median Prediction', linewidth = 5)
            # show vertical lines at the rf model prediction and the labeled oxidation state
            plt.vlines(self.rf_error_df.iloc[index]['Labels Test'], 0, height, color='blue', label='Labeled BV', linewidth=5)
            plt.vlines(np.mean(self.rf_error_df.iloc[index]['Full Predictions']), 0, height, color='red', label='Prediction',
                       linewidth=5,
                       linestyle='-')
            mean = np.mean(self.rf_error_df.iloc[index]['Full Predictions'])
            std = np.std(self.rf_error_df.iloc[index]['Full Predictions'])
            low = mean - std
            high = mean + std

            # plot prediction standard deviation as a horizontal line
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
        """
        A version of visualize_mixtures that focuses on displaying a set of mixture components in detail

        :param cu_metal_id: materials ID of the Cu metal material in the mixture (string)
        :param cu2o_id: materials ID of the Cu (I) material in the mixture (string)
        :param cuo_id: materials ID of the Cu (II) material in the mixture (string)
        :param include_title: whether to include the title in the plot (bool)
        :param include_ticks: whether to include the ticks in the plot (bool)
        :param savefigure: whether to save the figure as a pdf (bool)
        :return: none, plot is displayed
        """

        plt.figure(figsize=(8, 7))
        # extract materials names from dataframe and store as labels for the plot
        energies = np.asarray(
            self.spectra_df.loc[self.spectra_df['mpid_string'] == cu_metal_id][
                'new Scaled Energies use'])[0]
        label_cu_metal = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] ==
                                                        cu_metal_id]['pretty_formula'])[0] + ' BV = ' + str(0)
        label_cu2o = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] ==
                                                    cu2o_id]['pretty_formula'])[0] + ' BV = ' + str(1)
        label_cuo = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] ==
                                                   cuo_id]['pretty_formula'])[0] + ' BV = ' + str(2)

        # extract spectra and plot
        cu_metal = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cu_metal_id]['TEAM_1_aligned_925_970'])[0]
        cu2o = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cu2o_id]['TEAM_1_aligned_925_970'])[0]
        cuo = np.asarray(self.spectra_df.loc[self.spectra_df['mpid_string'] == cuo_id]['TEAM_1_aligned_925_970'])[0]

        plt.plot(energies, cu_metal, label=label_cu_metal, linewidth=4)
        plt.plot(energies, cu2o, label=label_cu2o, linewidth=4)
        plt.plot(energies, cuo, label=label_cuo, linewidth=4)
        if include_title:
            plt.title('Random Sample Cu(0), Cu(I) and Cu(II)', fontsize=20)

        font = font_manager.FontProperties(
            style='normal', size=22, weight='bold')
        # plt.ylabel(' Intensty', fontsize = 36, fontweight='bold')
        if include_ticks:
            plt.xlabel('Energy (eV)', fontsize=36, fontweight = 'bold')
            plt.xticks([930, 950, 970], fontsize=36, fontweight = 'bold')
        else:
            plt.xticks([])
        plt.yticks([], fontsize=36)
        plt.legend(prop=font)

        if savefigure:
            plt.savefig('Mixture Components ' + str([cu_metal_id, cu2o_id, cuo_id]) + '.pdf', bbox_inches='tight',
                    transparent=True)

    def add_interp_spec_to_ERROR_df(self, interpolation_ranges, add_zeros=False, energy_zeros=920):
        """
        Interpolate the spectra in the test set to be on a different energy range. This is built to accompany the
        simulated noise exploration functions, as the experimental spectra for this work are on a 0.03 eV energy
        scale and this allows the direct comparison of simulated noise addition to our experimental procedures
        :param interpolation_ranges: list of energy spacings to add to the simulated data test set (list/ndarray)
        :param add_zeros: whether to add zeros at the low energy end of the spectrum (pre onset edge) (bool)
        :param energy_zeros: the energy for the added zeros above to start
        :return: Interpolated spectra are added to the error df attribute
        """

        count = 0
        for interpolation_range in interpolation_ranges: # runs over all inputted energy spacings
            print(interpolation_range)
            energy = np.arange(925, 970.1, 0.1)[0:451]
            energy[450] = 970 # ensure no floating point errors on the top value
            interp_spec = []
            interp_energy_for_df = []
            interp_energies = np.arange(925, 970 + interpolation_range, interpolation_range)
            # add zeros to the front of the spectrum by adding the necessary energy values
            if add_zeros:
                interp_energies_front = np.arange(energy_zeros, 925, interpolation_range)
                interp_energies_front = list(interp_energies_front)
                interp_energies_front.reverse()

            # interpolate all spectra in test set to new energy axis
            for spec in np.asarray(self.rf_error_df['XAS Spectrum']):
                # print(count)
                f = interpolate.interp1d(energy, spec)
                interp_energies_final = []
                for en in interp_energies:
                    en_rounded = round(en, 3) # ensure no floating point errors in new energies
                    if en_rounded <= 970:
                        interp_energies_final.append(en_rounded)
                # print(interp_energies_final)
                interped_spec = f(interp_energies_final)
                # add zeros to the front of the interpolated spectrum (if applicable)
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
            # create new columns in the test set dataframe to contain the interpolated spectra and new energies
            self.rf_error_df['Interpolated_spec_' + str(interpolation_range)] = interp_spec
            self.rf_error_df['Interpolated_spec_energies_' + str(interpolation_range)] = interp_energy_for_df



    def scale_experimental_spectra(self, intens, energies, scale = 0.1):
        """
        Scales an experimental spectrum so it on an even specified energy scale
        :param intens: Spectrum intensities to scale (list/ndarray)
        :param energies: Spectrum energies to scale (list/ndarray)
        :param scale: Energy spacing for scaled spectrum (float)
        :return: defines 'interped_intens' and 'interped_energies' attributes for scaled spectrum
        """
        # set interp range within the energy bounds of the inputted spectrum
        energies_interp = np.arange(round(min(energies)+0.5, 1),round(max(energies)-0.5, 1), scale)
        energies_interp_use = []
        for i in energies_interp:
            energies_interp_use.append(round(i, 1))
        f = interp1d(energies, intens)
        interp_intens = f(energies_interp)

        # scale intensity to higher energy tail (no longer does anything from an analytical perspective due to
        # predictions of cumulative spectrum, but still valuable for visual spectral comparisons)
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
                                  savefigure = False, theory_indicies = []):
        """
        Wrapper function for predicting all XAS/EELS spectra in a folder. All spectra in that folder must be of the
        same file type (ie csv or dm4). This function calls the 'predict_experiment_random_forest' function for each
        spectrum in the folder with the specified set of conditions constant for each spectrum.
        :param folder_path: filepath to the spectra folder (string)
        :param shifts: energy axis scaling to be done to the spectra (float)
        :param smoothings: parameters for smoothing with savgol filter (window size and polynomial order) (list
        of list of int)
        :param edge_points: points in the tail of the spectrum to average over to create a normalization value (int)
        :param spectra_type: either TEAM I (opened with ncempy's dm4 reader) or csv (opened with pandas' read_csv)
        (string)
        :param cumulative_spec: whether to turn the experimental spectrum into a cumulative spectrum for prediction
        (bool)
        :param theory_column: column in the dataframe containing the simulated spectrum that will match the experimental
        spectrum for visualization and comparison purposes (string)
        :param energies_range: the range of energy axis values for the experimental spectrum to be able to match
        simulations (list of int)
        :param show_hist: whether to show a histogram of predictions of each decision tree (bool)
        :param show_plots: whether to show a series of plots illustrating each trainsformation done to the spectrum
        as the function runs (bool)
        :param show_inputted_spectrum: whether to plot the inputted spectrum for visualization purposes (bool)
        :param print_details: whether to print the predicted oxidation state and standard deviation (bool)
        :param savefigure: whether to save all these plots (bool)
        :return:

        """
        # grab paths to spectra
        paper_paths = glob.glob(folder_path+'/*')

        # loop through all sets of parameters passed to the wrapper function and predict a spectrum for each set
        predictions_set = []
        for path in paper_paths:
            for shift in shifts:
                for smooth in smoothings:
                    for edge in edge_points:
                        if print_details:
                            print('Predicting From ' + path)
                            print('Energy Axis Shift = ' + str(shift))
                            print('Smoothing Parameters = ' + str(smooth))
                        if 'Cu Metal' in path:
                            self.predict_experiment_random_forest(path, theory_indicies[0], 'Cu Metal',
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
                            self.predict_experiment_random_forest(path, theory_indicies[1], 'Cu2O',
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
                            self.predict_experiment_random_forest(path, theory_indicies[2], 'CuO',
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
                            self.predict_experiment_random_forest(path, 1644, 'Unlabeled Cu',
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
        # build prediction dataframe of all combinations of conditions predicted here. This is used in broader
        # visualization methods
        self.prediction_df = pd.DataFrame(predictions_set,
                                columns=['Prediction', 'Predictions Std', 'True', 'Num Tail Points', 'Smoothing Window',
                                         'Smoothing Poly Order', 'Predicted Spectrum', 'Theory Index',
                                         'Spectrum filepath', 'Material', 'Spectrum Energy Shift'])

    def reset_mixed_valent_series(self):
        """
        Resets the mixed valent predictions so a fresh set can be established. This is done for mixed valent prediction
        analysis and visualization
        :return:
        """
        self.mixed_valent_pred = []

    def predict_Experimental_Cu_non_integers(self, cu_metal=1, cu2o=0, cuo=0,
                                             show_predicted_spectrum = True,
                                 show_plots = False, print_predictions = False,
                                folder_path = 'C:/Users/smgls/Materials_database/Cu_deconvolved_spectra',
                                smoothing_params = [51,3],
                                energies_range = [925, 970],
                                exp_scale = 0.1):
        """
        Generates an experimental mixed valent spectrum for a set of integer valent ratios and predicts that spectrum.
        Generates both raw spectrum and cumulative spectrum but only predicts the cumulative spectrum
        :param cu_metal: amount the Cu metal material contributes to the mixture (float between 0 and 1)
        :param cu2o: amount the Cu(I) material contributes to the mixture (float between 0 and 1)
        :param cuo: amount the Cu(II) material contributes to the mixture (float between 0 and 1)
        :param show_predicted_spectrum: whether to make a plot showing the spectrum fed into the model (bool)
        :param show_plots: whether to show a broader set of plots showing how the mixture spectrum is comprised
        of the integer spectra (bool)
        :param print_predictions: whether to print the prediction and prediction std (bool)
        :param folder_path: path to folder containing integer valent spectra
        :param smoothing_params: window size and polynomial order for savgol filter smoothing (list of int)
        :param energies_range: energy range to crop the experimental spectra to (list of int)
        :param exp_scale: energy axis scale for the experimental spectrum (float)
        :return: predicts mixed valent spectrum and stores its predictions in 'mixed_valent_pred'
        """

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
        # for each experimental spectrum provided, load and process in the same way as the 'predict_experiment'
        # functions. First smooth spectra, then scale to 0.1 eV, then crop to 925-970 eV energy range
        for exp_spectrum in paper_paths:
            # load spectrum
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


            # smooth spectrum
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

            # scale to 0.1 eV spacing
            self.scale_experimental_spectra(intens, energies, scale = exp_scale)

            # crop to 925-970 eV range
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

            # generate cumulative spectrum
            self.standard_cum_spec = temp_intens / max(temp_intens)

            # save generated spectrum to dictionary labeled by spectrum type
            if 'Cu Metal' in exp_spectrum:
                self.exp_spectra_raw['Cu Metal'] = self.intensities_final
                self.exp_spectra_cumulative['Cu Metal'] = self.standard_cum_spec
            if 'Cu2O' in exp_spectrum:
                self.exp_spectra_raw['Cu2O'] = self.intensities_final
                self.exp_spectra_cumulative['Cu2O'] = self.standard_cum_spec

            if 'CuO' in exp_spectrum:
                self.exp_spectra_raw['CuO'] = self.intensities_final
                self.exp_spectra_cumulative['CuO'] = self.standard_cum_spec

        # generate mixture spectra based on provided mixture components
        mixed_spec = self.exp_spectra_raw['Cu Metal'] * cu_fractions[0] + \
                     self.exp_spectra_raw['Cu2O'] * cu_fractions[1] + \
                     self.exp_spectra_raw['CuO'] * cu_fractions[2]

        self.mixed_cum_spec = self.exp_spectra_cumulative['Cu Metal'] * cu_fractions[0] + \
                     self.exp_spectra_cumulative['Cu2O'] * cu_fractions[1] + \
                     self.exp_spectra_cumulative['CuO'] * cu_fractions[2]

        # normalize cumulative spectrum
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


        if show_predicted_spectrum:
            plt.figure(figsize=(6,5))
            plt.plot(self.final_energies, self.mixed_cum_spec, linewidth = 3)
            plt.title('Predicted Spectrum', fontsize = 24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

        # predict mixture spectrum and extract predictions for each decision tree
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

        # store prediction statistics
        self.prediction = round(pred[0], 2)
        self.prediction_std = round(predictions_std, 2)
        self.true_val = true_val

        self.mixed_valent_pred.append([self.prediction, self.prediction_std, self.true_val])

    def predict_Cu_non_integers(self, cu_metal=1, cu2o=0, cuo=0, cu_int_indicies = (),
                                show_standards = False, show_plots = False, energy_col ='new Scaled Energies use',
                                theory_col = 'TEAM_1_aligned_925_970',
                                predict_col = 'Cumulative_Spectra_TEAM_1_aligned_925_970',
                                show_predictions = False):
        """
        This function is very similar to 'predict_Experimental_Cu_non_integers' except its using simulated spectra
        instead
        :param cu_metal: amount the Cu metal material contributes to the mixture (float between 0 and 1)
        :param cu2o: amount the Cu(I) material contributes to the mixture (float between 0 and 1)
        :param cuo: amount the Cu(II) material contributes to the mixture (float between 0 and 1)
        :param cu_int_indicies: indicies of the dataframe to use as the integer valent spectra (list of int)
        :param show_standards: whether to plot the integer valent spectra (bool)
        :param show_plots: whether to show a series of other plots showing how the mixture spectra are formed (bool)
        :param energy_col: column from the dataframe containing the energy values (string)
        :param theory_col: column from the dataframe containing the raw spectrum (string)
        :param predict_col: column from the dataframe containing the spectrum to predict (string)
        :param show_predictions: whether to print the predictions (bool)
        :return: predictions are stored in 'mixed_valent_pred'
        """

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


        # generate mixture spectrum from components of integer valent spectra
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

        # generate cumulative spectrum in the same way
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
        # normalize cumulative spectrum
        self.mixed_cum_spec = self.mixed_cum_spec / max(self.mixed_cum_spec)

        if show_plots:
            plt.figure(figsize=(6,5))
            plt.plot(energies, self.mixed_cum_spec, linewidth = 3)
            plt.title('Predicted Spectrum', fontsize = 24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

        # predict mixture cumulative spectrum and extract predictions from each decision tree
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

        # store prediction information
        self.mixed_valent_pred.append([self.prediction, self.prediction_std, self.true_val])

    def simulated_mixed_valent(self, catagory = '0-1', colorbar_range = [0.1, 0.5], savefigure=False):
        """
        Generates and predicts binary simulated mixture spectra in a range of 50 mixtures from purely the lower
        oxidation state to purely the higher oxidation state. Plots the results by generating a predicted vs true plot
        where the scatter plot colors correspond to the prediction standard deviation and a dashed black line shows
        the location of perfect predictions.
        :param catagory: types of mixtures to generate and predict, 0-1, 1-2 or 0-2 (string) only supports binary
        mixtures
        :param colorbar_range: the range of values for the color bar showing prediction standard deviation (list of
        float)
        :param savefigure: whether to save the plot as a pdf (bool)
        :return: None, summary plot is shown and attributes are set based on 'predict_Cu_non_integers' results
        """
        min_std = colorbar_range[0]
        max_std = colorbar_range[1]
        self.reset_mixed_valent_series()
        # set list of 50 mixture ratios
        first_comp = list(np.linspace(1, 0, 51))
        second_comp = list(np.linspace(0, 1, 51))

        cu_metal_index = self.spectra_df.loc[self.spectra_df['mpid_string'] == 'mp-30'].index[0]
        cu2o_index = self.spectra_df.loc[self.spectra_df['mpid_string'] == 'mp-361'].index[0]
        cuo_index = self.spectra_df.loc[self.spectra_df['mpid_string'] == 'mp-704645'].index[0]

        if catagory == '0-1':
            # if mixtures of 0 and 1 fix Cu(II) to zero in 'predict_Cu_non_integers'
            for i in range(0, len(first_comp)):
                self.predict_Cu_non_integers(first_comp[i], second_comp[i], 0, cu_int_indicies=(cu_metal_index, cu2o_index, cuo_index))
            plt.figure(figsize=(8, 7))
            plt.title('Simulated Mixtures Cu(0) to Cu(I)', fontsize=23)

        elif catagory == '1-2':
            # if mixtures of 1 and 2 fix Cu(0) to zero in 'predict_Cu_non_integers'
            for i in range(0, len(first_comp)):
                self.predict_Cu_non_integers(0, first_comp[i], second_comp[i], cu_int_indicies=(cu_metal_index, cu2o_index, cuo_index))
            plt.figure(figsize=(8, 7))
            plt.title('Simulated Mixtures Cu(I) to Cu(II)', fontsize=23)

        elif catagory == '0-2':
            # if mixtures of 0 and 2 fix Cu(I) to zero in 'predict_Cu_non_integers'
            for i in range(0, len(first_comp)):
                self.predict_Cu_non_integers(first_comp[i], 0, second_comp[i], cu_int_indicies=(cu_metal_index, cu2o_index, cuo_index))
            plt.figure(figsize=(8, 7))
            plt.title('Simulated Mixtures Cu(0) to Cu(II)', fontsize=23)

        # build list of prediction details based on values stored in 'predict_Cu_non_integers'
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
        font = matplotlib.font_manager.FontProperties(size=22)
        text.set_font_properties(font)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(22)
            # t.set_weight('bold')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlabel('True Value', fontsize=24)
        plt.ylabel('Bond Valance Prediction', fontsize=24)
        if savefigure:
            plt.savefig('Simulated Mixtures ' + catagory + '.pdf', bbox_inches='tight', transparent=True)

    def experimental_mixed_valent(self, catagory = '0-1', smoothing_params = [51,3], show_plots = False,
                                  show_predicted_spectrum = False, colorbar_range = [0.1, 0.5], savefigure=False):
        """
        This function is very similar to 'simulated_mixed_valent' except its using experimetnal spectra instead

        :param catagory: types of mixtures to generate and predict, 0-1, 1-2 or 0-2 (string) only supports
        binary mixtures
        :param smoothing_params: window size and polynomial order for savgol filter smoothing (list of int)
        :param show_predicted_spectrum: whether to make a plot showing the spectrum fed into the model (bool)
        :param show_plots: whether to show a broader set of plots showing how the mixture spectrum is comprised
        of the integer spectra (bool)
        :param colorbar_range: the range of values for the color bar showing prediction standard deviation
        (list of float)
        :param savefigure: whether to save the plot as a pdf (bool)
        :return:
        """
        min_std = colorbar_range[0]
        max_std = colorbar_range[1]
        first_comp = list(np.linspace(1,0,51))
        second_comp = list(np.linspace(0,1,51))
        self.reset_mixed_valent_series()

        if catagory == '0-1':
            # if mixtures of 0 and 1 fix Cu(II) to zero in 'predict_Cu_non_integers'
            for i in range(0, len(first_comp)):
                # print(first_comp[i] + second_comp[i])
                self.predict_Experimental_Cu_non_integers(cu_metal=first_comp[i], cu2o=second_comp[i], cuo=0,
                                                                 show_plots=show_plots,
                                                          show_predicted_spectrum=show_predicted_spectrum,
                                                             smoothing_params=smoothing_params)
            trues = []
            # generate true values based on the values returned for the integer spectra from our RF model, rather
            # than just assigning 0, 1 and 2. This is because we can't  expect the model to do better on mixture
            # spectra than it does on integers and, as described in the manuscript, we believe the prediction of 0.3
            # for the Cu metal spectrum is more accurate than its nominal label of zero.
            for i in range(0, len(first_comp)):
                trues.append(first_comp[i] * 0.3 + second_comp[i] * 1.07)
            # trues = np.asarray(test_rf_obj.mixed_valent_pred).T[2]
            plt.figure(figsize=(8, 7))
            plt.title('Experimental Cu(0) to Cu(I)', fontsize=26)
            plt.xticks([0.3,0.5,0.7,0.9,1.1])
            plt.yticks([0.3,0.5,0.7,0.9,1.1])
        elif catagory == '1-2':
            # if mixtures of 1 and 2 fix Cu(0) to zero in 'predict_Cu_non_integers'
            for i in range(0, len(first_comp)):
                # print(first_comp[i] + second_comp[i])
                self.predict_Experimental_Cu_non_integers(cu_metal=0, cu2o=first_comp[i], cuo=second_comp[i],
                                                                 show_plots=show_plots,
                                                          show_predicted_spectrum=show_predicted_spectrum,
                                                                 smoothing_params=smoothing_params)
            trues = []
            for i in range(0, len(first_comp)):
                trues.append(first_comp[i] * 1.07 + second_comp[i] * 2.12)
            # trues = np.asarray(test_rf_obj.mixed_valent_pred).T[2]
            plt.figure(figsize=(8, 7))
            plt.title('Experimental Cu(I) to Cu(II)', fontsize=26)


        elif catagory == '0-2':
            for i in range(0, len(first_comp)):
                # print(first_comp[i] + second_comp[i])
                self.predict_Experimental_Cu_non_integers(cu_metal=first_comp[i], cu2o=0, cuo=second_comp[i],
                                                                 show_plots=show_plots,
                                                          show_predicted_spectrum=show_predicted_spectrum,
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
        font = matplotlib.font_manager.FontProperties(size=28)
        text.set_font_properties(font)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(28)
            # t.set_weight('bold')
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.xlabel('True Value', fontsize=28)
        plt.ylabel('Bond Valance Prediction', fontsize=28)
        if savefigure:
            plt.savefig('Experimental Mixtures ' + catagory + '.pdf', bbox_inches='tight', transparent=True)

    def visualize_shift(self, material = 'All', show_stds = False, show_table = False, savefigure=False,
                        show_shift_labels = False, shift_labels = (0.9, 1.2, 1.2), spectrum_type = 'XAS'):
        """
        Generates a scatter plot showing how shifting the energy axis changes the spectrum's predicted oxidation
        state. This function rests on having already run 'predict_experiment_folder' on a spectrum at a series of
        different energy shifts.

        :param material: whether to show all Cu standards (Cu Metal, Cu2O and CuO) or an unlabeled spectrum (string,
        'All' or 'Unlabeled')
        :param show_stds: whether to print out a list of the lowest prediction standard deviations across a shift
        series (bool)
        :param show_table: whether to show a table of all prediction results across a shift series (bool)
        :param savefigure: whether to save the figures as a pdf (bool)
        :param show_shift_labels: Whether to draw vertical lines showing the 'true' location of the energy axis
        in the shift series (bool)
        :param shift_labels: the location of the above vertical lines (list/ndarray of float)
        :return: shows shift plot and generates a list of vales for each prediction
        """
        if material == 'All':
            mins = []
            maxes = []
            for mat in ['Cu Metal', 'Cu2O', 'CuO']:
                # extract predictions from each standard
                subdf = self.prediction_df.loc[self.prediction_df['Material'] == mat]
                prediction_std = np.asarray(subdf['Predictions Std'])

                mins.append(min(np.asarray(prediction_std)))
                maxes.append(max(np.asarray(prediction_std)))

            min_std = min(mins)
            max_std = max(maxes)

            bv = 0
            label_count = 0
            label_fontsizes = [21, 21, 19]
            for mat in ['Cu Metal', 'Cu2O', 'CuO']:
                # make scatterplot of each shift series
                full_output = [['Shift (eV)', 'Prediction', 'Prediction STD', 'True Oxidation State']]
                subdf = self.prediction_df.loc[self.prediction_df['Material'] == mat]
                shifts = np.asarray(subdf['Spectrum Energy Shift'])
                predictions = np.asarray(subdf['Prediction'])
                prediction_std = np.asarray(subdf['Predictions Std'])
                if mat == 'CuO':
                    plt.figure(figsize=(10, 7))
                else:
                    plt.figure(figsize=(8, 7))
                # print(len(count))
                sc = plt.scatter(shifts, predictions, s=200, c=prediction_std, vmin = round(min_std-0.05,1),
                            vmax = round(max_std+0.05, 1),)
                plt.title(mat + ' ' + spectrum_type, fontsize=36)
                if mat == 'Cu2O':
                    plt.title('Cu$_2$O' + ' ' + spectrum_type, fontsize=36)

                plt.xticks([-1.5, 0.0, 1.5], fontsize=36)
                ax = plt.gca()
                ax.get_xticks() # will get the current ticks
                ax.xaxis.set_minor_locator(AutoMinorLocator())

                ax.tick_params(which='major', width=4, length = 10)
                ax.tick_params(which='minor', width=2, length = 5)

                ax.xaxis.set_minor_locator(MultipleLocator(0.1))
                if spectrum_type == 'XAS':
                    plt.xlabel('Spectrum Shift (eV)', fontsize=36)

                if mat == 'CuO':
                    cb = plt.colorbar(sc, label='Prediction Std')
                    cb.set_ticks(np.linspace(round(min_std-0.05,1), round(max_std+0.05, 1), 4))
                    ax = cb.ax
                    text = ax.yaxis.label
                    font = matplotlib.font_manager.FontProperties(size=36)
                    text.set_font_properties(font)
                    cb.ax.set_yticklabels([0.1, 0.3, 0.5, 0.7], fontsize=36)


                # if mat == 'CuO':
                plt.ylim([-0.05,2.3])
                # plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
                if mat == 'Cu Metal':
                    plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=36)
                    plt.ylabel('Bond Valance Prediction', fontsize = 36)
                else:
                    plt.tick_params(labelleft=False)

                if show_shift_labels:
                    plt.vlines(0.0, -0.05, 2.3, linestyles='--', linewidth = 3, color = 'k',
                               label = 'Prediction Raw Spectrum')
                    plt.vlines(shift_labels[label_count], -0.05, 2.3, linestyles='--', linewidth = 3, color = 'r',
                               label = 'Prediction Manual Alignment')
                    plt.legend(fontsize = label_fontsizes[label_count])
                    label_count += 1


                if savefigure:
                    plt.savefig(spectrum_type + ' Spectrum Shift Analysis ' + mat + '.pdf', bbox_inches='tight', transparent=True)

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
        """
        Generates a scatter plot showing how changing the degree of smoothing changes the spectrum's predicted oxidation
        state. This function rests on having already run 'predict_experiment_folder' on a spectrum at a series of
        different smoothing values.

        :param material: whether to show all Cu standards (Cu Metal, Cu2O and CuO) or an unlabeled spectrum (string,
        'All' or 'Unlabeled')
        :param show_stds: whether to print out a list of the lowest prediction standard deviations across a shift
        series (bool)
        :param show_table: whether to show a table of all prediction results across a shift series (bool)
        :param savefigure: whether to save the figures as a pdf (bool)
        :return: shows smoothing plot and generates a list of vales for each prediction
        """
        if material == 'All':
            mins = []
            maxes = []
            for mat in ['Cu Metal', 'Cu2O', 'CuO']:
                # extract predictions from each standard
                subdf = self.prediction_df.loc[self.prediction_df['Material'] == mat]
                prediction_std = np.asarray(subdf['Predictions Std'])

                mins.append(min(np.asarray(prediction_std)))
                maxes.append(max(np.asarray(prediction_std)))

            min_std = min(mins)
            max_std = max(maxes)
            bv = 0

            for mat in ['Cu Metal', 'Cu2O', 'CuO']:
                # make scatterplot of each smoothing series
                full_output = [['Smoothing Window (eV)', 'Prediction', 'Prediction STD', 'True Oxidation State']]

                subdf = self.prediction_df.loc[self.prediction_df['Material'] == mat]
                shifts = np.asarray(subdf['Smoothing Window'])
                predictions = np.asarray(subdf['Prediction'])
                prediction_std = np.asarray(subdf['Predictions Std'])
                plt.figure(figsize=(8, 7))
                # print(len(count))
                sc = plt.scatter(shifts*0.03, predictions, s=200, c=prediction_std, vmin=round(min_std - 0.05, 1),
                                 vmax=round(max_std + 0.05, 1))
                plt.vlines(1.5, 0.15, 2.3, linewidth = 3, color = 'darkorange', linestyles=['--'],
                           label = 'Default Smoothing')
                if mat == 'CuO':
                    plt.legend(fontsize=23.25, loc='lower right')
                else:
                    plt.legend(fontsize=23.25)
                cb = plt.colorbar(sc, label='Prediction Std')
                cb.set_ticks([0.15, 0.3, 0.45])
                ax = cb.ax
                text = ax.yaxis.label
                font = matplotlib.font_manager.FontProperties(size=26)
                text.set_font_properties(font)
                for t in cb.ax.get_yticklabels():
                    t.set_fontsize(24)
                    # t.set_weight('bold')
                if mat == 'Cu2O':
                    mat = 'Cu$_2$O'
                plt.title(mat + ' vs Smoothing', fontsize=28)
                plt.xticks(fontsize=24)
                # if mat == 'CuO':
                plt.ylim([0.15, 2.3])
                # plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
                plt.yticks(fontsize=24)
                plt.xlabel('Smoothing Window (eV)', fontsize=26)
                plt.ylabel('Bond Valance Prediction', fontsize=26)

                for i in range(0, len(shifts)):
                    full_output.append([shifts[i]*0.03, round(predictions[i], 2), prediction_std[i], bv])
                bv += 1

                if show_stds:
                    stds = pd.DataFrame(np.asarray([prediction_std, shifts, predictions]).T, columns = ['std', 'Smoothing Window', 'predictions'])
                    print(stds.sort_values('std').head())
                if show_table:
                    print(tabulate(full_output, headers='firstrow', tablefmt='fancy_grid'))
                if mat == 'Cu$_2$O':
                    mat = 'Cu2O'
                if savefigure:
                    plt.savefig('Spectrum Smoothing Analysis ' + mat + '.pdf', bbox_inches='tight', transparent=True)

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
                                          exp_spectrum_type = 'TEAM I',
                                          smoothing_parms = (), exp_scale = 0.1,
                                         points_to_average = 20, cumulative_spectrum = True,
                                         spectrum_energy_shift = 0.0, energies_range = [925, 970],
                                         theory_column = 'XAS_aligned_925_970', print_prediction = True,
                                         use_dnn = False, dnn_model = None, show_plots = False, show_hist = False,
                                         show_inputted_spectrum = True, save_all_figures = False, savefigure = False):

        """
        This function takes in an experimental EELS/XAS spectrum and predicts its oxidation state using our random
        forest model. It returns the predicted state, the prediction standard deviation and can also show plots with
        detailed statistics regarding the prediction.

        :param exp_spectrum: Filepath to the experimental EELS/XAS Spectrum (string)
        :param theory_index: index in the dataframe corresponding to the simulated spectrum which matches this spectrum.
        This is used to assign a true value to the oxidation state of the experimental spectrum (int)
        :param material: name of the material the spectrum was taken from (string)
        :param exp_spectrum_type: either TEAM I (opened with ncempy's dm4 reader) or csv (opened with pandas' read_csv)
        (string)
        :param smoothing_parms: parameters for smoothing with savgol filter (window size and polynomial order) (list
        of int)
        :param exp_scale: energy axis scale for the experimental spectrum (float)
        :param points_to_average: points in the tail of the spectrum to average over to create a normalization value
        (int)
        :param cumulative_spectrum: whether to turn the experimental spectrum into a cumulative spectrum for prediction
        (bool)
        :param spectrum_energy_shift: whether and how much to shift the experimental spectrum's energy axis (float)
        :param energies_range: the range of energy axis values for the experimental spectrum to be able to match
        simulations (list of int)
        :param theory_column: column in the dataframe containing the simulated spectrum that will match the experimental
        spectrum for visualization and comparison purposes (string)
        :param print_prediction: whether to print the predicted oxidation state and standard deviation (bool)
        :param use_dnn: whether to use a dnn model rather than random forest (bool)
        :param dnn_model: the dnn model paired with the above (sequential dnn model)
        :param show_plots: whether to show a series of plots illustrating each trainsformation done to the spectrum
        as the function runs (bool)
        :param show_hist: whether to show a histogram of predictions of each decision tree (bool)
        :param show_inputted_spectrum: whether to plot the inputted spectrum for visualization purposes (bool)
        :param savefigure: whether to save all these plots (bool)
        :return:
        """
        # define attributes
        self.cumulative_spectrum = cumulative_spectrum
        self.theory_index = theory_index
        self.exp_spec_path = exp_spectrum
        self.material = material
        self.points_to_average = points_to_average
        self.spectrum_energy_shift = spectrum_energy_shift

        if exp_spectrum_type == 'TEAM I':
            # load data
            output = nio.dm.dmReader(exp_spectrum)
            intens = output['data']
            energies = output['coords'][0]

            if show_plots:
                plt.figure(figsize=(8, 7))
                plt.xticks([900, 920, 940, 960, 980], fontsize=36)
                plt.yticks(fontsize=36)
                plt.ylabel('Intensity', fontsize=36)
                plt.xlabel('Energy (eV)', fontsize=36)
                plt.title('Raw Experimental Spectrum', fontsize=36)
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
                if save_all_figures:
                    plt.savefig(material + ' Raw Spectrum.pdf', bbox_inches='tight', transparent=True)
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


            # smooth specctrum
            self.smoothing_params = smoothing_parms
            intens = savgol_filter(intens, self.smoothing_params[0], self.smoothing_params[1])




        if exp_spectrum_type == 'csv':
            # load data
            output = pd.read_csv(exp_spectrum)
            # Baseline subtract spectrum. This is because the csv file used in this work is extracted from literature
            # and its baseline is far above zero due to the axes of the literature extraction.
            # TODO This will need to be updated before broad use with csv data is possible
            intens_temp = output['Intensity'] - min(output['Intensity'])
            intens = intens_temp / intens_temp[len(intens_temp) - 5]
            energies = output['Energy (eV)']
            if show_plots:
                plt.xticks([900, 920, 940, 960, 980], fontsize=36)
                plt.yticks(fontsize=36)
                plt.ylabel('Intensity', fontsize=36)
                plt.xlabel('Energy (eV)', fontsize=36)
                plt.title('Raw Experimental Spectrum', fontsize=22)
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


            self.scale_experimental_spectra(intens, energies, scale=exp_scale) # this is being done before smoothing
            # here because the extracted spectrum isn't on an even scale, which makes
            # assigning smoothing parameters to a meaningful energy range very challenging
            intens = self.interped_intens
            energies = self.interped_energies


            # smooth spectrum
            self.smoothing_params = smoothing_parms
            intens = savgol_filter(intens, self.smoothing_params[0],
                                                     self.smoothing_params[1])

        if show_plots:
            plt.figure(figsize=(8, 7))
            plt.xticks([900, 920, 940, 960, 980], fontsize=36)
            plt.yticks(fontsize=36)
            plt.ylabel('Intensity', fontsize=36)
            plt.xlabel('Energy (eV)', fontsize=36)
            plt.title('Post Smoothing', fontsize=36)
            # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
            plt.plot(energies, intens / max(intens),
                     label='Experimental Spectrum', linewidth=3, zorder=10)
            # plt.plot(self.spectra_df.iloc[theory_index][self.energy_col],
            #          self.spectra_df.iloc[theory_index][theory_column][0:451] / max(
            #              self.spectra_df.iloc[theory_index][theory_column][0:451]),
            #          label='From FEFF', linewidth=3)
            if save_all_figures:
                plt.savefig(material + ' Post Smoothing Spectrum.pdf', bbox_inches='tight', transparent=True)
            plt.show()

        # scale spectrum to have 0.1 eV energy axis
        self.scale_experimental_spectra(intens, energies, scale = exp_scale) # has no effect for the csv extracted xas
        # paper, since that has already been scaled to 0.1 eV

        if show_plots:
            plt.figure(figsize=(8, 7))
            plt.plot(self.interped_energies, self.interped_intens / max(self.interped_intens),
                     label='Experimental Spectrum', linewidth=3, zorder=10)
            plt.xticks([900, 920, 940, 960, 980], fontsize=36)
            plt.yticks(fontsize=36)
            plt.ylabel('Intensity', fontsize=36)
            plt.xlabel('Energy (eV)', fontsize=36)
            plt.title('Post Interpolation', fontsize=36)
            if save_all_figures:
                plt.savefig(material + ' Post Interpolated Spectrum.pdf', bbox_inches='tight', transparent=True)
            plt.show()

        # shift spectrum if inputted
        if self.spectrum_energy_shift != 0.0:
            diff = self.spectrum_energy_shift * 10 # will need a factor of 10 more points than the inputted shift
            # due to the 0.1 eV spacing. For example, to shift 0.2 eV that would require two additional points
            spectrum_use = list(self.interped_intens)
            energies_use = list(energies)
            energies_min = 924.9
            energies_max = max(energies_use)
            if diff > 0:
                # to shift in the positive direction, keep energy axis fixed but add zeros to the low energy edge and
                # knock off points at the end. NOTE if the experimental spectrum starts well before 925, this shifting
                # procedure will not result in a line of meaningless zeros in the low energy edge. Instead, the lower
                # energy region of the inputted spectrum will just be shifted upwards
                points_to_add = np.zeros((int(diff)))
                for point in points_to_add:
                    spectrum_use.insert(0, point)
                    energies_use.insert(0, energies_min)
                    energies_min = round(energies_min - 0.1, 1)
                spectrum_use = spectrum_use[0:len(self.interped_intens)]
            if diff < 0:
                # to shift in the negative direction, add points to the end of the spectrum and knock points off in the
                # beginning. The points added at the end are an average of the last 20 points in the spectrum. Similar
                # to the case outlined above, this approach will not result in generating meaningless features at the
                # high energy edge of the spectrum if the high energy point of the inputted spectrum is higher than
                # 970 eV
                points_to_add = np.zeros((int(np.abs(diff))))
                intens_max = np.mean(spectrum_use[len(spectrum_use) - points_to_average:len(spectrum_use)])
                for point in points_to_add:
                    energies_max = round(energies_max + 0.1, 1)
                    spectrum_use.append(intens_max)
                    energies_use.append(energies_max)

                spectrum_use = spectrum_use[int(np.abs(diff)):len(spectrum_use)]

            spectrum_use = np.asarray(spectrum_use)
            # scale spectrum so it's normalized to high energy tail (this no longer means anything for the cumulative
            # spectrum prediction, since those are normalized to one, but it is still helpful for visualization and
            # comparison to other spectra, so this operation is still performed)
            self.interped_intens = spectrum_use / np.mean(spectrum_use[len(spectrum_use) - points_to_average:len(spectrum_use)])

        # this is a holdover from when the spectra were broadened as a part of the analysis. It no longer does anything
        # significant to the spectrum
        self.broadened_intens = self.interped_intens


        self.broadened_intens =  self.broadened_intens / np.mean(self.broadened_intens[len(self.broadened_intens) - points_to_average:len(self.broadened_intens)])

        # set energy axis to 925-970
        self.final_energies = np.arange(energies_range[0], energies_range[1]+0.2, 0.1)[0:((energies_range[1]-energies_range[0])*10)+1]

        # interoplate experimental spectrum so it is also on this axis
        energies_interp_use = []
        for i in self.final_energies :
            energies_interp_use.append(round(i, 1))
        self.final_energies = energies_interp_use
        f = interp1d(self.interped_energies, self.broadened_intens)

        self.intensities_final = f(self.final_energies)

        if show_plots:
            plt.figure(figsize=(8, 7))
            plt.xticks(fontsize=36)
            plt.yticks(fontsize=36)
            plt.xlabel('Energy (eV)', fontsize=36)
            plt.ylabel('Intensity', fontsize = 36)
            plt.title('Post Cropping', fontsize=36)
            # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
            plt.plot(self.final_energies, self.intensities_final,
                     label='Experimental Spectrum', linewidth=4, zorder=10)
            if save_all_figures:
                plt.savefig(material + ' Post Cropping Spectrum.pdf', bbox_inches='tight', transparent=True)
            plt.show()
            # plt.plot(self.spectra_df.iloc[theory_index][self.energy_col],
            #          self.spectra_df.iloc[theory_index][theory_column][0:451],
            #          label='From FEFF', linewidth=4)
            plt.figure(figsize=(8, 7))
            plt.xticks([930, 950, 970], fontsize=36)
            plt.yticks(fontsize=36)
            plt.ylabel('Intensity', fontsize=36)
            plt.xlabel('Energy (eV)', fontsize=36)
            if material == 'Cu2O':
                material = 'Cu$_2$O'
            plt.title('EELS Spectrum ' + material, fontsize=30)
            if material == 'Cu$_2$O':
                material = 'Cu2O'
            # plt.plot(interp_energies_final,interped_intens_smoothed, label = 'Experiment Smoothed', linewidth = 3)
            plt.plot(self.final_energies, self.intensities_final,
                     label='Experimental Spectrum', linewidth=4, zorder=10)
            if savefigure:
                plt.savefig(material + ' Exp Spectrum.pdf', bbox_inches='tight', transparent=True)
            plt.show()

        # another somewhat meaningless rescaling, since it will all be undone by the cumulative spectrum operation
        self.intensities_final = self.intensities_final / np.mean(self.intensities_final[len(self.intensities_final) - points_to_average:len(self.intensities_final)])

        # compute the cumulative spectrum
        if self.cumulative_spectrum:
            temp_intens = []
            for k in range(0, len(self.intensities_final)):
                temp_intens.append(sum(self.intensities_final[0:k]))
            self.intensities_final = temp_intens/max(temp_intens)
        # print(self.intensities_final)
        if show_inputted_spectrum:
            plt.figure(figsize=(8, 7))
            # plt.xlabel('Energy (eV)', fontsize=36)
            # plt.ylabel('Intensity', fontsize = 36)
            # plt.xticks([930, 950, 970], fontsize=36)
            # plt.yticks([0, 0.5, 1], fontsize=36, fontweight='bold')
            plt.yticks([], fontsize = 36)
            plt.xticks([])
            # plt.title('Cumulative Spectrum', fontsize=36)

            # self.intensities_final = self.intensities_final - min(self.intensities_final)

            # print(self.intensities_final)
            plt.plot(self.final_energies, self.intensities_final,
                          linewidth=5, zorder=10)


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

        # predict spectrum using the random forest model and extract predictions from each decision tree
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

        # store predictions
        self.prediction = round(pred[0], 2)
        self.prediction_std = round(predictions_std, 2)
        self.true_val = self.spectra_df.iloc[theory_index]['BV Used For Alignment']

        if show_hist:
            plt.figure(figsize=(8, 7))
            if material == 'Cu2O':
                material = 'Cu$_2$O'
            plt.title('Prediction Histogram ' + material, fontsize=30)
            if material == 'Cu$_2$O':
                material = 'Cu2O'



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
            plt.xlabel('Prediction', fontsize=36)
            plt.ylabel('Num Trees', fontsize=36)

            plt.xticks([0, 0.5, 1, 1.5, 2, 2.5], fontsize=36)
            plt.yticks([100, 200, 300], fontsize=36)
            font = font_manager.FontProperties(
                style='normal', size=26)
            plt.legend(prop=font)
            if savefigure:
                plt.savefig('Prediction Histogram ' + material[0:5] + '.pdf', bbox_inches='tight', transparent=True)
            # plt.legend(fontsize=20,  loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.plot(np.arange(0,max(errors),0.1), np.arange(0,max(errors),0.1), color = 'k', linewidth = 3, linestyle = '--')
            plt.show()

    def show_feature_importances(self, material_type='Cu', savefigure=False):
        """
        Visualize a plot of the feature importances for the model. The feature importances are plotted against the
        spectra energy axis, as each feature corresponds to the intensity at that energy
        :param material_type: Element trained on (string)
        :param savefigure: whether to save the plot as a pdf (bool)
        :return: none, shows plot
        """
        plt.figure(figsize=(8, 6))
        plt.xticks([930, 950, 970], fontsize=32)
        plt.yticks([0.0, 0.05, 0.1, 0.15], fontsize=32)
        # plt.ylabel('Normalized Variance Reduction', fontsize=30)
        plt.ylabel('Importance', fontsize=36)
        plt.xlabel('Energy (eV)', fontsize=36)
        plt.title('Feature Importances', fontsize=36)

        # extract and plot feature importances
        plt.plot(np.asarray(self.spectra_df.iloc[0][self.energy_col]),
                 self.rf_model.feature_importances_, linewidth=5, label=material_type, color='#ff7f0e')
        # plt.legend(fontsize=24)
        if savefigure:
            plt.savefig('Feature Importances.pdf', bbox_inches='tight', transparent=True)
        plt.show()


    def show_errors_histogram(self, nbins = 50, title = 'Error Histogram', show_rmse = True, show_type = 'Abs Error',
                              savefigure=False, error_df = None, yticks = None):
        """
        Shows a histogram of absolute value errors for the predictions in the test set. The RMSE is drawn in a solid
        green line and labeled with text.
        :param nbins: number of bins in the histogram (int)
        :param title: title of the plot (string)
        :param show_rmse: whether to show the vertical line indicating the RMSE (bool)
        :param show_type: whether to show the absolute error 'Abs Error' or the square error 'MSE' (string)
        :param savefigure: whether to save the plot as a pdf (bool)
        :param error_df: defaults to using the error dataframe generated by the model training procedure, but a
        different one can be inputted here (pandas dataframe)
        :param yticks: the y ticks to use in the plot (list of int or None)
        :return: none, plot is shown
        """

        if type(error_df) == type(None):
            error_df = self.rf_error_df


        # extract errors from dataframe
        errors = error_df['Errors']

        # calculate RMSE and MSE
        MAE = np.mean(errors)
        MSE_list = np.square(errors)
        MSE = np.mean(MSE_list)
        RMSE = math.sqrt(MSE)

        print('RMSE ' + str(RMSE))
        plt.figure(figsize=(8, 6))
        plt.title(title, fontsize=24)
        if show_type == 'Abs Error':
            hist = plt.hist(errors, bins=nbins)
        elif show_type == 'MSE':
            hist = plt.hist(np.square(errors), bins=nbins)

        if show_rmse:
            # add solid green line showing RMSE
            plt.vlines(RMSE, max(hist[0]), min(hist[0]), color='limegreen', linewidth=5, label='RMSE')
            plt.vlines(MAE, max(hist[0]), min(hist[0]), color='red', linewidth=5, label='MAE')
            plt.text(RMSE + 0.25, max(hist[0]) - 0.2 * max(hist[0]), 'RMSE = ' + str(round(RMSE, 2)),
                     horizontalalignment='left', fontsize=28)
            plt.text(RMSE + 0.25, max(hist[0]) - 0.1 * max(hist[0]), 'MAE = ' + str(round(MAE, 2)),
                     horizontalalignment='left', fontsize=28)
        plt.xticks([0.0, 0.4, 0.8, 1.2], fontsize=32)
        if type(yticks) != list:
            plt.yticks([0.0, 250, 500, 750], fontsize=32)

        else:
            plt.yticks(yticks, fontsize=32)
        # plt.ylabel('Normalized Variance Reduction', fontsize=30)
        plt.ylabel('Frequency', fontsize=36)
        plt.xlabel('Abs Error', fontsize=36)
        # plt.title('Feature Importances ' + self.predicted_col, fontsize=22)
        plt.title('Error Histogram', fontsize=36)
        if savefigure:
            plt.savefig('Error Histogram.pdf',  bbox_inches='tight', transparent=True)
        plt.show()


    def show_mixture_samples_accuracy(self, savefigure = False):
        """
        Shows the R2 and RMSE plots for the integer samlpes that make up the generated mixed valent spectra
        :return: None, shows plot
        """
        # find the materials IDs for the materials that make up the mixed valent samples
        mixture_ids = []
        for i in range(0, len(self.spectra_df)):
            mp_id = self.spectra_df.iloc[i]['mpid_string']
            if type(mp_id) == list:
                for m_id in mp_id:
                    if m_id not in mixture_ids:
                        mixture_ids.append(m_id)

        # extract these spectra and predict the sub dataset using 'predict_set_of_spectra'
        mixture_df_slice = self.spectra_df.loc[self.spectra_df['mpid_string'].isin(mixture_ids)]
        error_df_mixtures = self.predict_set_of_spectra(mixture_df_slice)

        # show r2 and rmse
        self.show_r2(error_df=error_df_mixtures, savefigure=savefigure)
        self.show_errors_histogram(error_df=error_df_mixtures, yticks=[0, 50, 100, 150], savefigure=savefigure)

    def plot_errors_vs_std(self, scatter_spot_multiplier = 15):
        """
        Plots each spectrum in the test set by its error vs its prediction standard deviation
        :param scatter_spot_multiplier: size of the scatter plots (int)
        :return: None, shows plot
        """
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

    def show_r2(self, scatter_spot_multiplier = 15, savefigure=False, error_df = None, show_value_counts_plot = False,
                include_integer=True):
        """
        Show the R2 plot for a set of predictions. Predictions and true oxidation states are rounded to the nearest 0.1.
        Scatter plot sizes are scaled to the number of data points at that point and are colored by the average
        prediction standard deviation for the data points at that point.
        :param scatter_spot_multiplier: size of scatter plot points (int)
        :param savefigure: whether to save the figure as a pdf (bool)
        :param error_df: defaults to the error df from the training function, but a new one can be inputted here
        (pandas dataframe)
        :param show_value_counts_plot: whether to show another plot with scatter points colored by number of values
        rather than prediction standard deviation (bool)
        :return: None, plot is shown
        """

        if type(error_df) == type(None):
            error_df = self.rf_error_df
        print('num spectra = ' +str(len(error_df)))

        print('model accuracy (R^2) on simulated test data ' + str(self.accuracy))

        true = []
        pred = []
        count = []
        condensed_stds = []
        # extract predictions and labels and round them to 0.1
        for i in np.asarray(error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts().index):
            if include_integer==False:
                if round(i[1], 1) in [0.0,1.0,2.0]:
                    pass
                else:
                    pred.append(round(i[0], 1))
                    true.append(round(i[1], 1))
                    condensed_stds.append(np.mean(error_df.loc[
                                                      (error_df['Predictions Rounded'] == round(i[0], 1)) & (
                                                              error_df['Labels Test Rounded'] == round(i[1], 1))][
                                                      'Predictions Std']))
            else:
                pred.append(round(i[0], 1))
                true.append(round(i[1], 1))
                condensed_stds.append(np.mean(error_df.loc[
                                                  (error_df['Predictions Rounded'] == round(i[0], 1)) & (
                                                          error_df['Labels Test Rounded'] == round(i[1], 1))][
                                                  'Predictions Std']))

        for i in np.asarray(error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts().index):
            if include_integer == False:
                if round(i[1], 1) in [0.0, 1.0, 2.0]:
                    pass
                else:
                    count.append(error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts()[i])
            else:
                count.append(error_df[['Predictions Rounded', 'Labels Test Rounded']].value_counts()[i])

        print(len(count))
        print(len(pred))
        count = np.asarray(count)
        print(count)
        if show_value_counts_plot:
            # create scatter plot with points colored by number of datapoints at that position
            plt.figure(figsize=(8, 6))
            plt.scatter(true, pred, s=count * scatter_spot_multiplier, c=count)

            cb = plt.colorbar(label='Num Predictions')
            ax = cb.ax
            text = ax.yaxis.label
            print(text)
            font = matplotlib.font_manager.FontProperties(size=32)
            text.set_font_properties(font)
            cb.ax.set_yticklabels(fontsize=32)

            min_plot = round(min(error_df['Labels Test']) - 0.5, 0)
            max_plot = round(max(error_df['Labels Test']) + 1.5, 0)
            plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3,
                     linestyle='--')
            plt.title('Model Performance', fontsize=24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

            plt.ylabel('Bond Valance Prediction', fontsize=22)
            plt.xlabel('True Bond Valance', fontsize=22)
            plt.show()

        # create scatter plot with points colored by average prediction std across the datapoints at that position
        plt.figure(figsize=(8, 6))

        plt.xticks([0.0, 1.0, 2.0, 3.0], fontsize=32)
        plt.yticks([0.0, 1.0, 2.0, 3.0], fontsize=32)
        # plt.ylabel('Normalized Variance Reduction', fontsize=30)
        plt.ylabel('Predicted Valence', fontsize=36)
        plt.xlabel('True Valence', fontsize=36)
        # plt.title('Feature Importances ' + self.predicted_col, fontsize=22)
        plt.title('Model Performance', fontsize=36)

        min_plot = round(min(error_df['Labels Test']) - 0.5, 0)
        max_plot = round(max(error_df['Labels Test']) + 1.5, 0)
        plt.plot(np.arange(min_plot, max_plot, 1), np.arange(min_plot, max_plot, 1), color='k', linewidth=3,
                 linestyle='--')

        plt.scatter(true, pred, s=count * scatter_spot_multiplier, c=condensed_stds)
        cb = plt.colorbar(label='Prediction Std', ticks = [0.1, 0.2, 0.3, 0.4, 0.5])
        ax = cb.ax
        text = ax.yaxis.label
        font = matplotlib.font_manager.FontProperties(size=28)
        text.set_font_properties(font)
        for t in cb.ax.yaxis.get_ticklabels():
            # t.set_weight("bold")
            t.set_fontsize(32)

        if savefigure:
            plt.savefig('r^2 plot.pdf',  bbox_inches='tight', transparent=True)

    def predictions_from_threshold(self, threshold, show_plot=False):
        """
        Determines R2 and RMSE for a subset of the test set where all predictions with a standard deviation greater
        than a specified threshold are removed
        :param threshold: standard deviation threshold (float)
        :param show_plot: whether to show r2 and rmse plots (bool)
        :return: list of r2, number of test data points under the threshold and rmse
        """
        # grab samples under threshold
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
        # calculate RMSE and Errors
        MSE = np.square(np.subtract(labels_test, predictions)).mean()
        RMSE = math.sqrt(MSE)
        errors = np.abs(np.subtract(labels_test, predictions))

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
        """
        Wrapper function which runs 'predictions_from_threshold' for a series of thresholds and displays the results,
        focusing on how the threshold changes R2, RMSE and the percentage of the test set predicted
        :param thresholds: standard deviation thresholds to analyze (list/ndarray of float)
        :param ylims: ylims of the threshold plot (list/ndarray of float)
        :param width_multiplier: with of bar plot scaling (float)
        :param text_height: height of plot text above bar plot (float)
        :param text_fontsize: fontsize of bar plot text (int/flaot)
        :param savefigure: whether to save the figure as a pdf
        :param show_type: text to show above the bars, either 'percentage_predicted' or RMSE (string)
        :param color_scheme: color scheme for bars (string)
        :param yticks_to_use: yticks for the barplots
        :return: None, plots are shown
        """

        r2s = []
        percentage_predicted = []
        rmse = []
        # generate lists of results across thresholds
        for thresh in thresholds:
            output = self.predictions_from_threshold(thresh, show_plot=False)
            # print(output[0])
            # print(thresh)
            r2s.append(output[0])
            # print(output[1])
            percentage_predicted.append(output[1] / len(self.rf_error_df))
            rmse.append(output[2])

        # scale data for color mapping
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
        font = matplotlib.font_manager.FontProperties(size=22)
        text.set_font_properties(font)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(22)
            # t.set_weight('bold')

        plt.xticks(fontsize=22)
        if len(yticks_to_use) == 0:
            plt.yticks(fontsize=22)
        else:
            plt.yticks(yticks_to_use, fontsize = 22)
        plt.xlabel('Std Cutoff', fontsize=22)
        plt.ylabel('Accuracy (R^2)', fontsize=22)
        plt.title('Std Cutoff Threshold vs Accuracy', fontsize=22)
        plt.ylim(ylims)
        for i in range(0, len(r2s)):
            if show_type == 'percentage_predicted':
                plt.text(thresholds[i], r2s[i] + text_height, str(round(percentage_predicted[i], 2)), horizontalalignment='center',
                 fontsize=text_fontsize)
            if show_type == 'RMSE':
                plt.text(thresholds[i], r2s[i] + text_height, str(round(rmse[i], 2)), horizontalalignment='center',
                 fontsize=text_fontsize)
        if savefigure:
            plt.savefig(show_type + 'bar_chart_with_colorbar.pdf', bbox_inches='tight')
        plt.show()

    def augment_df_with_mixtures(self, len_mixtures=100, len_combinations=20):
        """
        Wrapper function which adds a specified number of mixture samples to our spectral dataset.
        :param len_mixtures: number of integer spectra to combine into mixture spectra. This parameter will random
        sample the integer spectra 9 total times this number
        :param len_combinations: for each combination of integer spectra, the number of different mixture spectra to
        generate (ie a mixture of Cu(0), Cu2O and CuO will be mixed with different ratios 20 times if this value is 20)
        :return: Updates dataframe with mixtures
        """
        # 100 evenly spaced components from 0 to 1 which will be random sampled to build out the mixture spectra
        cu_metal = list(np.linspace(1, 0, 101))
        cu_1 = list(np.linspace(0, 1, 101))
        cu_2 = list(np.linspace(0, 1, 101))

        # extract integer valent spectra from full dataset
        zeros = self.spectra_df.loc[self.spectra_df['BV Used For Alignment'] == 0.0]
        ones = self.spectra_df.loc[self.spectra_df['BV Used For Alignment'] == 1.0]
        twos = self.spectra_df.loc[self.spectra_df['BV Used For Alignment'] == 2.0]

        # generate mixture dataset based on three different types of mixtures:
            # mixtures of all three oxidation states, 0, 1 and 2
            # mixtures of just 0 and 1
            # mixtures of just 1 and 2

        mixture_df = self.add_mixed_valent_spectra(zeros, ones, twos, mixture_type='all', len_mixtures=len_mixtures,
                                              len_combinations=len_combinations, cu_metal=cu_metal, cu_1=cu_1,
                                                   cu_2=cu_2)
        mixture_0_1 = self.add_mixed_valent_spectra(zeros, ones, twos, mixture_type='0-1', len_mixtures=len_mixtures,
                                               len_combinations=len_combinations, cu_metal=cu_metal, cu_1=cu_1,
                                                    cu_2=cu_2)
        mixture_1_2 = self.add_mixed_valent_spectra(zeros, ones, twos, mixture_type='1-2', len_mixtures=len_mixtures,
                                               len_combinations=len_combinations, cu_metal=cu_metal, cu_1=cu_1,
                                                    cu_2=cu_2)

        # compile these into dataframes and add to existing data
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
        """
        Adds mixed valent spectra to an existing spectra dataset using integer valent spectra stored in that dataset
        :param zeros: dataframe of just Cu metal (pandas dataframe)
        :param ones: dataframe of just Cu(I) (pandas dataframe)
        :param twos: dataframe of just Cu(II) (pandas dataframe)
        :param mixture_type: type of mixture to create, 'all', '0-1', or '1-2'. This indicates the oxidation states
        which will be blended into the mixture (string)
        :param len_mixtures: number of times to run through this mixture generation process. Each one of these 100
        mixtures has a randomly drawn set of three integer oxidation states (even if one of those three isn't being
        used) (int)
        :param len_combinations: For a given set of three integer valent spectra, the number of different ratios to
        use to make a mixture  (int)
        :param cu_metal: values to random sample for Cu metal's mixture ratios (list/ndarray)
        :param cu_1: values to random sample for Cu metal's mixture ratios (list/ndarray)
        :param cu_2: values to random sample for Cu metal's mixture ratios (list/ndarray)
        :return:
        """

        mixture_df = []
        np.random.seed(32)

        for k in range(0, len_mixtures):
            # for each mixture (in this case there are 100) we draw a random Cu(0), a random Cu(I) and a random Cu(II)
            # therefore 900 draws are occuring, since this is happening over all three mixture types over the full
            # mixture df generation process, and the mixture spectra are comprised of 541 unique integer valence
            # spectra. 359 draws result in the same spectrum as a previous draw at this random seed
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
                # draw random ratios for each of the three oxidation states for 'all' case
                metal_fractions = np.random.choice(cu_metal, len_combinations)
                cuo_fractions = np.random.choice(cu_2, len_combinations)

            elif mixture_type == '0-1':
                metal_fractions = np.random.choice(cu_metal, len_combinations)
                # Cu(II) is zero for '0-1' mixtures, however, for notational purposes the Cu(II) is still considered
                # part of the mixtures, just at zero contribution
                cuo_fractions = np.zeros((len_combinations))

            elif mixture_type == '1-2':
                # Cu(0) is zero for '1-2' mixtures, however, for notational purposes the Cu(0) is still considered
                # part of the mixtures, just at zero contribution
                metal_fractions = np.zeros((len_combinations))
                cuo_fractions = np.random.choice(cu_2, len_combinations)

            # every mixture contains a non zero Cu(I)
            cu2o_fractions = np.random.choice(cu_1, len_combinations)


            for j in range(0, len_combinations):
                sum_comp = metal_fractions[j] + cu2o_fractions[j] + cuo_fractions[j]
                # if a mixture sums to zero contributions by random chance it's skipped to avoid a divide by zero error
                if sum_comp == 0:
                    pass
                # make sure components add to one and are scaled properly
                else:
                    components = [metal_fractions[j], cu2o_fractions[j], cuo_fractions[j]] / sum_comp
                    # print(components)

                    # generate mixture spectra by multiplying components by their ratios
                    mixed_spec = zero['TEAM_1_aligned_925_970'] * components[0] + \
                                 one['TEAM_1_aligned_925_970'] * components[1] + \
                                 two['TEAM_1_aligned_925_970'] * components[2]

                    mixed_cum_spec = zero['Cumulative_Spectra_TEAM_1_aligned_925_970'] * components[0] + \
                                     one['Cumulative_Spectra_TEAM_1_aligned_925_970'] * components[1] + \
                                     two['Cumulative_Spectra_TEAM_1_aligned_925_970'] * components[2]

                    mixed_cum_spec = mixed_cum_spec / max(mixed_cum_spec)

                    # old test that is hopefully no longer relevant
                    if np.isnan((max(mixed_cum_spec))):
                        print(metal_fractions[j], cu2o_fractions[j], cuo_fractions[j])
                        print(sum_comp)
                        print(components)
                        raise ValueError

                    # label mixture spectra by material id and ratios of components
                    mat_ids = [zero.mpid_string, one.mpid_string, two.mpid_string]
                    formulas = [zero.pretty_formula, one.pretty_formula, two.pretty_formula]
                    # print(mat_ids)
                    bv = round(components[0] * 0 + components[1] * 1 + components[2] * 2, 2)
                    # print(bv)

                    mixture_df.append(
                        [mixed_spec, mat_ids, formulas, bv, mixed_cum_spec, zero['new Scaled Energies use'], components])

        return mixture_df


    def predict_set_of_spectra(self, df_slice, using_error_df = False):
        """
        Runs predictions on a subset of a pandas dataframe containing spectra and oxidation state labels
        :param df_slice: dataset to predict (pandas dataframe)
        :param using_error_df: whether the inputted dataframe is an error dataframe (bool)
        :return: dataframe containing predictions, spectra and other labels/prediction statistics
        """
        df_slice.reset_index(inplace = True)
        # extract spectra and labels
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

        # predict extracted spectra
        predictions = self.rf_model.predict(spectra_to_predict)
        accuracy = self.rf_model.score(spectra_to_predict, labels_to_predict)
        print('Accuracy = ' + str(accuracy))

        # compile predictions from each tree
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

        # store predictions and other labels in dataframe, extract info based on whether it is an error df or not
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
                                             test_fraction=0.25,  show_uncertianty=True,num_trees=500,
                                             energy_col='new Scaled Energies use', max_features='auto'):
        """
        This function trains the random forest model based on the pandas dataframe provided when the object was
        initialized. It then generates attributes corresponding to the model itself, a dataframe describing the
        predictions, the training set and the test set.

        :param bv_column: the column in the dataframe containing the oxidation state labels (string)
        :param spectra_to_predict: the column in the dataframe containing the spectra (string)
        :param test_fraction: fraction of the dataset used for testing (float between 0 and 1)
        :param show_uncertianty: whether to include detailed statistical metrics about each prediction in the final
        output (bool)
        :param num_trees: number of trees in the random forest (int)
        :param energy_col: the column in the dataframe containing the energy values corresponding to the spectrum
        (string)
        :param max_features: the maximum features each decision tree in the random forest has access to (string or int,
        if string one of 'auto' or 'sqrt')
        :return: None, output saved as attributes
        """
        # label spectra and energy value
        self.energy_col = energy_col
        self.predicted_col = spectra_to_predict
        error_df = None
        # load spectra df from filepath if this hasn't been done yet
        if type(self.spectra_df) == type(None):
            print('no spectra df, loading ' + self.spectra_df_filepath)
            self.load_spectra_df()

        # drop the stable forms of Cu metal, Cu2O and CuO to ensure validation experimental spectra aren't baised
        # by having these in the training set
        cu_metal_index = self.spectra_df.loc[self.spectra_df['mpid_string'] == 'mp-30'].index[0]
        cu2o_index = self.spectra_df.loc[self.spectra_df['mpid_string'] == 'mp-361'].index[0]
        cuo_index = self.spectra_df.loc[self.spectra_df['mpid_string'] == 'mp-704645'].index[0]

        self.spectra_df_no_oxides = self.spectra_df.drop([cu_metal_index, cu2o_index, cuo_index])
        # self.spectra_df_no_oxides = self.spectra_df
        self.spectra_df_no_oxides = self.spectra_df_no_oxides.reset_index()

        # extract oxidation states and spectra from dataframe
        bond_valences = self.spectra_df_no_oxides[bv_column]
        scaled_spectra = np.asarray(self.spectra_df_no_oxides[spectra_to_predict])

        # split into training and test sets
        spectra_train, spectra_test, labels_train, labels_test = train_test_split(scaled_spectra, bond_valences,
                                                                                  test_size=test_fraction,
                                                                                  random_state=32)

        # process spectra so they're all arrays
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

        # train model
        rf_model = RandomForestRegressor(n_estimators=num_trees, n_jobs=-1, max_features=max_features,
                                         random_state=32)

        rf_model.fit(self.spectra_train, self.labels_train)
        accuracy = rf_model.score(np.asarray(updated_spectra_test), np.asarray(labels_test))
        print('model accuracy (R^2) on simulated test data ' + str(accuracy))
        self.accuracy = accuracy

        # predict test set
        predictions = rf_model.predict(updated_spectra_test)

        if show_uncertianty:
            # build prediction standard deviation by storing the predictions of each decision tree
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

            # compile absolute value of errors
            errors = np.abs(labels_test - predictions)
            errors = np.asarray(errors)

            error_list = []

            # store other labels for test set
            task_ids = np.asarray(self.spectra_df_no_oxides.iloc[labels_test.index]['mpid_string'])
            stable = np.asarray(self.spectra_df_no_oxides.iloc[labels_test.index]['is_stable'])
            theoretical = np.asarray(self.spectra_df_no_oxides.iloc[labels_test.index]['is_theoretical'])

            # build test set df with predictions and other statistics
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

        # label attributes
        self.rf_error_df = error_df
        self.rf_model = rf_model
        self.rf_training_set = [updated_spectra_train, labels_train]
        self.rf_test_set = [updated_spectra_test, labels_test]




def flatten(list1):
    return [item for sublist in list1 for item in sublist]
def scale_spectra_flex(df_w_spectra, zero_energy='default', energy_col='Energies',
                       intensity_col='Spectrum', broadened_col=None, output_col_energy='Scaled Energy (eV)',
                       output_col_intensity='Scaled Intensity', show_plot = False, savefigure = False):
    """
    Scale a set of spectra so they all start and end at the same point and have a constant energy resolution
    :param df_w_spectra: dataframe with the unscaled spectra (pandas dataframe)
    :param zero_energy: energy value for each spectrum to start at. Must be below the lowest energy start for all
    the spectra in the dataframe (int)
    :param energy_col: column of the dataframe containing the energy axis (string)
    :param intensity_col: column of the dataframe containing the intensities (string)
    :param broadened_col: column of the dataframe containing broadened intensities (if applicable) (string or None)
    :param output_col_energy: the name to give the scaled energy column created by this function (string)
    :param output_col_intensity: the name to give the scaled intensity column created by this function (string)
    :param show_plot: whether to show plots illustrating the scaling process (bool)
    :return: dataframe updated with scaled spectra columns
    """
    # define accumulators to hold new spectra/energies
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
    # make sure that the maximum value of the interpolation is below the maximum value of the lowest energy range
    # spectrum
    interp_max = round(min(maxes) - 0.05, 1)
    for i in range(0, len(df_w_spectra)):
        full_energies = []
        full_intens = []

        energy = df_w_spectra.iloc[i][energy_col]
        intensity = df_w_spectra.iloc[i][intensity_col]
        if broadened_col != None:
            broadened_intensity = df_w_spectra.iloc[i]['Corrected Broadened Intensities exp alignment']
        interp_min = round(min(energy) + 0.05, 1)

        # interpolate spectrum to have new energy axis and start/stop point
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
            plt.title('Baseline Illustration', fontsize=28)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlabel('Energy (eV)', fontsize=26)
            plt.ylabel('Intensity', fontsize=26)
            # plt.plot(interp_energies_final, interped_intens, label = 'interpolated')
            # plt.legend(fontsize = 16)
            # plt.show()
            print('interp min = ' + str(interp_min))

        if zero_energy == 'default':
            zero_energy = interp_min - 1

        # set the start of the spectrum to have zero intensity at the pre determined start energy
        x = [zero_energy, interp_energies_final[
            0]]  # two given datapoints to which the exponential function with power pw should fit
        y = [10 ** -10, interped_intens[0]]

        if broadened_col != None:
            y_broadened = [10 ** -10, interped_broadened_intens[0]]

        # fit a function with a 6th order polynomial to bring the true first value of each spectrum down to zero
        # smoothly at the pre determined start point
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
            # print(ys - min(ys))
            # print(xf)
            # print(interp_energies)
            if savefigure:
                plt.savefig('Scaled Baseline Example.pdf', bbox_inches='tight',
                            transparent=True)
            plt.show()

        interp_energies_rounded = [round(num, 1) for num in interp_energies_final]
        extrapolated_energies_rounded = [round(num, 1) for num in xf]

        if np.isnan(ys[0]):
            ys = np.zeros((len(ys)))
        full_energies = list(extrapolated_energies_rounded) + list(interp_energies_rounded)
        full_intens = list(ys) + list(interped_intens)
        full_intens = full_intens - min(full_intens)
        # make sure first value is actually zero!
        if min(full_intens) > 10 ** -13:
            print('fail')

        if broadened_col != None:
            full_intens_broad = list(ys_broad) + list(interped_broadened_intens)
            full_intens_broad = full_intens_broad - min(full_intens_broad)
            if min(full_intens_broad) > 10 ** -13:
                print('fail')

        spectra_energies_scaled.append(full_energies)
        spectra_intensities_scaled.append(full_intens)
        if broadened_col != None:
            spectra_broadened_scaled.append(full_intens_broad)

    # update inputted dataframe with new columns
    df_w_spectra[output_col_energy] = spectra_energies_scaled
    df_w_spectra[output_col_intensity] = spectra_intensities_scaled
    if broadened_col != None:
        df_w_spectra['Aligned Scaled Broadened'] = spectra_broadened_scaled

    return df_w_spectra

def build_L2_3(l3, l2, show_plot=True, savefigure = False):

    """
    Builds and L2,3 spectrum from inputted L2 and L3 spectra
    :param l3: L3 spectrum (ndarray)
    :param l2: L2 spectrum (ndarray)
    :param show_plot: whether to show the generated L2,3 spectrum and component spectra
    :return: The L2,3 spectrum, list of ndarray first entery energies, second spectrum
    """
    # interpolate L2 and L3 spectra so they start at the same point and are on a common energy axis
    interp_min_L3 = round(min(l3.T[0]) + 0.15, 1)
    interp_max_L3 = round(max(l3.T[0]) - 0.15, 1)
    L3_energies = np.arange(interp_min_L3, interp_max_L3, 0.1)

    interp_min_L2 = round(min(l2.T[0]) + 0.15, 1)
    interp_max_L2 = round(max(l2.T[0]) - 0.15, 1)
    L2_energies = np.arange(interp_min_L2, interp_max_L2, 0.1)

    # plt.vlines(l3_fermi, 0,1.5, color = 'green')
    f_l2 = interpolate.interp1d(l2.T[0], l2.T[1])
    f_l3 = interpolate.interp1d(l3.T[0], l3.T[1])

    interped_l3 = f_l3(L3_energies)
    interped_l2 = f_l2(L2_energies)


    # scale L2 spectrum so it has a zero at the same starting point as the L3. Generate a set of points that make this
    # added energy region zero in the L2 spectrum
    zero_energy = interp_min_L3
    x = [zero_energy, L2_energies[0]]
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

    # generate L2,3 spectrum by simple summation of the L2 and L3, now that they have the same start and stop point and
    # a common energy axis
    L2_3 = interped_l3 + full_intens[0:len(L3_energies)]
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(L3_energies, interped_l3, label='L3', linewidth = 3)
        plt.plot(full_energies_final[0:len(L3_energies)], full_intens[0:len(L3_energies)], label='L2', linewidth = 3)
        plt.plot(L3_energies, L2_3, label='L2,3', linewidth = 3)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(fontsize=18)
        plt.xlabel('Energy (eV)', fontsize = 26)
        plt.ylabel('Intensity', fontsize = 26)
        plt.title('L$_2$, L$_3$, L$_2$,$_3$ For mp-30 (Cu(0))', fontsize=28)
        if savefigure:
            plt.savefig('L2,3 example.pdf', bbox_inches='tight',
                        transparent=True)
        plt.show()

    L3_energies_rounded = []
    for i in L3_energies:
        L3_energies_rounded.append(round(i, 1))

    return [L3_energies_rounded, L2_3]


def visualize_full_noise_test_set(noise_dfs, interp_ranges, show_err = True, savefigure=False):

    """
    Visualization function for the data generated by a simulated noise analysis on the simulated test spectra
    :param noise_dfs: dataframes containing test set accuracies for different noise levels and random states (pandas
    dataframe)
    :param interp_ranges: test spectra energy axes present in the dataframe (float/list of float)
    :param show_err: whether to show error bars indicating the standard deviation across different random states
    :param savefigure: whether to save the plot as a pdf
    :return: none plot is generated
    """

    if type(noise_dfs) != list:
        noise_dfs = [noise_dfs]
    if type(interp_ranges) != list:
        interp_ranges = [interp_ranges]
    for vis in ['R2', 'RMSE']:
        count = -1
        plt.figure(figsize=(8,7))
        # generate means and standard deviations from different noise levels
        for noise_df in noise_dfs:
            count += 1
            mean_01 = np.mean(np.asarray(noise_df.loc[noise_df['noise_std'] == 1000][vis]))
            mean_05 = np.mean(np.asarray(noise_df.loc[noise_df['noise_std'] == 200][vis]))
            mean_1 = np.mean(np.asarray(noise_df.loc[noise_df['noise_std'] == 100][vis]))
            mean_2 = np.mean(np.asarray(noise_df.loc[noise_df['noise_std'] == 50][vis]))

            std_01 = np.std(np.asarray(noise_df.loc[noise_df['noise_std'] == 1000][vis]))
            std_05 = np.std(np.asarray(noise_df.loc[noise_df['noise_std'] == 200][vis]))
            std_1 = np.std(np.asarray(noise_df.loc[noise_df['noise_std'] == 100][vis]))
            std_2 = np.std(np.asarray(noise_df.loc[noise_df['noise_std'] == 50][vis]))


            if vis == 'R2':
                # plt.title('R2 vs Noise', fontsize = 36)
                plt.xlabel('Noise STD', fontsize = 36)
                plt.ylabel('R$^2$', fontsize = 36)
                plt.xticks([0,0.1, 0.2], fontsize = 36)
                plt.yticks([0.3,0.6,0.9], fontsize = 36)
                plt.ylim([0.28, 0.95])
                plt.xlim([-0.01, 0.225])
                print('R2s')
                print(mean_01, mean_05, mean_1, mean_2)
                plt.scatter([0, 0.01, 0.05, 0.1, 0.2], [0.9, mean_01, mean_05, mean_1, mean_2], color = 'k', s=200, zorder=5)

                if show_err:
                    plt.plot([0, 0.01, 0.05, 0.1, 0.2], [0.9, mean_01, mean_05, mean_1, mean_2], color = 'k')
                    eb1 = plt.errorbar([0, 0.01, 0.05, 0.1, 0.2], [0.9, mean_01, mean_05, mean_1, mean_2], yerr=[0, std_01, std_05, std_1, std_2],
                                 ecolor='k', errorevery=1, capsize=15, linewidth = 4, label = str(interp_ranges[count]))
                    eb1[-1][0].set_linestyle(':')
                    if savefigure:
                        plt.savefig('R2 Noise Profile '+str(interp_ranges[count])+'.pdf',  bbox_inches='tight', transparent=True)

                else:
                    plt.plot([0, 0.01, 0.05, 0.1, 0.2], [0.9, mean_01, mean_05, mean_1, mean_2],
                             linewidth = 4, label = str(interp_ranges[count]))

            if vis == 'RMSE':
                # plt.title('RMSE vs Noise', fontsize = 36)
                plt.xlabel('Noise STD', fontsize = 36)
                plt.ylabel('RMSE', fontsize = 36)
                plt.xticks([0,0.1, 0.2], fontsize = 36)
                plt.yticks([0.2,0.35,0.5], fontsize = 36)
                plt.ylim([0.18, 0.525])
                plt.xlim([-0.01, 0.225])
                print('RMSEs')
                print(mean_01, mean_05, mean_1, mean_2)
                plt.scatter([0, 0.01, 0.05, 0.1, 0.2], [0.2, mean_01, mean_05, mean_1, mean_2], color = 'k', s=200, zorder=5)

                if show_err:
                    plt.plot([0, 0.01, 0.05, 0.1, 0.2], [0.2, mean_01, mean_05, mean_1, mean_2], color = 'k')
                    eb1 = plt.errorbar([0, 0.01, 0.05, 0.1, 0.2], [0.2, mean_01, mean_05, mean_1, mean_2], yerr=[0, std_01, std_05, std_1, std_2],
                                 ecolor='k', errorevery=1, capsize=15, linewidth = 4, label = str(interp_ranges[count]))
                    eb1[-1][0].set_linestyle(':')
                    if savefigure:
                        plt.savefig('RMSE Noise Profile '+str(interp_ranges[count])+'.pdf',  bbox_inches='tight', transparent=True)

                else:
                    plt.plot([0, 0.01, 0.05, 0.1, 0.2], [0.2, mean_01, mean_05, mean_1, mean_2],
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
    """
    Augment a set of data with poisson noise
    :param x: list of points to be augmented with noise (list/ndarray)
    :param std: noise standard deviation (the noise is divided by this parameter, so a higher value is less noisy data)
    (float)
    :param random_state: random seed used for this noise profile (int)
    :return: noisy spectrum (ndarray)
    """
    np.random.seed(random_state)

    noise = np.random.poisson(100, len(x))-100
    noise = noise/std

    x_noisy = np.asarray(x) + np.asarray(noise)

    return x_noisy

def gaussian_noise(x, mu, std, random_state=32):
    """
    Augment a set of data with gaussian noise

    :param x: int/float - value to be augmented with gaussian noise
    :param mu: average of the gaussian (float)
    :param std: standard deviation of the gaussian  (float)
    :param random_state: random seed used for this noise profile (int)
    :return: noisy spectrum (ndarray)
    """
    np.random.seed(random_state)

    noise = []
    for entry in x:
        noise.append(np.random.normal(mu, std, size=1)[0])

    x_noisy = np.asarray(x) + np.asarray(noise)

    return x_noisy

def spectrum(E,osc,sigma,x):
    """
    Generates a spectrum broadened by gaussian broadening
    :param E: energy vales for the spectrum (list/ndarray)
    :param osc: intensity values for the spectrum (list/ndarray)
    :param sigma: gausian distribution standard deviation (float)
    :param x: energy vales for the spectrum (list/ndarray)
    :return: broadened spectrum (list)
    """
    gE=[]
    for Ei in x:
        tot=0
        for Ej,os in zip(E,osc):
            tot+=os*np.exp(-((((Ej-Ei)/sigma)**2)))
        gE.append(tot)
    return gE
