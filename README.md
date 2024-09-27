# ML_XAS_EELS


The code in this repo accompanies the paper:
Gleason, S.P., Lu, D. & Ciston, J. Prediction of the Cu oxidation state from EELS and XAS spectra using supervised machine learning. npj Comput Mater 10, 221 (2024).
https://www.nature.com/articles/s41524-024-01408-1

The "Paper_Figures.ipynb" notebook reproduces all the plots that comprise the 6 main text figures and 16 SI figures in the paper. Headers designate which figures are being produced by the code in that region.

"Analysis_objects_and_functions.py" contains an object, called eels_rf_setup, which runs most of the code discussed in this work. A few additional visualization functions are included at the end.

Of the six directories, three contain spectra analyzed by this work (Cu_deconvolved_spectra, Additional_Literature_Spectra and xas paper). The fourth, Figures, contains the saved plots from "Paper_Figures.ipynb" and the final figure files. The fifth, Dataset_generation, contains the scripts used to generate the data used by this work. The sixth, vizualization files, contains a few small files used in "Paper_Figures.ipynb". The project data files can be found at https://drive.google.com/drive/folders/1hsffSo7_6LB5TfsfH9-85lvbodeZTDUC?usp=sharing

In particular, "Cu_reproducable_alignment_df_extracted_110222.joblib" is a dataframe containing the simulated XAS spectra and is needed to run "Paper_Figures.ipynb" 

Relevant versions: python version 3.10.12 pandas version 1.5.3 numpy version 1.26.2 sklearn version 1.0.2 matplotlib version 3.7.1 joblib version 1.1.0. A .yml file is also included and can be used to generate an environment for running this repo. 
