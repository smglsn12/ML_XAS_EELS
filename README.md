# ML_XAS_EELS


The code in this repo accompanies "Prediction of the Cu oxidation state from EELS and XAS spectra using supervised machine learning."

The "Paper_Figures.ipynb" notebook reproduces all the plots that comprise the 6 main text figures and 16 SI figures in the paper. Headers designate which figures are being produced by the code in that region.

"Analysis_objects_and_functions.py" contains an object, called eels_rf_setup, which runs most of the code discussed in this work. A few additional visualization functions are included at the end.

The remaining files in this repo are the small datasets included for visualization purposes.

Of the five directories, three contain spectra analyzed by this work (Cu_deconvolved_spectra, Additional_Literature_Spectra and xas paper), the fourth contains the saved plots from "Paper_Figures.ipynb" and the final figure files, and the fifth contains the scripts used to generate the data used by this work. The data files can be found at https://drive.google.com/drive/folders/1hsffSo7_6LB5TfsfH9-85lvbodeZTDUC?usp=sharing

Relevant versions: python version 3.10.12 pandas version 1.5.3 numpy version 1.24.3 sklearn version 1.0.2 matplotlib version 3.7.1 joblib version 1.1.0
