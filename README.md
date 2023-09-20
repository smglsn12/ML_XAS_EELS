# ML_XAS_EELS


The code in this repo accompanies "Supervised Machine Learning Prediction of the Cu Oxidation State from EELS Spectra." 

The "Paper_Figures.ipynb" notebook reproduces all the plots that comprise the 6 main text figures and 10 SI figures in the paper. Headers designate which figures are being produced by the code in that region.

"Analysis_objects_and_functions.py" contains an object, called eels_rf_setup, which runs most of the code discussed in this work. A few additional visualization functions are included at the end. 

The remaining files in this repo are the small datasets included for visualization purposes. 

Of the three directories, two contain spectra analyzed by this work (Cu_deconvolved_spectra and xas paper) and the third contains the saved plots from "Paper_Figures.ipynb" and the final figure files 

Relevant versions:
python version 3.10.12 
pandas version 1.5.3
numpy version 1.24.3
sklearn version 1.0.2
matplotlib version 3.7.1
joblib version 1.1.0
