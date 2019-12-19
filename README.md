# Final-Thesis-Repository
This repository contains all Python scripts and Jupyter Notebooks that were used in the final stages of my undergraduate thesis. 

To summarise, the thesis involved the analysis of Stage N2 portions of 8 hour EEG recordings across 15 patients in order to extract 'sleep spindles'. 
The 'Quadratic Parameter Sinusoid' or QPS (Palliyali et. al. 2015) was used as a way to extract 6 quadratic polynomial coefficients that served as statistical 
descriptors of features of the extracted spindles (and non-spindles) such as their amplitudes, envelope symmetry, frequency, phase and more. 
The way this was achieved was using non-linear least squares (NLLS) via the Levenberg-Marquadt Algorithm (LM) as a way to perform a best fit of the model to the raw captured spindle. 
The equation for the QPS model is defined below:

```s(t) = exp(a + bt + ct^2) * cos(d + et+ ft^2) ```

The coefficients are:

- a: Controls the log-amplitude of the synthetic spindle
- b: Controls when the onset occurs. Generates a time-shift to the instance where the highest amplitude occurs.
- c: Affects the symmetry of the Gaussian envelope
- d: Phase of the carrier component
- e: Angular frequency of the carrier component
- f: Linear perturbation about the centre angular frequency, e.

The main goal of the thesis was to use these 6 parameters as learning features in a simple feed-forward neural network in order to classify whether or not an acquired raw portion of an EEG signal is a spindle or not. 
The conclusion to the study showed that while the QPS model was a great way to reconstruct spindles and extract valuable coefficient data, there is no guarantee the non-linear regression will work since parameter 
initialisation is highly dependent on whether or not it is known (for certain) if the acquired raw section of the EEG is a spindle or not.

This project is no longer actively pursued but is uploaded here as a record of my thesis project. The raw PSG files used (DREAMS and MASS) are
NOT uploaded due to their large size and because the files are restricted to public use unless authorisation is granted. More information on the 
MASS database can be read in the website linked below.

http://massdb.herokuapp.com/en/

