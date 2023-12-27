## README

Code relative to the experiments of the paper "A Generalization of the Convolution Theorem and its Connections to Non-Stationarity and the Graph Frequency Domain", by Alberto Natali and Geert Leus. 


Toolbox required: CVX, GSPBOX. Both free and third-party toolbox.

The repository contains the following files (we indicate with 'S' the files relative to the synthetic data, and with 'R' the ones relative to the real data. Files which are used for both are without any letter.):


- [S] **main_from_sh.m**: MATLAB file which can be run for debugging purposes and visualization of the learning process. It can optionally be launched by a bash script.

- [S] **avg_main_from_sh**.m: MATLAB file intended to be launched from a bash script with some of the parameters initialized. It does not display plots and messages, thus it is not for debugging.


- [S] **visualization.m**: MATLAB file to load the (synthetic) numerical results and display/save the plots of interest. 


- [R] **real_data.m**: script just used for debugging the theoretical part
- [R] **mnist.m**: script to run the MNIST section
- [R] **traffic_volume.m**: script to run the Traffic Volume section

- **utils**: folder with MATLAB functions needed for the main code execution.
