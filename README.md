# djICA
This repository contains the distributed joint ICA code written for the old coinstac simulator (v2.3). It contains the following files:
1. djica_local.py - for computing the local PCA and gradient on local data on local data and sending the proxy data matrix to the master.
2. djica_master.py - for aggregation of the proxy data matrix and gradient sent by local sites and releasing the inverse mixing matrix W
3. computation.js - computation specification JavaScript file
4. declaration.js - declaration file specifying the local site names,  JavaScript file

The code runs fine on the simulator, but it fails to converge. Needs debugging why the gradient descent is not converging.
