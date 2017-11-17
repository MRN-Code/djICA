import json;
import argparse
from os import listdir
from os.path import isfile, join
import sys
import numpy as np


parser = argparse.ArgumentParser(description='read beta vector from single site and do beta averaging!')
parser.add_argument('--run', type=str,  help='grab coinstac args')
args = parser.parse_args()
args.run = json.loads(args.run)

#inspect what args were passed
#runInputs = json.dumps(args.run, sort_keys=True, indent=4, separators=(',', ': '))
#sys.stderr.write(runInputs + "\n")

#if 'remoteResult' in args.run and \
#    'data' in args.run['remoteResult'] and \
#    username in args.run['remoteResult']['data']:
#    sys.exit(0); # no-op!  we already contributed our data



user_results = args.run['userResults']

n_site = len(user_results)
K = 2

flagPCA = True # flag for Global PCA
flagICA = True # flag for Global ICA
for i in range(0, n_site):
    if 'data' in user_results[i] and user_results[i]['data']['PCA_complete'] == False:
        flagPCA = False
        
    if 'data' in user_results[i] and user_results[i]['data']['Grad_complete'] == False:
        flagICA = False
        

if flagPCA and not(flagICA): # perform global PCA; all sites sent the corresponding P matrix
    en_sites_true = []
    C = 0
    C_central = 0
    for i in range(0, n_site):
        en_sites_true.append(user_results[i]['data']['en'])
        Ps = np.array(user_results[i]['data']['P'])
        C_central += np.dot(Ps, Ps.T)
        C += np.array(user_results[i]['data']['C'])
    
    # consensus subspace    
    Ua, Sa, Va = np.linalg.svd(C_central)
    Uak = Ua[:, :K]
    var_cons = np.trace(np.dot(Uak.T, np.dot(C, Uak)))
    var_true = sum(en_sites_true)
    
    sys.stderr.write("Done with PCA! consensus subspace energy : {}".format(var_cons)+"\n")
    sys.stderr.write("Done with PCA! true subspace energy : {}".format(var_true)+"\n")
    
    # initialize W and b
    W = np.eye(K)
    b = np.zeros([K, 1])
    rho = 0.015 / np.log(Ps.shape[1]) 
    itr = 0
    
    # send these values to local sites
    sys.stderr.write("Sending initial values to local sites ...")    
    computationOutput = json.dumps({'W' : W.tolist(), 'U' : Uak.tolist(), 'b' : b.tolist(), 'rho' : rho, 'itr' : itr, 'var_cons': var_cons, 'var_true': var_true}, sort_keys=True, indent=4, separators=(',', ': '))
    
    # send results
    sys.stdout.write(computationOutput)
    
if flagICA: # perform global ICA step; all sites sent the corresponding matrices
    sys.stderr.write("djICA process ")    
    maxItr = 10
    thr = 1e8
    if 'previousData' in args.run and 'itr' in args.run['previousData']:
        itr = args.run['previousData']['itr']
        W = np.array(args.run['previousData']['W'])
        b = np.array(args.run['previousData']['b'])
        U = np.array(args.run['previousData']['U'])
        rho = args.run['previousData']['rho']
    
    gradSum = 0.0
    biasGradSum = 0.0    
    
    for i in range(0, n_site):
        gradSum += np.array(user_results[i]['data']['G'])
        biasGradSum += np.array(user_results[i]['data']['h'])

    sys.stderr.write("Iteration : {}".format(itr)+"\n")
#    sys.stderr.write("Type of biasGradSum: {}".format(type(biasGradSum))+"\n")
#    sys.stderr.write("Type of b: {}".format(type(b))+"\n")
    
    if itr <= maxItr:
        sys.stderr.write("\n Updating W")    
        W = np.add(W, gradSum, casting='unsafe')
        sys.stderr.write("\n Updating b")    
        b = np.add(b, biasGradSum, casting='unsafe')
        itr += 1

        # check blowout and update rho if needed
        if np.max(np.abs(W)) >= thr:
            sys.stderr.write("\n Blowout detected. Restarting...")    
            rho = rho * 0.8
            # initialize W and b again
            W = np.eye(K)
            b = np.zeros([K, 1])
            
        # send these values to local sites
        computationOutput = json.dumps({'complete': False, 'W' : W.tolist(), 'U' : U.tolist(), 'b' : b.tolist(), 'rho' : rho, 'itr' : itr}, sort_keys=True, indent=4, separators=(',', ': '))
    
        # send results
        sys.stdout.write(computationOutput)
    else:
        res_file = 'mixing_matrix.npz'
        A = np.load(res_file)['arr_0']
        Ahat = np.linalg.pinv(np.dot(W, U.T))
        err = np.linalg.norm(A - Ahat, 'fro')
        sys.stderr.write("Done with djICA! Error : {}".format(err)+"\n")
        # send these values to local sites
        computationOutput = json.dumps({'complete': True, 'W' : W.tolist(), 'U' : U.tolist(), 'b' : b.tolist(), 'rho' : rho, 'itr' : itr}, sort_keys=True, indent=4, separators=(',', ': '))
    
        # send results
        sys.stdout.write(computationOutput)

    
