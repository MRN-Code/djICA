import json;
import argparse
from os import listdir
from os.path import isfile, join
import sys
import numpy as np

################### helper functions ###################

def mySigmoid(X):
    tmp = 1 + np.exp(-X)
    return np.divide(1, tmp)

########################################################

parser = argparse.ArgumentParser(description='help read in my data from my local machine!')
parser.add_argument('--run', type=str,  help='grab coinstac args')
args = parser.parse_args()
args.run = json.loads(args.run)

username = args.run['username']

# inspect what args were passed
# runInputs = json.dumps(args.run, sort_keys=True, indent=4, separators=(',', ': '))
# sys.stderr.write(runInputs + "\n")

if 'remoteResult' in args.run and \
    'data' in args.run['remoteResult'] and \
    username in args.run['remoteResult']['data']:
    sys.exit(0); # no-op!  we already contributed our data

passedDir = args.run['userData']['dirs'][0]
sys.stderr.write("reading files from dir: " + passedDir)

files = [f for f in listdir(passedDir) if isfile(join(passedDir, f))]


if ('data' in args.run['remoteResult']) and ('U' in args.run['remoteResult']['data']):
    # subsequent iteration - project data samples on the consensus subspace and take a gradient step
    for f in files:
        if not(f == '.DS_Store'):
            # load data
            X = np.load(join(passedDir,f))
            D, N = X.shape
            block = (N/20) ** (0.5)
            ones = np.ones([N, 1])
            
            # project data on the sonsensus subspace
            U = np.array(args.run['remoteResult']['data']['U'])
            Xred = np.dot(U.T, X)
            BI = block * np.eye(Xred.shape[0])
            
            # take grdient step
            W = np.array(args.run['remoteResult']['data']['W'])
            b = np.array(args.run['remoteResult']['data']['b'])
            rho = np.array(args.run['remoteResult']['data']['rho'])
            
            Z = np.dot(W, Xred) + np.dot(b, ones.T)
            Y = mySigmoid(Z)
            G = rho * np.dot(BI + np.dot(1 - 2*Y, Z.T), W)
            h = rho * np.sum(1 - 2*Y, axis = 1)
            h = h.reshape([Xred.shape[0], 1])
    
    sys.stderr.write("\nSending gradient values to central ...")    
#    computationOutput = json.dumps({'G' : G.tolist(), 'h' : h.tolist(), 'PCA_complete' : True, 'Grad_complete': True, 'rho' : rho, 'W' : W.tolist(), 'b' : b.tolist()}, sort_keys=True, indent=4, separators=(',', ': '))
    computationOutput = json.dumps({'G' : G.tolist(), 'h' : h.tolist(), 'PCA_complete' : True, 'Grad_complete': True}, sort_keys=True, indent=4, separators=(',', ': '))
    
    # send results
    sys.stdout.write(computationOutput)
        
        
else:    
    # very first iteration - doing local PCA only
    for f in files:
        if not(f == '.DS_Store'):
            X = np.load(join(passedDir,f))
            d, n = X.shape
            K = 4
            C = (1.0 / n) * np.dot(X, X.T)
            U, S, V = np.linalg.svd(C)
            Uk = U[:, :K]
            Sk = np.diag(S)[:K, :K]
            P = np.dot(Uk, np.sqrt(Sk))
            en = np.trace(np.dot(Uk.T, np.dot(C, Uk)))

    sys.stderr.write("Sending PCA results to central ...")    
    computationOutput = json.dumps({'P': P.tolist(), 'en': en, 'C': C.tolist(), 'PCA_complete' : True, 'Grad_complete': False}, sort_keys=True, indent=4, separators=(',', ': '))
    
    # send results
    sys.stdout.write(computationOutput)



