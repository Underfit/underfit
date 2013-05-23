import numpy as np
from numpy import random


#This function gets a sample_size'd observation from the K-dimensional model 
#It generates observations one by one and then counts the observations that occured in each category
def get_obs(model, sample_size, K):
    log = []
    for i in xrange(sample_size):
        rv = random.uniform(0.0,1.0)
        for h in xrange(K):
            rv -= model[h]
            if rv < 0:
                log.append(h)                
                break
    obs = []    
    for i in xrange(K):
        obs.append(0)
    
    for o in log:
        obs[o] += 1
    return obs

#This function returns the standardized entropy (between 0 and 1) of the passed observation vector
def entropy(obs, K):
    h = 0.0
    N = sum(obs)
    for i in xrange(K):
        p = np.float(obs[i])/N
        if p != 0:
            h += - p*np.log(p)/np.log(K)
    return h
 
    
import cPickle

#Load the models from the .pkl file
fmodels = open("models.pkl", "rb")
models = cPickle.load(fmodels)
fmodels.close()

#Constants for computational trial
maxK = 8
NUMTRIALS = 100000
numNBINS = 10
numHBINS = 10
HbinWidth = 1.0/numHBINS

#The following code generated NUMTRIALS observations for each K (2 to maxK),
#and it bins them according to the sample size and entropy, before putting them into matrices
#l will be an k by numNBINS by numHBINS list with entries of the form [obs, verse],
#where obs and verse are matrices

l = []
for k in xrange(2,maxK + 1):
    l.append([])
    dN = np.power(2,k-1)
    print k
    for i in xrange(numNBINS):
        l[k-2].append([])
        nMin = 1 + i*dN
        nMax = 1 + (1 + i)*dN

        bins = []
        for x in xrange(numHBINS):
            bins.append([[],[]])                    
        for j in xrange(NUMTRIALS/numNBINS):
            
            N = random.randint(nMin, nMax)
            vnum = random.randint(0, models[k-2].shape[0])
            verse = models[k-2][vnum].tolist()[0]
            obs = get_obs(verse, N, k)
            h = entropy(obs, k)
            bins[np.int(0.999999*h/HbinWidth)][0].append(obs)            
            bins[np.int(0.999999*h/HbinWidth)][1].append(vnum)

        for x in xrange(numHBINS):
            mbins = [np.asmatrix(bins[x][0]),np.asarray(bins[x][1])]
            l[k-2][i].append(mbins)

#Finally pickling the result for later use
f = open("obs.pkl", "wb")
cPickle.dump(l, f)
f.close()