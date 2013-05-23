import numpy as np
from scipy.special import gammaln
import theano 

INF = 1000.

def multinomial_coefficient(Obs, K, num_Obs):
    Ns_p1 = np.dot(Obs,np.ones((K,1))) + np.ones((num_Obs,1))   
    Obs_p1 = Obs + np.ones((num_Obs,K))
    lnmlti = gammaln(Ns_p1) - np.dot(gammaln(Obs_p1),np.ones((K,1)))
    return np.exp(lnmlti)

def multinomial_probabilities(M, Obs, K, num_M, num_Obs):
    #Find the zeros so we can use log probabilities, with -INF as log(0)
    mZEROs = np.cast['float32'](M == 0.0)
    lnM = np.log(M + mZEROs) - INF*mZEROs
    probs = np.dot(np.ones((num_M,1)),multinomial_coefficient(Obs,K,num_Obs).T)*np.exp(np.dot(lnM,Obs.T))
    return probs

def direct_pvalues(M, Obs, K, num_M, num_Obs):    
    Probs = multinomial_probabilities(M,Obs,K, num_M, num_Obs)
    Ns = np.dot(Obs,np.ones((K,1)))       
    pvs = np.zeros((num_M,num_Obs), dtype=theano.config.floatX)
    
    for i in xrange(num_M):
        for j in xrange(num_Obs):
            pvs[i,j]  = _pvalue(Probs[i,j], M[i], Ns[j], K)
             
    return pvs

def _pvalue(prob, model, N, K):
    pval = [0]
    l = []
    recursive_pvalue(l,pval, prob, model, int(N[0]), K, K)
    return np.cast[theano.config.floatX](pval[0])


def recursive_pvalue(l,pvalue, prob, model, N, K, catsleft):
    if catsleft == 1:
        l.append(N)
        pr = multinomial_probabilities(model, np.asmatrix(l), K, 1,1)
        l.pop()        
        if (pr <= prob) :
            pvalue[0] += pr 
        
    else:
        for i in xrange(N+1): # i can be zero 
            l.append(i)
            recursive_pvalue(l, pvalue, prob, model, N - i, K, catsleft - 1)
            l.pop()
        
    
'''
Demonstrating functionality:
''
import cPickle as pickle
K = 2
f = open('../models.pkl', 'rb')
l = pickle.load(f)[0]
f.close()
num_M = len(l)

M = np.asarray(l)
Obs = np.asarray([[1,2],[3,2],[0,2]])
num_Obs = 3


f = direct_pvalues(M, Obs, K, num_M, num_Obs)
print np.around(f, 2)
#'''
