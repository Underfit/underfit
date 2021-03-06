import numpy as np
import theano 
import theano.tensor as T
try:
    from theano.sandbox.linalg.kron import kron
except:
    try:
        from theano.toolbox import kron
    except:
        kron = theano.numpy.kron

from scipy import stats

INF = 10000

from theanoChi2pv_proper import chi2pv

#This function returns a num_M by num_Obs matrix of chi^2 test statistics 
#M is num_M by K
#Obs is num_Obs by K
def chi2_test_statistic(M, Obs, K, num_M, num_Obs):
    #Getting frequencies from observations
    Ns = T.dot(Obs,T.ones((K,1)))
    p = Obs/Ns
        
    #Find the zeros so we can deal with them later
    pZEROs = T.eq(p, 0)
    mZEROs = T.eq(M, 0)
    
    #log probabilities, with -INF as log(0)
    lnM = T.log(M + mZEROs) - INF*mZEROs
    lnp = T.log(p + pZEROs) - INF*pZEROs


    #Using kroneker products so every row of M hits every row of P in the difference klnM - kln
    O_ones = T.ones((num_Obs,1))
    M_ones = T.ones((num_M,1))
    klnM = kron(lnM,O_ones)
    klnP = kron(M_ones, lnp)
    klnP_M = klnP - klnM
    kObs = kron(M_ones, Obs)
    
    G = 2.0*T.dot(klnP_M ,kObs.T)
    
    G = G*T.identity_like(G)
    G = T.dot(G,T.ones((num_M*num_Obs,1)))   
    G = T.reshape(G,(num_M,num_Obs))
    
    #The following quotient improves the convergence to chi^2 by an order of magnitude
    #source: http://en.wikipedia.org/wiki/Multinomial_test
    
    #numerator = T.dot(- 1.0/(M + 0.01),T.ones((K,1))) - T.ones((num_M,1))    
    #q1 = T.ones((num_M,num_Obs)) + T.dot(numerator,1.0/Ns.T/6.0)/(K-1.0)
        
    return G#/q1 

#This function returns a num_M by num_Obs pvalue matrix
#M is num_M by K
#Obs is num_Obs by K
def chi2_pvalues(M, Obs, K, num_M, num_Obs):
    
    X = chi2_test_statistic(M, Obs, K, num_M, num_Obs)

    Pvs = chi2pv(X, K-1) #Elemwise(Chi2PV())(X, K-1)

    return Pvs
    
'''
Demonstrating functionality:
''

from generate_models import *
import cPickle as pickle


f=open('../models.pkl')
l = pickle.load(f)
f.close()

print 'got models'

l = l[0]

num_M = len(l)
print num_M
M = T.matrix('M')
Obs = T.matrix('Obs')

#M = theano.shared(np.asarray(l))
#Obs = theano.shared(np.asarray([[1,2],[3,2],[0,2]]))

num_Obs = 3
K = 2

print 'calling function'
fM = theano.function([M, Obs],chi2_pvalues(M, Obs, K, num_M, num_Obs))
models = l
print models
obs = np.asarray([[1,2],[3,2],[0,2]])
print obs

pvs = fM(models, obs)
print np.round(pvs, 2)
print pvs.shape
#print stats.chi2.sf(x, K-1)
#print np.round(fM(models, obs),2)
#'''
