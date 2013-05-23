import numpy as np
import theano 
from theano import tensor as T

#Global variable for making log(0) = -INF
INF = 10000

#This function returns multinomial coefficients
#Obs is num_Obs by K
#Output is num_Obs by 1
def multinomial_coefficient(Obs, K, num_Obs):
    Ns_p1 = T.dot(Obs,T.ones((K,1))) + T.ones((num_Obs,1))   
    Obs_p1 = Obs + T.ones((num_Obs, K))
    lnmlti = T.gammaln(Ns_p1) - T.dot(T.gammaln(Obs_p1),T.ones((K,1)))
    return T.exp(lnmlti)

#This function returns the multinomial likelihood for each model in M on each observation set in Obs
#M is num_M by K
#Obs is num_Obs by K
#Output is num_M by num_Obs
def multinomial_probabilities(M, Obs, K, num_M, num_Obs):
    #Fixing zeros
    mZEROs = T.eq(M, 0)
    lnM = T.log(M + mZEROs) - INF*mZEROs
        
    a = T.dot(T.ones((num_M,1)),multinomial_coefficient(Obs,K,num_Obs).T)
    b = T.exp(T.dot(lnM,Obs.T))
    
    probs =a*b

    return probs
    
    
     
'''
Demonstrating functionality:
''

from generate_models import *
l = generate_models(2,0.1)
num_M = len(l)
print np.asarray(l)
M = theano.shared(np.asarray(l))
Obs = theano.shared(np.asarray([[1,2],[3,2],[0,2]]))
num_Obs = 3
K = 2

fM = theano.function([],multinomial_probabilities(M, Obs, K, num_M, num_Obs))
print np.round(fM(),2)
#'''